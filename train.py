"""Train relational distillation from FOVNet B-rep faces to mesh-based students.

Supports two student backbones:
  diffusion_net  — DiffusionNet on triangle meshes (default, original B2Mesh)
  pd_meshnet     — Primal-Dual MeshNet on triangle meshes
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import BRepMeshDataset, collate_brep_samples, collate_distill_samples
from .losses import feature_alignment_loss, relational_distillation_loss, scatter_mean_by_index
from .models import (
    DiffusionMeshStudent,
    FOVNetFaceTeacher,
    classification_stats,
    freeze_module,
    load_fovnet_teacher_checkpoint,
    mean_iou,
)
from .visualization import load_class_names, save_prediction_meshes, write_legend

# PD-MeshNet student (optional — loaded lazily so missing deps don't break diffusion_net mode)
try:
    from .PD_MeshNet import PDMeshNetStudent, collate_pd_samples
    from .PD_MeshNet.datasets import MFCADPlusPlusDualPrimal, Fusion360DualPrimal
    from .PD_MeshNet.utils import create_dual_primal_batch
    _HAS_PD_MESHNET = True
except Exception:  # pragma: no cover
    _HAS_PD_MESHNET = False

# PyG Data detection for device-movement
try:
    from torch_geometric.data import Data as _PyGData
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


DATASET_DEFAULTS = {
    "fusion360": {
        "data_root": "/data2/gwlee/fovnet/data/fusion360/s2.0.1",
        "brep_graph_root": None,
        "brep_seg_dir": "breps/seg",
        "split_file": "train_test_new.json",
        "mesh_dir_template": "meshes",
        "num_classes": 8,
        "op_cache_dir": "/data2/gwlee/diffusion-net/fusion360ver/outputs/op_cache_k128",
    },
    "mfcad++": {
        "data_root": "/data2/gwlee/fovnet/data/mfcad++",
        "brep_graph_root": "{data_root}",
        "brep_seg_dir": None,
        "split_file": "{split}.txt",
        "mesh_dir_template": "meshes/{split}",
        "num_classes": 25,
        "op_cache_dir": "/data2/gwlee/diffusion-net/mfcad_ver/outputs/op_cache_k128",
    },
}

# PD-MeshNet specific dataset settings (graph feature dims are fixed by GraphCreator)
_PD_IN_CHANNELS = {"primal": 1, "dual": 7}  # face-area + dihedral/local-index features
_PD_CACHE_DIRS = {
    "fusion360": "/data2/gwlee/fovnet/data/fusion360/s2.0.1/pd_cache",
    "mfcad++": "/data2/gwlee/fovnet/data/mfcad++/pd_cache",
}


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--student_type",
        choices=["diffusion_net", "pd_meshnet"],
        default="diffusion_net",
        help="Student backbone: diffusion_net (default) or pd_meshnet.",
    )
    parser.add_argument(
        "--no_teacher",
        action="store_true",
        help="Train student with mesh segmentation loss only — no FOVNet teacher, no distillation. "
             "--teacher_ckpt is not required and distill/align weights are ignored.",
    )
    parser.add_argument(
        "--seg_loss",
        choices=["focal", "ce"],
        default="ce",
        help="Segmentation loss for the student. "
             "'ce' (default) handles class imbalance; 'ce' matches original PD-MeshNet.",
    )
    # mfcad++
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="fusion360")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--brep_graph_root", default=None)
    parser.add_argument("--brep_seg_dir", default=None)
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--mesh_dir_template", default=None)
    parser.add_argument("--output_dir", default="B2Mesh/outputs")
    parser.add_argument(
        "--teacher_ckpt",
        default=None,
        help="Explicit FOVNet teacher checkpoint file. If omitted, the teacher is trained first.",
    )
    parser.add_argument("--teacher_epochs", type=int, default=50)
    parser.add_argument("--teacher_lr", type=float, default=5e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--teacher_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--teacher_device",
        default=None,
        help="Device for the DGL/FOVNet teacher. Defaults to --device; use cpu as a fallback if DGL CUDA is unstable.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--input_features", choices=["xyz", "hks"], default="xyz")
    parser.add_argument("--k_eig", type=int, default=128)
    parser.add_argument("--op_cache_dir", default=None)
    parser.add_argument(
        "--eigen_cache_dir",
        default=None,
        help="Alias for --op_cache_dir; directory for DiffusionNet cached eigen/operators.",
    )
    parser.add_argument("--no_op_cache", action="store_true")
    parser.add_argument("--num_classes", type=int, default=None)
    # DiffusionNet student args
    parser.add_argument("--student_width", type=int, default=128)
    parser.add_argument("--student_blocks", type=int, default=4)
    # PD-MeshNet student args
    parser.add_argument(
        "--pd_conv_channels",
        type=int,
        nargs="+",
        default=[64, 128],
        help="Primal output channels per PD-MeshNet encoder block.",
    )
    parser.add_argument(
        "--pd_cache_dir",
        default=None,
        help="Directory to cache PD-MeshNet primal/dual graphs. Defaults to dataset pd_cache dir.",
    )
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument(
        "--lr_scheduler",
        choices=["cosine", "step", "none"],
        default="cosine",
        help="LR scheduler for student (and teacher when trained from scratch).",
    )
    parser.add_argument("--lr_min", type=float, default=1e-5, help="Minimum LR for cosine scheduler.")
    parser.add_argument("--lr_step_size", type=int, default=10, help="StepLR: decay every N epochs.")
    parser.add_argument("--lr_gamma", type=float, default=0.5, help="StepLR: multiplicative decay factor.")
    parser.add_argument("--rkd_mode", choices=["distance", "angle", "distance_angle"], default="distance_angle")
    parser.add_argument("--distill_weight", type=float, default=1.0)
    parser.add_argument("--align_weight", type=float, default=0)
    parser.add_argument("--mesh_loss_weight", type=float, default=1.0)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--test_only", action="store_true",
                        help="Skip training and run test phase only. Requires --student_ckpt.")
    parser.add_argument("--student_ckpt", default=None,
                        help="Path to a student checkpoint (.pt) to load before test.")
    parser.add_argument("--visualize_test_count", type=int, default=30)
    parser.add_argument("--visualize_dir", default=None)
    parser.add_argument("--no_vision", action="store_true")
    parser.add_argument("--no_ov", action="store_true")
    parser.add_argument("--no_iv", action="store_true")
    parser.add_argument("--no_uv", action="store_true")
    parser.add_argument("--no_face_feat", action="store_true")
    parser.add_argument("--global_uv", action="store_true")
    parser.add_argument("--vision_az", type=int, default=12)
    parser.add_argument("--vision_el", type=int, default=6)
    return parser.parse_args()


def apply_dataset_defaults(args: argparse.Namespace) -> None:
    defaults = DATASET_DEFAULTS[args.dataset]
    for key in ("data_root", "brep_graph_root", "brep_seg_dir", "split_file", "mesh_dir_template", "num_classes"):
        if getattr(args, key) is None:
            value = defaults[key]
            if isinstance(value, str):
                value = value.replace("{data_root}", str(args.data_root))
            setattr(args, key, value)

    if args.eigen_cache_dir is not None:
        args.op_cache_dir = args.eigen_cache_dir

    # PD-MeshNet cache dir default
    if args.student_type == "pd_meshnet" and args.pd_cache_dir is None:
        args.pd_cache_dir = _PD_CACHE_DIRS.get(args.dataset)


def resolve_op_cache_dir(args: argparse.Namespace) -> None:
    """Choose an operator cache directory without clobbering the shared k128 cache."""

    # PD-MeshNet does not use DiffusionNet eigen-operators; skip op_cache for it.
    if args.student_type == "pd_meshnet":
        args.op_cache_dir = None
        return
    if args.no_op_cache:
        args.op_cache_dir = None
        return
    if args.op_cache_dir is not None:
        return
    if args.k_eig == 128:
        args.op_cache_dir = DATASET_DEFAULTS[args.dataset]["op_cache_dir"]
    else:
        args.op_cache_dir = f"/data2/gwlee/B2Mesh/op_cache_{args.dataset.replace('+', 'p')}_k{args.k_eig}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def set_current_cuda_device(device: torch.device) -> None:
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)


def _is_pyg_data(value: Any) -> bool:
    """Return True if value is a PyTorch Geometric Data/Batch object."""
    return _HAS_PYG and isinstance(value, _PyGData)


def move_sample_to_device(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in sample.items():
        if key == "brep_graph":
            out[key] = value.to(device)
        elif torch.is_tensor(value):
            out[key] = value.to(device)
        elif _is_pyg_data(value):          # primal_graph / dual_graph (PD-MeshNet)
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def move_sample_to_devices(
    sample: dict[str, Any],
    teacher_device: torch.device,
    student_device: torch.device,
) -> dict[str, Any]:
    out = {}
    for key, value in sample.items():
        if key == "brep_graph":
            out[key] = value.to(teacher_device)
        elif key == "brep_labels":
            out[key] = value.to(teacher_device)
        elif torch.is_tensor(value):
            out[key] = value.to(student_device)
        elif _is_pyg_data(value):          # primal_graph / dual_graph (PD-MeshNet)
            out[key] = value.to(student_device)
        else:
            out[key] = value               # petdni (dict), name (str), etc. stay as-is
    return out


def iter_samples(batch: dict[str, Any] | list[dict[str, Any]]):
    if isinstance(batch, list):
        yield from batch
    else:
        yield batch


def _make_pd_loaders(args: argparse.Namespace):
    """Build DataLoaders using PD-MeshNet datasets (MFCAD++ or Fusion360)."""
    if not _HAS_PD_MESHNET:
        raise ImportError("PD-MeshNet dependencies not available. Check B2Mesh/PD_MeshNet/.")

    dataset_cls = {
        "mfcad++": MFCADPlusPlusDualPrimal,
        "fusion360": Fusion360DualPrimal,
    }[args.dataset]

    common = dict(
        data_root=args.data_root,
        cache_graphs=True,
        cache_dir=args.pd_cache_dir,
    )
    if args.dataset == "fusion360":
        common["brep_graph_root"] = args.brep_graph_root
        common["split_file"] = args.split_file

    def make_ds(split, limit):
        kw = dict(**common, split=split)
        # Fusion360DualPrimal accepts `limit`; MFCADPlusPlusDualPrimal does not.
        if limit is not None and args.dataset == "fusion360":
            kw["limit"] = limit
        # No teacher → no need to load B-rep graphs for each sample.
        kw["load_brep"] = not args.no_teacher
        ds = dataset_cls(**kw)
        # For datasets without native limit support, slice the name list manually.
        if limit is not None and args.dataset != "fusion360":
            ds._names = ds._names[:limit]
        return ds

    loader_args = dict(
        batch_size=1,           # PD-MeshNet processes one mesh at a time
        num_workers=args.num_workers,
        collate_fn=collate_pd_samples,
        pin_memory=False,
    )
    return (
        DataLoader(make_ds("train", args.limit_train), shuffle=True, **loader_args),
        DataLoader(make_ds("val", args.limit_val), shuffle=False, **loader_args),
        DataLoader(make_ds("test", args.limit_test), shuffle=False, **loader_args),
    )


def make_loaders(args: argparse.Namespace, load_mesh: bool = True):
    if args.student_type == "pd_meshnet":
        return _make_pd_loaders(args)

    common = dict(
        data_root=args.data_root,
        brep_graph_root=args.brep_graph_root,
        brep_seg_dir=args.brep_seg_dir,
        split_file=args.split_file,
        mesh_dir_template=args.mesh_dir_template,
        k_eig=args.k_eig,
        op_cache_dir=args.op_cache_dir,
        input_features=args.input_features,
        load_mesh=load_mesh,
    )
    train_set = BRepMeshDataset(split="train", limit=args.limit_train, **common)
    val_set = BRepMeshDataset(split="val", limit=args.limit_val, **common)
    test_set = BRepMeshDataset(split="test", limit=args.limit_test, **common)
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_distill_samples,
        pin_memory=False,
    )
    return (
        DataLoader(train_set, shuffle=True, **loader_args),
        DataLoader(val_set, shuffle=False, **loader_args),
        DataLoader(test_set, shuffle=False, **loader_args),
    )


def make_teacher_loaders(args: argparse.Namespace):
    common = dict(
        data_root=args.data_root,
        brep_graph_root=args.brep_graph_root,
        brep_seg_dir=args.brep_seg_dir,
        split_file=args.split_file,
        mesh_dir_template=args.mesh_dir_template,
        k_eig=args.k_eig,
        op_cache_dir=args.op_cache_dir,
        input_features=args.input_features,
        load_mesh=False,
    )
    train_set = BRepMeshDataset(split="train", limit=args.limit_train, **common)
    val_set = BRepMeshDataset(split="val", limit=args.limit_val, **common)

    if len(train_set) == 0:
        hint = (
            " For pd_meshnet mode with MFCAD++, B-rep graphs (.bin) may not exist — "
            "pass --teacher_ckpt to use a pre-trained FOVNet checkpoint instead."
            if args.student_type == "pd_meshnet"
            else ""
        )
        raise RuntimeError(
            f"Teacher training dataset is empty (no .bin B-rep graphs found at "
            f"'{train_set.graph_dir}').{hint}"
        )

    loader_args = dict(
        batch_size=args.teacher_batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_brep_samples,
        pin_memory=False,
    )
    return (
        DataLoader(train_set, shuffle=True, drop_last=len(train_set) > args.teacher_batch_size, **loader_args),
        DataLoader(val_set, shuffle=False, drop_last=False, **loader_args),
    )


def make_models(
    args: argparse.Namespace,
    teacher_device: torch.device,
    student_device: torch.device,
):
    teacher = FOVNetFaceTeacher(
        num_classes=args.num_classes,
        vision=not args.no_vision,
        vision_az=args.vision_az,
        vision_el=args.vision_el,
        local_uv=not args.global_uv,
        segmentation=True,
        use_face_feat=not args.no_face_feat,
        use_uv=not args.no_uv,
        use_ov=not args.no_ov,
        use_iv=not args.no_iv,
        graph_emb_dim=args.embedding_dim,
    )

    if args.student_type == "pd_meshnet":
        if not _HAS_PD_MESHNET:
            raise ImportError("PD-MeshNet dependencies not available.")
        student = PDMeshNetStudent(
            in_channels_primal=_PD_IN_CHANNELS["primal"],
            in_channels_dual=_PD_IN_CHANNELS["dual"],
            num_classes=args.num_classes,
            embedding_dim=args.embedding_dim,
            conv_primal_out_res=args.pd_conv_channels,
            conv_dual_out_res=args.pd_conv_channels,
        )
    else:
        student = DiffusionMeshStudent(
            input_features=args.input_features,
            embedding_dim=args.embedding_dim,
            num_classes=args.num_classes,
            width=args.student_width,
            blocks=args.student_blocks,
        )

    # no_teacher: skip teacher construction entirely to save memory.
    if args.no_teacher:
        return None, student.to(student_device)

    return teacher.to(teacher_device), student.to(student_device)


def make_optimizer(args: argparse.Namespace, student: torch.nn.Module):
    return torch.optim.AdamW(student.parameters(), lr=args.lr)


def make_teacher_optimizer(args: argparse.Namespace, teacher: torch.nn.Module):
    return torch.optim.AdamW(teacher.parameters(), lr=args.teacher_lr)


def make_scheduler(
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=args.lr_min
        )
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    return None  # "none"


def forward_teacher_loss(
    teacher: FOVNetFaceTeacher,
    sample: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
    logits, _ = teacher(sample["brep_graph"])
    labels = sample["brep_labels"]
    loss = F.cross_entropy(logits, labels)
    return loss, {"loss": float(loss.detach().cpu())}, logits, labels


def _prepare_pd_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Wrap primal/dual graphs into a proper PyG batch (needed by PD-MeshNet encoder)."""
    pg, dg, petdni = create_dual_primal_batch(
        [sample["primal_graph"]], [sample["dual_graph"]], [sample["petdni"]]
    )
    return {**sample, "primal_graph": pg, "dual_graph": dg, "petdni": petdni}


def forward_losses(
    teacher: FOVNetFaceTeacher | None,
    student: torch.nn.Module,
    sample: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
    mesh_labels = sample["mesh_labels"]

    # --- Student forward ---
    if args.student_type == "pd_meshnet":
        sample = _prepare_pd_sample(sample)

    student_logits, student_tri_emb = student(sample)

    # Segmentation loss: focal (default) or plain CE (original PD-MeshNet).
    if args.seg_loss == "ce":
        mesh_ce = F.cross_entropy(student_logits, mesh_labels)
    else:
        mesh_ce = focal_loss(student_logits, mesh_labels)

    # --- Distillation losses (skipped when teacher=None or weights=0) ---
    has_brep = (
        teacher is not None
        and "brep_graph" in sample
        and sample["brep_graph"] is not None
        and (args.distill_weight > 0 or args.align_weight > 0)
    )
    if has_brep:
        with torch.no_grad():
            _, teacher_face_emb = teacher(sample["brep_graph"])
        teacher_face_emb = teacher_face_emb.to(student_tri_emb.device)
        student_brep_emb, face_mask = scatter_mean_by_index(
            student_tri_emb,
            sample["fidx"],
            output_size=teacher_face_emb.shape[0],
        )
        rel = relational_distillation_loss(teacher_face_emb, student_brep_emb, face_mask, mode=args.rkd_mode)
        align = feature_alignment_loss(teacher_face_emb, student_brep_emb, face_mask)
    else:
        rel = torch.tensor(0.0, device=student_tri_emb.device)
        align = torch.tensor(0.0, device=student_tri_emb.device)

    loss = (
        args.mesh_loss_weight * mesh_ce
        + args.distill_weight * rel
        + args.align_weight * align
    )

    parts = {
        "loss": float(loss.detach().cpu()),
        "mesh_ce": float(mesh_ce.detach().cpu()),
        "rel": float(rel.detach().cpu()),
        "align": float(align.detach().cpu()),
    }
    return loss, parts, student_logits, mesh_labels


def run_teacher_epoch(
    teacher: FOVNetFaceTeacher,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    num_classes: int,
    desc: str,
) -> dict[str, float]:
    train = optimizer is not None
    teacher.train(train)

    totals = {"loss": 0.0}
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    correct = 0
    total = 0
    n_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=desc, unit="batch"):
            samples = [move_sample_to_device(s, device) for s in iter_samples(batch)]
            if train:
                optimizer.zero_grad(set_to_none=True)

            batch_loss = 0.0
            for sample in samples:
                use_eval_for_small_batch = train and sample["brep_labels"].numel() < 2
                if use_eval_for_small_batch:
                    teacher.eval()
                loss, parts, logits, labels = forward_teacher_loss(teacher, sample)
                if use_eval_for_small_batch:
                    teacher.train(True)
                batch_loss = batch_loss + loss / len(samples)
                c, t, conf = classification_stats(logits, labels, num_classes=num_classes)
                correct += c
                total += t
                confusion += conf
                for key, value in parts.items():
                    totals[key] += value
                n_samples += 1

            if train:
                batch_loss.backward()
                optimizer.step()

    denom = max(n_samples, 1)
    metrics = {key: value / denom for key, value in totals.items()}
    metrics["brep_acc"] = correct / max(total, 1)
    metrics["brep_miou"] = mean_iou(confusion)
    return metrics


def run_epoch(
    teacher: FOVNetFaceTeacher,
    student: DiffusionMeshStudent,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    teacher_device: torch.device,
    student_device: torch.device,
    args: argparse.Namespace,
    desc: str,
) -> dict[str, float]:
    train = optimizer is not None
    if teacher is not None:
        teacher.eval()
    student.train(train)

    totals = {"loss": 0.0, "mesh_ce": 0.0, "rel": 0.0, "align": 0.0}
    confusion = torch.zeros((args.num_classes, args.num_classes), dtype=torch.long)
    correct = 0
    total = 0
    n_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=desc, unit="batch"):
            samples = [move_sample_to_devices(s, teacher_device, student_device) for s in iter_samples(batch)]
            if train:
                optimizer.zero_grad(set_to_none=True)

            batch_loss = 0.0
            for sample in samples:
                loss, parts, logits, labels = forward_losses(teacher, student, sample, args)
                batch_loss = batch_loss + loss / len(samples)
                c, t, conf = classification_stats(logits, labels, num_classes=args.num_classes)
                correct += c
                total += t
                confusion += conf
                for key, value in parts.items():
                    totals[key] += value
                n_samples += 1

            if train:
                batch_loss.backward()
                optimizer.step()

    denom = max(n_samples, 1)
    metrics = {key: value / denom for key, value in totals.items()}
    metrics["mesh_acc"] = correct / max(total, 1)
    metrics["mesh_miou"] = mean_iou(confusion)
    return metrics


def save_teacher_checkpoint(
    path: Path,
    teacher: FOVNetFaceTeacher,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "teacher": teacher.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def load_teacher_checkpoint(path: Path, teacher: FOVNetFaceTeacher, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    teacher.load_state_dict(ckpt["teacher"])
    return ckpt


def save_checkpoint(
    path: Path,
    teacher: FOVNetFaceTeacher,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "student": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "metrics": metrics,
    }
    if teacher is not None:
        ckpt["teacher"] = teacher.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    teacher: FOVNetFaceTeacher | None,
    student: torch.nn.Module,
    teacher_device: torch.device,
    student_device: torch.device,
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if teacher is not None and "teacher" in ckpt:
        teacher.load_state_dict(ckpt["teacher"])
        teacher.to(teacher_device)
    student.load_state_dict(ckpt["student"])
    student.to(student_device)
    return ckpt


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _metrics_row(
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    lr: float,
    best_val: float,
    train_time_s: float | None = None,
    val_time_s: float | None = None,
) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "epoch": epoch,
        "lr": lr,
        "best_val_loss": best_val,
    }
    row.update({f"train_{key}": value for key, value in train_metrics.items()})
    row.update({f"val_{key}": value for key, value in val_metrics.items()})
    if train_time_s is not None:
        row["train_time_s"] = round(train_time_s, 3)
    if val_time_s is not None:
        row["val_time_s"] = round(val_time_s, 3)
    if train_time_s is not None and val_time_s is not None:
        row["epoch_time_s"] = round(train_time_s + val_time_s, 3)
    return row


def append_metrics_log(log_dir: Path, row: dict[str, float | int]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = log_dir / "metrics.jsonl"
    csv_path = log_dir / "metrics.csv"

    with jsonl_path.open("a") as f:
        f.write(json.dumps(row) + "\n")

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_test_log(output_dir: Path, metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row = {"split": "test", **metrics}
    with (output_dir / "test_metrics.json").open("w") as f:
        json.dump(row, f, indent=2)
    with (output_dir / "test_metrics.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")


def write_timing_summary(
    output_dir: Path,
    train_times: list[float],
    val_times: list[float],
    test_time_s: float | None = None,
    n_test_samples: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, float | int] = {
        "total_train_time_s": round(sum(train_times) + sum(val_times), 3),
        "total_epoch_train_time_s": round(sum(train_times), 3),
        "total_epoch_val_time_s": round(sum(val_times), 3),
        "n_epochs": len(train_times),
        "mean_epoch_time_s": round((sum(train_times) + sum(val_times)) / max(len(train_times), 1), 3),
        "mean_train_time_s": round(sum(train_times) / max(len(train_times), 1), 3),
        "mean_val_time_s": round(sum(val_times) / max(len(val_times), 1), 3),
    }
    if test_time_s is not None:
        summary["test_time_s"] = round(test_time_s, 3)
    if test_time_s is not None and n_test_samples:
        summary["test_samples_per_sec"] = round(n_test_samples / test_time_s, 3)
    with (output_dir / "timing_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"timing | total {summary['total_train_time_s']:.1f}s "
        f"mean epoch {summary['mean_epoch_time_s']:.1f}s "
        f"(train {summary['mean_train_time_s']:.1f}s / val {summary['mean_val_time_s']:.1f}s)"
        + (f" | test {summary['test_time_s']:.1f}s ({summary.get('test_samples_per_sec', 0):.1f} samples/s)" if test_time_s else "")
    )


def print_metrics(
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    train_time_s: float | None = None,
    val_time_s: float | None = None,
) -> None:
    time_str = ""
    if train_time_s is not None and val_time_s is not None:
        time_str = f" | {train_time_s:.1f}s+{val_time_s:.1f}s={train_time_s + val_time_s:.1f}s"
    msg = (
        f"epoch {epoch:03d} | "
        f"train loss {train_metrics['loss']:.4f} acc {train_metrics['mesh_acc']:.4f} miou {train_metrics['mesh_miou']:.4f} | "
        f"val loss {val_metrics['loss']:.4f} acc {val_metrics['mesh_acc']:.4f} miou {val_metrics['mesh_miou']:.4f}"
        f"{time_str}"
    )
    print(msg)


def print_teacher_metrics(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
    msg = (
        f"teacher epoch {epoch:03d} | "
        f"train loss {train_metrics['loss']:.4f} acc {train_metrics['brep_acc']:.4f} miou {train_metrics['brep_miou']:.4f} | "
        f"val loss {val_metrics['loss']:.4f} acc {val_metrics['brep_acc']:.4f} miou {val_metrics['brep_miou']:.4f}"
    )
    print(msg)


def _student_forward(student: torch.nn.Module, sample: dict[str, Any], args: argparse.Namespace):
    """Call student.forward() with the correct sample preparation per student type."""
    if args.student_type == "pd_meshnet":
        sample = _prepare_pd_sample(sample)
    return student(sample)


def save_all_test_predictions(
    student: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path,
) -> int:
    """Save pred/gt npy for every test sample (no PLY, no limit)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    student.eval()
    saved = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="save predictions", unit="batch"):
            for raw_sample in iter_samples(batch):
                sample = move_sample_to_devices(raw_sample, torch.device("cpu"), device)
                logits, _ = _student_forward(student, sample, args)
                logits_cpu = logits.detach().cpu()
                pred_np = logits_cpu.argmax(dim=-1).numpy()
                gt_np = sample["mesh_labels"].detach().cpu().numpy()
                logits_np = logits_cpu.numpy()
                sample_dir = out_dir / sample["name"]
                sample_dir.mkdir(parents=True, exist_ok=True)
                np.save(sample_dir / f"{sample['name']}_pred.npy", pred_np)
                np.save(sample_dir / f"{sample['name']}_gt.npy", gt_np)
                np.save(sample_dir / f"{sample['name']}_logits.npy", logits_np)
                saved += 1
    return saved


def save_test_visualizations(
    student: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path,
) -> int:
    if args.visualize_test_count <= 0:
        return 0

    student.eval()
    write_legend(out_dir / "legend.csv", load_class_names(args.data_root))
    saved = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="visualize", unit="batch"):
            for raw_sample in iter_samples(batch):
                sample = move_sample_to_devices(raw_sample, torch.device("cpu"), device)
                logits, _ = _student_forward(student, sample, args)
                pred_labels = logits.argmax(dim=-1)
                save_prediction_meshes(
                    out_dir,
                    sample["name"],
                    sample["verts"],
                    sample["faces"],
                    sample["mesh_labels"],
                    pred_labels,
                )
                saved += 1
                if saved >= args.visualize_test_count:
                    return saved
    return saved


def main() -> None:
    args = parse_args()

    apply_dataset_defaults(args)
    resolve_op_cache_dir(args)
    set_seed(args.seed)
    student_device = resolve_device(args.device)
    teacher_device = resolve_device(args.teacher_device) if args.teacher_device else student_device
    set_current_cuda_device(teacher_device)
    set_current_cuda_device(student_device)

    teacher_tag = "None" if args.no_teacher else "FOVNet"
    student_tag = {"diffusion_net": "DiffusionNet", "pd_meshnet": "PDNet"}[args.student_type]
    dataset_tag = args.dataset.replace("++", "pp")  # mfcad++ → mfcadpp (filesystem-safe)
    input_tag = args.input_features
    run_name = f"{teacher_tag}_{student_tag}_{dataset_tag}_{input_tag}_{time.strftime('%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Student type: {args.student_type}  |  teacher: {'disabled' if args.no_teacher else 'enabled'}")
    print(f"Segmentation loss: {args.seg_loss}")
    if args.student_type == "diffusion_net":
        print(f"Using DiffusionNet op cache: {args.op_cache_dir}")
    else:
        print(f"Using PD-MeshNet graph cache: {args.pd_cache_dir}")
    if not args.no_teacher:
        print(f"Using teacher device: {teacher_device}; student device: {student_device}")

    train_loader, val_loader, test_loader = make_loaders(args, load_mesh=True)
    teacher, student = make_models(args, teacher_device, student_device)

    if args.no_teacher:
        print("--no_teacher: skipping teacher setup, distillation disabled.")
    elif args.teacher_ckpt:
        if not Path(args.teacher_ckpt).is_file():
            raise FileNotFoundError(f"--teacher_ckpt must be a checkpoint file: {args.teacher_ckpt}")
        print(f"Loading FOVNet teacher checkpoint: {args.teacher_ckpt}")
        load_fovnet_teacher_checkpoint(teacher, args.teacher_ckpt)
    else:
        print("No --teacher_ckpt provided; training FOVNet teacher first.")
        teacher_train_loader, teacher_val_loader = make_teacher_loaders(args)
        teacher_optimizer = make_teacher_optimizer(args, teacher)
        teacher_scheduler = make_scheduler(args, teacher_optimizer, args.teacher_epochs)
        teacher_dir = output_dir / "teacher"
        best_teacher_val = float("inf")
        for epoch in range(1, args.teacher_epochs + 1):
            epoch_lr = current_lr(teacher_optimizer)
            teacher_train_metrics = run_teacher_epoch(
                teacher,
                teacher_train_loader,
                teacher_optimizer,
                teacher_device,
                args.num_classes,
                "teacher-train",
            )
            teacher_val_metrics = run_teacher_epoch(
                teacher,
                teacher_val_loader,
                None,
                teacher_device,
                args.num_classes,
                "teacher-val",
            )
            is_best_teacher = teacher_val_metrics["loss"] < best_teacher_val
            if is_best_teacher:
                best_teacher_val = teacher_val_metrics["loss"]
            if teacher_scheduler is not None:
                teacher_scheduler.step()
            print_teacher_metrics(epoch, teacher_train_metrics, teacher_val_metrics)
            append_metrics_log(
                teacher_dir,
                _metrics_row(
                    epoch,
                    teacher_train_metrics,
                    teacher_val_metrics,
                    epoch_lr,
                    best_teacher_val,
                ),
            )

            save_teacher_checkpoint(teacher_dir / "last.pt", teacher, epoch, args, teacher_val_metrics)
            if is_best_teacher:
                save_teacher_checkpoint(teacher_dir / "best.pt", teacher, epoch, args, teacher_val_metrics)

        load_teacher_checkpoint(teacher_dir / "best.pt", teacher, teacher_device)
        print(f"Loaded best trained teacher from {teacher_dir / 'best.pt'}")

    if args.test_only:
        if not args.student_ckpt:
            raise ValueError("--test_only requires --student_ckpt")
        ckpt_path = Path(args.student_ckpt)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"--student_ckpt not found: {ckpt_path}")
        print(f"[test_only] Loading student checkpoint: {ckpt_path}")
        load_checkpoint(ckpt_path, teacher if not args.no_teacher else None, student, teacher_device, student_device)
        if teacher is not None:
            freeze_module(teacher)
        t0 = time.perf_counter()
        test_metrics = run_epoch(teacher, student, test_loader, None, teacher_device, student_device, args, "test")
        test_time_s = time.perf_counter() - t0
        test_metrics["test_time_s"] = round(test_time_s, 3)
        n_test = len(test_loader.dataset)
        test_metrics["test_samples_per_sec"] = round(n_test / test_time_s, 3)
        write_test_log(output_dir, test_metrics)
        print(
            "test | "
            f"loss {test_metrics['loss']:.4f} "
            f"acc {test_metrics['mesh_acc']:.4f} "
            f"miou {test_metrics['mesh_miou']:.4f} "
            f"| {test_time_s:.1f}s ({test_metrics['test_samples_per_sec']:.1f} samples/s)"
        )
        vis_dir = Path(args.visualize_dir) if args.visualize_dir else output_dir / "test_visualizations"
        saved = save_test_visualizations(student, test_loader, student_device, args, vis_dir)
        if saved:
            print(f"saved {saved} test visualizations to {vis_dir}")
        pred_dir = output_dir / "test_predictions"
        n_saved = save_all_test_predictions(student, test_loader, student_device, args, pred_dir)
        print(f"saved {n_saved} test predictions (npy) to {pred_dir}")
        write_timing_summary(output_dir, [], [], test_time_s=test_time_s, n_test_samples=n_test)
        return

    if teacher is not None:
        freeze_module(teacher)
    optimizer = make_optimizer(args, student)
    scheduler = make_scheduler(args, optimizer, args.epochs)

    best_val = float("inf")
    epoch_train_times: list[float] = []
    epoch_val_times: list[float] = []
    for epoch in range(1, args.epochs + 1):
        epoch_lr = current_lr(optimizer)
        t0 = time.perf_counter()
        train_metrics = run_epoch(teacher, student, train_loader, optimizer, teacher_device, student_device, args, "train")
        train_time_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        val_metrics = run_epoch(teacher, student, val_loader, None, teacher_device, student_device, args, "val")
        val_time_s = time.perf_counter() - t0
        epoch_train_times.append(train_time_s)
        epoch_val_times.append(val_time_s)
        is_best = val_metrics["loss"] < best_val
        if is_best:
            best_val = val_metrics["loss"]
        if scheduler is not None:
            scheduler.step()
        print_metrics(epoch, train_metrics, val_metrics, train_time_s, val_time_s)
        append_metrics_log(
            output_dir,
            _metrics_row(epoch, train_metrics, val_metrics, epoch_lr, best_val, train_time_s, val_time_s),
        )

        save_checkpoint(output_dir / "last.pt", teacher, student, optimizer, epoch, args, val_metrics, scheduler)
        if is_best:
            save_checkpoint(output_dir / "best.pt", teacher, student, optimizer, epoch, args, val_metrics, scheduler)

    test_time_s: float | None = None
    if not args.skip_test:
        best_path = output_dir / "best.pt"
        if best_path.exists():
            load_checkpoint(best_path, teacher, student, teacher_device, student_device)
        t0 = time.perf_counter()
        test_metrics = run_epoch(teacher, student, test_loader, None, teacher_device, student_device, args, "test")
        test_time_s = time.perf_counter() - t0
        test_metrics["test_time_s"] = round(test_time_s, 3)
        n_test = len(test_loader.dataset)
        test_metrics["test_samples_per_sec"] = round(n_test / test_time_s, 3)
        write_test_log(output_dir, test_metrics)
        print(
            "test | "
            f"loss {test_metrics['loss']:.4f} "
            f"acc {test_metrics['mesh_acc']:.4f} "
            f"miou {test_metrics['mesh_miou']:.4f} "
            f"| {test_time_s:.1f}s ({test_metrics['test_samples_per_sec']:.1f} samples/s)"
        )
        vis_dir = Path(args.visualize_dir) if args.visualize_dir else output_dir / "test_visualizations"
        saved = save_test_visualizations(student, test_loader, student_device, args, vis_dir)
        if saved:
            print(f"saved {saved} test visualizations to {vis_dir}")
        pred_dir = output_dir / "test_predictions"
        n_saved = save_all_test_predictions(student, test_loader, student_device, args, pred_dir)
        print(f"saved {n_saved} test predictions (npy) to {pred_dir}")

    write_timing_summary(
        output_dir,
        epoch_train_times,
        epoch_val_times,
        test_time_s=test_time_s,
        n_test_samples=len(test_loader.dataset) if not args.skip_test else None,
    )


if __name__ == "__main__":
    main()
