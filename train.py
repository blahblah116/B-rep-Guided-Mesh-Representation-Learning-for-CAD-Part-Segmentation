"""Train relational distillation from FOVNet B-rep faces to DiffusionNet mesh faces."""

from __future__ import annotations

import argparse
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

from .data import Fusion360BRepMeshDataset, collate_brep_samples, collate_distill_samples
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


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
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
    parser.add_argument("--teacher_lr", type=float, default=1e-3)
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
    parser.add_argument("--student_width", type=int, default=128)
    parser.add_argument("--student_blocks", type=int, default=4)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--rkd_mode", choices=["distance", "angle", "distance_angle"], default="distance_angle")
    parser.add_argument("--distill_weight", type=float, default=1.0)
    parser.add_argument("--align_weight", type=float, default=0)
    parser.add_argument("--mesh_loss_weight", type=float, default=1.0)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--visualize_test_count", type=int, default=20)
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


def resolve_op_cache_dir(args: argparse.Namespace) -> None:
    """Choose an operator cache directory without clobbering the shared k128 cache."""

    if args.no_op_cache:
        args.op_cache_dir = None
        return
    if args.op_cache_dir is not None:
        return
    if args.k_eig == 128:
        args.op_cache_dir = DATASET_DEFAULTS[args.dataset]["op_cache_dir"]
    else:
        args.op_cache_dir = f"distillationgwlee/op_cache_{args.dataset.replace('+', 'p')}_k{args.k_eig}"


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


def move_sample_to_device(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in sample.items():
        if key == "brep_graph":
            out[key] = value.to(device)
        elif torch.is_tensor(value):
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
        else:
            out[key] = value
    return out


def iter_samples(batch: dict[str, Any] | list[dict[str, Any]]):
    if isinstance(batch, list):
        yield from batch
    else:
        yield batch


def make_loaders(args: argparse.Namespace, load_mesh: bool = True):
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
    train_set = Fusion360BRepMeshDataset(split="train", limit=args.limit_train, **common)
    val_set = Fusion360BRepMeshDataset(split="val", limit=args.limit_val, **common)
    test_set = Fusion360BRepMeshDataset(split="test", limit=args.limit_test, **common)
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
    train_set = Fusion360BRepMeshDataset(split="train", limit=args.limit_train, **common)
    val_set = Fusion360BRepMeshDataset(split="val", limit=args.limit_val, **common)
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

    student = DiffusionMeshStudent(
        input_features=args.input_features,
        embedding_dim=args.embedding_dim,
        num_classes=args.num_classes,
        width=args.student_width,
        blocks=args.student_blocks,
    )
    return teacher.to(teacher_device), student.to(student_device)


def make_optimizer(args: argparse.Namespace, student: torch.nn.Module):
    return torch.optim.AdamW(student.parameters(), lr=args.lr)


def make_teacher_optimizer(args: argparse.Namespace, teacher: torch.nn.Module):
    return torch.optim.AdamW(teacher.parameters(), lr=args.teacher_lr)


def forward_teacher_loss(
    teacher: FOVNetFaceTeacher,
    sample: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
    logits, _ = teacher(sample["brep_graph"])
    labels = sample["brep_labels"]
    loss = F.cross_entropy(logits, labels)
    return loss, {"loss": float(loss.detach().cpu())}, logits, labels


def forward_losses(
    teacher: FOVNetFaceTeacher,
    student: DiffusionMeshStudent,
    sample: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
    graph = sample["brep_graph"]
    mesh_labels = sample["mesh_labels"]

    with torch.no_grad():
        _, teacher_face_emb = teacher(graph)

    student_logits, student_tri_emb = student(sample)
    teacher_face_emb = teacher_face_emb.to(student_tri_emb.device)
    mesh_ce = focal_loss(student_logits, mesh_labels)
    student_brep_emb, face_mask = scatter_mean_by_index(
        student_tri_emb,
        sample["fidx"],
        output_size=teacher_face_emb.shape[0],
    )

    rel = relational_distillation_loss(teacher_face_emb, student_brep_emb, face_mask, mode=args.rkd_mode)
    align = feature_alignment_loss(teacher_face_emb, student_brep_emb, face_mask)
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
    student: DiffusionMeshStudent,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "teacher": teacher.state_dict(),
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    teacher: FOVNetFaceTeacher,
    student: DiffusionMeshStudent,
    teacher_device: torch.device,
    student_device: torch.device,
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    teacher.load_state_dict(ckpt["teacher"])
    student.load_state_dict(ckpt["student"])
    teacher.to(teacher_device)
    student.to(student_device)
    return ckpt


def print_metrics(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
    msg = (
        f"epoch {epoch:03d} | "
        f"train loss {train_metrics['loss']:.4f} acc {train_metrics['mesh_acc']:.4f} miou {train_metrics['mesh_miou']:.4f} | "
        f"val loss {val_metrics['loss']:.4f} acc {val_metrics['mesh_acc']:.4f} miou {val_metrics['mesh_miou']:.4f}"
    )
    print(msg)


def print_teacher_metrics(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
    msg = (
        f"teacher epoch {epoch:03d} | "
        f"train loss {train_metrics['loss']:.4f} acc {train_metrics['brep_acc']:.4f} miou {train_metrics['brep_miou']:.4f} | "
        f"val loss {val_metrics['loss']:.4f} acc {val_metrics['brep_acc']:.4f} miou {val_metrics['brep_miou']:.4f}"
    )
    print(msg)


def save_all_test_predictions(
    student: DiffusionMeshStudent,
    loader: DataLoader,
    device: torch.device,
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
                logits, _ = student(sample)
                pred_np = logits.argmax(dim=-1).detach().cpu().numpy()
                gt_np = sample["mesh_labels"].detach().cpu().numpy()
                sample_dir = out_dir / sample["name"]
                sample_dir.mkdir(parents=True, exist_ok=True)
                np.save(sample_dir / f"{sample['name']}_pred.npy", pred_np)
                np.save(sample_dir / f"{sample['name']}_gt.npy", gt_np)
                saved += 1
    return saved


def save_test_visualizations(
    student: DiffusionMeshStudent,
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
                logits, _ = student(sample)
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

    output_dir = Path(args.output_dir) / time.strftime("%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Using DiffusionNet op cache: {args.op_cache_dir}")
    print(f"Using teacher device: {teacher_device}; student device: {student_device}")

    train_loader, val_loader, test_loader = make_loaders(args, load_mesh=True)
    teacher, student = make_models(args, teacher_device, student_device)

    if args.teacher_ckpt:
        if not Path(args.teacher_ckpt).is_file():
            raise FileNotFoundError(f"--teacher_ckpt must be a checkpoint file: {args.teacher_ckpt}")
        print(f"Loading FOVNet teacher checkpoint: {args.teacher_ckpt}")
        load_fovnet_teacher_checkpoint(teacher, args.teacher_ckpt)
    else:
        print("No --teacher_ckpt provided; training FOVNet teacher first.")
        teacher_train_loader, teacher_val_loader = make_teacher_loaders(args)
        teacher_optimizer = make_teacher_optimizer(args, teacher)
        teacher_dir = output_dir / "teacher"
        best_teacher_val = float("inf")
        for epoch in range(1, args.teacher_epochs + 1):
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
            print_teacher_metrics(epoch, teacher_train_metrics, teacher_val_metrics)

            save_teacher_checkpoint(teacher_dir / "last.pt", teacher, epoch, args, teacher_val_metrics)
            if teacher_val_metrics["loss"] < best_teacher_val:
                best_teacher_val = teacher_val_metrics["loss"]
                save_teacher_checkpoint(teacher_dir / "best.pt", teacher, epoch, args, teacher_val_metrics)

        load_teacher_checkpoint(teacher_dir / "best.pt", teacher, teacher_device)
        print(f"Loaded best trained teacher from {teacher_dir / 'best.pt'}")

    freeze_module(teacher)
    optimizer = make_optimizer(args, student)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(teacher, student, train_loader, optimizer, teacher_device, student_device, args, "train")
        val_metrics = run_epoch(teacher, student, val_loader, None, teacher_device, student_device, args, "val")
        print_metrics(epoch, train_metrics, val_metrics)

        save_checkpoint(output_dir / "last.pt", teacher, student, optimizer, epoch, args, val_metrics)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(output_dir / "best.pt", teacher, student, optimizer, epoch, args, val_metrics)

    if not args.skip_test:
        best_path = output_dir / "best.pt"
        if best_path.exists():
            load_checkpoint(best_path, teacher, student, teacher_device, student_device)
        test_metrics = run_epoch(teacher, student, test_loader, None, teacher_device, student_device, args, "test")
        print(
            "test | "
            f"loss {test_metrics['loss']:.4f} "
            f"acc {test_metrics['mesh_acc']:.4f} "
            f"miou {test_metrics['mesh_miou']:.4f}"
        )
        vis_dir = Path(args.visualize_dir) if args.visualize_dir else output_dir / "test_visualizations"
        saved = save_test_visualizations(student, test_loader, student_device, args, vis_dir)
        if saved:
            print(f"saved {saved} test visualizations to {vis_dir}")
        pred_dir = output_dir / "test_predictions"
        n_saved = save_all_test_predictions(student, test_loader, student_device, pred_dir)
        print(f"saved {n_saved} test predictions (npy) to {pred_dir}")


if __name__ == "__main__":
    main()
