"""Evaluate a saved checkpoint on the full test set and report per-class IoU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import Fusion360BRepMeshDataset, collate_distill_samples
from .models import (
    DiffusionMeshStudent,
    FOVNetFaceTeacher,
    classification_stats,
)
from .train import iter_samples, move_sample_to_devices


CLASS_NAMES = {
    "fusion360": [
        "ExtrudeSide",
        "ExtrudeEnd",
        "CutSide",
        "CutEnd",
        "Fillet",
        "Chamfer",
        "RevolveSide",
        "RevolveEnd",
    ],
    "mfcad++": [
        "Chamfer",
        "Through hole",
        "Triangular passage",
        "Rectangular passage",
        "6-sided passage",
        "Triangular through slot",
        "Rectangular through slot",
        "Circular through slot",
        "Rectangular through step",
        "2-sided through step",
        "Slanted through step",
        "O-ring",
        "Blind hole",
        "Triangular pocket",
        "Rectangular pocket",
        "6-sided pocket",
        "Circular end pocket",
        "Rectangular blind slot",
        "Vertical circular end blind slot",
        "Horizontal circular end blind slot",
        "Triangular blind step",
        "Circular blind step",
        "Rectangular blind step",
        "Round",
        "Stock",
    ],
}


def per_class_iou(confusion: torch.Tensor):
    confusion = confusion.float()
    ious = []
    for c in range(confusion.shape[0]):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        denom = tp + fp + fn
        iou = (tp / denom).item() if denom > 0 else float("nan")
        ious.append(iou)
    return ious


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", help="Output directory containing best.pt and args.json")
    parser.add_argument("--ckpt", default="best.pt", help="Checkpoint filename inside ckpt_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit_test", type=int, default=None)
    cli = parser.parse_args()

    ckpt_dir = Path(cli.ckpt_dir)
    with (ckpt_dir / "args.json").open() as f:
        saved = json.load(f)

    # Override a few fields from CLI
    saved["device"] = cli.device
    if cli.limit_test is not None:
        saved["limit_test"] = cli.limit_test
    args = SimpleNamespace(**saved)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_classes = args.num_classes
    class_names = CLASS_NAMES.get(args.dataset, [str(i) for i in range(num_classes)])

    test_set = Fusion360BRepMeshDataset(
        split="test",
        data_root=args.data_root,
        brep_graph_root=args.brep_graph_root,
        brep_seg_dir=args.brep_seg_dir,
        split_file=args.split_file,
        mesh_dir_template=args.mesh_dir_template,
        k_eig=args.k_eig,
        op_cache_dir=args.op_cache_dir,
        input_features=args.input_features,
        load_mesh=True,
        limit=getattr(args, "limit_test", None),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_distill_samples,
    )
    print(f"Dataset: {args.dataset}  |  Classes: {num_classes}  |  Test samples: {len(test_set)}")

    teacher = FOVNetFaceTeacher(
        num_classes=num_classes,
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
        num_classes=num_classes,
        width=args.student_width,
        blocks=args.student_blocks,
    )

    ckpt_path = ckpt_dir / cli.ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu")
    teacher.load_state_dict(ckpt["teacher"])
    student.load_state_dict(ckpt["student"])
    teacher.to(device).eval()
    student.to(device).eval()
    print(f"Loaded checkpoint: {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="eval", unit="batch"):
            for raw_sample in iter_samples(batch):
                sample = move_sample_to_devices(raw_sample, device, device)
                logits, _ = student(sample)
                _, _, conf = classification_stats(logits, sample["mesh_labels"], num_classes=num_classes)
                confusion += conf

    ious = per_class_iou(confusion)
    valid_ious = [v for v in ious if v == v]  # exclude nan
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

    col = max(len(n) for n in class_names) + 2
    print()
    print(f"{'Class':<{col}} {'GT Points':>10} {'IoU':>8}")
    print("-" * (col + 20))
    for c, name in enumerate(class_names):
        count = int(confusion[c, :].sum())
        iou_str = f"{ious[c]*100:>7.2f}%" if count > 0 else "    N/A"
        print(f"{name:<{col}} {count:>10,} {iou_str}")
    print("-" * (col + 20))
    total = int(confusion.sum())
    print(f"{'mIoU':<{col}} {'Total:':>5}{total:>6,} {miou*100:>7.2f}%")


if __name__ == "__main__":
    main()
