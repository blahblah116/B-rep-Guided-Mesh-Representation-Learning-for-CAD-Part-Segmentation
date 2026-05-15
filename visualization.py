"""Save colored mesh predictions for inspection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


DEFAULT_COLORS = np.array(
    [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
    ],
    dtype=np.uint8,
)


def load_class_names(data_root: str | Path) -> list[str]:
    path = Path(data_root) / "segment_names.json"
    if path.exists():
        with path.open("r") as f:
            return json.load(f)
    feature_labels = Path(data_root) / "feature_labels.txt"
    if feature_labels.exists():
        names = []
        with feature_labels.open("r") as f:
            for line in f:
                stripped = line.strip()
                if " - " in stripped and stripped[0].isdigit():
                    names.append(stripped.split(" - ", 1)[1])
        if names:
            return names
    return [str(i) for i in range(len(DEFAULT_COLORS))]


def colorize_faces(labels: np.ndarray, colors: np.ndarray = DEFAULT_COLORS) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    return colors[labels % len(colors)]


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def write_face_colored_ply(
    path: str | Path,
    verts: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = np.asarray(verts)
    faces = np.asarray(faces, dtype=np.int64)
    face_colors = np.asarray(face_colors, dtype=np.uint8)

    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {verts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for vert in verts:
            f.write(f"{vert[0]} {vert[1]} {vert[2]}\n")
        for tri, color in zip(faces, face_colors):
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]} {color[0]} {color[1]} {color[2]}\n")


def write_legend(path: str | Path, class_names: list[str], colors: np.ndarray = DEFAULT_COLORS) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("class_id,class_name,red,green,blue\n")
        for class_id, name in enumerate(class_names):
            color = colors[class_id % len(colors)]
            f.write(f"{class_id},{name},{int(color[0])},{int(color[1])},{int(color[2])}\n")


def save_prediction_meshes(
    out_dir: str | Path,
    sample_name: str,
    verts: torch.Tensor,
    faces: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_labels: torch.Tensor,
) -> None:
    """Save prediction, ground-truth, and error maps as face-colored PLY files."""

    out_dir = Path(out_dir) / sample_name
    verts_np = tensor_to_numpy(verts)
    faces_np = tensor_to_numpy(faces).astype(np.int64)
    gt_np = tensor_to_numpy(gt_labels).astype(np.int64)
    pred_np = tensor_to_numpy(pred_labels).astype(np.int64)

    write_face_colored_ply(out_dir / f"{sample_name}_pred.ply", verts_np, faces_np, colorize_faces(pred_np))
    write_face_colored_ply(out_dir / f"{sample_name}_gt.ply", verts_np, faces_np, colorize_faces(gt_np))

    error_colors = np.zeros((faces_np.shape[0], 3), dtype=np.uint8)
    error_colors[:] = np.array([30, 170, 80], dtype=np.uint8)
    error_colors[pred_np != gt_np] = np.array([220, 40, 40], dtype=np.uint8)
    write_face_colored_ply(out_dir / f"{sample_name}_error.ply", verts_np, faces_np, error_colors)

    np.save(out_dir / f"{sample_name}_pred.npy", pred_np)
    np.save(out_dir / f"{sample_name}_gt.npy", gt_np)
