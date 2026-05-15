"""B-rep/mesh paired datasets for distillation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

import dgl
from dgl.data.utils import load_graphs
from . import diffusion_net

try:
    import potpourri3d as pp3d
except ImportError:  # pragma: no cover - fallback is used only without pp3d.
    pp3d = None


def _read_obj_tri_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read vertices and triangular faces from an OBJ file."""

    if pp3d is not None:
        return pp3d.read_mesh(str(path))

    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                face = [int(tok.split("/")[0]) - 1 for tok in parts[1:]]
                if len(face) != 3:
                    raise ValueError(f"{path} contains a non-triangular face: {line.strip()}")
                faces.append(face)
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _to_float32_graph(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    for key in list(graph.ndata.keys()):
        if key != "y":
            graph.ndata[key] = graph.ndata[key].float()
    for key in list(graph.edata.keys()):
        graph.edata[key] = graph.edata[key].float()
    return graph


class Fusion360BRepMeshDataset(Dataset):
    """Paired sample: FOVNet B-rep graph plus OBJ mesh and face correspondence."""

    def __init__(
        self,
        data_root: str | Path = "/data2/gwlee/fovnet/data/fusion360/s2.0.1",
        split: str = "train",
        brep_graph_root: str | Path | None = None,
        brep_seg_dir: str | Path | None = "breps/seg",
        graph_subdir: str = "graphs",
        split_file: str = "train_test_new.json",
        mesh_dir_template: str = "meshes",
        k_eig: int = 128,
        op_cache_dir: str | Path | None = None,
        input_features: str = "xyz",
        limit: int | None = None,
        validate_files: bool = True,
        load_mesh: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.mesh_dir = self._resolve_template_path(mesh_dir_template, split)
        self.brep_seg_dir = (
            None if brep_seg_dir is None else self._resolve_template_path(brep_seg_dir, split)
        )
        self.split = split
        self.graph_root = Path(brep_graph_root) if brep_graph_root else self.data_root.parent
        self.graph_dir = self.graph_root / split / graph_subdir
        self.k_eig = k_eig
        self.op_cache_dir = None if op_cache_dir is None else str(Path(op_cache_dir))
        self.input_features = input_features
        self.load_mesh = load_mesh

        if input_features not in ("xyz", "hks"):
            raise ValueError("input_features must be one of: xyz, hks")

        self.names = self._load_split_names(split_file, split)
        if limit is not None:
            self.names = self.names[: int(limit)]

        if validate_files:
            self.names = self._filter_existing(self.names)
            if not self.names:
                raise FileNotFoundError(f"No valid paired samples found for split '{split}'.")

    def _resolve_template_path(self, value: str | Path, split: str) -> Path:
        path = Path(str(value).format(split=split))
        return path if path.is_absolute() else self.data_root / path

    def _load_split_names(self, split_file: str, split: str) -> list[str]:
        split_path = self._resolve_template_path(split_file, split)
        if split_path.suffix == ".json":
            with split_path.open("r") as f:
                split_data = json.load(f)
            if split not in split_data:
                raise ValueError(f"split '{split}' not found in {split_path}")
            return [str(name) for name in split_data[split]]

        with split_path.open("r") as f:
            names = [line.strip() for line in f if line.strip()]
        return [Path(name).stem for name in names]

    def _filter_existing(self, names: list[str]) -> list[str]:
        out = []
        for name in names:
            required = [self.graph_dir / f"{name}.bin"]
            if self.brep_seg_dir is not None:
                required.append(self.brep_seg_dir / f"{name}.seg")
            if self.load_mesh:
                required.extend(
                    [
                        self.mesh_dir / f"{name}.obj",
                        self.mesh_dir / f"{name}.fidx",
                    ]
                )
            if all(path.exists() for path in required):
                out.append(name)
        return out

    def __len__(self) -> int:
        return len(self.names)

    def _load_brep(self, name: str) -> tuple[dgl.DGLGraph, torch.Tensor]:
        graph_path = self.graph_dir / f"{name}.bin"
        graph = load_graphs(str(graph_path))[0][0]
        if self.brep_seg_dir is not None:
            labels_np = np.loadtxt(self.brep_seg_dir / f"{name}.seg", dtype=np.int64, ndmin=1)
            labels = torch.tensor(np.ascontiguousarray(labels_np)).long()
        elif "y" in graph.ndata:
            labels = graph.ndata["y"].long().view(-1)
        else:
            raise FileNotFoundError(
                f"{name}: no B-rep labels found. Provide --brep_seg_dir or use graphs with ndata['y']."
            )

        if graph.number_of_nodes() != labels.shape[0]:
            raise ValueError(
                f"{name}: graph has {graph.number_of_nodes()} nodes, "
                f"but B-rep labels have {labels.shape[0]} labels"
            )
        graph.ndata["y"] = labels
        return _to_float32_graph(graph), labels

    def _load_mesh(self, name: str, brep_labels: torch.Tensor) -> dict[str, torch.Tensor]:
        verts_np, faces_np = _read_obj_tri_mesh(self.mesh_dir / f"{name}.obj")
        fidx_np = np.loadtxt(self.mesh_dir / f"{name}.fidx", dtype=np.int64, ndmin=1)
        if faces_np.shape[0] != fidx_np.shape[0]:
            raise ValueError(f"{name}: OBJ has {faces_np.shape[0]} faces, fidx has {fidx_np.shape[0]} rows")

        verts = torch.tensor(np.ascontiguousarray(verts_np)).float()
        faces = torch.tensor(np.ascontiguousarray(faces_np)).long()
        fidx = torch.tensor(np.ascontiguousarray(fidx_np)).long()

        if int(fidx.max()) >= brep_labels.shape[0] or int(fidx.min()) < 0:
            raise ValueError(f"{name}: fidx range is outside B-rep face count {brep_labels.shape[0]}")

        mesh_seg_path = self.mesh_dir / f"{name}.seg"
        if mesh_seg_path.exists():
            mesh_labels_np = np.loadtxt(mesh_seg_path, dtype=np.int64, ndmin=1)
            mesh_labels = torch.tensor(np.ascontiguousarray(mesh_labels_np)).long()
        else:
            mesh_labels = brep_labels[fidx]

        if mesh_labels.shape[0] != faces.shape[0]:
            raise ValueError(f"{name}: mesh labels do not match triangle face count")

        verts = diffusion_net.geometry.normalize_positions(verts)
        # scipy.sparse.linalg.eigsh requires k < num_vertices. MFCAD++ has
        # small meshes, so cap k per sample just like diffusion-net/mfcad_ver.
        effective_k = min(self.k_eig, max(0, verts.shape[0] - 2))
        if effective_k != self.k_eig:
            print(f"[WARN] {self.split}/{name}: using k_eig={effective_k} for {verts.shape[0]} vertices")
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(
            verts,
            faces,
            k_eig=effective_k,
            op_cache_dir=self.op_cache_dir,
        )
        return {
            "verts": verts,
            "faces": faces,
            "frames": frames,
            "mass": mass,
            "L": L,
            "evals": evals,
            "evecs": evecs,
            "gradX": gradX,
            "gradY": gradY,
            "fidx": fidx,
            "mesh_labels": mesh_labels,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        name = self.names[idx]
        graph, brep_labels = self._load_brep(name)
        sample: dict[str, Any] = {
            "name": name,
            "brep_graph": graph,
            "brep_labels": brep_labels,
        }
        if not self.load_mesh:
            return sample
        sample.update(self._load_mesh(name, brep_labels))
        return sample


def build_mesh_features(sample: dict[str, Any], input_features: str) -> torch.Tensor:
    if input_features == "xyz":
        return sample["verts"]
    if input_features == "hks":
        return diffusion_net.geometry.compute_hks_autoscale(sample["evals"], sample["evecs"], 16)
    raise ValueError("input_features must be one of: xyz, hks")


def _with_batched_brep_graph(sample: dict[str, Any]) -> dict[str, Any]:
    sample = dict(sample)
    sample["brep_graph"] = dgl.batch([sample["brep_graph"]])
    return sample


def collate_distill_samples(batch: list[dict[str, Any]]) -> dict[str, Any] | list[dict[str, Any]]:
    """Keep variable-size samples separate; training iterates through the list."""

    batch = [_with_batched_brep_graph(sample) for sample in batch]
    return batch[0] if len(batch) == 1 else batch


def collate_brep_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch B-rep-only samples for FOVNet teacher training."""

    return {
        "name": [sample["name"] for sample in batch],
        "brep_graph": dgl.batch([sample["brep_graph"] for sample in batch]),
        "brep_labels": torch.cat([sample["brep_labels"] for sample in batch]),
    }
