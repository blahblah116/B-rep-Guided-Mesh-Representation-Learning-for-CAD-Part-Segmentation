"""Fusion360 Gallery dataset for PD-MeshNet with B-rep pairing.

Fusion360 OBJ meshes contain quad faces; this dataset triangulates them
automatically before building primal-dual graphs.
"""

from __future__ import annotations

import json
import os.path as osp
import pickle as pkl
from pathlib import Path
from typing import Any

import numpy as np
from ..utils._pymesh_compat import Mesh as _Mesh, load_mesh as _load_mesh, form_mesh as _form_mesh
import torch
import dgl
from dgl.data.utils import load_graphs

from ..datasets import BaseDualPrimalDataset
from ..utils import GraphCreator, preprocess_mesh
from ..data import augmentation, post_augmentation


def _triangulate_quad_mesh(mesh: "pymesh.Mesh") -> "pymesh.Mesh":
    """Split each quad face into 2 triangles (0-1-2, 0-2-3)."""
    verts = mesh.vertices
    quads = mesh.faces
    if quads.shape[1] == 3:
        return mesh  # Already triangulated
    if quads.shape[1] != 4:
        raise ValueError(f"Expected triangle or quad mesh, got faces with {quads.shape[1]} vertices.")

    tris = np.empty((len(quads) * 2, 3), dtype=np.int64)
    tris[0::2] = quads[:, [0, 1, 2]]
    tris[1::2] = quads[:, [0, 2, 3]]
    return _form_mesh(verts, tris)


def _duplicate_labels_for_quads(labels: np.ndarray, original_faces: np.ndarray) -> np.ndarray:
    """Duplicate per-quad labels so each resulting triangle pair gets the same label."""
    if original_faces.shape[1] == 3:
        return labels
    # Each quad → 2 triangles
    dup = np.empty(len(labels) * 2, dtype=labels.dtype)
    dup[0::2] = labels
    dup[1::2] = labels
    return dup


def _duplicate_fidx_for_quads(fidx: np.ndarray, original_faces: np.ndarray) -> np.ndarray:
    """Duplicate fidx so each triangle pair from a quad maps to the same B-rep face."""
    if original_faces.shape[1] == 3:
        return fidx
    dup = np.empty(len(fidx) * 2, dtype=fidx.dtype)
    dup[0::2] = fidx
    dup[1::2] = fidx
    return dup


class Fusion360DualPrimal(BaseDualPrimalDataset):
    """Fusion360 Gallery mesh segmentation as primal-dual graph pairs.

    Each sample contains:
      - primal_graph / dual_graph / petdni: PD-MeshNet graph structures
      - brep_graph:  DGL B-rep face graph (teacher input)
      - brep_labels: per-B-rep-face class labels
      - mesh_labels: per-triangle class labels (supervision for student)
      - fidx:        triangle → B-rep face index mapping
      - name:        sample ID string

    Expected on-disk layout (Fusion360 Gallery v2 / s2.0.1)::

        data_root/
          meshes/{id}.obj   - OBJ mesh (may contain quads → auto-triangulated)
          meshes/{id}.seg   - per-face class labels
          meshes/{id}.fidx  - per-face B-rep face index
          train_test_new.json           - split JSON
          {split}/graphs/{id}.bin       - DGL B-rep graph (optional)

    Args:
        data_root (str | Path): Root directory (e.g. ``.../fusion360/s2.0.1``).
        brep_graph_root (str | Path | None): Parent dir that holds
            ``{split}/graphs/``. Defaults to ``data_root.parent``.
        split (str): One of ``"train"``, ``"val"``, ``"test"``.
        split_file (str): JSON file name relative to data_root.
        mesh_dir (str): Subdirectory under data_root for mesh files.
        graph_subdir (str): Subdirectory under ``{brep_graph_root}/{split}/``.
        single_dual_nodes, undirected_dual_edges,
        primal_features_from_dual_features,
        prevent_nonmanifold_edges: GraphCreator options.
        num_augmentations (int): Augmented copies per sample.
        vertices_scale_mean, vertices_scale_var,
        edges_flip_fraction, slide_vertices_fraction: Augmentation params.
        load_brep (bool): Load DGL B-rep graph if available.
        cache_graphs (bool): Cache computed primal/dual graphs.
        cache_dir (str | Path | None): Graph cache directory.
        limit (int | None): Cap the number of samples (useful for debug).
    """

    NUM_CLASSES = 8  # Fusion360 Gallery segment types

    def __init__(
        self,
        data_root: str | Path,
        brep_graph_root: str | Path | None = None,
        split: str = "train",
        split_file: str = "train_test_new.json",
        mesh_dir: str = "meshes",
        graph_subdir: str = "graphs",
        single_dual_nodes: bool = True,
        undirected_dual_edges: bool = True,
        primal_features_from_dual_features: bool = False,
        prevent_nonmanifold_edges: bool = True,
        num_augmentations: int = 1,
        vertices_scale_mean: float | None = None,
        vertices_scale_var: float | None = None,
        edges_flip_fraction: float | None = None,
        slide_vertices_fraction: float | None = None,
        load_brep: bool = True,
        cache_graphs: bool = True,
        cache_dir: str | Path | None = None,
        limit: int | None = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self._mesh_dir = self.data_root / mesh_dir
        self._brep_root = (
            Path(brep_graph_root) if brep_graph_root is not None else self.data_root.parent
        )
        self._graph_dir = self._brep_root / split / graph_subdir
        self._single_dual_nodes = single_dual_nodes
        self._undirected_dual_edges = undirected_dual_edges
        self._primal_features_from_dual_features = primal_features_from_dual_features
        self._prevent_nonmanifold_edges = prevent_nonmanifold_edges
        self._num_augmentations = num_augmentations
        self._vertices_scale_mean = vertices_scale_mean
        self._vertices_scale_var = vertices_scale_var
        self._edges_flip_fraction = edges_flip_fraction
        self._slide_vertices_fraction = slide_vertices_fraction
        self._load_brep = load_brep
        self._cache_graphs = cache_graphs
        self._cache_dir = (
            Path(cache_dir) if cache_dir is not None
            else self.data_root / "pd_cache" / split
        )
        self._input_parameters = {
            "data_root": str(self.data_root),
            "split": split,
            "single_dual_nodes": single_dual_nodes,
            "undirected_dual_edges": undirected_dual_edges,
            "primal_features_from_dual_features": primal_features_from_dual_features,
            "prevent_nonmanifold_edges": prevent_nonmanifold_edges,
            "num_augmentations": num_augmentations,
        }

        self._names = self._load_split_names(split_file, limit)
        super().__init__(
            root=str(self.data_root),
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    # ------------------------------------------------------------------
    # BaseDualPrimalDataset interface
    # ------------------------------------------------------------------
    @property
    def input_parameters(self) -> dict[str, Any]:
        return self._input_parameters

    @property
    def raw_file_names(self) -> list[str]:
        return ["meshes"]

    @property
    def processed_file_names(self) -> list[str]:
        return []

    def download(self) -> None:
        pass

    def process(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_split_names(self, split_file: str, limit: int | None) -> list[str]:
        split_path = self.data_root / split_file
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        with split_path.open() as f:
            data = json.load(f)
        key = self.split
        if key not in data:
            # Some JSON files use "test" for both val/test
            if key == "val" and "test" in data:
                key = "test"
            else:
                raise ValueError(f"Split '{self.split}' not found in {split_path}")
        raw_names = [str(n) for n in data[key]]

        # Filter to names that have OBJ + seg on disk
        names = []
        for name in raw_names:
            if (self._mesh_dir / f"{name}.obj").exists() and (self._mesh_dir / f"{name}.seg").exists():
                names.append(name)
        if limit is not None:
            names = names[:limit]
        if not names:
            raise FileNotFoundError(
                f"No valid OBJ+seg pairs found for split '{self.split}' in {self._mesh_dir}"
            )
        return names

    def _cache_paths(self, name: str, aug_idx: int):
        base = self._cache_dir / f"{name}_aug{aug_idx}"
        return (
            base.with_suffix(".primal.pt"),
            base.with_suffix(".dual.pt"),
            base.with_suffix(".petdni.pkl"),
            base.parent / f"{base.stem}_verts.npy",
            base.parent / f"{base.stem}_faces.npy",
        )

    def _load_or_compute_graphs(self, name: str, aug_idx: int):
        p_path, d_path, petdni_path, verts_path, faces_path = self._cache_paths(name, aug_idx)

        if self._cache_graphs and all(p.exists() for p in (p_path, d_path, petdni_path, verts_path, faces_path)):
            primal_graph = torch.load(p_path)
            dual_graph = torch.load(d_path)
            with open(petdni_path, "rb") as f:
                petdni = pkl.load(f)
            verts = torch.from_numpy(np.load(verts_path)).float()
            faces = torch.from_numpy(np.load(faces_path)).long()
            return primal_graph, dual_graph, petdni, None, verts, faces

        mesh_path = self._mesh_dir / f"{name}.obj"
        mesh = _load_mesh(str(mesh_path))
        original_faces = mesh.faces.copy()

        # Triangulate if needed
        mesh = _triangulate_quad_mesh(mesh)
        mesh = preprocess_mesh(
            input_mesh=mesh,
            prevent_nonmanifold_edges=self._prevent_nonmanifold_edges,
        )

        aug_mesh = augmentation(
            mesh=mesh,
            vertices_scale_mean=self._vertices_scale_mean,
            vertices_scale_var=self._vertices_scale_var,
            edges_flip_fraction=self._edges_flip_fraction,
        )
        aug_mesh = post_augmentation(
            mesh=aug_mesh,
            slide_vertices_fraction=self._slide_vertices_fraction,
        )

        gc = GraphCreator(
            mesh=aug_mesh,
            single_dual_nodes=self._single_dual_nodes,
            undirected_dual_edges=self._undirected_dual_edges,
            primal_features_from_dual_features=self._primal_features_from_dual_features,
            prevent_nonmanifold_edges=self._prevent_nonmanifold_edges,
        )
        primal_graph, dual_graph = gc.create_graphs()
        petdni = gc.primal_edge_to_dual_node_idx

        verts = torch.from_numpy(np.ascontiguousarray(aug_mesh.vertices)).float()
        faces = torch.from_numpy(np.ascontiguousarray(aug_mesh.faces)).long()

        if self._cache_graphs:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(primal_graph, p_path)
            torch.save(dual_graph, d_path)
            with open(petdni_path, "wb") as f:
                pkl.dump(petdni, f)
            np.save(verts_path, aug_mesh.vertices)
            np.save(faces_path, aug_mesh.faces)

        return primal_graph, dual_graph, petdni, original_faces, verts, faces

    def _load_labels_and_fidx(
        self, name: str, primal_graph, original_faces
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seg_np = np.loadtxt(self._mesh_dir / f"{name}.seg", dtype=np.int64, ndmin=1)
        fidx_path = self._mesh_dir / f"{name}.fidx"
        if fidx_path.exists():
            fidx_np = np.loadtxt(fidx_path, dtype=np.int64, ndmin=1)
        else:
            fidx_np = np.arange(len(seg_np), dtype=np.int64)

        # Duplicate for quads if we still have the original face info
        if original_faces is not None and original_faces.shape[1] == 4:
            seg_np = _duplicate_labels_for_quads(seg_np, original_faces)
            fidx_np = _duplicate_fidx_for_quads(fidx_np, original_faces)

        n = primal_graph.num_nodes
        labels = torch.tensor(seg_np[:n]).long()
        fidx = torch.tensor(fidx_np[:n]).long()
        return labels, fidx

    def _load_brep_graph(self, name: str) -> tuple["dgl.DGLGraph | None", "torch.Tensor | None"]:
        bin_path = self._graph_dir / f"{name}.bin"
        if not bin_path.exists():
            return None, None
        graph = load_graphs(str(bin_path))[0][0]
        for key in list(graph.ndata.keys()):
            if key != "y":
                graph.ndata[key] = graph.ndata[key].float()
        for key in list(graph.edata.keys()):
            graph.edata[key] = graph.edata[key].float()
        brep_labels = graph.ndata["y"].long().view(-1) if "y" in graph.ndata else None
        return graph, brep_labels

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def len(self) -> int:
        return len(self._names) * self._num_augmentations

    def get(self, idx: int) -> dict[str, Any]:
        name_idx = idx // self._num_augmentations
        aug_idx = idx % self._num_augmentations
        name = self._names[name_idx]

        primal_graph, dual_graph, petdni, original_faces, verts, faces = self._load_or_compute_graphs(name, aug_idx)
        mesh_labels, fidx = self._load_labels_and_fidx(name, primal_graph, original_faces)
        primal_graph.y = mesh_labels

        sample: dict[str, Any] = {
            "name": name,
            "primal_graph": primal_graph,
            "dual_graph": dual_graph,
            "petdni": petdni,
            "mesh_labels": mesh_labels,
            "verts": verts,   # for PLY visualization
            "faces": faces,   # for PLY visualization
            "fidx": fidx,
        }

        if self._load_brep:
            brep_graph, brep_labels = self._load_brep_graph(name)
            if brep_graph is not None:
                sample["brep_graph"] = brep_graph
                sample["brep_labels"] = brep_labels

        return sample

    def __repr__(self) -> str:
        return f"Fusion360DualPrimal(split={self.split}, n={len(self._names)})"
