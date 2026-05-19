"""MFCAD++ dataset for PD-MeshNet (primal-dual graph pairs) with B-rep pairing."""

from __future__ import annotations

import os
import os.path as osp
import glob
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


class MFCADPlusPlusDualPrimal(BaseDualPrimalDataset):
    """MFCAD++ mesh segmentation dataset as primal-dual graph pairs.

    Each sample contains:
      - primal_graph / dual_graph / petdni: PD-MeshNet graph structures
      - brep_graph:  DGL B-rep face graph (teacher input)
      - brep_labels: per-B-rep-face class labels
      - mesh_labels: per-triangle class labels (supervision for student)
      - fidx:        triangle → B-rep face index mapping (for distillation)
      - name:        sample ID string

    Expected on-disk layout::

        data_root/
          meshes/{split}/{id}.obj   - triangulated OBJ mesh
          meshes/{split}/{id}.seg   - per-face integer labels
          meshes/{split}/{id}.fidx  - per-face B-rep face index
          {split}/graphs/{id}.bin   - DGL B-rep graph (optional)
          {split}.txt               - list of sample IDs (one per line)

    Args:
        data_root (str | Path): Root directory of the MFCAD++ dataset.
        split (str): One of ``"train"``, ``"val"``, ``"test"``.
        single_dual_nodes (bool): GraphCreator option.
        undirected_dual_edges (bool): GraphCreator option.
        primal_features_from_dual_features (bool): GraphCreator option.
        prevent_nonmanifold_edges (bool): GraphCreator option.
        num_augmentations (int): Number of augmented copies per sample.
        vertices_scale_mean, vertices_scale_var (float | None): Vertex scaling augmentation.
        edges_flip_fraction (float | None): Edge-flip fraction augmentation.
        slide_vertices_fraction (float | None): Vertex-sliding fraction augmentation.
        graph_subdir (str): Subdirectory under ``{split}/`` that holds ``.bin`` graphs.
        load_brep (bool): If True, load the DGL B-rep graph for each sample.
        cache_graphs (bool): If True, cache computed primal/dual graphs to disk.
        cache_dir (str | Path | None): Directory for cached graphs. Defaults to
            ``data_root/pd_cache/{split}``.
    """

    NUM_CLASSES = 25  # MFCAD++ has 25 machining-feature classes (0-indexed)

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        single_dual_nodes: bool = True,
        undirected_dual_edges: bool = True,
        primal_features_from_dual_features: bool = False,
        prevent_nonmanifold_edges: bool = True,
        num_augmentations: int = 1,
        vertices_scale_mean: float | None = None,
        vertices_scale_var: float | None = None,
        edges_flip_fraction: float | None = None,
        slide_vertices_fraction: float | None = None,
        graph_subdir: str = "graphs",
        load_brep: bool = True,
        cache_graphs: bool = True,
        cache_dir: str | Path | None = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
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

        self._mesh_dir = self.data_root / "meshes" / split
        self._graph_dir = self.data_root / split / graph_subdir
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

        self._names = self._load_split_names()
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
        return [f"meshes/{self.split}"]

    @property
    def processed_file_names(self) -> list[str]:
        return []  # We manage caching ourselves

    def download(self) -> None:
        pass  # MFCAD++ is expected to be placed manually

    def process(self) -> None:
        pass  # Graph construction is on-the-fly / per-sample cached

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_split_names(self) -> list[str]:
        split_file = self.data_root / f"{self.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with split_file.open() as f:
            raw = [line.strip() for line in f if line.strip()]
        # Each line may be a full path or just the stem; extract stem
        names = []
        for entry in raw:
            stem = Path(entry).stem
            obj_path = self._mesh_dir / f"{stem}.obj"
            seg_path = self._mesh_dir / f"{stem}.seg"
            if obj_path.exists() and seg_path.exists():
                names.append(stem)
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
            return primal_graph, dual_graph, petdni, verts, faces

        # Load mesh
        mesh_path = self._mesh_dir / f"{name}.obj"
        mesh = _load_mesh(str(mesh_path))
        mesh = preprocess_mesh(
            input_mesh=mesh,
            prevent_nonmanifold_edges=self._prevent_nonmanifold_edges,
        )

        # Data augmentation
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

        # Build primal-dual graphs
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

        return primal_graph, dual_graph, petdni, verts, faces

    def _load_labels_and_fidx(self, name: str, primal_graph) -> tuple[torch.Tensor, torch.Tensor]:
        seg_np = np.loadtxt(self._mesh_dir / f"{name}.seg", dtype=np.int64, ndmin=1)
        labels = torch.tensor(seg_np).long()

        fidx_path = self._mesh_dir / f"{name}.fidx"
        if fidx_path.exists():
            fidx_np = np.loadtxt(fidx_path, dtype=np.int64, ndmin=1)
            fidx = torch.tensor(fidx_np).long()
        else:
            # No B-rep correspondence: identity mapping (each face → itself)
            fidx = torch.arange(primal_graph.num_nodes, dtype=torch.long)

        # GraphCreator may remove non-manifold faces, so num_nodes can differ
        # from original face count. Truncate labels to match.
        n = primal_graph.num_nodes
        if labels.shape[0] > n:
            labels = labels[:n]
        if fidx.shape[0] > n:
            fidx = fidx[:n]

        return labels, fidx

    def _load_brep_graph(self, name: str) -> tuple[dgl.DGLGraph | None, torch.Tensor | None]:
        bin_path = self._graph_dir / f"{name}.bin"
        if not bin_path.exists():
            return None, None
        graph = load_graphs(str(bin_path))[0][0]
        for key in list(graph.ndata.keys()):
            if key != "y":
                graph.ndata[key] = graph.ndata[key].float()
        for key in list(graph.edata.keys()):
            graph.edata[key] = graph.edata[key].float()
        if "y" in graph.ndata:
            brep_labels = graph.ndata["y"].long().view(-1)
        else:
            brep_labels = None
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

        primal_graph, dual_graph, petdni, verts, faces = self._load_or_compute_graphs(name, aug_idx)
        mesh_labels, fidx = self._load_labels_and_fidx(name, primal_graph)

        # Attach labels to primal graph so they travel with it
        primal_graph.y = mesh_labels

        sample: dict[str, Any] = {
            "name": name,
            "primal_graph": primal_graph,
            "dual_graph": dual_graph,
            "petdni": petdni,
            "mesh_labels": mesh_labels,
            "fidx": fidx,
            "verts": verts,   # for PLY visualization
            "faces": faces,   # for PLY visualization
        }

        if self._load_brep:
            brep_graph, brep_labels = self._load_brep_graph(name)
            if brep_graph is not None:
                sample["brep_graph"] = brep_graph
                sample["brep_labels"] = brep_labels

        return sample

    def __repr__(self) -> str:
        return f"MFCADPlusPlusDualPrimal(split={self.split}, n={len(self._names)})"
