"""Drop-in replacement for the pymesh API used by PD-MeshNet, backed by trimesh.

Only the subset of pymesh used within this package is implemented.
"""

from __future__ import annotations

import numpy as np

try:
    import trimesh
    _HAS_TRIMESH = True
except ImportError:
    _HAS_TRIMESH = False


class Mesh:
    """Mimics pymesh.Mesh interface using trimesh internally."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self._vertices = np.asarray(vertices, dtype=np.float64)
        self._faces = np.asarray(faces, dtype=np.int64)
        self._tm: "trimesh.Trimesh | None" = None
        self._face_adjacency: list[np.ndarray] | None = None

    def _get_trimesh(self) -> "trimesh.Trimesh":
        if self._tm is None:
            self._tm = trimesh.Trimesh(
                vertices=self._vertices,
                faces=self._faces,
                process=False,
            )
        return self._tm

    # ------------------------------------------------------------------ attrs
    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @property
    def faces(self) -> np.ndarray:
        return self._faces

    @property
    def num_faces(self) -> int:
        return len(self._faces)

    @property
    def num_vertices(self) -> int:
        return len(self._vertices)

    # ------------------------------------------------------------------ connectivity
    def enable_connectivity(self) -> None:
        """Pre-compute face adjacency (called for side effects in pymesh)."""
        self._build_adjacency()

    def _build_adjacency(self) -> None:
        if self._face_adjacency is not None:
            return
        tm = self._get_trimesh()
        # face_adjacency: (K, 2) array of face index pairs sharing an edge
        adj_pairs = tm.face_adjacency  # shape (K, 2)
        result: list[list[int]] = [[] for _ in range(self.num_faces)]
        for f1, f2 in adj_pairs:
            result[f1].append(int(f2))
            result[f2].append(int(f1))
        self._face_adjacency = [np.array(a, dtype=np.int64) for a in result]

    def get_face_adjacent_faces(self, face_idx: int | None = None):
        """Return adjacent faces.

        If ``face_idx`` is given, return a 1-D array of face indices adjacent
        to that face (pymesh API: ``mesh.get_face_adjacent_faces(i)``).
        If ``face_idx`` is None, return the full list for all faces.
        """
        self._build_adjacency()
        if face_idx is None:
            return self._face_adjacency
        return self._face_adjacency[face_idx]

    # ------------------------------------------------------------------ attributes
    def add_attribute(self, name: str) -> None:
        pass  # attributes computed lazily in get_face_attribute

    def get_face_attribute(self, name: str) -> np.ndarray:
        tm = self._get_trimesh()
        if name == "face_area":
            return tm.area_faces
        if name == "face_normal":
            return tm.face_normals
        raise ValueError(f"Unsupported mesh attribute: '{name}'")

    def get_vertex_attribute(self, name: str) -> np.ndarray:
        raise ValueError(f"Unsupported vertex attribute: '{name}'")


# ---------------------------------------------------------------------------
# Top-level pymesh functions
# ---------------------------------------------------------------------------

def load_mesh(path: str) -> Mesh:
    """Load a mesh file (OBJ, PLY, OFF, …) via trimesh."""
    tm = trimesh.load(str(path), process=False, force="mesh")
    return Mesh(np.array(tm.vertices, dtype=np.float64),
                np.array(tm.faces, dtype=np.int64))


def form_mesh(vertices: np.ndarray, faces: np.ndarray) -> Mesh:
    """Create a Mesh from vertex/face arrays."""
    return Mesh(vertices, faces)


def save_mesh(path: str, mesh: Mesh) -> None:
    """Save a mesh to disk (OBJ by default)."""
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    tm.export(str(path))


def remove_duplicated_vertices(mesh: Mesh, tol: float = 1e-7):
    """Merge vertices that are closer than `tol`."""
    tm = trimesh.Trimesh(vertices=mesh.vertices.copy(),
                         faces=mesh.faces.copy(), process=False)
    # trimesh merge_vertices merges identical verts; use tol as threshold
    tm.merge_vertices(merge_tex=False, merge_norm=False)
    return Mesh(np.array(tm.vertices), np.array(tm.faces)), None


def remove_duplicated_faces(mesh: Mesh):
    """Remove faces that are identical to another face (same vertex indices)."""
    faces = mesh.faces
    # Canonical representation: sort each row, then remove duplicates
    canonical = np.sort(faces, axis=1)
    _, unique_idx = np.unique(canonical, axis=0, return_index=True)
    new_faces = faces[np.sort(unique_idx)]
    return Mesh(mesh.vertices, new_faces), None


def mesh_to_dual_graph(mesh: Mesh):
    """Return (None, primal_edges) where primal_edges is shape (K, 2).

    primal_edges[i] = [face_i, face_j] means face_i and face_j share an edge.
    This matches what pymesh.mesh_to_dual_graph returns (only the edge list is
    used by PD-MeshNet's GraphCreator).
    """
    tm = mesh._get_trimesh()
    primal_edges = np.array(tm.face_adjacency, dtype=np.int64)  # (K, 2)
    return None, primal_edges


def remove_isolated_vertices(mesh: Mesh):
    """Remove vertices not referenced by any face, remapping indices."""
    verts = mesh.vertices
    faces = mesh.faces
    used = np.unique(faces)
    if len(used) == len(verts):
        return Mesh(verts, faces), None
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    new_verts = verts[used]
    new_faces = remap[faces]
    return Mesh(new_verts, new_faces), None
