"""PoissonNet student for B2Mesh distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .layers import PoissonNet
from .poisson_ops import compute_face_areas, compute_gradient_coeffs, build_cholesky_solver


class PoissonMeshStudent(nn.Module):
    """PoissonNet on triangle meshes with per-face output and B2Mesh interface.

    Reuses verts/faces/mass/L already computed by BRepMeshDataset._load_mesh()
    — no torch_mesh_ops required.

    Operator caching:
      - G_coeffs and M are saved to <solver_cache_dir>/<name>.pt on first encounter
        and reloaded on subsequent runs.
      - The cholespy solver (not serialisable) is kept in self._solver_cache for
        the lifetime of the process, so each mesh is factorised at most once per run.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_classes: int = 8,
        width: int = 128,
        blocks: int = 4,
        solver_cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.encoder = PoissonNet(
            C_in=3,
            C_out=embedding_dim,
            C_width=width,
            n_blocks=blocks,
            outputs_at="faces",
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, num_classes),
        )
        self._cache_dir = Path(solver_cache_dir) if solver_cache_dir else None
        self._solver_cache: dict[str, Any] = {}

    def _get_operators(
        self, name: str | None, verts: torch.Tensor, faces: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (G_coeffs, M), loading from disk cache when available."""
        if self._cache_dir is not None and name is not None:
            cache_path = self._cache_dir / f"{name}.pt"
            if cache_path.exists():
                cached = torch.load(cache_path, map_location=device, weights_only=True)
                return cached["G_coeffs"], cached["M"]
            G_coeffs = compute_gradient_coeffs(verts, faces)
            M = compute_face_areas(verts, faces).repeat_interleave(2).unsqueeze(0)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"G_coeffs": G_coeffs.cpu(), "M": M.cpu()}, cache_path)
            return G_coeffs, M

        G_coeffs = compute_gradient_coeffs(verts, faces)
        M = compute_face_areas(verts, faces).repeat_interleave(2).unsqueeze(0)
        return G_coeffs, M

    def _get_solver(self, name: str | None, L: torch.Tensor, V: int) -> Any:
        """Return cholespy solver, building and caching in memory on first call."""
        key = name if name is not None else id(L)
        if key not in self._solver_cache:
            self._solver_cache[key] = build_cholesky_solver(L, V)
        return self._solver_cache[key]

    def forward(self, sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        verts = sample["verts"]   # (V, 3)
        faces = sample["faces"]   # (F, 3)
        mass = sample["mass"]     # (V,)
        L = sample["L"]           # cotangent Laplacian, sparse (V, V)
        name: str | None = sample.get("name")

        device = verts.device
        V = verts.shape[0]

        G_coeffs, M = self._get_operators(name, verts, faces, device)
        solver = self._get_solver(name, L, V)

        x_in = verts.unsqueeze(0)   # (1, V, 3)  — PoissonNet always uses xyz

        face_emb = self.encoder(
            x_in=x_in,
            M=M,
            G_coeffs=G_coeffs,
            solver=[solver],
            faces=faces.unsqueeze(0),
            vertex_mass=mass.unsqueeze(0),
        ).squeeze(0)                    # (F, embedding_dim)

        logits = self.classifier(face_emb)  # (F, num_classes)
        return logits, face_emb
