"""
Geometric mesh operators for PoissonNet — replaces torch_mesh_ops.

All operators are computed in pure PyTorch from (verts, faces).
The cotangent Laplacian and vertex mass are reused from DiffusionNet's
get_operators() which is already run by BRepMeshDataset.
"""

from __future__ import annotations

import torch
import cholespy


def compute_face_areas(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """(V,3), (F,3) -> (F,) triangle areas."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return torch.linalg.cross(v1 - v0, v2 - v0).norm(dim=-1) / 2.0


def compute_gradient_coeffs(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Per-face intrinsic gradient operator coefficients.

    For face f = [i, j, k], the gradient of basis function phi_i within f is:
        grad3d_i = (n x e_opp_i) / (2*A_f)
    projected onto the face's local 2D frame (x_ax, y_ax).

    Returns: (F, 2, 3)
        coeffs[f, 0, local] = x-component of grad(phi_local) in face f
        coeffs[f, 1, local] = y-component of grad(phi_local) in face f

    Usage: (G_coeffs @ x_verts)[f] gives the 2D gradient of x at face f.
    """
    v0 = verts[faces[:, 0]]   # (F, 3)
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    e01 = v1 - v0             # (F, 3)
    n_unnorm = torch.linalg.cross(e01, v2 - v0)   # (F, 3)
    area2 = n_unnorm.norm(dim=-1, keepdim=True).clamp_min(1e-10)  # (F, 1) = 2*A
    n = n_unnorm / area2      # unit face normal

    # Local 2D frame: x along first edge, y = n x x
    x_ax = e01 / e01.norm(dim=-1, keepdim=True).clamp_min(1e-10)  # (F, 3)
    y_ax = torch.linalg.cross(n, x_ax)                             # (F, 3)

    # Edges opposite to each vertex: e_opp[v_local] is the edge NOT touching v_local
    e_opp = torch.stack([v2 - v1, v0 - v2, v1 - v0], dim=1)       # (F, 3, 3)

    # 3D gradient: grad3d_i = (n x e_opp_i) / (2*A)
    n_exp = n.unsqueeze(1).expand(-1, 3, -1)                        # (F, 3, 3)
    grad3d = torch.linalg.cross(n_exp, e_opp) / area2.unsqueeze(1) # (F, 3, 3)

    # Project onto local frame
    gx = (grad3d * x_ax.unsqueeze(1)).sum(-1)  # (F, 3)
    gy = (grad3d * y_ax.unsqueeze(1)).sum(-1)  # (F, 3)

    return torch.stack([gx, gy], dim=1)        # (F, 2, 3)


def grad_forward(
    G_coeffs: torch.Tensor,
    faces: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute G @ x: vertex signals → interleaved face gradients.

    G_coeffs : (F, 2, 3)
    faces     : (F, 3)  int
    x         : (B, V, C) or (V, C)
    returns   : (B, 2F, C)  interleaved [dx_0, dy_0, dx_1, dy_1, ...]
    """
    squeeze = x.ndim == 2
    if squeeze:
        x = x.unsqueeze(0)
    B, _, C = x.shape
    F = faces.shape[0]

    # Gather vertex features at the 3 corners of each face: (B, F, 3, C)
    x_vf = x[:, faces, :]  # works because faces is (F, 3)

    # Dot with gradient coefficients
    gx = (G_coeffs[None, :, 0, :, None] * x_vf).sum(2)  # (B, F, C)
    gy = (G_coeffs[None, :, 1, :, None] * x_vf).sum(2)  # (B, F, C)

    out = torch.zeros(B, 2 * F, C, device=x.device, dtype=x.dtype)
    out[:, 0::2, :] = gx
    out[:, 1::2, :] = gy

    return out.squeeze(0) if squeeze else out


def grad_backward(
    G_coeffs: torch.Tensor,
    faces: torch.Tensor,
    M: torch.Tensor,
    grads: torch.Tensor,
    num_verts: int,
) -> torch.Tensor:
    """
    Compute G^T @ (M * grads): weighted face gradients → vertex divergence.

    G_coeffs : (F, 2, 3)
    faces     : (F, 3)  int
    M         : (B, 2F) interleaved face areas
    grads     : (B, 2F, C) interleaved face gradients
    returns   : (B, V, C)
    """
    B, _, C = grads.shape
    F = faces.shape[0]

    gx = grads[:, 0::2, :]  # (B, F, C)
    gy = grads[:, 1::2, :]
    Mx = M[:, 0::2]          # (B, F)
    My = M[:, 1::2]

    result = torch.zeros(B, num_verts, C, device=grads.device, dtype=grads.dtype)
    for li in range(3):
        v_idx = faces[:, li]                          # (F,)
        cx = G_coeffs[:, 0, li]                       # (F,)
        cy = G_coeffs[:, 1, li]                       # (F,)
        contrib = (
            (Mx * cx[None]).unsqueeze(-1) * gx        # (B, F, C)
            + (My * cy[None]).unsqueeze(-1) * gy
        )
        result.scatter_add_(
            1,
            v_idx[None, :, None].expand(B, F, C),
            contrib,
        )
    return result


def build_cholesky_solver(
    L: torch.Tensor,
    num_verts: int,
) -> cholespy.CholeskySolverF:
    """
    Build a cholespy Cholesky solver from the cotangent Laplacian L.

    L is the sparse cotangent Laplacian returned by DiffusionNet's
    get_operators() — it is positive semi-definite.  A small diagonal
    regulariser is added to make it strictly positive definite.
    """
    # L may be on CUDA; cholespy needs COO indices on CPU.
    L_cpu = L.cpu().coalesce()

    # Scale eps relative to the mean diagonal of L so the regulariser is
    # meaningful regardless of mesh size / edge length scale.
    idx = L_cpu.indices()
    diag_mean = L_cpu.values()[idx[0] == idx[1]].abs().mean().item()
    eps = max(1e-5 * diag_mean, 1e-8)

    diag_idx = torch.arange(num_verts, dtype=torch.long)
    for _ in range(6):
        diag_sp = torch.sparse_coo_tensor(
            torch.stack([diag_idx, diag_idx]),
            torch.full((num_verts,), eps, dtype=torch.float32),
            (num_verts, num_verts),
        ).coalesce()

        L_reg = (L_cpu + diag_sp).coalesce()
        ii = L_reg.indices()[0]
        jj = L_reg.indices()[1]
        vals = L_reg.values().float()

        try:
            solver = cholespy.CholeskySolverF(
                num_verts, ii, jj, vals, cholespy.MatrixType.COO
            )
            return solver
        except Exception:
            eps *= 10.0

    raise RuntimeError("Cholesky decomposition failed after 6 retries.")
