"""
PoissonNet layers adapted for B2Mesh.

Key difference from the original: instead of passing a dense G matrix (2F, V)
computed by torch_mesh_ops, we pass G_coeffs (F, 2, 3) and use
poisson_ops.grad_forward / grad_backward which are memory-efficient and
require no custom CUDA extension.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .poisson_ops import grad_forward, grad_backward


# ---------------------------------------------------------------------------
# Helpers copied verbatim from poissonnet/networks/common.py
# (no torch_mesh_ops dependency in these)
# ---------------------------------------------------------------------------

class PoissonSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, solver, rhs: torch.Tensor):
        ctx.solver = solver
        ctx.device = rhs.device
        rhs_cpu = rhs.cpu()
        x_cpu = torch.zeros_like(rhs_cpu)
        solver.solve(rhs_cpu, x_cpu)
        return x_cpu.to(ctx.device)

    @staticmethod
    def backward(ctx, grad_output):
        f_grad = None
        if ctx.needs_input_grad[1]:
            grad_cpu = grad_output.contiguous().cpu()
            f_grad_cpu = torch.zeros_like(grad_cpu)
            ctx.solver.solve(grad_cpu, f_grad_cpu)
            f_grad = f_grad_cpu.to(ctx.device)
        del ctx.solver
        return None, f_grad


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(x.new_full(shape, keep))
        return x * mask / keep


class PoissonNetNorm(nn.Module):
    def __init__(self, mode: str, hidden_size: int, eps: float = 1e-12) -> None:
        super().__init__()
        assert mode in ("vertex", "function")
        self.mode = mode
        self.d = -1 if mode == "vertex" else -2
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, mass: torch.Tensor | None = None) -> torch.Tensor:
        if mass is not None and self.mode == "function":
            mass_sum = mass.sum(1, keepdim=True)
            mass = mass.unsqueeze(-1)
            u = (mass * x).sum(1) / (mass_sum + 1e-12)
            s = (mass * (x - u) ** 2).sum(1) / (mass_sum + 1e-12)
        else:
            u = x.mean(dim=self.d, keepdim=True)
            s = (x - u).pow(2).mean(dim=self.d, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight.view(1, 1, -1) + self.bias.view(1, 1, -1)


class Mlp(nn.Module):
    def __init__(self, in_c: int, out_c: int, width: int, drop: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_c, width), nn.GELU(), nn.Dropout(drop), nn.Linear(width, out_c)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PoissonBlockMLP(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        width: int,
        drop: float = 0.0,
        drop_path: float = 0.0,
        norm: bool = False,
        extra_feats: int = 0,
    ) -> None:
        super().__init__()
        self.norm_pde = PoissonNetNorm("function", in_c // 2) if norm else nn.Identity()
        in_c = in_c + extra_feats
        self.mlp1 = Mlp(in_c, width, width, drop=drop)
        self.ls1 = nn.Parameter(torch.ones(out_c))
        self.dp1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.mlp2 = Mlp(width, out_c, width, drop=drop)
        self.ls2 = nn.Parameter(torch.ones(out_c))
        self.dp2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        pde_sol: torch.Tensor,
        extra_features: torch.Tensor | None = None,
        mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm_pde = self.norm_pde(pde_sol, mass) if isinstance(self.norm_pde, PoissonNetNorm) else pde_sol
        parts = [x, norm_pde]
        if extra_features is not None:
            parts.append(extra_features)
        mlp_in = torch.cat(parts, dim=-1)
        out = x + self.dp1(self.ls1 * self.mlp1(mlp_in))
        return out + self.dp2(self.ls2 * self.mlp2(out))


def vertices_to_faces(x: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """(B,V,C) + (B,F,3) -> (B,F,C) by averaging the 3 corner features."""
    x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
    f_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
    return torch.gather(x_gather, 1, f_gather).mean(dim=-1)


# ---------------------------------------------------------------------------
# Complex-valued gradient MLP (from poissonnet/networks/PoissonNet.py)
# ---------------------------------------------------------------------------

class ComplexLayer(nn.Module):
    def __init__(self, in_c: int, out_c: int, nonlin: bool = True) -> None:
        super().__init__()
        self.lin_real = nn.Linear(in_c, out_c, bias=False)
        self.lin_imag = nn.Linear(in_c, out_c, bias=False)
        self.nonlin = nonlin
        if nonlin:
            self.mag_bias = nn.Parameter(torch.zeros(out_c))
            self.gelu = nn.GELU()

    def forward(self, re: torch.Tensor, im: torch.Tensor):
        yr = self.lin_real(re) - self.lin_imag(im)
        yi = self.lin_real(im) + self.lin_imag(re)
        if self.nonlin:
            r = torch.sqrt(yr ** 2 + yi ** 2 + 1e-8)
            scale = self.gelu(r + self.mag_bias) / (r + 1e-8)
            yr, yi = yr * scale, yi * scale
        return yr, yi


class ComplexMLP(nn.Module):
    def __init__(self, in_c: int, out_c: int, width: int, num_layers: int = 2) -> None:
        super().__init__()
        sizes = [in_c] + [width] * (num_layers - 1) + [out_c]
        self.layers = nn.ModuleList(
            [ComplexLayer(sizes[i], sizes[i + 1], nonlin=(i < num_layers - 1))
             for i in range(num_layers)]
        )

    def forward(self, f: torch.Tensor, x_face: torch.Tensor | None = None) -> torch.Tensor:
        """f: (B, 2F, C) interleaved; returns (B, 2F, C) interleaved."""
        re, im = f[:, 0::2, :], f[:, 1::2, :]
        for layer in self.layers:
            re, im = layer(re, im)
        B, F, C = re.shape
        out = torch.empty(B, 2 * F, C, device=f.device, dtype=f.dtype)
        out[:, 0::2, :] = re
        out[:, 1::2, :] = im
        return out


# ---------------------------------------------------------------------------
# PoissonBlock — uses G_coeffs instead of dense G
# ---------------------------------------------------------------------------

class PoissonBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, width: int, extra_feats: int = 0, config: dict | None = None) -> None:
        super().__init__()
        cfg = config or {}
        self.grad_mlp = ComplexMLP(in_c, out_c, width, num_layers=cfg.get("cmlp_nlayers", 2))
        self.vert_mlp = PoissonBlockMLP(
            in_c=in_c + out_c,
            out_c=out_c,
            width=width,
            drop=cfg.get("dropout_p", 0.0),
            drop_path=cfg.get("drop_path", 0.0),
            norm=cfg.get("mlp_norm", False),
            extra_feats=extra_feats,
        )
        self.mass_norm = cfg.get("mass_norm", False)

    def forward(
        self,
        x_in: torch.Tensor,        # (B, V, C)
        M: torch.Tensor,            # (B, 2F) interleaved face areas
        G_coeffs: torch.Tensor,     # (F, 2, 3) gradient coefficients
        solver: list,               # [B] cholespy solvers
        faces: torch.Tensor,        # (B, F, 3)
        vertex_mass: torch.Tensor,  # (B, V)
        extra_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, V, C = x_in.shape
        assert B == 1, "PoissonMeshStudent only supports batch_size=1"
        faces_2d = faces.squeeze(0)  # (F, 3)

        # Forward gradient: vertex signals → interleaved face gradients
        grads_in = grad_forward(G_coeffs, faces_2d, x_in)   # (B, 2F, C)
        x_face = vertices_to_faces(x_in, faces)              # (B, F, C)
        grads = self.grad_mlp(grads_in, x_face)              # (B, 2F, C)

        # Solve Poisson: L u = G^T (M * grads)
        rhs = grad_backward(G_coeffs, faces_2d, M, grads, V)   # (B, V, C)

        u = torch.empty_like(rhs)
        for b in range(B):
            for j in range(0, C, 128):
                u[b, :, j:j+128] = PoissonSolver.apply(solver[b], rhs[b, :, j:j+128].contiguous())

        # Nullify area-weighted mean
        u = u - (u * vertex_mass.unsqueeze(-1)).sum(1, keepdim=True) / vertex_mass.sum(1, keepdim=True).unsqueeze(-1)

        out = self.vert_mlp(x_in, u, extra_features, mass=vertex_mass if self.mass_norm else None)
        return out, grads


# ---------------------------------------------------------------------------
# PoissonNet — top-level model (linear head, outputs_at='faces')
# ---------------------------------------------------------------------------

def _output_at(x: torch.Tensor, faces: torch.Tensor, mass: torch.Tensor, domain: str) -> torch.Tensor:
    if domain in ("vertices", "verts"):
        return x
    if domain == "faces":
        return vertices_to_faces(x, faces)
    if domain == "global_mean":
        return (x * mass.unsqueeze(-1)).sum(-2) / mass.sum(-1, keepdim=True)
    raise ValueError(f"Unknown domain: {domain}")


class PoissonNet(nn.Module):
    def __init__(
        self,
        C_in: int,
        C_out: int,
        C_width: int = 128,
        n_blocks: int = 4,
        outputs_at: str = "faces",
        config: dict | None = None,
    ) -> None:
        super().__init__()
        self.outputs_at = outputs_at
        cfg = config or {}
        self.proj_in = nn.Linear(C_in, C_width)
        self.blocks = nn.ModuleList(
            [PoissonBlock(C_width, C_width, C_width, config=cfg) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(C_width, C_out)

    def forward(
        self,
        x_in: torch.Tensor,        # (B, V, C_in)
        M: torch.Tensor,            # (B, 2F)
        G_coeffs: torch.Tensor,     # (F, 2, 3)
        solver: list,               # [B] cholespy solvers
        faces: torch.Tensor,        # (B, F, 3)
        vertex_mass: torch.Tensor,  # (B, V)
    ) -> torch.Tensor:
        x = self.proj_in(x_in)
        for block in self.blocks:
            x, _ = block(x, M, G_coeffs, solver, faces, vertex_mass)
        out = self.head(x)
        return _output_at(out, faces, vertex_mass, self.outputs_at)
