"""Losses for face-level mesh/B-rep distillation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_mask(
    teacher: torch.Tensor,
    student: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask is not None:
        teacher = teacher[mask]
        student = student[mask]
    return teacher, student


def rkd_distance_loss(
    teacher: torch.Tensor,
    student: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Distance-wise RKD loss over face pairs."""

    if teacher.shape[0] < 2:
        return teacher.new_zeros(())

    teacher_dist = torch.pdist(teacher.detach(), p=2)
    student_dist = torch.pdist(student, p=2)

    # From Original RKD(0518)
    teacher_dist = teacher_dist / teacher_dist.mean().clamp_min(eps)
    student_dist = student_dist / student_dist.mean().clamp_min(eps)
    return F.smooth_l1_loss(student_dist, teacher_dist)


def rkd_angle_loss(
    teacher: torch.Tensor,
    student: torch.Tensor,
) -> torch.Tensor:
    """Angle-wise RKD loss over distinct face triplets."""

    n_faces = teacher.shape[0]
    if n_faces < 3:
        return teacher.new_zeros(())

    teacher_vec = teacher.detach().unsqueeze(0) - teacher.detach().unsqueeze(1)
    student_vec = student.unsqueeze(0) - student.unsqueeze(1)
    teacher_vec = F.normalize(teacher_vec, p=2, dim=-1)
    student_vec = F.normalize(student_vec, p=2, dim=-1)

    teacher_angle = torch.bmm(teacher_vec, teacher_vec.transpose(1, 2))
    student_angle = torch.bmm(student_vec, student_vec.transpose(1, 2))

    idx = torch.arange(n_faces, device=teacher.device)
    anchor = idx.view(n_faces, 1, 1)
    j = idx.view(1, n_faces, 1)
    k = idx.view(1, 1, n_faces)
    distinct = (anchor != j) & (anchor != k) & (j != k)

    return F.smooth_l1_loss(student_angle[distinct], teacher_angle[distinct])


def relational_distillation_loss(
    teacher: torch.Tensor,
    student: torch.Tensor,
    mask: torch.Tensor | None = None,
    mode: str = "distance_angle",
) -> torch.Tensor:
    """RKD loss between teacher/student face embeddings.

    ``teacher`` and ``student`` are face embeddings with shape ``[F, C]``.
    ``mask`` can be used to ignore B-rep faces which have no mesh triangles.
    ``mode`` is one of ``distance``, ``angle``, or ``distance_angle``.
    """

    teacher, student = _apply_mask(teacher, student, mask)
    if mode == "distance":
        return rkd_distance_loss(teacher, student)
    if mode == "angle":
        return rkd_angle_loss(teacher, student)
    if mode == "distance_angle":
        return rkd_distance_loss(teacher, student) + 2 * rkd_angle_loss(teacher, student)
    raise ValueError(f"Unknown RKD mode: {mode}")


class FCRDLoss(nn.Module):
    """Face-level cross-modal contrastive distillation loss.

    Pulls together teacher and student embeddings that share the same B-rep
    face label (SupCon-style multi-positive), while pushing apart pairs with
    different labels. Objects where all faces share one label are skipped
    (no meaningful negatives exist).
    """

    def __init__(self, embedding_dim: int, proj_dim: int = 128, tau: float = 0.07) -> None:
        super().__init__()
        self.proj_t = nn.Linear(embedding_dim, proj_dim, bias=False)
        self.proj_s = nn.Linear(embedding_dim, proj_dim, bias=False)
        self.tau = tau

    def forward(
        self,
        teacher_emb: torch.Tensor,  # (F, embedding_dim)
        student_emb: torch.Tensor,  # (F, embedding_dim)
        labels: torch.Tensor,       # (F,)  B-rep face labels
        mask: torch.Tensor,         # (F,)  bool — faces with mesh coverage
    ) -> torch.Tensor:
        teacher_emb = teacher_emb[mask]
        student_emb = student_emb[mask]
        labels      = labels[mask]

        # Need at least two distinct labels to have any negatives.
        if labels.unique().shape[0] < 2:
            return teacher_emb.new_zeros(())

        z_t = F.normalize(self.proj_t(teacher_emb), dim=-1)   # (F', proj_dim)
        z_s = F.normalize(self.proj_s(student_emb), dim=-1)   # (F', proj_dim)

        # sim[i, j] = cosine similarity between teacher face i and student face j
        sim      = z_t @ z_s.T / self.tau                          # (F', F')
        pos_mask = (labels[:, None] == labels[None, :]).float()     # (F', F')

        # log P(j | i) = sim[i,j] - log Σ_k exp(sim[i,k])
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # (F', F')

        # Average log-prob over all positives for each anchor.
        loss = -(log_prob * pos_mask).sum(1) / pos_mask.sum(1)
        return loss.mean()


def scatter_mean_by_index(
    values: torch.Tensor,
    index: torch.Tensor,
    output_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average ``values`` into ``output_size`` groups using integer ``index``."""

    valid = (index >= 0) & (index < output_size)
    out = values.new_zeros((output_size, values.shape[-1]))
    counts = values.new_zeros((output_size, 1))
    if valid.any():
        idx = index[valid].long()
        out.index_add_(0, idx, values[valid])
        counts.index_add_(0, idx, torch.ones((idx.shape[0], 1), device=values.device, dtype=values.dtype))
    mask = counts.squeeze(-1) > 0
    out = out / counts.clamp_min(1.0)
    return out, mask
