"""Losses for face-level mesh/B-rep distillation."""

from __future__ import annotations

import torch
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


def feature_alignment_loss(
    teacher: torch.Tensor,
    student: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Optional direct normalized feature alignment."""

    if mask is not None:
        teacher = teacher[mask]
        student = student[mask]
    if teacher.numel() == 0:
        return teacher.new_zeros(())
    return F.mse_loss(F.normalize(student, dim=-1), F.normalize(teacher.detach(), dim=-1))


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
