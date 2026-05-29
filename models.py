"""Teacher and student models for B-rep/mesh distillation."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import diffusion_net
from .fovnet import FOVNet


def build_mesh_features(sample: dict[str, Any], input_features: str) -> torch.Tensor:
    if input_features == "xyz":
        return sample["verts"]
    if input_features == "hks":
        return diffusion_net.geometry.compute_hks_autoscale(sample["evals"], sample["evecs"], 16)
    raise ValueError("input_features must be one of: xyz, hks")


class FOVNetFaceTeacher(nn.Module):
    """FOVNet segmentation model which also returns per-B-rep-face embeddings."""

    def __init__(self, num_classes: int = 8, **kwargs: Any) -> None:
        super().__init__()
        kwargs.setdefault("segmentation", True)
        self.model = FOVNet(num_classes=num_classes, **kwargs)

    def forward(self, graph):
        m = self.model

        # Use local_scope so ndata/edata mutations do not persist on the original graph.
        # Without this, popping unused keys would destroy the sample's brep_graph in-place,
        # breaking any second access to the same sample (e.g., gradient accumulation).
        with graph.local_scope():
            if m.use_uv:
                for key in ("x", "x_local"):
                    if key in graph.ndata and graph.ndata[key].dim() == 4 and graph.ndata[key].shape[-1] == 7:
                        graph.ndata[key] = graph.ndata[key].permute(0, 3, 1, 2).contiguous()
            if m.vision and "vision_grids" in graph.ndata:
                vg = graph.ndata["vision_grids"]
                if vg.dim() == 4 and vg.shape[-1] == 6:
                    graph.ndata["vision_grids"] = vg.permute(0, 3, 1, 2).contiguous()

            keep = {m.uv_key, "vision_grids"} | ({"face_feat"} if m.use_face_feat else set())
            for key in list(graph.ndata.keys()):
                if key not in keep:
                    graph.ndata.pop(key)
            for key in list(graph.edata.keys()):
                graph.edata.pop(key)

            parts = []
            if m.use_uv and m.uv_key in graph.ndata:
                parts.append(m.surf_encoder(graph.ndata[m.uv_key]))
            vg = graph.ndata.get("vision_grids")
            if m.vision and vg is not None:
                if m.ov_encoder:
                    parts.append(m.ov_encoder(vg[:, m.ov_channels]))
                if m.iv_encoder:
                    parts.append(m.iv_encoder(vg[:, m.iv_channels]))
            if m.use_face_feat and "face_feat" in graph.ndata:
                parts.append(graph.ndata["face_feat"][:, :7])

            hidden = m.shared_fc(torch.cat(parts, dim=1))
            node_emb, graph_emb = m.graph_encoder(graph, hidden)
            expanded = graph_emb.repeat_interleave(graph.batch_num_nodes().to(graph_emb.device), dim=0)
            logits = m.seg(torch.cat((node_emb, expanded), dim=1))

        return logits, node_emb


class DiffusionMeshStudent(nn.Module):
    """DiffusionNet vertex encoder with triangle-face outputs and classifier."""

    def __init__(
        self,
        input_features: str = "xyz",
        embedding_dim: int = 128,
        num_classes: int = 8,
        width: int = 128,
        blocks: int = 4,
        dropout: bool = True,
    ) -> None:
        super().__init__()
        c_in = {"xyz": 3, "hks": 16}[input_features]
        self.input_features = input_features
        self.encoder = diffusion_net.layers.DiffusionNet(
            C_in=c_in,
            C_out=embedding_dim,
            C_width=width,
            N_block=blocks,
            last_activation=None,
            outputs_at="faces",
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        features = build_mesh_features(sample, self.input_features)
        face_embeddings = self.encoder(
            features,
            sample["mass"],
            L=sample["L"],
            evals=sample["evals"],
            evecs=sample["evecs"],
            gradX=sample["gradX"],
            gradY=sample["gradY"],
            faces=sample["faces"],
        )
        logits = self.classifier(face_embeddings)
        return logits, face_embeddings


_TEACHER_TRAIN_ONLY_HPARAMS = {"lr", "train_random_rotation"}


def _clean_state_dict(state: dict) -> OrderedDict:
    cleaned = OrderedDict()
    for key, value in state.items():
        if key.startswith("model."):
            cleaned[key[len("model."):]] = value
        elif key.startswith("teacher.model."):
            cleaned[key[len("teacher.model."):]] = value
        elif key.startswith("teacher."):
            cleaned[key[len("teacher."):]] = value
        else:
            cleaned[key] = value
    return cleaned


def fovnet_teacher_from_checkpoint(
    ckpt_path: str | Path,
    emb_dim: int = 128,
) -> tuple["FOVNetFaceTeacher", dict]:
    """Create FOVNetFaceTeacher from a Lightning checkpoint.

    Architecture hyperparameters (vision, az, el, uv, num_classes,
    srf_emb_dim, vision_emb_dim, graph_emb_dim …) are read directly from the
    checkpoint's hyper_parameters.  emb_dim is used as a fallback for all three
    embedding dims only when the checkpoint predates the --emb_dim flag.

    Returns (teacher, hparams) where hparams is the full dict from the ckpt.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})

    model_kw = {k: v for k, v in hparams.items() if k not in _TEACHER_TRAIN_ONLY_HPARAMS}
    model_kw.setdefault("srf_emb_dim", emb_dim)
    model_kw.setdefault("vision_emb_dim", emb_dim)
    model_kw.setdefault("graph_emb_dim", emb_dim)
    model_kw.setdefault("segmentation", True)

    teacher = FOVNetFaceTeacher(**model_kw)
    teacher.model.load_state_dict(_clean_state_dict(ckpt.get("state_dict", ckpt)), strict=True)
    return teacher, hparams


def load_fovnet_teacher_checkpoint(model: FOVNetFaceTeacher, ckpt_path: str | Path, strict: bool = False) -> None:
    """Load weights only into an already-constructed FOVNetFaceTeacher.

    Prefer fovnet_teacher_from_checkpoint() when possible — it reads
    architecture hyperparameters from the checkpoint instead of relying on
    the caller to match them manually.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.model.load_state_dict(_clean_state_dict(ckpt.get("state_dict", ckpt)), strict=strict)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def classification_stats(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 8) -> tuple[int, int, torch.Tensor]:
    preds = logits.argmax(dim=-1)
    correct = int((preds == labels).sum().detach().cpu())
    total = int(labels.numel())
    inds = labels.detach().cpu() * num_classes + preds.detach().cpu()
    counts = torch.bincount(inds, minlength=num_classes * num_classes)
    return correct, total, counts.reshape(num_classes, num_classes)


def mean_iou(confusion: torch.Tensor) -> float:
    confusion = confusion.float()
    ious = []
    for cls in range(confusion.shape[0]):
        tp = confusion[cls, cls]
        fp = confusion[:, cls].sum() - tp
        fn = confusion[cls, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            ious.append((tp / denom).item())
    return float(sum(ious) / len(ious)) if ious else 0.0
