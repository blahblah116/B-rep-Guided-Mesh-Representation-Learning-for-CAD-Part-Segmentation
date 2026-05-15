"""FOVNet architecture for B-Rep 3D model learning.

Combines vision grids (ray-casting hemispheres), UV-grid surface
parametrizations, and face geometric features for classification
and per-face segmentation.
"""

import torch
import torch.nn.functional as F
from torch import nn
import lightning.pytorch as pl
import torchmetrics
from . import encoders as enc


# ── Classification / segmentation heads ─────────────────────────────────────
class _MLPHead(nn.Module):
    """Three-layer MLP with BN and dropout."""

    def __init__(self, in_dim, num_classes, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128, bias=False),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        return self.net(x)


# ── Core model ──────────────────────────────────────────────────────────────
class FOVNet(nn.Module):
    """Multi-representation B-Rep encoder with optional vision, UV, and face features."""

    def __init__(
        self, num_classes, vision=False, vision_az=12, vision_el=6,
        ov_channels=(0, 1, 2), iv_channels=(3, 4, 5), srf_emb_dim=128, 
        vision_emb_dim=128, graph_emb_dim=128, dropout=0.3, local_uv=False,
        input_graph_dim=64, segmentation=False,
        use_face_feat=False, use_uv=True, use_ov=True, use_iv=True,
    ):
        super().__init__()
        self.vision = vision
        self.ov_channels = list(ov_channels)
        self.iv_channels = list(iv_channels)
        self.uv_key = "x_local" if local_uv else "x"
        self.segmentation = segmentation
        self.use_face_feat = use_face_feat
        self.use_uv = use_uv
        self.use_ov = use_ov
        self.use_iv = use_iv

        # Build encoders and accumulate feature dimensions
        feat_dims = []
        self.ov_encoder = self.iv_encoder = self.surf_encoder = None

        if vision and use_ov:
            self.ov_encoder = enc.VisionGridEncoder(vision_az, vision_el, len(ov_channels), vision_emb_dim)
            feat_dims.append(vision_emb_dim)
        if vision and use_iv:
            self.iv_encoder = enc.VisionGridEncoder(vision_az, vision_el, len(iv_channels), vision_emb_dim)
            feat_dims.append(vision_emb_dim)
        if use_uv:
            self.surf_encoder = enc.SurfaceEncoder(in_channels=7, output_dims=srf_emb_dim)
            feat_dims.append(srf_emb_dim)
        if use_face_feat:
            feat_dims.append(7)

        total_raw = sum(feat_dims)
        self.shared_fc = enc.combination_fc(total_raw, input_graph_dim, bias=False)
        self.graph_encoder = enc.GraphEncoder(input_graph_dim, graph_emb_dim)
        self.clf = _MLPHead(graph_emb_dim, num_classes, dropout)
        self.seg = _MLPHead(2 * graph_emb_dim, num_classes, dropout)

    def forward(self, g):
        # Permute to NCHW for conv layers
        if self.use_uv:
            for k in ("x", "x_local"):
                if k in g.ndata:
                    g.ndata[k] = g.ndata[k].permute(0, 3, 1, 2).contiguous()
        if self.vision and "vision_grids" in g.ndata:
            g.ndata["vision_grids"] = g.ndata["vision_grids"].permute(0, 3, 1, 2).contiguous()

        # Drop unneeded features to save memory
        keep = {self.uv_key, "vision_grids"} | ({"face_feat"} if self.use_face_feat else set())
        for k in list(g.ndata.keys()):
            if k not in keep:
                g.ndata.pop(k)
        for k in list(g.edata.keys()):
            g.edata.pop(k)

        # Encode each feature stream
        parts = []
        if self.use_uv and self.uv_key in g.ndata:
            parts.append(self.surf_encoder(g.ndata[self.uv_key]))
        vg = g.ndata.get("vision_grids")
        if self.vision and vg is not None:
            # vg shape: (N, 6, H, W) where channels are:
            # 0-2: OV (occupancy, distance, dot), 3-5: IV (occupancy, distance, dot)
            if self.ov_encoder:
                parts.append(self.ov_encoder(vg[:, self.ov_channels]))
            if self.iv_encoder:
                parts.append(self.iv_encoder(vg[:, self.iv_channels]))
        if self.use_face_feat and "face_feat" in g.ndata:
            parts.append(g.ndata["face_feat"][:, :7])

        hidden = self.shared_fc(torch.cat(parts, dim=1))
        node_emb, graph_emb = self.graph_encoder(g, hidden)

        if self.segmentation:
            expanded = graph_emb.repeat_interleave(g.batch_num_nodes().to(graph_emb.device), dim=0)
            return self.seg(torch.cat((node_emb, expanded), dim=1))
        return self.clf(graph_emb)


# ── Lightning wrapper ──────────────────────────────────────────────────────
class FOVNetModule(pl.LightningModule):

    def __init__(self, **kwargs):
        kwargs.pop("weights_only", None)
        super().__init__()
        self.save_hyperparameters()

        model_kw = {k: v for k, v in self.hparams.items() if k not in ("lr", "train_random_rotation")}
        self.model = FOVNet(**model_kw)

        nc = self.hparams.num_classes
        seg = self.hparams.get("segmentation", False)
        self.metrics = nn.ModuleDict()
        for stage in ("train", "val", "test"):
            m = nn.ModuleDict({"acc": torchmetrics.Accuracy(task="multiclass", num_classes=nc)})
            if seg:
                m["iou"] = torchmetrics.JaccardIndex(task="multiclass", num_classes=nc)
            self.metrics[f"{stage}_metrics"] = m

    def forward(self, g):
        return self.model(g)

    def _step(self, batch, stage):
        g = batch["graph"].to(self.device)
        seg = self.hparams.get("segmentation", False)
        labels = (g.ndata["y"] if seg else batch["label"]).to(self.device).long()
        bs = len(g.batch_num_nodes())

        logits = self(g)
        loss = F.cross_entropy(logits, labels)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, batch_size=bs)

        preds = torch.softmax(logits, -1) if seg else logits.argmax(1)
        for name, metric in self.metrics[f"{stage}_metrics"].items():
            metric(preds, labels)
            self.log(f"{stage}_{name}", metric, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def test_step(self, batch, _):
        self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
