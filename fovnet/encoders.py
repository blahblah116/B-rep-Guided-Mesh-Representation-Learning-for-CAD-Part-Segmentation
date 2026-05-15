"""Feature encoders for FOVNet: vision grids, UV surfaces, and graph (GAT)."""

import torch
import torch.nn.functional as F
from torch import nn
from dgl.nn.pytorch.glob import MaxPooling
from dgl.nn.pytorch import GATConv


# ── Building blocks ─────────────────────────────────────────────────────────
def _conv2d(in_c, out_c, ks, padding=0, bias=False, drop=0.0):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=ks, padding=padding, bias=bias),
        nn.BatchNorm2d(out_c), nn.LeakyReLU(), nn.Dropout(drop),
    )

def _fc(in_f, out_f, bias=False):
    return nn.Sequential(
        nn.Linear(in_f, out_f, bias=bias), nn.BatchNorm1d(out_f), nn.LeakyReLU(), nn.Dropout(0.1),
    )

def combination_fc(in_f, out_f, bias=False):
    return nn.Sequential(
        nn.Linear(in_f, 256, bias=bias), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, out_f, bias=bias), nn.BatchNorm1d(out_f), nn.ReLU(), nn.Dropout(0.1),
    )

def _kaiming_init(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

# ── Surface encoder (2-D UV grid) ──────────────────────────────────────────
class SurfaceEncoder(nn.Module):
    def __init__(self, in_channels=7, output_dims=64):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = _conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = _conv2d(32, 64, 3, padding=1)
        self.conv3 = _conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _fc(128, output_dims)
        _kaiming_init(self)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        return self.fc(self.pool(x).flatten(1))

# ── Vision-grid encoder ───────────────────────────────────────
class VisionGridEncoder(nn.Module):
    def __init__(self, input_az=12, input_el=6, in_channels=3, output_dims=128):
        super().__init__()
        self.el, self.az = input_el, input_az
        self.conv1 = nn.Conv2d(in_channels + 2, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _fc(64, output_dims)

    def forward(self, x):
        B, dev = x.size(0), x.device
        el = torch.linspace(0, 1, self.el, device=dev).view(1, 1, self.el, 1).expand(B, 1, self.el, self.az)
        az = torch.linspace(0, 1, self.az, device=dev).view(1, 1, 1, self.az).expand(B, 1, self.el, self.az)
        x = torch.cat([x, el, az], dim=1)
        # Circular + zero padding, then conv
        for conv, bn in ((self.conv1, self.bn1), (self.conv2, self.bn2)):
            x = F.pad(x, (1, 1, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, 1, 1), mode="constant", value=0)
            x = F.relu(bn(conv(x)))
        return self.fc(self.pool(x).flatten(1))

# ── Graph encoder (GAT) ────────────────────────────────────────────────────
class GraphEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, n_layers=3, n_heads=4,
                 feat_drop=0, attn_drop=0, residual=True):
        super().__init__()
        layers = [GATConv(in_dim, hidden, n_heads, feat_drop=feat_drop,
                          attn_drop=attn_drop, residual=residual,
                          activation=F.elu, allow_zero_in_degree=True)]
        for _ in range(n_layers - 2):
            layers.append(GATConv(hidden * n_heads, hidden, n_heads,
                                  feat_drop=feat_drop, attn_drop=attn_drop,
                                  residual=residual, activation=F.elu,
                                  allow_zero_in_degree=True))
        layers.append(GATConv(hidden * n_heads, out_dim, 1,
                              feat_drop=feat_drop, attn_drop=attn_drop,
                              residual=False, activation=None,
                              allow_zero_in_degree=True))
        self.layers = nn.ModuleList(layers)
        self.pool = MaxPooling()

    def forward(self, g, h):
        for layer in self.layers:
            h = layer(g, h)
            h = h.flatten(1) if h.dim() == 3 else h
        return h, self.pool(g, h)
