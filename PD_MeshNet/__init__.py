"""PD-MeshNet student backbone for B2Mesh distillation.

Core network code is adapted from:
  https://github.com/MIT-SPARK/PD-MeshNet

New additions:
  - datasets/mfcad_plus_plus.py  — MFCAD++ dataset
  - datasets/fusion360.py        — Fusion360 Gallery dataset
  - student.py                   — PDMeshNetStudent + DataLoader helpers
"""

from .student import PDMeshNetStudent, collate_pd_samples
from .datasets import MFCADPlusPlusDualPrimal, Fusion360DualPrimal

__all__ = [
    "PDMeshNetStudent",
    "collate_pd_samples",
    "MFCADPlusPlusDualPrimal",
    "Fusion360DualPrimal",
]
