# B2Mesh: B-Rep Guided Mesh Segmentation

Triangle mesh segmentation using B-Rep graph knowledge distillation.  
A FOVNet teacher trained on B-Rep graphs guides a mesh-only student (DiffusionNet or PD-MeshNet) at train time — at inference only the mesh is required.

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate distillation
```

---

## Data Preparation

### Fusion360 (recommended)

Download the [Fusion 360 Gallery Segmentation dataset](https://github.com/AutodeskAILab/Fusion-360-Gallery-Dataset) and place it as:

```
data/fusion360/s2.0.1/
├── breps/seg/          # per-face B-Rep segmentation labels
├── meshes/             # .obj + .seg + .fidx triangle mesh files
└── train_test_new.json # train/val/test split file
```

### MFCAD++ (optional)

Download MFCAD++ and convert STEP files to mesh format:

```bash
python -m B2Mesh.preprocessing.mfcadpp_to_mesh \
    --data_root /path/to/mfcad++ \
    --out_dir   /path/to/mfcad++/meshes
```

---

## Training

### 1. DiffusionNet student only (no teacher)

```bash
python -m B2Mesh.train \
    --no_teacher \
    --student_type diffusion_net \
    --dataset fusion360 \
    --data_root /path/to/fusion360/s2.0.1 \
    --input_features hks \
    --op_cache_dir /path/to/op_cache \
    --epochs 50 \
    --lr 1e-3
```

### 2. DiffusionNet student + FOVNet teacher (distillation)

```bash
python -m B2Mesh.train \
    --student_type diffusion_net \
    --dataset fusion360 \
    --data_root /path/to/fusion360/s2.0.1 \
    --teacher_ckpt /path/to/fovnet_best.ckpt \
    --input_features hks \
    --op_cache_dir /path/to/op_cache \
    --epochs 50 \
    --distill_weight 1.0
```

### 3. PD-MeshNet student + FOVNet teacher (distillation)

```bash
python -m B2Mesh.train \
    --student_type pd_meshnet \
    --dataset fusion360 \
    --data_root /path/to/fusion360/s2.0.1 \
    --teacher_ckpt /path/to/fovnet_best.ckpt \
    --input_features xyz \
    --pd_cache_dir /path/to/pd_cache \
    --epochs 50
```

### Key training arguments

| Argument | Default | Description |
|---|---|---|
| `--student_type` | `diffusion_net` | Student backbone: `diffusion_net` or `pd_meshnet` |
| `--dataset` | `fusion360` | Dataset: `fusion360` or `mfcad++` |
| `--input_features` | `xyz` | Mesh input features: `xyz` or `hks` |
| `--no_teacher` | off | Train student without distillation |
| `--teacher_ckpt` | None | Path to pretrained FOVNet checkpoint |
| `--epochs` | 50 | Number of training epochs |
| `--lr` | 1e-3 | Student learning rate |
| `--seg_loss` | `ce` | Segmentation loss: `ce` or `focal` |
| `--distill_weight` | 1.0 | Weight for distillation loss |
| `--op_cache_dir` | None | Cache dir for DiffusionNet eigen-operators |
| `--pd_cache_dir` | None | Cache dir for PD-MeshNet graphs |
| `--skip_test` | off | Skip test evaluation after training |

---

## Test Only

Run evaluation on a pretrained checkpoint without training:

```bash
python -m B2Mesh.train \
    --test_only \
    --student_ckpt /path/to/outputs/<run_name>/best.pt \
    --no_teacher \
    --student_type diffusion_net \
    --dataset fusion360 \
    --data_root /path/to/fusion360/s2.0.1 \
    --input_features hks \
    --op_cache_dir /path/to/op_cache
```

---

## Outputs

Each run creates a directory under `B2Mesh/outputs/<run_name>/`:

```
<run_name>/
├── args.json                  # full argument config
├── best.pt                    # best checkpoint (by val loss)
├── last.pt                    # last epoch checkpoint
├── metrics.jsonl              # per-epoch train/val metrics
├── test_metrics.json          # final test metrics
├── timing_summary.json        # training time breakdown
├── test_predictions/
│   └── <sample_name>/
│       ├── <sample_name>_pred.npy     # predicted class index per face
│       ├── <sample_name>_gt.npy       # ground truth class index per face
│       └── <sample_name>_logits.npy   # raw logits per face (num_faces × num_classes)
└── test_visualizations/       # colored PLY mesh files (up to --visualize_test_count)
```

---

## Project Structure

```
B2Mesh/
├── train.py            # main training & evaluation script
├── data.py             # dataset and dataloader
├── models.py           # model construction
├── losses.py           # distillation / segmentation losses
├── visualization.py    # PLY mesh export
├── eval_per_class.py   # per-class metric evaluation
├── diffusion_net/      # DiffusionNet backbone
├── fovnet/             # FOVNet teacher
├── PD_MeshNet/         # PD-MeshNet backbone
└── preprocessing/
    └── mfcadpp_to_mesh.py  # MFCAD++ STEP → mesh converter
```
