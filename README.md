# pathotrain

> [!WARNING]
> For research use only. Not a medical device.

Training on whole slide images (WSI), built on
[patchology](https://github.com/twpkevin06222/patchology) for tiling.

pathotrain turns whole slides and their annotations into tiles and trains
models on them: tissue segmentation, cell detection, virtual staining and
self-supervised pretraining of pathology foundation models. Every task runs
from one YAML config on a shared trainer (Lightning Fabric: one or more GPUs,
mixed precision, checkpoints, early stopping, MLflow and CSV logs, Optuna
sweeps), and trained models export to ONNX for inference.

## Installation

pathotrain is not on PyPI yet; install it from source in editable mode.
Python 3.12 or newer and an NVIDIA GPU are expected.

```bash
python -m venv .venv && source .venv/bin/activate

# 1. PyTorch built for CUDA 13.0 (the PyPI default build may not match your GPU driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. patchology (tiling), also not on PyPI
pip install "patchology @ git+https://github.com/twpkevin06222/patchology"
# or, from a local checkout: pip install -e ../patchology

# 3. pathotrain
git clone https://github.com/twpkevin06222/pathotrain.git
cd pathotrain
pip install -e .
```

The core install covers `segmentation`, `pathofm`, `dataloader`, `engine`,
`registration` and `experiment`. The other modules need an extra. The full
list of extras:

| Extra | Installs | Needed for |
|---|---|---|
| `detection` | rfdetr 1.11.2 (pinned), pycocotools, torch-hungarian | `pathotrain.detection` |
| `histossl` | peft | LoRA adaptation in `pathotrain.histossl` |
| `pathogen` | diffusers | `pathotrain.pathogen` |
| `export` | onnx, onnxruntime | ONNX export of any trained model |
| `all` | every extra above | everything |

```bash
pip install -e ".[detection]"          # one extra
pip install -e ".[detection,export]"   # a combination
pip install -e ".[all]"                # everything
```

Foundation-model weights (about 4.5 GB each) download from Hugging Face into
`~/.cache/huggingface` on first use; set `HF_HOME` to put them elsewhere. None
of them is gated, so no login is needed.

## Getting started

Each trainable module ships an example config. Copy it, set the data paths
and train:

```bash
cp src/pathotrain/segmentation/config.example.yaml my_run.yaml   # edit data paths
python -m pathotrain.segmentation.train my_run.yaml
python -m pathotrain.segmentation.eval runs/<name>/checkpoints/best.ckpt
python -m pathotrain.segmentation.export runs/<name>/checkpoints/best.ckpt model.onnx
```

Every run writes to its `output_dir`:

```
runs/<name>/
  config.yaml          the config as run (a changed config is refused unless resuming)
  train.log            console log
  checkpoints/         best.ckpt (weights + config), last.ckpt (full state, for --resume)
  logs/version_0/      metrics.csv, hparams.csv
```

Set `mlflow_db: ~/project/mlflow.db` in the config to also log to MLflow
(`mlflow ui --backend-store-uri sqlite:///$HOME/project/mlflow.db`). The other
modules follow the same pattern (`python -m pathotrain.<module>.train
config.yaml`); see each module's README for its data layout and options.

## Trainable modules

| Module | Task | Framework | Example config |
|---|---|---|---|
| [`segmentation`](src/pathotrain/segmentation/README.md) | Tissue / region segmentation from GeoJSON annotations | [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) (U-Net, U-Net++, DeepLabV3+, UPerNet, SegFormer, ...), [nnU-Net](https://github.com/MIC-DKFZ/dynamic-network-architectures) networks, and `pathofm` backbones as encoders | `segmentation/config.example.yaml` |
| [`detection`](src/pathotrain/detection/README.md) | Cell / object detection | [RF-DETR](https://github.com/roboflow/rf-detr) nano to large (DINOv2 backbone, DETR decoder), COCO-pretrained | `detection/config.example.yaml` |
| [`pathogen`](src/pathotrain/pathogen/README.md) | Virtual staining (e.g. H&E to IHC) from registered tile pairs | [diffusers](https://github.com/huggingface/diffusers): latent diffusion with a frozen SD VAE and a UNet (DDPM / DDIM), or SiT with rectified flow and REPA | `pathogen/config.example.yaml` |
| [`histossl`](src/pathotrain/histossl/README.md) | Self-supervised (continued) pretraining of pathology foundation models | DINOv2 (DINO + iBOT + KoLeo / KDE, incl. the OpenMidnight recipe), teacher-JEPA, LeJEPA; full, last-N or LoRA ([peft](https://github.com/huggingface/peft)) adaptation; linear, kNN and dense probes | `histossl/configs/*.example.yaml` |

## Helper modules

These support the trainable modules and have no training of their own.

| Module | What it provides |
|---|---|
| [`dataloader`](src/pathotrain/dataloader/README.md) | PyTorch datasets for slides and tiles: segmentation (GeoJSON masks), detection, classification (class folders or CSV), unlabelled tiles with multi-crop views, paired tiles; tile export to disk; augmentation (HED / HSV stain jitter, flips, rotations, blur, JPEG, RandAugment) |
| `engine` | The trainer all tasks share: `Task` contract, `Trainer` on Lightning Fabric (devices, DDP, mixed precision, gradient accumulation and clipping), `ModelCheckpoint` / `EarlyStopping` callbacks, MLflow and CSV loggers, warmup-cosine schedule, Optuna `sweep`, YAML config loading |
| [`pathofm`](src/pathotrain/pathofm/README.md) | Pathology foundation models as frozen or trainable backbones, with their model-card preprocessing and tile / dense embeddings (see below) |
| [`registration`](src/pathotrain/registration/README.md) | Rigid alignment of a non-H&E slide (IHC, special stain) to its H&E slide, with an overlay check; feeds paired tiles to `pathogen` |
| `experiment.py` | The plumbing behind every config-driven `train` / `eval` / `export` command: `start` (copy the config, start `train.log`, seed), `fit` (wire trainer, callbacks, loggers and data loaders from the config), `evaluate` / `log_eval`, `export_onnx` (ONNX graph with its preprocessing metadata) and `register_onnx` (MLflow model registry), and config `normalisation` presets |

### Foundation models in `pathofm`

Only open models: MIT or Apache-2.0, not gated. Revisions are pinned, so the
weights never change under a run.

| Name | Weights | Architecture | Normalisation |
|---|---|---|---|
| `midnight-12k` | [kaiko-ai/midnight](https://huggingface.co/kaiko-ai/midnight) | DINOv2 ViT-g/14, 1.1B parameters | mean = std = 0.5 |
| `spectra-midnight-12k` | [medarc/Midnight-12k-SPECTRA](https://huggingface.co/medarc/Midnight-12k-SPECTRA) | as midnight-12k, tuned for scanner, stain and centre robustness | mean = std = 0.5 |
| `openmidnight-1.1` | [medarc/OpenMidnight-1.1](https://huggingface.co/medarc/OpenMidnight-1.1) | DINOv2 ViT-g/14 with 4 registers, converted from timm on load | ImageNet |

```python
from pathotrain.pathofm import build_encoder

backbone, transform = build_encoder("openmidnight-1.1")  # model + its model-card preprocessing
```

## Logging

Logs (via loguru) show `INFO` and above by default. Each library has its own
level, so you can turn one up without the other:

```bash
export PATHOTRAIN_LOG_LEVEL=DEBUG    # pathotrain: DEBUG, ALL for everything, WARNING for less
export PATCHOLOGY_LOG_LEVEL=WARNING  # patchology (tiling) messages
```

If your application configures loguru itself (removes the default handler
before importing pathotrain), that setup is left untouched.

## License

Apache License 2.0, see [LICENSE](LICENSE).
