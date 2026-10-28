# pathotrain - segmentation

## Supported architectures

`build_model(name, num_classes, in_channels=3, **kwargs)` returns a plain
`nn.Module` that outputs `(B, num_classes, H, W)` logits. Extra keyword
arguments go to the library's own builder.

| `name` | Architecture | Library | Notes |
|---|---|---|---|
| `smp:unet` | U-Net | SMP | |
| `smp:unetplusplus` | U-Net++ | SMP | |
| `smp:manet` | MA-Net | SMP | |
| `smp:linknet` | LinkNet | SMP | |
| `smp:fpn` | FPN | SMP | |
| `smp:pspnet` | PSPNet | SMP | |
| `smp:pan` | PAN | SMP | |
| `smp:deeplabv3` | DeepLabV3 | SMP | |
| `smp:deeplabv3plus` | DeepLabV3+ | SMP | |
| `smp:upernet` | UPerNet | SMP | |
| `smp:segformer` | SegFormer | SMP | |
| `smp:dpt` | DPT | SMP | timm ViT encoders only (`encoder_name="tu-..."`) |
| `nnunet:plain` | nnU-Net U-Net (2D) | dynamic_network_architectures | no pretrained weights |
| `nnunet:resenc` | nnU-Net ResEnc U-Net (2D) | dynamic_network_architectures | no pretrained weights |

- **SMP models** take `encoder_name` (default `resnet34`, see the
  [SMP encoder list](https://smp.readthedocs.io/en/latest/encoders.html), or
  any timm model as `tu-<name>`) and `encoder_weights` (default `"imagenet"`,
  `None` for random initialisation). ImageNet weights match the default
  normalisation in `train_transform` / `eval_transform`.
- **timm backbones:** timm comes with SMP, so any timm model with feature
  maps can be the encoder. Prefix its name with `tu-`. Find names with
  `timm.list_models("convnext*", pretrained=True)`.

  ```python
  # ConvNeXt-Tiny encoder; "imagenet" loads timm's default pretrained weights.
  build_model("smp:unet", num_classes=6, encoder_name="tu-convnext_tiny", encoder_weights="imagenet")
  build_model("smp:fpn", num_classes=6, encoder_name="tu-efficientnet_b3", encoder_weights="imagenet")

  # DPT needs a ViT encoder. dynamic_img_size=True lets a 224-pretrained ViT
  # take 512 px tiles (without it, any size other than 224 raises).
  build_model(
      "smp:dpt",
      num_classes=6,
      encoder_name="tu-vit_small_patch16_224",
      encoder_weights="imagenet",
      dynamic_img_size=True,
  )
  ```

  Use the normalisation the backbone was trained with. ImageNet mean/std is
  the default in `train_transform`; pass `mean=` / `std=` for other backbones
  (CLIP-pretrained ones, for example).
- **nnU-Net models** build only the network, not the nnU-Net pipeline. They take
  `n_stages` (default 7), `base_features` (32) and `max_features` (512). The
  tile side must be divisible by `2 ** (n_stages - 1)`, which is 64 by default.
  Deep supervision is off.

## Pathology foundation-model encoders

Two open foundation models from `pathotrain.pathofm` can serve as the encoder
of any SMP decoder except DPT (SMP restricts DPT to timm encoders). Both are
MIT licensed and not gated, so no Hugging Face login is needed.

| `encoder_name` | Weights | Notes |
|---|---|---|
| `midnight-12k` | [kaiko-ai/midnight](https://huggingface.co/kaiko-ai/midnight) | DINOv2 ViT-g/14, 1.1B parameters, 1536-dim tokens |
| `spectra-midnight-12k` | [medarc/Midnight-12k-SPECTRA](https://huggingface.co/medarc/Midnight-12k-SPECTRA) | the same network, tuned for robustness to scanner, stain and centre |

```python
from pathotrain.dataloader.augmentation import eval_transform, train_transform
from pathotrain.pathofm import MIDNIGHT_MEAN, MIDNIGHT_STD
from pathotrain.segmentation import build_model

model = build_model("smp:unet", num_classes=6, encoder_name="midnight-12k")
model = build_model(
    "smp:upernet",
    num_classes=6,
    encoder_name="spectra-midnight-12k",
    encoder_kwargs={"freeze": False, "layers": (10, 20, 30, 40)},
)

# Midnight normalises with mean = std = 0.5, not ImageNet statistics.
train_tf = train_transform(mean=MIDNIGHT_MEAN, std=MIDNIGHT_STD)
val_tf = eval_transform(mean=MIDNIGHT_MEAN, std=MIDNIGHT_STD)
```

**Classification vs segmentation embedding.** The model card defines two different outputs:

- **Classification embedding** (`pathofm.classification_embedding`): the CLS
  token concatenated with the mean patch token, `(B, 3072)`. It is one vector
  per tile, with no spatial information.
- **Segmentation embedding** (`pathofm.segmentation_embedding`): the
  last-layer patch tokens after the final LayerNorm, reshaped to
  `(B, 1536, H/14, W/14)` (16 × 16 for a 224 px tile).

The encoder (`ViTPyramidEncoder`) builds on the segmentation embedding. SMP
decoders need features at strides 4, 8, 16 and 32, so it takes the patch
tokens of four layers (by default 10, 20, 30 and 40 of 40), applies the final
LayerNorm to each, and resamples them DPT-style:

- the layer-10 tokens are upsampled to stride 4;
- the layer-20 tokens are upsampled to stride 8;
- the layer-30 tokens stay at stride 16;
- the layer-40 tokens are downsampled to stride 32.

The layer-40 feature is exactly the card's segmentation embedding. Use
`layers=(40, 40, 40, 40)` to build every scale from it alone.

Notes:
- **Tile size:** the tile is resized so that the token grid is H/16 × W/16
  (512 px is fed as 448 px), which lines the grid up with stride 16. The tile
  side must be divisible by 32.
- **`freeze=True` (default):** the backbone is fixed and runs without
  gradients, so only the resampling and the decoder train. Checkpoints still
  include the full backbone (about 4.5 GB in fp32).
- **`freeze=False`:** fine-tunes all 1.1B parameters. Enable
  `model.encoder.backbone.gradient_checkpointing_enable()` and expect several
  times the memory.
- **Other ViTs:** `encoder_name` also accepts a loaded ViT module with the
  transformers interface (e.g. any `Dinov2Model`).

## Training

`pathotrain.engine` is a training loop on [Lightning Fabric](https://lightning.ai/docs/fabric/)
that knows nothing about the task: a `Task` gives it the loss, metrics and
optimizer. Fabric handles devices, DDP/FSDP and mixed precision.
`pathotrain.segmentation` builds the networks (SMP or nnU-Net architectures)
and the `SegmentationTask`.

```python
from lightning_fabric import seed_everything
from torch.utils.data import DataLoader

from pathotrain.engine import EarlyStopping, MLflowLogger, ModelCheckpoint, Trainer, TrainerConfig
from pathotrain.segmentation import SegmentationTask, build_model

seed_everything(0, workers=True)
model = build_model("smp:unet", num_classes=6, encoder_name="resnet34", encoder_weights="imagenet")
# or build_model("nnunet:resenc", num_classes=6)  # nnU-Net 2D network, trained from scratch
task = SegmentationTask(model, num_classes=6, lr=1e-3)

trainer = Trainer(
    TrainerConfig(max_epochs=50, precision="bf16-mixed", strategy="ddp", devices=4),
    callbacks=[ModelCheckpoint("checkpoints"), EarlyStopping(patience=10)],
    loggers=[MLflowLogger(experiment="segmentation")],
)
metrics = trainer.fit(
    task,
    DataLoader(train, batch_size=16, shuffle=True, num_workers=8),
    DataLoader(val, batch_size=16, num_workers=8),
)
```

- `val/loss`, `val/miou` and `val/iou_<class>` are logged after every epoch. Pixels set to
  255 (outside the ROI, or uncovered by augmentation) are excluded from both
  the loss and the metrics.
- Model selection: `ModelCheckpoint` keeps `best.ckpt` by `val/miou` (higher
  is better), and `EarlyStopping` and `OptunaPruning` watch the same metric. Pass
  `monitor=` (and `mode="min"` for losses) to change it, e.g.
  `ModelCheckpoint("checkpoints", monitor="val/iou_tumor")`. `OptunaPruning`
  has no `mode`; it takes the direction from the study (`sweep(direction=...)`).
- MLflow writes to `MLFLOW_TRACKING_URI`, e.g. `sqlite:///mlflow.db`. Browse it with
  `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

### Checkpoints

`ModelCheckpoint(dirpath, monitor="val/miou", mode="max", log_artifact=False)` writes:

| File | When | Contents |
|---|---|---|
| `best.ckpt` | `monitor` beats the best so far (strictly; NaN never counts) | weights only: `model`, `epoch`, `step`, `monitor`, `value` |
| `last.ckpt` | after every epoch | `model`, `optimizer`, `scheduler`, `epoch`, `step` and callback state, for resuming |

- Resume with `trainer.fit(..., ckpt_path="checkpoints/last.ckpt")`. The best
  value and epoch, and `EarlyStopping`'s patience counter, are restored, so the
  first resumed epoch only replaces `best.ckpt` if it really is better.
- At the end of training, `best/<monitor>`, `best/epoch` and the
  `best_checkpoint` path are logged to every logger. With `log_artifact=True`,
  `best.ckpt` is also uploaded to the MLflow run.
- `model` is a plain state dict, so loading needs no Fabric:

  ```python
  ckpt = torch.load("checkpoints/best.ckpt", map_location="cpu", weights_only=False)
  model = build_model("smp:unet", num_classes=6, encoder_name="resnet34", encoder_weights=None)
  model.load_state_dict(ckpt["model"])
  ```

### CSV logs

Add a `CSVLogger` to save metrics, loss and learning rate per epoch or per step
for a run. `TrainerConfig.log_interval` chooses which:

```python
from pathotrain.engine import CSVLogger

trainer = Trainer(
    TrainerConfig(max_epochs=50, log_interval="epoch"),  # or "step", with log_every=50
    callbacks=[ModelCheckpoint("checkpoints")],
    loggers=[CSVLogger("logs"), MLflowLogger()],
)
```

Each run writes to `logs/pathotrain/version_<n>/`:

- `metrics.csv` has one row per log, with columns `step`, `epoch`, `train/loss`,
  `train/lr`, `val/loss`, `val/miou`, `val/iou_<class>` and finally `best/...`.
  - `log_interval="epoch"` (default): one row per epoch. `train/loss` is the mean
    over the epoch's steps (and all GPUs), and `train/lr` is the rate at the end
    of the epoch.
  - `log_interval="step"`: a `train/loss` and `train/lr` row every `log_every`
    optimizer steps, plus one validation row per epoch.
- `hparams.csv` holds the `name,value` pairs of the trainer config, `hparams=`
  and `best_checkpoint`.

### Hyperparameter sweeps

`sweep` runs an Optuna study. The objective receives the per-trial callbacks
and loggers and passes them to its `Trainer`:

- **Pruning:** `OptunaPruning` on `monitor` (`val/miou` by default) lets the
  pruner (median by default) stop trials that fall behind.
- **MLflow:** each trial is a child run `trial-<n>` under one parent run per study.
- **Checkpoints:** by default, **no model weights are saved**. Pass
  `checkpoint_dir="sweep_ckpt"` to keep `best.ckpt` / `last.ckpt` per trial in
  `sweep_ckpt/trial-<n>/`.
- **Trials CSV:** `csv_path="trials.csv"` rewrites a CSV after every trial with
  one row per trial: `number`, `state` (`COMPLETE`, `PRUNED`, `FAIL`),
  `value`, `duration_s` and one column per hyperparameter.

```python
import optuna
from pathotrain.engine import sweep


def objective(trial, callbacks, loggers):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    arch = trial.suggest_categorical("arch", ["smp:unet", "smp:deeplabv3plus"])
    task = SegmentationTask(build_model(arch, 6, encoder_weights="imagenet"), 6, lr=lr)
    trainer = Trainer(TrainerConfig(max_epochs=30, devices=1), callbacks=callbacks, loggers=loggers)
    return trainer.fit(task, train_loader, val_loader, hparams=trial.params)["val/miou"]


study = sweep(
    objective,
    n_trials=50,
    study_name="seg-v1",
    storage="sqlite:///sweep.db",
    sampler=optuna.samplers.GPSampler(),  # Gaussian-process BO; TPE by default
    csv_path="trials.csv",
)
```

Each trial runs on a single device. To use several GPUs, start one process per
GPU (`CUDA_VISIBLE_DEVICES=i`) with the same `study_name` and `storage`. They
all keep `trials.csv` up to date for the whole study.

## License

Apache License 2.0, see [LICENSE](LICENSE).
