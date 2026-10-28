# pathotrain - detection

Object detection with [RF-DETR](https://github.com/roboflow/rf-detr) on the
pathotrain engine (Fabric: devices, DDP, mixed precision, checkpoints, MLflow
and CSV logs, sweeps). It uses rfdetr's network, loss and post-processing, not
its own Lightning trainer.

```bash
pip install "pathotrain[detection]"   # rfdetr (pinned), pycocotools
```

## Models

| Name | Input (default) | Params | Licence |
|---|---|---|---|
| `rfdetr:nano` | 384 px | 30.5 M | Apache-2.0 |
| `rfdetr:small` | 512 px | 32.1 M | Apache-2.0 |
| `rfdetr:medium` | 576 px | 33.7 M | Apache-2.0 |
| `rfdetr:large` | 704 px | 33.9 M | Apache-2.0 |

All are DINOv2 (windowed attention) backbones with a DETR decoder. The larger
and smallest RF-DETR sizes (XL, 2XL, Atto, Femto, Pico) are under the PML
licence and are not offered. With `pretrained=True` (default) the COCO
checkpoint is downloaded once to `~/.cache/rfdetr` and the class head is
re-initialised for your classes.

## Training

```python
from torch.utils.data import DataLoader

from pathotrain.detection import DetectionTask, build_detector, collate
from pathotrain.engine import EarlyStopping, MLflowLogger, ModelCheckpoint, Trainer, TrainerConfig

detector = build_detector("rfdetr:medium", num_classes=2)
task = DetectionTask(detector, warmup_epochs=1, class_names=["lymphocyte", "tumour"])
trainer = Trainer(
    TrainerConfig(max_epochs=50, grad_clip=0.1, precision="bf16-mixed"),
    callbacks=[ModelCheckpoint("checkpoints", monitor="val/map"), EarlyStopping(monitor="val/map")],
    loggers=[MLflowLogger(experiment="detection")],
)
trainer.fit(
    task,
    # persistent_workers: no worker restart per epoch (train only, so the val
    # workers are not held in RAM for the whole run)
    DataLoader(
        train, batch_size=8, shuffle=True, num_workers=8, persistent_workers=True, collate_fn=collate
    ),
    DataLoader(val, batch_size=8, num_workers=8, collate_fn=collate),
)
```

**What a dataset must return**, per sample `(image, target)`:

- `image`: `(3, H, W)` float tensor, ImageNet-normalised
  (`pathotrain.dataloader.augmentation.IMAGENET_MEAN` / `IMAGENET_STD`), with
  `H` and `W` divisible by 32. Tiles do not need to be the model's default size.
- `target`: `{"boxes": (N, 4) x1, y1, x2, y2 in pixels, "labels": (N,) in 0 .. num_classes - 1}`.
  `N` may be 0; float labels or `(0,)` boxes from `torch.tensor([])` are accepted.

`collate` stacks the images and keeps the targets as a list.

**Logged:** `train/loss` with `train/loss_ce`, `train/loss_bbox` and
`train/loss_giou`; per epoch `val/loss`, COCO `val/map` (IoU 0.5:0.95),
`val/map_50`, `val/map_75`, `val/mar_100` and `val/ap_<class>`.

**Optimiser:** RF-DETR's recipe. AdamW with `lr=1e-4` for the heads,
`lr * 0.7` for the decoder, `lr_encoder=1.5e-4` for the backbone with layer-wise
decay 0.8 towards the input, weight decay `1e-4` (none on norms, biases and
embeddings), gradient clipping at 0.1. The schedule is pathotrain's warmup plus
cosine; RF-DETR's model EMA is not used.
