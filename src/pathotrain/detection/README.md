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

`train` and `val` can be `pathotrain.dataloader.DetectionDataset` (slides and
GeoJSON annotations, tiled on the fly) with `train_transform(boxes=True)` and
`eval_transform(boxes=True)`; see the [dataloader README](../dataloader/README.md#detection-dataloader).

**What a dataset must return**, per sample `(image, target)`:

- `image`: `(3, H, W)` float tensor, ImageNet-normalised
  (`pathotrain.dataloader.augmentation.IMAGENET_MEAN` / `IMAGENET_STD`), with
  `H` and `W` divisible by 32. Tiles do not need to be the model's default size.
- `target`: `{"boxes": (N, 4) x1, y1, x2, y2 in pixels, "labels": (N,) in 0 .. num_classes - 1}`.
  `N` may be 0; float labels or `(0,)` boxes from `torch.tensor([])` are accepted.

`collate` stacks the images and keeps the targets as a list.

**Logged:** `train/loss` with `train/loss_ce`, `train/loss_bbox` and
`train/loss_giou`; per epoch `val/loss`, COCO `val/map` (IoU 0.5:0.95),
`val/map_50`, `val/map_75`, `val/mar_300` and `val/ap_<class>`, counting up
to 300 predictions per image (RF-DETR's output) rather than COCO's 100.

**Optimiser:** RF-DETR's recipe. AdamW with `lr=1e-4` for the heads,
`lr * 0.7` for the decoder, `lr_encoder=1.5e-4` for the backbone with layer-wise
decay 0.8 towards the input, weight decay `1e-4` (none on norms, biases and
embeddings), gradient clipping at 0.1. The schedule is pathotrain's warmup plus
cosine; RF-DETR's model EMA is not used.

### From a config file

The same setup without writing a script: one YAML file per experiment. Copy
[`config.example.yaml`](config.example.yaml), which lists every section with its defaults.

```bash
python -m pathotrain.detection.train config.yaml
python -m pathotrain.detection.train config.yaml --resume runs/det01/checkpoints/last.ckpt
```

```yaml
name: det01                 # MLflow experiment
seed: 0
output_dir: runs/det01      # config.yaml, checkpoints/, logs/
mean: IMAGENET_MEAN
std: IMAGENET_STD
model: {name: "rfdetr:nano", num_classes: 2}
task: {warmup_epochs: 1, class_names: [class01, class02]}
trainer: {max_epochs: 50, precision: bf16-mixed}   # grad_clip defaults to 0.1
early_stopping: {patience: 10}                     # monitors val/map
loader: {batch_size: 8, num_workers: 8}
data:
  class_map: {class01: 0, class02: 1}              # the rest goes to DetectionTileGenerator
  roi_class: roi
  min_box_visibility: 0.6
  train: {slides: [data/train/slides], annotations: [data/train/geojson], tiling: {tile_size: 384, tile_mpp: 0.5}}
  val: {root: tiles/valid}                         # or a COCO folder from export_detection_tiles
```

- The config is copied to `output_dir/config.yaml`, uploaded to the MLflow run,
  logged as flat params and stored in both checkpoints. A changed config for
  the same `output_dir` is refused unless resuming.
- `checkpoint` and `early_stopping` monitor `val/map` by default.
- `model.num_classes` must match the classes of each split (`class_map` labels,
  or a COCO folder's categories); a mismatch fails before training.
- `load_model("runs/det01/checkpoints/best.ckpt")` returns the trained
  `Detector` and its config.

### Evaluation

Score a checkpoint on a held-out split. `best.ckpt` was chosen on `val`, so
report a `test` split: add one under `data` (as `val`) and pass that config.

```bash
python -m pathotrain.detection.eval runs/det01/checkpoints/best.ckpt --config config.yaml
python -m pathotrain.detection.eval runs/det01/checkpoints/best.ckpt --split val
```

- The model, normalisation and task settings come from the checkpoint; `data`,
  `loader` and `trainer` too, unless `--config` gives them.
- Prints the metrics as JSON keyed by the split (`test/map`, `test/ap_<class>`,
  ...). Runs on one device. `evaluate(checkpoint, split, config)` returns them.
- Scores are per tile: tile the split with `tile_overlap: 0` so no object counts twice.

### ONNX export

```bash
pip install "pathotrain[detection,export]"   # adds onnx, onnxruntime
python -m pathotrain.detection.export runs/det01/checkpoints/best.ckpt model.onnx
```

- Input `image` (batch, 3, size, size) float32, normalised as in training. The
  graph includes the post-processing and returns the top 300 detections per
  image, best first: `boxes` (batch, 300, 4) `[x1, y1, x2, y2]` tile pixels,
  `scores` (batch, 300) and `labels` (batch, 300). Threshold the scores.
- The batch is dynamic; the side is fixed to the training tile (or `--size`, a
  multiple of 32), since the backbone's position embeddings are frozen to it.
- The `pathotrain` metadata key holds JSON with `mean`, `std`, `tile_size`,
  `tile_mpp`, `class_names` and `class_map`. Export checks ONNX Runtime against
  PyTorch on random images.
- Weights are fp32. Convert to fp16 or quantise to int8 from this file, then
  check the result with the same tiles.
