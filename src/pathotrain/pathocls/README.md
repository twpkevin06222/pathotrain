# pathocls

Tile classification you can train, score, export and serve: a linear or MLP head on a
foundation-model backbone's tile embedding (CLS token and mean patch token). The
backbone (`pathofm`, or a histossl checkpoint or export) is frozen, a linear probe, or
fine-tuned.

```bash
cp src/pathotrain/pathocls/config.example.yaml cls.yaml          # edit data paths
python -m pathotrain.pathocls.train cls.yaml
python -m pathotrain.pathocls.eval runs/cls01/checkpoints/best.ckpt --config cls.yaml
python -m pathotrain.pathocls.export runs/cls01/checkpoints/best.ckpt out/model.onnx \
    --mpp 0.5 --icc
```

## Data

Each split is class folders (`{root: dir}`, `dir/<class>/*.png`) or a CSV with `path`
and `label` columns (`{csv: file}`, paths relative to the CSV or to `root`), read with
`pathotrain.dataloader.ClassTileDataset`. Tiles are resized and centre cropped to the
backbone's input size (224 px) and normalised as it expects. The class order is
`task.class_names`, default the train split's labels sorted; the run's `config.yaml`
records it, with `model.num_classes`.

## Training

Three ways, set by `model.freeze` and an `augmentation` section:

| | config | backbone runs | checkpoint holds |
|---|---|---|---|
| cached probe (default) | frozen, no `augmentation` | once per split | head |
| online probe | frozen, `augmentation: {...}` | every step, no backward | head |
| fine-tuning | `model.freeze: false` | every step, trained | backbone and head |

- Cached: the head trains on the embeddings in memory, so an epoch takes
  milliseconds; `loader.num_workers: 0` is right for it.
- Online: train tiles are resized, randomly cropped to the input size, flipped,
  rotated by 90 degree steps and H&E RandAugmented (`train_transform`: `n`, `m`,
  `blur`, `noise`, `jpeg`); val tiles are prepared as for embedding. Use
  `precision: bf16-mixed` and `loader.num_workers: 8`.
- Fine-tuning trains the backbone at `task.backbone_lr` (default `lr / 10`; 1e-5 is a
  safer start for a ViT-g) with gradient checkpointing; augmentation is optional.
- The head standardises the embeddings with the train split's mean and std (online,
  from up to 4096 un-augmented train tiles; stored in the head, so the export carries
  them), then applies one linear layer, or with `model.head: {hidden: 512, dropout:
  0.1}` an MLP (one GELU hidden layer, dropout before the output layer); trained with
  AdamW and a cosine schedule.
- Metrics: `val/acc`, `val/balanced_acc` (mean recall per class), `val/macro_f1`,
  `val/auroc` (macro one-vs-rest), `val/recall_<class>`, `val/loss`; the same as
  histossl's linear probe, so numbers compare. `best.ckpt` follows
  `val/balanced_acc`.
- `load_model(ckpt)` rebuilds the whole classifier (backbone and head) from any of
  the three checkpoints; a head-only one reloads the backbone from `model.name`.

## Export

The ONNX graph is the whole classifier: `image` (batch, 3, 224, 224), normalised with
the backbone's mean and std, to `logits` (batch, num_classes). It is written as a
pathoinfer model folder with a classification `model_card.json`. Tile folders record
no resolution or colour handling, so `--mpp` and `--icc`/`--no-icc` are required;
`--mpp` is the resolution of the 224 px input (448 px tiles at 0.25 um/px give 0.5).

## Results

MHIST (HP vs SSA, 2175 train / 977 test tiles), midnight-12k, defaults of
`config.example.yaml` (50 epochs, lr 1e-3), test as the val split:

| | bal. acc | AUROC |
|---|---|---|
| pathocls, last epoch | 0.816 | 0.891 |
| pathocls, best epoch (chosen on test, optimistic) | 0.826 | 0.897 |
| histossl linear probe (L-BFGS, L2 chosen on a train holdout) | 0.806 | 0.906 |
| MLP head, last epoch (`head: {hidden: 512, dropout: 0.1}`, cached, 50 epochs) | 0.765 | 0.885 |
| online probe, last epoch (`augmentation: {n: 2, m: 3}`, 10 epochs, bf16) | 0.795 | 0.907 |
| fine-tuned, last epoch (`backbone_lr: 1e-5`, 1 warmup epoch, batch 32, 10 epochs, bf16) | 0.866 | 0.953 |

On an RTX 5090: the cached probe's embedding pass took 30 s and its 50 head epochs about
1 s; the online probe about 15 s per epoch; fine-tuning about 50 s per epoch. Augmenting
a frozen backbone did not help here, and the MLP head overfit the 2175 tiles (train loss
0.001); fine-tuning helped. Try an MLP on larger sets, or with more dropout. A fine-tuned midnight-12k
checkpoint is large: `best.ckpt` 4.5 GB, `last.ckpt` 13.6 GB (with the optimizer
state), so keep the disk free.

## Not yet

Layer-wise lr decay and partial freezing for fine-tuning, timm backbones and
class-weighted sampling: see the design note in the pathotrain handoff.
