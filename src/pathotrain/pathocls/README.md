# pathocls

Tile classification as a linear probe you can train, score, export and serve: a frozen
foundation-model backbone (`pathofm`, or a histossl checkpoint or export), its tile
embedding (CLS token and mean patch token) computed once per tile, and a linear head
trained on those embeddings.

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

- The backbone is frozen and runs once per split. The head trains on the embeddings
  in memory, so an epoch takes milliseconds; `loader.num_workers: 0` is right for it.
- The head standardises the embeddings with the train split's mean and std (stored
  in the head, so the export carries them), then applies one linear layer, trained
  with AdamW and a cosine schedule.
- Metrics: `val/acc`, `val/balanced_acc` (mean recall per class), `val/macro_f1`,
  `val/auroc` (macro one-vs-rest), `val/recall_<class>`, `val/loss`; the same as
  histossl's linear probe, so numbers compare. `best.ckpt` follows
  `val/balanced_acc`.
- Checkpoints hold the head and the config; `load_model(ckpt)` rebuilds the whole
  classifier (backbone and head).

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

The embedding pass took 30 s on an RTX 5090; the 50 head epochs about 1 s.

## Not yet

Augmentation (the backbone would then run every step), fine-tuning the backbone, timm
backbones, an MLP head and class-weighted sampling: see the design note in the
pathotrain handoff.
