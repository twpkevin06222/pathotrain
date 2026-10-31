# pathotrain - histossl

> [!WARNING]
> For research use only. Not a medical device.

Self-supervised learning on pathology tiles: continued pretraining of Midnight
(teacher_jepa, dinov2) and LeJEPA from scratch. Work in progress: teacher_jepa
and dinov2 train; lejepa is planned (the config registry already knows its
rules). Needs the `histossl` extra for LoRA:

```bash
pip install "pathotrain[histossl]"   # peft
```

```python
import functools

import torch
from pathotrain.histossl import apply_adaptation, block_mask, build_backbone, param_groups, tokens

backbone = build_backbone("pretrained", id="midnight-12k")  # or ("scratch", arch="vit_b14")
model = apply_adaptation(backbone, "lora", rank=16)  # or "full", "frozen", "last_n" (n=4)
optimizer = torch.optim.AdamW(param_groups(model, lr=1e-4, weight_decay=0.05, layer_decay=0.9))
cls, patches = tokens(model, images, mask)  # (B, 1536), (B, 256, 1536) at 224 px
with model.disable_adapter():  # LoRA off: the frozen base model
    target = tokens(model, images)[1]
mask_fn = functools.partial(block_mask, ratio=(0.1, 0.5))  # for MultiCropTransform(mask=...)
```

- `build_backbone`: `pretrained` loads a `pathotrain.pathofm` model at its pinned
  revision; `scratch` builds `vit_s14`, `vit_b14` or `vit_l14` (patch 14, so 224 px
  is a 16 x 16 grid either way). Gradient checkpointing is on by default
  (non-reentrant, so LoRA adapters on a frozen trunk still get gradients).
- `tokens(model, x, mask)`: final CLS and patch tokens; masked patches take the
  model's `mask_token` (Dinov2's `bool_masked_pos`).
- `apply_adaptation`: LoRA targets among `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `fc1`, `fc2` (default `q_proj`, `v_proj`); `alpha` defaults to the rank.
- `param_groups`: AdamW groups over trainable weights, no weight decay on
  biases, norms, LayerScale and the CLS / mask / position tokens; optional
  layer-wise learning-rate decay.
- `block_mask` / `random_mask`: exactly `ratio` of the patches; `block_mask`
  masks rectangles (BEiT / iBOT style) and, with `p < 1`, leaves some views
  unmasked.

On one RTX 5090, Midnight-12k with LoRA rank 16 trains 3.9M of 1.14B weights; a
masked student plus an adapter-off teacher pass at batch 64 (224 px, bf16) peaks
at 9.6 GiB.

## Training: teacher_jepa

```bash
python -m pathotrain.histossl.train config.yaml   # start from teacher_jepa.example.yaml
python -m pathotrain.histossl.train config.yaml --resume runs/jepa01/checkpoints/last.ckpt
python -m pathotrain.histossl.eval.linear_probe runs/jepa01/checkpoints/best.ckpt \
    --train mhist/train.csv --test mhist/test.csv
```

The student (Midnight with LoRA, or its last N blocks) sees each tile with
40-60% of its patches block-masked; a small transformer predictor maps the
student's tokens to the frozen teacher's final patch tokens, and the loss is
smooth L1 at the masked patches (targets layer-normalised per token). With LoRA
the teacher is the student with its adapters off, so Midnight is held once.

- The config is checked before anything loads: unknown keys, and combinations
  a method cannot run (LoRA from scratch, an EMA teacher for teacher_jepa or
  LeJEPA, masking missing, patch sizes, FSDP, ...), fail together in one error.
- Validation reports `val/loss` on `data.val` (masks fixed per batch). With an
  `eval` section, every `eval.every` validations also run the linear probe and
  kNN, and `best.ckpt` follows `probe/balanced_acc`; otherwise `val/loss`.
- Both checkpoints store the config and a `provenance` entry: backbone and
  revision, init, normalisation, input size, adaptation (LoRA rank and targets,
  or N), method, git commit (`-dirty` with uncommitted changes) and pathotrain
  version.
- `load_backbone(ckpt)` returns the trained encoder as a plain `Dinov2Model`
  (LoRA merged), e.g. for the probe or `pathotrain.segmentation`.
- `data.samples_per_epoch` draws that many tiles per epoch (with replacement);
  `data.val.max_tiles` caps validation to evenly spaced tiles.

## Training: dinov2

```bash
python -m pathotrain.histossl.train config.yaml   # start from dinov2.example.yaml
```

Continued DINOv2 pretraining (Oquab et al. 2023) of Midnight: an EMA teacher
sees the 2 global crops of each tile, the student those (iBOT-masked) and 8
local crops. Losses, weighted as DINOv2: DINO (CLS prototype scores of every
view against the teacher's of each other global view), iBOT (patch scores at
masked patches) and KoLeo (spread of the global CLS tokens). Teacher targets
are Sinkhorn-Knopp centred. The head, centring and KoLeo are adapted from
facebookresearch/dinov2 (Apache-2.0).

- Midnight has no DINO heads, so they start from scratch: their last layer is
  held for `freeze_last_layer_epochs` (1), and `head_warmup_epochs` can train
  the heads alone first.
- Schedules: teacher momentum cosine 0.994 -> 1, teacher temperature 0.04 ->
  0.07 over `teacher_temp_warmup_epochs`, weight decay cosine 0.04 -> 0.2. The
  step count is saved with the teacher, so they resume where they stopped.
- The EMA covers the trained weights (LoRA adapters or last blocks, and the
  heads). The probe and `load_backbone` use the teacher, which DINOv2 evaluates
  and releases.
- Validation's `val/loss` uses each tile and its mirror image as the two global
  views, with fixed masks.

## Linear probe and kNN

Score a backbone's frozen tile embeddings (CLS + mean patch token) on a
labelled tile set, as class folders or `path,label` CSVs:

```bash
python -m pathotrain.histossl.eval.linear_probe midnight-12k \
    --train mhist/train.csv --test mhist/test.csv
```

- Linear probe: logistic regression on standardised features (L-BFGS). Its L2
  strength is picked by balanced accuracy on a stratified 20% of train, then
  refit on all of train and scored on test.
- kNN: 20 nearest train tiles by cosine similarity, votes weighted by
  `exp(similarity / 0.07)`.
- Metrics under `probe/` and `knn/`: `acc`, `balanced_acc`, `macro_f1`, `auroc`.
  In Python: `evaluate_features(model, train, test, num_classes)`.

Baselines on [MHIST](https://bmirds.github.io/MHIST/) (colorectal polyps, HP vs
SSA, 224 px; 2175 train / 977 test tiles, official split):

| Backbone | probe bal. acc | probe AUROC | kNN bal. acc | kNN AUROC |
|---|---|---|---|---|
| midnight-12k | 0.806 | 0.906 | 0.723 | 0.821 |
| spectra-midnight-12k | 0.801 | 0.903 | 0.753 | 0.853 |

MHIST ships one CSV for both splits (`Image Name`, `Majority Vote Label`,
`Partition`); split it into `train.csv` and `test.csv` with `path,label`
columns first. A self-supervised run should beat the first row to have helped.

## Tests

```bash
uv run pytest packages/pathotrain/tests/test_histossl.py packages/pathotrain/tests/test_histossl_train.py
```
