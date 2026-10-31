# pathotrain - histossl

> [!WARNING]
> For research use only. Not a medical device.

Self-supervised learning on pathology tiles: continued pretraining of Midnight
(teacher_jepa, dinov2) and LeJEPA from scratch. Work in progress; so far the
shared parts. Needs the `histossl` extra for LoRA:

```bash
pip install "pathotrain[histossl]"   # peft
```

```python
import functools

import torch
from pathotrain.histossl import apply_adaptation, block_mask, build_backbone, param_groups, tokens

backbone = build_backbone("pretrained", id="midnight-12k")   # or ("scratch", arch="vit_b14")
model = apply_adaptation(backbone, "lora", rank=16)         # or "full", "frozen", "last_n" (n=4)
optimizer = torch.optim.AdamW(param_groups(model, lr=1e-4, weight_decay=0.05, layer_decay=0.9))
cls, patches = tokens(model, images, mask)                   # (B, 1536), (B, 256, 1536) at 224 px
with model.disable_adapter():                                # LoRA off: the frozen base model
    target = tokens(model, images)[1]
mask_fn = functools.partial(block_mask, ratio=(0.1, 0.5))    # for MultiCropTransform(mask=...)
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

## Tests

```bash
uv run pytest packages/pathotrain/tests/test_histossl.py
```
