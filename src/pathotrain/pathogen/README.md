# pathotrain - pathogen

> [!WARNING]
> For research use only. Not a medical device.

Generative models for pathology, starting with virtual staining: translate a
tile in one stain (e.g. H&E) into another (e.g. HER2 IHC) from registered,
paired tiles. Training runs on pathotrain's engine (Lightning Fabric, MLflow
and CSV logging, checkpoints, early stopping).

```bash
pip install "pathotrain[pathogen]"   # diffusers
```

## Data

Two folders of tiles per split, paired by file stem. Tiles of a pair must be
registered and of one size; unpaired tiles are skipped with a warning.

```
HE/train/wsi003_id1002_x22528_y69632.jpg    IHC/train/wsi003_id1002_x22528_y69632.jpg
HE/val/...                                  IHC/val/...
```

```python
from pathotrain.dataloader import PairedTileDataset, paired_transform

train = PairedTileDataset(["HE/train"], ["IHC/train"], paired_transform(True, crop=512))
source, target = train[0]  # (3, 512, 512) float32 in [0, 1] each
```

`paired_transform` crops, flips and turns both tiles together; it never
changes colours, since the target's colours are what the model learns. Both
live in `pathotrain.dataloader` (see its README); the models scale tiles to the
VAE's `[-1, 1]` themselves.

## Baseline: latent diffusion

Stable Diffusion's recipe for image-to-image translation:

- A frozen VAE (`stabilityai/sd-vae-ft-mse`, MIT) maps tiles to latents at
  1/8 of the side (512 px to 4 x 64 x 64).
- A UNet (diffusers `UNet2DModel`, 106M parameters by default, trained from
  scratch) denoises the IHC latent with the H&E latent concatenated on the
  channel axis, as InstructPix2Pix conditions on its input image.
- Stable Diffusion's DDPM noise schedule, v-prediction by default; DDIM
  sampling (50 steps by default).

The VAE bounds what any model on its latents can reach: on HER2match val
tiles it reconstructs IHC at 31.1 dB PSNR and 0.80 SSIM.

### Training

```bash
cp $(python -c "import pathotrain.pathogen, pathlib; print(pathlib.Path(pathotrain.pathogen.__file__).parent)")/config.example.yaml config.yaml
python -m pathotrain.pathogen.train config.yaml
python -m pathotrain.pathogen.train config.yaml --resume runs/ldm01/checkpoints/last.ckpt
```

- Validation reports `val/loss` (the denoising loss with fixed noise, so epochs
  compare) and `val/psnr`, `val/ssim` of images sampled for the first
  `task.sample_batches` val batches. `best.ckpt` is chosen on `val/ssim`.
- `load_model("runs/ldm01/checkpoints/best.ckpt")` returns the task (in eval
  mode) and its config; `task.translate(source)` samples targets in `[0, 1]`.
- The VAE is frozen and runs in fp32; the UNet trains at the trainer's
  precision (bf16-mixed by default). On one RTX 5090, 512 px crops at batch 8
  take ~0.35 s per step, most of it encoding both tiles with the VAE.

### Translation

```bash
python -m pathotrain.pathogen.translate runs/ldm01/checkpoints/best.ckpt HE/test out/ --steps 50
```

Writes `out/<stem>.png` per tile, at the tile's own size (sides a multiple of
64 for the default UNet, 16 for SiT with 2 x 2 patches), on the GPU in bf16
when one is present.

In Python, `load_model` rebuilds the task from the config stored in the
checkpoint, with the EMA weights, in eval mode. Tiles go in as `[0, 1]` (not
mean/std normalised):

```python
import torch
from pathotrain.dataloader.base import to_image_tensor
from pathotrain.pathogen.train import load_model

task, cfg = load_model("runs/her2_sit01/checkpoints/best.ckpt")
task.cuda()
x = to_image_tensor(he_tile)[None].cuda()  # (H, W, 3) uint8 tile -> (1, 3, H, W) in [0, 1]
with torch.autocast("cuda", torch.bfloat16):
    ihc = task.translate(x, steps=25, generator=torch.Generator("cuda").manual_seed(0))
ihc = (ihc[0].float().permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
```

## SiT with rectified flow and REPA

`model.name: sit` swaps the UNet and DDPM for the current recipe; data, VAE,
validation, training and translation stay the same.

- **SiT** (Ma et al., 2024): a DiT-style transformer over 2 x 2 patches of the
  latent (a 512 px tile is 32 x 32 tokens), the time fed through adaLN-Zero.
  The source latent is concatenated to the noisy target latent, as in the
  baseline. Sizes S, B, L, XL: 32M, 130M, 457M, 674M parameters.
- **Rectified flow**: `x_t = (1 - t) x_0 + t noise`; the model predicts the
  velocity `noise - x_0`, with `t` drawn logit-normal (SD3). Sampling follows
  the ODE from noise in 25 Euler steps by default (`sampler: heun` for two
  model calls per step), against 50 DDIM steps for the baseline.
- **REPA** (Yu et al., 2025): an MLP projects block `repa.depth`'s tokens onto
  frozen Midnight's patch features of the clean tile, and `repa.weight * (1 -
  cosine similarity)` is added to the loss. Midnight sees the tile at 14 px per
  token (448 px for a 512 px crop), so its patches match the tokens one to one.
  `repa.image: source` aligns with Midnight's H&E features instead, for when
  its features of the target stain are weak. The encoder and projector are
  only for training: checkpoints keep the projector, `load_model` drops both.

```yaml
model:
  name: sit
  vae: stabilityai/sd-vae-ft-mse
  size: B
  repa: {encoder: midnight-12k, depth: 4, weight: 0.5, image: target}
task: {lr: 1.0e-4, sampler: euler, sample_steps: 25}
```

Training logs `flow_loss`, `repa_loss` and their sum `loss`; `val/loss` is the
flow loss alone. On one RTX 5090, 512 px crops at batch 8 take ~0.27 s per
step for SiT-B/2, ~0.40 s with REPA (Midnight's forward pass is ~0.11 s).

## EMA and the latent cache

**EMA.** Both models keep an exponential moving average of their weights
(`task.ema_decay`, 0.9999 by default, warmed up as `min(decay, (1 + n) / (10 +
n))`), updated after every optimizer step. Validation, `best.ckpt` selection,
`load_model` and `translate` use it; checkpoints store it under `"ema"` next to
the trained `"model"`, and a resumed run continues it.

**Latent cache.** The frozen VAE takes most of a training step. Turn on the
cache to encode the training tiles once instead:

```yaml
data:
  crop: 512
  cache: true          # default false
  cache_variants: 2    # optional, 1-8
  train: {source: [HE/train], target: [IHC/train]}
```

- The first run builds the cache before epoch 1; later runs on the same data
  reuse it. `python -m pathotrain.pathogen.cache config.yaml` builds it ahead of
  time. It lives in `~/.cache/pathotrain/latents/<fingerprint>` (or
  `data.cache_dir`), named after the VAE, the tile directories, the paired tile
  names and `cache_variants`, so changing any of them builds a new cache.
- Whole tiles are encoded; each step takes a random `crop / 8` window of their
  latents (and, for REPA, the same window of the image).
- The VAE is not equivariant to flips and turns (a flipped latent is ~60% off
  the latent of the flipped tile), so each augmentation orientation is encoded
  on its own: `cache_variants` of the 8 flips and turns, 2 (identity and
  horizontal flip) by default. Without the cache, augmentation uses all 8.
- HER2match train (11,610 pairs of 1024 px tiles), 2 variants: 5.7 GB, built
  in 73 min on one RTX 5090. Training steps (512 px crops, batch 8) then take
  0.032 s instead of 0.35 s for the UNet, and 0.18 s instead of 0.40 s for
  SiT-B/2 with REPA, which still runs Midnight on the image crop each step.
  Validation still encodes its tiles, so it becomes most of an epoch's time.
- Validation always encodes images, so its metrics compare with uncached runs.
  Under `torchrun`, rank 0 builds and the other ranks wait.

## Tests

```bash
uv run pytest packages/pathotrain/tests/test_pathogen.py
```
