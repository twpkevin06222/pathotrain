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
from pathogen import PairedTileDataset, paired_transform

train = PairedTileDataset(["HE/train"], ["IHC/train"], paired_transform(True, crop=512))
source, target = train[0]  # (3, 512, 512) float32 in [-1, 1] each
```

`paired_transform` crops, flips and turns both tiles together; it never
changes colours, since the target's colours are what the model learns.

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
python -m pathotrain.pathotrain.pathogen.train config.yaml
python -m pathotrain.pathotrain.pathogen.train config.yaml --resume runs/ldm01/checkpoints/last.ckpt
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
64 for the default UNet), on the GPU in bf16 when one is present.

## Tests

```bash
uv run pytest packages/pathotrain/tests/test_pathogen.py
```
