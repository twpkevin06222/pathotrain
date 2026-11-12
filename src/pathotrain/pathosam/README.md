# pathotrain - pathosam

> [!WARNING]
> For research use only. Not a medical device.

SAM 3 for prompted and interactive segmentation of pathology images: an annotator clicks
or boxes a cell and gets its mask. The package builds on transformers' SAM 3 models: it
scores them with prompts simulated from the ground truth, zero-shot or fine-tuned, and
fine-tunes the point and box model.

SAM 3 (`facebook/sam3`) is gated on Hugging Face and released under the SAM License (see
the [pathofm README](../pathofm/README.md#sam-3-dense-backbone-gated)).

## Zero-shot baselines

```bash
cp src/pathotrain/pathosam/configs/zeroshot.example.yaml zeroshot.yaml   # set the paths
python -m pathotrain.pathosam.eval zeroshot.yaml --split val --out val.json
python -m pathotrain.pathosam.eval zeroshot.yaml --split test --only tracker
```

The data is an `InstanceTileFolderDataset` (see the dataloader README). Each tile is
encoded once and reused for every prompt.

**`tracker`: SAM 3's point and box model (`Sam3TrackerModel`), one nucleus per prompt.**
Every nucleus of at least `min_size` px is prompted with:

- a box, for each jitter in `tracker.box_jitter` (each side moved by up to that share of
  the box's size);
- a click at its centre (the pixel farthest from its boundary), then corrective clicks up
  to `tracker.max_clicks`. Each round adds one click and feeds the previous mask back in.

A corrective click follows RITM's NoC protocol: it goes to the deepest pixel of either the
missed or the extra region, whichever is deeper. A click in a missed region is positive,
one in an extra region negative. The simulation is in `pathosam.prompts`, so fine-tuning
can reuse it.

| Metric | Meaning |
|---|---|
| `iou_box_<jitter>` | mean IoU from one box |
| `iou_click<k>` | mean IoU after k clicks |
| `noc85`, `noc90` | mean clicks to reach 85% or 90% IoU (`max_clicks` when never) |
| `fail85`, `fail90` | share of nuclei that never reach it |

**`exemplar`: SAM 3's concept model (`Sam3Model`) with one example box per nucleus type.**
For each type present in a tile, the box of one of its nuclei asks for "all nuclei like
this one". Every detection takes that type, and overlaps go to the higher score. The result
is scored like `cellseg.eval` (`bpq`, `mpq`, `pq_<type>`, `dq`, `sq`) at every threshold in
`exemplar.thresholds`, so it compares directly with the trained cellseg models. The example
nucleus itself counts, as an annotator's box would.

Metrics are keyed `<split>/<evaluation>/<metric>`.

## Fine-tuning the point and box model

```bash
cp src/pathotrain/pathosam/configs/interactive.example.yaml interactive.yaml   # set the paths
python -m pathotrain.pathosam.train interactive.yaml
python -m pathotrain.pathosam.eval zeroshot.yaml --split test --only tracker \
    --checkpoint runs/pathosam-interactive/checkpoints/best.ckpt
```

The image encoder stays frozen; the prompt encoder and mask decoder train (4.2M of 458M
parameters), and checkpoints hold only those. Each training step simulates an interactive
session for up to `task.objects` nuclei per tile, as SAM was trained:

- **First prompt:** a jittered box (with probability `task.box_share`), else one click
  sampled inside the nucleus.
- **Then corrective clicks:** `task.rounds - 1` of them, each sampled uniformly from the
  previous prediction's error. A click where the nucleus was missed is positive, one
  where the prediction spilled over is negative. The previous mask is fed back each
  round.
- **Loss, every round:** focal (weight 20) plus Dice on the mask, and the MSE between
  predicted and true IoU. A single first click predicts three masks: the mask loss trains
  only the best one, the IoU loss all three. Inference keeps the mask with the best
  predicted IoU, so an untrained IoU score picks the wrong mask. Training only the best
  mask's score left the other two drifting upward, and by epoch 3 the wrong mask was
  picked for 44% of first clicks.

Validation runs the same click protocol as `pathosam.eval` with `task.val_clicks` clicks.
`best.ckpt` is chosen on `val/noc90`, the fewest clicks to 90% IoU.

Tiles are normalised with mean = std = 0.5, as SAM 3 expects, and resized to 1008 px inside
the model. At 2 tiles x 16 nuclei per step, one RTX 5090 uses about 14 GB; 4 tiles nears
its 32 GB.

Anything distributed that is built on SAM 3, such as a fine-tuned checkpoint, falls under
the SAM License. A model trained on PanNuke is also non-commercial.

## Export for serving

```bash
python -m pathotrain.pathosam.export model/ --checkpoint runs/pathosam-interactive/checkpoints/best.ckpt \
    --mpp 0.25 --register pathosam
```

This writes the model folder that patholamina's sam3 service loads:

- **`model.safetensors`:** all of SAM 3 (detector and tracker) with the fine-tuned tracker
  merged in, plus SAM 3's `config.json`, processor and tokenizer files and `LICENSE`.
  `Sam3TrackerModel.from_pretrained(folder)` serves clicks and boxes, and
  `Sam3Model.from_pretrained(folder)` serves text prompts.
- **`model_card.json`:** task `interactive_segmentation`, `input.mpp` (the training mpp),
  `input.tile_size` (`--size`), and `prompt_vocabulary`: the text prompts SAM 3 finds nuclei
  with on H&E (`cell`, `neoplastic cell`; change them with `--prompt`, repeatable).

Tile folders record no resolution, so `--mpp` is required unless training tiled slides.
`--register NAME` logs the folder to MLflow (`--mlflow-db`, default the config's) and
registers it as a new version of NAME, with the card at the version's artifact root. Without
`--checkpoint`, the export is zero-shot SAM 3.

## ONNX export

```bash
python -m pathotrain.pathosam.export out/ --checkpoint runs/pathosam-interactive/checkpoints/best.ckpt --onnx
python -m pathotrain.pathosam.eval zeroshot.yaml --split test --only tracker --onnx out/
```

As with SAM's own ONNX export, the model is split in two:

- **`encoder.onnx` (1.9 GB):** runs once per tile. It takes the normalised tile
  (mean = std = 0.5) at `--size` px and returns three feature maps, `fpn0`, `fpn1` and
  `fpn2`.
- **`decoder.onnx` (17 MB):** runs once per click, for N objects (`objects` and `points`
  are dynamic axes).
  - Inputs: the feature maps, `points` in tile pixels, and `labels` (1 positive, 0
    negative, 2 and 3 a box's corners after the clicks, -10 padding). Also `mask_input`,
    the previous `low_res` logits, with `has_mask`.
  - Outputs: the single mask (`low_res`, `iou`) for every round after the first, and three
    candidates (`multi_low_res`, `multi_iou`) for an ambiguous first click.
  - Mask logits are 288 x 288. Resize them to 1008, then to the tile, and threshold at 0
    (`pathosam.model.to_tile`).

`pathosam.json` beside the graphs (also stored in each) records the tile size,
normalisation, input and output names, label codes and licence. Export checks both graphs
against PyTorch. `pathosam.export.OnnxTracker` runs them behind the same interface as the
PyTorch model, so `pathosam.eval --onnx` scores an export with the same click protocol.

On 756 PanNuke val nuclei, the fine-tuned export matched PyTorch within 0.001 IoU at every
click (ONNX Runtime with CUDA). The encoder's float32 drift is larger on random input than
on real tiles, so its check tolerance is 1e-2. The files are SAM 3's weights, under the
SAM License.

## Notes

- SAM 3's point/box model reads the same image-encoder ViT, with identical weights, as
  `pathofm`'s `sam3` backbone. An encoder fine-tuned in cellseg could therefore be loaded
  into it (not done yet).
- SAM 3 resizes every tile to 1008 px, so a 256 px PanNuke tile is upsampled about 4x.
