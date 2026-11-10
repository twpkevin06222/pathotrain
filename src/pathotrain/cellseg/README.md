# pathotrain - cellseg

Cell (nucleus) segmentation, written from the papers. The authors' code and trained
weights are not used. Only encoder weights are loaded: timm ImageNet weights, or a
`pathofm` foundation model.

Three ways to train:

| What | Command | Config |
|---|---|---|
| Semantic: a nucleus type per pixel | `python -m pathotrain.segmentation.train` | `config.example.yaml` |
| Instance: each nucleus and its type (HoVer-Net style) | `python -m pathotrain.cellseg.train` (`.eval`, `.export`) | `instance.example.yaml` |
| CellViT++ cell classifier on frozen foundation-model tokens | `python -m pathotrain.cellseg.classify` | `classify.example.yaml` |

## Models

| `model.name` | Paper | Encoder | Decoder |
|---|---|---|---|
| `cellseg:hovernet` | HoVer-Net, Graham et al. 2019 | pre-activation ResNet-50 (timm `resnetv2_50`), stride-1 stem | HoVer-Net dense decoder ("fast" mode, padded convs), one per branch |
| `cellseg:hovernext` | HoVer-NeXt, Baumann et al. 2024 | ConvNeXt-V2 (timm, `encoder`) | U-Net (SMP), shared by the branches |
| `cellseg:cellvit` | CellViT, Hörst et al. 2024 | `pathofm` ViT (`backbone`), four layers | CellViT transposed-conv decoder, one per branch |

A semantic model outputs the nuclear-type (NT) logits, `(B, num_classes, H, W)`. An
instance model (`instance=True`, which `cellseg.train` sets) returns HoVer-Net's three
branches as a dict: `np` (nucleus vs background, 2 logits), `hv` (horizontal and
vertical offsets from the nucleus centre, in [-1, 1]) and `nt`.

Options per model:

- **hovernet:** `encoder_weights: imagenet` (timm, Apache-2.0) or `null`. Input
  sides must be divisible by 8.
- **hovernext:** `encoder` (default `convnextv2_base`), `encoder_weights` (default
  `null`: timm's ConvNeXt-V2 ImageNet weights are CC-BY-NC-4.0). For an Apache-2.0
  pretrained encoder, use `encoder: convnext_base` with `encoder_weights: imagenet`.
  Other keys go to `smp.Unet`.
- **cellvit:** `backbone` (e.g. `midnight-12k`; then set `mean`/`std` to `MIDNIGHT_*`),
  `layers` (four 1-based layers, default evenly spaced), `freeze` (default `true`),
  `skip_dims`, `bottleneck_dim`. Input sides must be divisible by 16.

## Instance segmentation

`cellseg.train` uses HoVer-Net's loss: cross-entropy plus Dice on NP and NT, MSE and
gradient MSE on HV (`task.loss_weights`). Validation turns each tile's outputs into
nuclei with HoVer-Net's watershed (`cellseg.instances.postprocess`, options in
`task.postprocess`) and logs:

- `val/bpq`: binary PQ, the mean over tiles (PanNuke's convention).
- `val/mpq`: multi-class PQ+, each type's PQ over the whole set, averaged over types (CoNIC's).
- `val/pq_<type>`, and the binary `val/dq` (detection) and `val/sq` (segmentation quality).

`best.ckpt` is chosen on `val/mpq`. `python -m pathotrain.cellseg.eval best.ckpt` scores
the test split.

### ONNX export

```bash
python -m pathotrain.cellseg.export best.ckpt out/model.onnx --size 256 --mpp 0.25 --no-icc
```

The graph returns the three branches before any activation: `np`, `hv` and `nt`. The
watershed runs outside the graph. On one tile's outputs,
`cellseg.instances.instances_from_outputs(np[i], hv[i], nt[i])` returns the instance map
and each instance's type, the same code validation uses.

`model_card.json` is written beside the graph and embedded in it. pathoinfer has no
instance task, so the card is a segmentation card on `nt`: pathoinfer runs the file as a
per-pixel nucleus-type map, and `pathoinfer card validate` accepts the folder. Classes need
names (`task.class_names`). Tile folders record no resolution, so `--mpp`,
`--icc`/`--no-icc` and `--size` are required. A ViT-g CellViT writes its 4.8 GB of
weights to `model.onnx.data`, which must stay beside the graph.

## CellViT++ cell classifier

`cellseg.classify` embeds each ground-truth cell as the mean of a frozen `pathofm`
encoder's tokens over the 16 px patches it covers, weighted by its area in each. It then
trains a linear or MLP head (pathocls's `Head`) on those embeddings. New cell classes
need only this small head, not a new segmentation model. Metrics are pathocls's:
`val/balanced_acc`, `val/macro_f1`, `val/recall_<class>`, and so on.

At inference, `classify_cells(backbone, head, image, instances)` types the nuclei of a
predicted instance map. `load_head(best.ckpt)` returns the head and its config, which
names the backbone. The head trains on ground-truth cells; CellViT++ trains on detected
cells matched to the ground truth.

## Data

- **Semantic** (`segmentation.train`): `images` and `masks` folders, as in
  `segmentation`. `image_prefix` / `mask_prefix` are removed before pairing, so
  PanNuke's `img_X.png` pairs with `sem_X.png`.
- **Instance and classifier** (`CellDataset`): `images`, `instances` and `types`
  folders, each with an optional prefix. Instance masks hold one id per nucleus. With
  `contours: true`, they hold nucleus outlines instead, as in the PanNuke conversion at
  `pannuke_split`, and the ids are derived: nucleus pixels inside no outline, split into
  connected pieces, with outline pixels given to the nearest piece. Type masks hold the
  class per pixel (0 background, 255 ignored).
