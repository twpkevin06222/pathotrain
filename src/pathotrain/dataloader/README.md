# pathotrain - dataloader

> [!WARNING]
> For research use only. Not a medical device.

## Segmentation dataloader

Annotations are GeoJSON (QuPath style, level-0 pixels). Convert other formats
first, e.g. ASAP XML:

```bash
uv run python -m patchology.utilities.xml_to_geojson data/train/annotations data/train/geo_json_annotations
```

You will have to write your own script for converting proprietary formats to compatible GeoJSON format.

Pass a list of slide directories and a list of annotation directories; each
slide is paired with the `.geojson` of the same stem found in any of them.
Unmatched slides or annotations are skipped with a warning.

```
train/
  slides/                 slide_name-xx.tif
  geo_json_annotations/   slide_name-xx.geojson
```

Keyword arguments go to `patchology.SegmentationTileGenerator`:

```python
from patchology import TilingConfig

kwargs = dict(
    class_map={"invasive tumor": 1, "tumor-associated stroma": 2},
    roi_class="roi",
    object_mask_overlap=0.2,  # keep tiles with >= 20% ground truth
    tiling=TilingConfig(tile_size=512, tile_mpp=0.5),
)
```

Both datasets subclass `torch.utils.data.Dataset`.

**Lazy (on the fly):** tiles every slide once, then reads each tile and mask
from the slide when the DataLoader asks for it.

```python
from torch.utils.data import DataLoader
from pathotrain.dataloader import SegmentationDataset

slides = ["data/train/slides", "data/train_02/slides"]
annotations = ["data/train/geo_json_annotations", "data/train_02/geo_json_annotations"]
train = SegmentationDataset(slides, annotations, **kwargs)
val = SegmentationDataset(["data/val/slides"], ["data/val/geo_json_annotations"], **kwargs)
loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=4)
for image, mask in loader:  # (B, 3, H, W) float32 in [0, 1], (B, H, W) int64
    ...
```

**Offline:** write tiles and masks once per split, then train from disk.

```python
from pathotrain.dataloader import SegmentationTileFolderDataset, export_segmentation_tiles

export_segmentation_tiles(slides, annotations, "tiles", "train", **kwargs)
export_segmentation_tiles(
    ["data/val/slides"], ["data/val/geo_json_annotations"], "tiles", "val", **kwargs
)
train = SegmentationTileFolderDataset("tiles/train")
```

Layout: `tiles/<split>/images/<slide>_<x>_<y>.png`, `masks/<slide>_<x>_<y>.png`
(lossless labels) and `tiles.csv`.

Tiles from elsewhere, without `tiles.csv`: pass the two directories and files
are paired by stem (images without a mask, and masks without an image, are
skipped with a warning).

```python
bcss = SegmentationTileFolderDataset(images="tissue-bcss/images", masks="tissue-bcss/masks")
```

Images of different sizes cannot be batched as they are: crop them to one size
in the transform, e.g. `train_transform(crop=512)`.

Both datasets take `transform(tile, mask)`, called with the `(H, W, 3)` uint8
tile and `(H, W)` uint8 mask; its return value is the sample. Use it for
augmentation:

```python
def augment(tile, mask):
    out = aug(image=tile, mask=mask)  # e.g. any (image, mask) pipeline
    return out["image"], out["mask"]


train = SegmentationDataset(dirs, transform=augment, **kwargs)
```

With `roi_class`, use `CrossEntropyLoss(ignore_index=255)`.

### Augmentation

`pathotrain.dataloader.augmentation` (torchvision v2) gives a ready
`transform` for each split. Both end in the same normalisation (ImageNet
mean/std by default) and return `(3, H, W)` float32 and `(H, W)` int64 tensors.

```python
from pathotrain.dataloader.augmentation import eval_transform, train_transform

train = SegmentationDataset(slides, annotations, transform=train_transform(n=3, m=5), **kwargs)
val = SegmentationDataset(val_slides, val_annotations, transform=eval_transform(), **kwargs)
```

- `train_transform`: optional random crop, flips and 90 degree rotations, then
  H&E RandAugment (`n` random operations at magnitude `m`) from
  [pathology-he-auto-augment](https://github.com/DIAGNijmegen/pathology-he-auto-augment)
  (Faryna et al., MIDL 2021): translate, shear, rotate, brightness, sharpness,
  colour, contrast, equalize, identity, HSV shift and HED stain shift. The
  operations and magnitude ranges follow the reference, which used `n=3, m=5`;
  `m` above 15 destroys tissue structure. `n=0` keeps only flips and rotations.
- Scanner artifacts (off by default): `blur`, `noise` and `jpeg` are the
  probabilities of slight defocus blur, sensor noise and JPEG compression, e.g.
  `train_transform(blur=0.2, noise=0.2, jpeg=0.2)`. They change the tile only.
- `eval_transform`: optional centre crop and normalisation only.
- `crop`: tile larger than you train on (e.g. `tile_size=576`) and pass
  `crop=512` to both, for shift augmentation without border artefacts.
- Geometric operations fill uncovered mask pixels with 255, so use
  `CrossEntropyLoss(ignore_index=255)` with augmentation too.

`HEDJitter`, `HSVJitter`, `Enhance` and `RandomRot90` are torchvision v2 transforms, so
they also fit in your own `v2.Compose` (wrap inputs as `tv_tensors.Image` / `Mask`).

## Detection dataloader

`DetectionDataset` is the detection counterpart of `SegmentationDataset`: the
same slide and GeoJSON directories, with keyword arguments going to
`patchology.DetectionTileGenerator`. Each annotated object becomes the
bounding box of its polygon.

```python
from torch.utils.data import DataLoader
from patchology import TilingConfig
from pathotrain.dataloader import DetectionDataset
from pathotrain.dataloader.augmentation import eval_transform, train_transform
from pathotrain.detection import collate

kwargs = dict(
    class_map={"class01": 0, "class02": 1},  # labels 0 .. num_classes - 1
    roi_class="roi",
    roi_fill=(255, 255, 255),  # paint outside the ROI instead of skipping tiles that cross it
    min_box_visibility=0.6,  # keep an object cut by the tile edge if 60% of its box shows
    min_objects=1,           # keep tiles with at least one object (0 adds empty tiles)
    tiling=TilingConfig(tile_size=512, tile_mpp=0.5),
)
train = DetectionDataset(slides, annotations, transform=train_transform(boxes=True), **kwargs)
val = DetectionDataset(val_slides, val_annotations, transform=eval_transform(boxes=True), **kwargs)
loader = DataLoader(train, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate)
for images, targets in loader:  # (B, 3, H, W); one {"boxes", "labels"} dict per image
    ...
```

**Offline:** write tiles and a COCO `_annotations.coco.json` once per split,
then train from disk.

```python
from pathotrain.dataloader import DetectionTileFolderDataset, export_detection_tiles

export_detection_tiles(slides, annotations, "tiles", "train", **kwargs)
export_detection_tiles(val_slides, val_annotations, "tiles", "valid", **kwargs)
train = DetectionTileFolderDataset("tiles/train", transform=train_transform(boxes=True))
```

Layout: `tiles/<split>/<slide>_<x>_<y>.png` and `tiles/<split>/_annotations.coco.json`
(boxes `[x, y, w, h]` in tile pixels, `category_id = label + 1`, each image's
`slide`, `x`, `y`). This is Roboflow's layout, which RF-DETR's own trainer and
other COCO tools read; name the splits `train`, `valid` and `test` for them.
`DetectionTileFolderDataset` reads any COCO folder in this layout, with labels
in category id order (names in `ds.class_names`).

- A sample is the image and `{"boxes": (N, 4) float32 [x1, y1, x2, y2] pixels,
  "labels": (N,) int64}`, as `pathotrain.detection.DetectionTask` takes.
  `collate` keeps the targets as a list, since `N` differs per tile.
- `class_map` is required, with labels `0 .. num_classes - 1`; several
  annotation classes may share a label.
- `boxes=True` makes the transforms move boxes with the tile (flips, 90 degree
  turns, crop, RandAugment's translate, shear and rotate), clip them to it and
  drop those under 1 px. Rotation and shear grow a box to enclose its turned
  corners. A crop clips boxes without applying `min_box_visibility` again, so
  prefer tiling at the training size to cropping.

## Paired-tile dataloader

For image-to-image translation (virtual staining with `pathotrain.pathogen`):
two folders of registered tiles per split, paired by file stem. Tiles of a pair
must have one size; unpaired tiles are skipped with a warning.

```
HE/train/wsi003_id1002_x22528_y69632.jpg    IHC/train/wsi003_id1002_x22528_y69632.jpg
```

```python
from pathotrain.dataloader import PairedTileDataset, paired_transform

train = PairedTileDataset(["HE/train"], ["IHC/train"], paired_transform(True, crop=512))
val = PairedTileDataset(["HE/val"], ["IHC/val"], paired_transform(False, crop=512))
source, target = train[0]  # (3, 512, 512) float32 in [0, 1] each
```

`paired_transform(train, crop, resize)` applies the same geometry to both
tiles: an optional resize, a random (train) or centre (val) crop, and for
training flips and 90 degree turns. It never changes colours, since the
target's colours are what a translation model learns.

### Tiles from a registered slide pair

To make those folders from whole slides, register the IHC slide to its H&E
slide (`pathotrain.registration`, see its README) and export the pairs:

```python
from patchology import TilingConfig
from pathotrain.dataloader.registration import export_registered_tiles
from pathotrain.registration import register

reg = register("he_slides/0_HE_val.tiff", "ihc_slides/0_KI67_val.tiff")
export_registered_tiles(reg, "tiles/train", tiling=TilingConfig(tile_size=1024, tile_mpp=1.0))
# tiles/train/source/0_HE_val_<x>_<y>.png  H&E
# tiles/train/target/0_HE_val_<x>_<y>.png  IHC over the same tissue
```

The H&E is tiled by `patchology.TileGenerator` (keyword arguments go to it);
each IHC tile is read from the slide through the registration and warped onto
the H&E tile's grid, so no warped whole slide is written. Tiles are kept where
the H&E has tissue, whether or not the IHC section does (it may be torn or
missing there).

## Tests

```bash
uv run pytest packages/pathotrain/tests
```

## License

Apache License 2.0, see [LICENSE](LICENSE).
