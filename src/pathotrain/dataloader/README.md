# pathotrain

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
from pathotrain.dataloader import TileFolderDataset, export_tiles

export_tiles(slides, annotations, "tiles", "train", **kwargs)
export_tiles(["data/val/slides"], ["data/val/geo_json_annotations"], "tiles", "val", **kwargs)
train = TileFolderDataset("tiles/train")
```

Layout: `tiles/<split>/images/<slide>_<x>_<y>.png`, `masks/<slide>_<x>_<y>.png`
(lossless labels) and `tiles.csv`.

Both datasets take `transform(tile, mask)`, called with the `(H, W, 3)` uint8
tile and `(H, W)` uint8 mask; its return value is the sample. Use it for
augmentation:

```python
def augment(tile, mask):
    out = aug(image=tile, mask=mask)  # e.g. an albumentations Compose
    return out["image"], out["mask"]


train = SegmentationDataset(dirs, transform=augment, **kwargs)
```

With `roi_class`, use `CrossEntropyLoss(ignore_index=255)`.

## Tests

```bash
uv run pytest packages/pathotrain/tests
```

## License

Apache License 2.0, see [LICENSE](LICENSE).