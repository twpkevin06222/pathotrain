# pathotrain

Training on whole slide images (WSI), built on
[patchology](https://github.com/twpkevin06222/patchology) for tiling.

> [!WARNING]
> For research use only. Not a medical device.

## Segmentation dataloader

Each data directory holds `slides/` and `annotations/`. A slide is paired with
the annotation of the same stem (ASAP `.xml` or QuPath `.geojson`); unmatched
slides or annotations are skipped with a warning.

```
train/
  slides/       TCGA-XX.tif
  annotations/  TCGA-XX.xml
```

Keyword arguments go to `patchology.SegmentationTileGenerator`:

```python
from patchology import TilingConfig

kwargs = dict(
    class_map={"invasive tumor": 1, "tumor-associated stroma": 2},
    roi_class="roi",            # TIGER-style annotated region; mask is 255 outside it
    object_mask_overlap=0.2,    # keep tiles with >= 20% ground truth
    tiling=TilingConfig(tile_size=512, tile_mpp=0.5),
)
```

**Lazy (on the fly):** tiles every slide once, then reads each tile and mask
from the slide when the DataLoader asks for it.

```python
from torch.utils.data import DataLoader
from pathotrain.dataloader import SegmentationDataset

train = SegmentationDataset(["data/train", "data/train_02"], **kwargs)
val = SegmentationDataset(["data/val"], **kwargs)
loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=4)
for image, mask in loader:  # (B, 3, H, W) float32 in [0, 1], (B, H, W) int64
    ...
```

**Offline:** write tiles and masks once per split, then train from disk.

```python
from pathotrain.dataloader import TileFolderDataset, export_tiles

export_tiles(["data/train", "data/train_02"], "tiles", "train", **kwargs)
export_tiles(["data/val"], "tiles", "val", **kwargs)
train = TileFolderDataset("tiles/train")
```

Layout: `tiles/<split>/images/<slide>_<x>_<y>.png`, `masks/<slide>_<x>_<y>.png`
(lossless labels), `geojson/` (converted XML) and `tiles.csv`.

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
