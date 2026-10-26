"""Segmentation datasets built from directories of slides and their annotations.

Each input directory holds ``slides/`` and ``annotations/``; a slide is paired
with the annotation file of the same stem (ASAP ``.xml`` or ``.geojson``).
Slides and annotations without a partner are skipped with a warning.

Two ways to feed a PyTorch ``DataLoader``:

* Lazy: :class:`SegmentationDataset` tiles every slide in ``__init__`` and
  reads each ``(tile, mask)`` from the slide on demand.
* Offline: :func:`export_tiles` writes tiles and masks to
  ``out_dir/<split>/{images,masks}/`` and :class:`TileFolderDataset` reads them back.

Both datasets are map-style (``len``, ``ds[i]``), so they go straight into a
``DataLoader`` with any ``num_workers``. A sample is ``(image, mask)``: a
``(3, H, W)`` float32 image in ``[0, 1]`` and an ``(H, W)`` int64 mask, or
whatever ``transform(tile, mask)`` returns, given the ``(H, W, 3)`` uint8 tile
and ``(H, W)`` uint8 mask.

Example (TIGER-style annotations)::

    kwargs = dict(
        class_map={"invasive tumor": 1, "tumor-associated stroma": 2},
        roi_class="roi",
        tiling=TilingConfig(tile_size=512, tile_mpp=0.5),
    )
    train = SegmentationDataset(["data/train", "data/train_02"], **kwargs)
    loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=4)

    export_tiles(["data/train", "data/train_02"], "tiles", "train", **kwargs)
    train = TileFolderDataset("tiles/train")
"""

import csv
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Self

import numpy as np
from loguru import logger
from PIL import Image

from patchology import SegmentationTileGenerator
from patchology.utilities.xml_to_geojson import convert

ANNOTATION_SUFFIXES = (".geojson", ".json", ".xml")

Transform = Callable[[np.ndarray, np.ndarray], object]


def find_pairs(
    dirs: Sequence[str | Path],
    geojson_dir: str | Path | None = None,
    slides: str = "slides",
    annotations: str = "annotations",
) -> list[tuple[Path, Path]]:
    """``(slide, ground truth GeoJSON)`` pairs from ``<dir>/slides`` and ``<dir>/annotations``.

    ASAP XML is converted to ``<geojson_dir>/<stem>.geojson`` (a temporary
    directory by default), since the tile generators read GeoJSON.

    Raises:
        FileNotFoundError: A directory has no ``slides`` or ``annotations`` folder.
        ValueError: No slide in ``dirs`` has an annotation.
    """
    pairs = []
    for d in map(Path, dirs):
        slide_dir, ann_dir = d / slides, d / annotations
        for sub in (slide_dir, ann_dir):
            if not sub.is_dir():
                raise FileNotFoundError(f"{sub} is not a directory")
        found = {p.stem: p for p in sorted(slide_dir.iterdir()) if p.is_file()}
        anns = {
            p.stem: p for p in sorted(ann_dir.iterdir()) if p.suffix.lower() in ANNOTATION_SUFFIXES
        }
        for stem in sorted(found.keys() - anns.keys()):
            logger.warning("{}: slide {} has no annotation, skipped", d, found[stem].name)
        for stem in sorted(anns.keys() - found.keys()):
            logger.warning("{}: annotation {} has no slide, skipped", d, anns[stem].name)
        for stem in sorted(found.keys() & anns.keys()):
            ann = anns[stem]
            if ann.suffix.lower() == ".xml":
                if geojson_dir is None:
                    geojson_dir = tempfile.mkdtemp(prefix="pathotrain-geojson-")
                ann = convert(ann, geojson_dir)
            pairs.append((found[stem], ann))
    if not pairs:
        raise ValueError(f"no slide with a matching annotation in {list(dirs)}")
    return pairs


def _sample(tile: np.ndarray, mask: np.ndarray, transform: Transform | None) -> object:
    if transform is not None:
        return transform(tile, mask)
    image = np.ascontiguousarray(tile.transpose(2, 0, 1), dtype=np.float32) / 255
    return image, mask.astype(np.int64)


class SegmentationDataset:
    """Lazy ``(image, mask)`` tiles of every annotated slide in ``dirs``.

    Each slide is tiled once in ``__init__`` (tissue, ROI and ground-truth
    overlap selection); tiles and masks are read from the slide per item.
    Keyword arguments go to :class:`patchology.SegmentationTileGenerator`
    (``class_map``, ``roi_class``, ``object_mask_overlap``, ``tiling``, ...).
    """

    # shortcut: every slide stays open for the dataset's lifetime; for thousands of
    # slides, open them per item instead (fd limits).
    def __init__(
        self,
        dirs: Sequence[str | Path],
        transform: Transform | None = None,
        geojson_dir: str | Path | None = None,
        **generator_kwargs,
    ) -> None:
        self.transform = transform
        self.generators = []
        try:
            for slide, ground_truth in find_pairs(dirs, geojson_dir):
                self.generators.append(
                    SegmentationTileGenerator(slide, ground_truth, **generator_kwargs)
                )
        except Exception:
            self.close()
            raise
        # offsets[k] is the dataset index of generator k's first tile.
        self.offsets = np.cumsum([0] + [len(g) for g in self.generators])
        logger.info("{} tiles from {} slides", len(self), len(self.generators))

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def locate(self, index: int) -> tuple[SegmentationTileGenerator, int]:
        """The generator holding dataset ``index`` and the tile's index within it."""
        if not -len(self) <= index < len(self):
            raise IndexError(f"index {index} out of range for {len(self)} tiles")
        index %= len(self)
        k = int(np.searchsorted(self.offsets, index, side="right")) - 1
        return self.generators[k], index - int(self.offsets[k])

    def __getitem__(self, index: int) -> object:
        gen, i = self.locate(index)
        _, tile, mask = gen.get_tile(i)
        return _sample(tile, mask, self.transform)

    def close(self) -> None:
        for gen in self.generators:
            gen.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def export_tiles(
    dirs: Sequence[str | Path],
    out_dir: str | Path,
    split: str,
    fmt: str = "png",
    **generator_kwargs,
) -> Path:
    """Write every annotated slide's tiles and masks under ``out_dir/split`` and return it.

    Layout: ``images/<slide>_<x>_<y>.<fmt>``, ``masks/<slide>_<x>_<y>.png`` (always
    lossless), ``geojson/`` (converted XML) and ``tiles.csv`` (image, mask, slide,
    x, y; level-0 coordinates). Keyword arguments are as for :class:`SegmentationDataset`.
    """
    root = Path(out_dir) / split
    images, masks = root / "images", root / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    rows = []
    for slide, ground_truth in find_pairs(dirs, root / "geojson"):
        with SegmentationTileGenerator(slide, ground_truth, **generator_kwargs) as gen:
            for (x, y), tile, mask in gen:
                name = f"{slide.stem}_{x}_{y}"
                Image.fromarray(tile).save(images / f"{name}.{fmt}")
                Image.fromarray(mask).save(masks / f"{name}.png")
                rows.append((f"images/{name}.{fmt}", f"masks/{name}.png", slide.name, x, y))
        logger.info("{}: wrote {} tiles to {}", slide.name, len(gen), root)
    with open(root / "tiles.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("image", "mask", "slide", "x", "y"))
        writer.writerows(rows)
    return root


class TileFolderDataset:
    """``(image, mask)`` pairs written by :func:`export_tiles`, read from ``root/tiles.csv``."""

    def __init__(self, root: str | Path, transform: Transform | None = None) -> None:
        self.root = Path(root)
        self.transform = transform
        with open(self.root / "tiles.csv", newline="") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> object:
        row = self.rows[index]
        tile = np.asarray(Image.open(self.root / row["image"]).convert("RGB"))
        mask = np.asarray(Image.open(self.root / row["mask"]))
        return _sample(tile, mask, self.transform)
