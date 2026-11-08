"""Segmentation datasets built from directories of slides and GeoJSON annotations.

A slide is paired with the GeoJSON annotation of the same stem, searched across
all slide and annotation directories (see :func:`find_pairs`). Convert other
formats first, e.g. ASAP XML with ``python -m patchology.utilities.xml_to_geojson``.

Two ways to feed a PyTorch ``DataLoader``:

* Lazy: :class:`SegmentationDataset` tiles every slide in ``__init__`` and
  reads each ``(tile, mask)`` from the slide on demand.
* Offline: :func:`export_segmentation_tiles` writes tiles and masks to
  ``out_dir/<split>/{images,masks}/`` and :class:`SegmentationTileFolderDataset` reads them back.

Both are map-style ``torch.utils.data.Dataset`` subclasses for a ``DataLoader``
with any ``num_workers``. A sample is ``(image, mask)``: a ``(3, H, W)`` float32
tensor in ``[0, 1]`` and an ``(H, W)`` int64 tensor, or
whatever ``transform(tile, mask)`` returns, given the ``(H, W, 3)`` uint8 tile
and ``(H, W)`` uint8 mask.

Example::

    kwargs = dict(
        class_map={"invasive tumor": 1, "tumor-associated stroma": 2},
        roi_class="roi",
        tiling=TilingConfig(tile_size=512, tile_mpp=0.5),
    )
    slides = ["data/train/slides", "data/train_02/slides"]
    annotations = ["data/train/geo_json_annotations", "data/train_02/geo_json_annotations"]
    train = SegmentationDataset(slides, annotations, **kwargs)
    loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=4)

    export_segmentation_tiles(slides, annotations, "tiles", "train", **kwargs)
    train = SegmentationTileFolderDataset("tiles/train")
"""

import csv
import functools
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

from patchology import SegmentationTileGenerator
from pathotrain.dataloader.base import (
    SlideDataset,
    by_stem,
    find_pairs,
    stack_rows,
    to_image_tensor,
)

Transform = Callable[[np.ndarray, np.ndarray], object]


def _class_columns(width: int) -> list[str]:
    """``tiles.csv`` column names of the per-label pixel shares."""
    return [f"class_{k}" for k in range(width)]


def _sample(tile: np.ndarray, mask: np.ndarray, transform: Transform | None) -> object:
    if transform is not None:
        return transform(tile, mask)
    return to_image_tensor(tile), torch.from_numpy(mask.astype(np.int64))


class SegmentationDataset(SlideDataset):
    """Lazy ``(image, mask)`` tiles of every slide in ``slide_dirs`` with a GeoJSON annotation.

    Each slide is tiled once in ``__init__`` (tissue, ROI and ground-truth
    overlap selection); tiles and masks are read from the slide per item.
    Keyword arguments go to :class:`patchology.SegmentationTileGenerator`
    (``class_map``, ``roi_class``, ``object_mask_overlap``, ``tiling``, ...).

    Each generator's thumbnail and tissue masks are released after tiling (see
    ``release_overviews``); ``generators[k].overlay(slide=generators[k].slide)``
    still draws its tiles and ground truth. For tissue masks, build a
    ``SegmentationTileGenerator`` for that slide yourself.
    """

    generator_cls = SegmentationTileGenerator

    @property
    def class_fraction(self) -> np.ndarray:
        """``(tiles, labels)`` share of each tile's pixels per label value (column 0 is
        background, ignored pixels in no column), from the generators' ``class_fraction``."""
        return stack_rows([g.class_fraction for g in self.generators])

    def __getitem__(self, index: int) -> object:
        gen, i = self.locate(index)
        _, tile, mask = gen.get_tile(i)
        return _sample(tile, mask, self.transform)


def export_segmentation_tiles(
    slide_dirs: Sequence[str | Path],
    annotation_dirs: Sequence[str | Path],
    out_dir: str | Path,
    split: str,
    fmt: str = "png",
    **generator_kwargs,
) -> Path:
    """Write every annotated slide's tiles and masks under ``out_dir/split`` and return it.

    Layout: ``images/<slide>_<x>_<y>.<fmt>``, ``masks/<slide>_<x>_<y>.png`` (always
    lossless) and ``tiles.csv`` (image, mask, slide,
    x, y; level-0 coordinates; then ``class_<k>``, the tile's share of pixels with
    label ``k``, see :attr:`SegmentationDataset.class_fraction`). Keyword arguments are
    as for :class:`SegmentationDataset`.

    Raises:
        FileExistsError: ``out_dir/split`` is not empty (a re-export would leave the
            old tiles beside the new index).
    """
    root = Path(out_dir) / split
    if root.is_dir() and any(root.iterdir()):
        # A second export would replace the index but leave the first one's tiles behind.
        raise FileExistsError(f"{root} is not empty; delete it or export to another split")
    images, masks = root / "images", root / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    rows, fractions = [], []
    for slide, ground_truth in find_pairs(slide_dirs, annotation_dirs):
        with SegmentationTileGenerator(slide, ground_truth, **generator_kwargs) as gen:
            fractions.append(gen.class_fraction)
            for (x, y), tile, mask in gen:
                name = f"{slide.stem}_{x}_{y}"
                Image.fromarray(tile).save(images / f"{name}.{fmt}")
                Image.fromarray(mask).save(masks / f"{name}.png")
                rows.append((f"images/{name}.{fmt}", f"masks/{name}.png", slide.name, x, y))
        logger.info("{}: wrote {} tiles to {}", slide.name, len(gen), root)
    fraction = stack_rows(fractions)
    with open(root / "tiles.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("image", "mask", "slide", "x", "y", *_class_columns(fraction.shape[1])))
        writer.writerows(
            (*row, *(round(float(v), 6) for v in frac))
            for row, frac in zip(rows, fraction, strict=True)
        )
    return root


class SegmentationTileFolderDataset(Dataset):
    """``(image, mask)`` tile pairs from disk.

    Either ``root`` written by :func:`export_segmentation_tiles` (read from ``root/tiles.csv``),
    or ``images`` and ``masks`` directories whose files are paired by stem
    (``a.png`` with ``a.png``, or ``a.jpg`` with ``a.png``); files without a
    partner are skipped with a warning.

    Example::

        train = SegmentationTileFolderDataset("tiles/train")
        tumor = SegmentationTileFolderDataset(images="tumor/images", masks="tumor/masks")
    """

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Transform | None = None,
        *,
        images: str | Path | None = None,
        masks: str | Path | None = None,
        ignore_index: int = 255,
    ) -> None:
        self.transform = transform
        self.ignore_index = ignore_index
        if (root is None) == (images is None) or (images is None) != (masks is None):
            raise ValueError("pass either root or both images and masks")
        if root is not None:
            root = Path(root)
            with open(root / "tiles.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            self.pairs = [(root / r["image"], root / r["mask"]) for r in rows]
            width = sum(1 for k in (rows[0] if rows else {}) if k.startswith("class_"))
            if width:  # written by export_segmentation_tiles; older exports lack it
                columns = _class_columns(width)
                self.class_fraction = np.array([[float(r[c]) for c in columns] for r in rows])
            return
        tiles, labels = by_stem([images], None), by_stem([masks], None)
        for stem in sorted(tiles.keys() - labels.keys()):
            logger.warning("image {} has no mask, skipped", tiles[stem])
        for stem in sorted(labels.keys() - tiles.keys()):
            logger.warning("mask {} has no image, skipped", labels[stem])
        self.pairs = [(tiles[s], labels[s]) for s in sorted(tiles.keys() & labels.keys())]
        if not self.pairs:
            raise ValueError(f"no image in {images} matches a mask in {masks}")

    def __len__(self) -> int:
        return len(self.pairs)

    @functools.cached_property
    def class_fraction(self) -> np.ndarray:
        """``(tiles, labels)`` share of each mask's pixels per label value (column 0 is
        background, ``ignore_index`` in no column): from ``tiles.csv`` when it has it,
        else read from every mask once, on first use."""
        counts = []
        for _, mask in self.pairs:
            values = np.asarray(Image.open(mask)).ravel()
            count = np.bincount(values, minlength=self.ignore_index + 1).astype(float)
            count[self.ignore_index] = 0
            counts.append(count / values.size)
        width = max((len(c) for c in counts), default=1)
        fraction = np.array([np.pad(c, (0, width - len(c))) for c in counts]).reshape(-1, width)
        labelled = np.flatnonzero(fraction.any(axis=0))
        return fraction[:, : (labelled[-1] + 1 if labelled.size else 1)]

    @functools.cached_property
    def image_area(self) -> np.ndarray:
        """``(tiles,)`` pixels per image, from the image headers, read once on first use."""
        return np.array([np.prod(Image.open(image).size) for image, _ in self.pairs])

    def __getitem__(self, index: int) -> object:
        image, mask = self.pairs[index]
        tile = np.asarray(Image.open(image).convert("RGB"))
        return _sample(tile, np.asarray(Image.open(mask)), self.transform)
