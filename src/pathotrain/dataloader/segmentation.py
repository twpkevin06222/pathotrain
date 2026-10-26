"""Segmentation datasets built from directories of slides and GeoJSON annotations.

A slide is paired with the GeoJSON annotation of the same stem, searched across
all slide and annotation directories (see :func:`find_pairs`). Convert other
formats first, e.g. ASAP XML with ``python -m patchology.utilities.xml_to_geojson``.

Two ways to feed a PyTorch ``DataLoader``:

* Lazy: :class:`SegmentationDataset` tiles every slide in ``__init__`` and
  reads each ``(tile, mask)`` from the slide on demand.
* Offline: :func:`export_tiles` writes tiles and masks to
  ``out_dir/<split>/{images,masks}/`` and :class:`TileFolderDataset` reads them back.

Both are map-style ``torch.utils.data.Dataset`` subclasses for a ``DataLoader``
with any ``num_workers``. A sample is ``(image, mask)``: a ``(3, H, W)`` float32
tensor in ``[0, 1]`` and an ``(H, W)`` int64 tensor, or
whatever ``transform(tile, mask)`` returns, given the ``(H, W, 3)`` uint8 tile
and ``(H, W)`` uint8 mask.

Example (TIGER-style annotations)::

    kwargs = dict(
        class_map={"invasive tumor": 1, "tumor-associated stroma": 2},
        roi_class="roi",
        tiling=TilingConfig(tile_size=512, tile_mpp=0.5),
    )
    slides = ["data/train/slides", "data/train_02/slides"]
    annotations = ["data/train/geo_json_annotations", "data/train_02/geo_json_annotations"]
    train = SegmentationDataset(slides, annotations, **kwargs)
    loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=4)

    export_tiles(slides, annotations, "tiles", "train", **kwargs)
    train = TileFolderDataset("tiles/train")
"""

import csv
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Self

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

from patchology import SegmentationTileGenerator

GEOJSON_SUFFIXES = (".geojson", ".json")

Transform = Callable[[np.ndarray, np.ndarray], object]


def _by_stem(dirs: Sequence[str | Path], suffixes: tuple[str, ...] | None) -> dict[str, Path]:
    """Files in ``dirs`` keyed by stem (all files if ``suffixes`` is None)."""
    files: dict[str, Path] = {}
    for d in map(Path, dirs):
        if not d.is_dir():
            raise FileNotFoundError(f"{d} is not a directory")
        for p in sorted(d.iterdir()):
            if not p.is_file() or (suffixes is not None and p.suffix.lower() not in suffixes):
                continue
            if p.stem in files:
                raise ValueError(f"{p.stem} is in both {files[p.stem].parent} and {d}")
            files[p.stem] = p
    return files


def find_pairs(
    slide_dirs: Sequence[str | Path], annotation_dirs: Sequence[str | Path]
) -> list[tuple[Path, Path]]:
    """``(slide, GeoJSON)`` pairs matched by file stem across all of the given directories.

    Annotations must already be GeoJSON (convert ASAP XML with
    ``python -m patchology.utilities.xml_to_geojson``). Slides and annotations
    without a partner are skipped with a warning.

    Raises:
        FileNotFoundError: A directory does not exist.
        ValueError: Two slides or two annotations share a stem, or nothing matches.
    """
    slides = _by_stem(slide_dirs, None)
    annotations = _by_stem(annotation_dirs, GEOJSON_SUFFIXES)
    for stem in sorted(slides.keys() - annotations.keys()):
        logger.warning("slide {} has no annotation, skipped", slides[stem])
    for stem in sorted(annotations.keys() - slides.keys()):
        logger.warning("annotation {} has no slide, skipped", annotations[stem])
    pairs = [(slides[s], annotations[s]) for s in sorted(slides.keys() & annotations.keys())]
    if not pairs:
        raise ValueError(f"no slide in {list(slide_dirs)} matches an annotation")
    return pairs


def _sample(tile: np.ndarray, mask: np.ndarray, transform: Transform | None) -> object:
    if transform is not None:
        return transform(tile, mask)
    image = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1))).float() / 255
    return image, torch.from_numpy(mask.astype(np.int64))


class SegmentationDataset(Dataset):
    """Lazy ``(image, mask)`` tiles of every slide in ``slide_dirs`` with a GeoJSON annotation.

    Each slide is tiled once in ``__init__`` (tissue, ROI and ground-truth
    overlap selection); tiles and masks are read from the slide per item.
    Keyword arguments go to :class:`patchology.SegmentationTileGenerator`
    (``class_map``, ``roi_class``, ``object_mask_overlap``, ``tiling``, ...).
    """

    # shortcut: every slide stays open for the dataset's lifetime; for thousands of
    # slides, open them per item instead (fd limits).
    def __init__(
        self,
        slide_dirs: Sequence[str | Path],
        annotation_dirs: Sequence[str | Path],
        transform: Transform | None = None,
        **generator_kwargs,
    ) -> None:
        self.transform = transform
        self.generators = []
        try:
            for slide, ground_truth in find_pairs(slide_dirs, annotation_dirs):
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
    x, y; level-0 coordinates). Keyword arguments are as for :class:`SegmentationDataset`.
    """
    root = Path(out_dir) / split
    images, masks = root / "images", root / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    rows = []
    for slide, ground_truth in find_pairs(slide_dirs, annotation_dirs):
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


class TileFolderDataset(Dataset):
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
