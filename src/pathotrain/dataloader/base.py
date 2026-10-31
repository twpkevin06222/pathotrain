"""Shared by the slide datasets: pairing slides with their GeoJSON annotations,
and a dataset over one tile generator per slide."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Self

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

from patchology import TileGenerator

GEOJSON_SUFFIXES = (".geojson", ".json")


def by_stem(dirs: Sequence[str | Path], suffixes: tuple[str, ...] | None) -> dict[str, Path]:
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
    slides = by_stem(slide_dirs, None)
    annotations = by_stem(annotation_dirs, GEOJSON_SUFFIXES)
    for stem in sorted(slides.keys() - annotations.keys()):
        logger.warning("slide {} has no annotation, skipped", slides[stem])
    for stem in sorted(annotations.keys() - slides.keys()):
        logger.warning("annotation {} has no slide, skipped", annotations[stem])
    pairs = [(slides[s], annotations[s]) for s in sorted(slides.keys() & annotations.keys())]
    if not pairs:
        raise ValueError(f"no slide in {list(slide_dirs)} matches an annotation")
    return pairs


def to_image_tensor(tile: np.ndarray) -> torch.Tensor:
    """``(H, W, 3)`` uint8 tile to a ``(3, H, W)`` float32 tensor in ``[0, 1]``."""
    return torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1))).float() / 255


class SlideDataset(Dataset):
    """Lazy tiles of every slide in ``slide_dirs`` with a GeoJSON annotation, through one
    ``generator_cls`` per slide; subclasses turn a generator's tile into a sample."""

    generator_cls: type[TileGenerator]

    # shortcut: every slide stays open for the dataset's lifetime; for thousands of
    # slides, open them per item instead (fd limits).
    def __init__(
        self,
        slide_dirs: Sequence[str | Path],
        annotation_dirs: Sequence[str | Path],
        transform: Callable | None = None,
        **generator_kwargs,
    ) -> None:
        self.transform = transform
        self._open(find_pairs(slide_dirs, annotation_dirs), generator_kwargs)

    def _open(self, sources: Sequence[tuple], generator_kwargs: dict) -> None:
        """One ``generator_cls(*source, **generator_kwargs)`` per source, e.g.
        ``(slide, annotation)``."""
        self.generators = []
        try:
            for source in sources:
                gen = self.generator_cls(*source, **generator_kwargs)
                # Free thumbnail and masks (~40 MB a slide) as soon as tiling is done.
                gen.release_overviews()
                self.generators.append(gen)
        except Exception:
            self.close()
            raise
        # offsets[k] is the dataset index of generator k's first tile.
        self.offsets = np.cumsum([0] + [len(g) for g in self.generators])
        logger.info("{} tiles from {} slides", len(self), len(self.generators))

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def locate(self, index: int) -> tuple[TileGenerator, int]:
        """The generator holding dataset ``index`` and the tile's index within it."""
        if not -len(self) <= index < len(self):
            raise IndexError(f"index {index} out of range for {len(self)} tiles")
        index %= len(self)
        k = int(np.searchsorted(self.offsets, index, side="right")) - 1
        return self.generators[k], index - int(self.offsets[k])

    def close(self) -> None:
        for gen in self.generators:
            gen.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
