"""Unlabelled tiles for self-supervised learning (:mod:`pathotrain.histossl`).

Two sources, as for the labelled datasets:

* :class:`UnlabelledSlideDataset`: every slide in ``slide_dirs``, tiled once by
  :class:`patchology.TileGenerator` (tissue detection, artifact removal), tiles
  read on demand. No annotations.
* :class:`UnlabelledTileFolderDataset`: image files in ``tile_dirs``.

A sample is what ``transform`` makes of the ``(3, H, W)`` uint8
``tv_tensors.Image`` tile, or the tile as a float32 tensor in ``[0, 1]`` without
one. :class:`MultiCropTransform` makes the global and local views of DINO-style
methods, and :func:`multicrop_collate` batches them::

    transform = MultiCropTransform(global_crops=2, local_crops=8)
    train = UnlabelledSlideDataset(["data/slides"], transform, tiling=TilingConfig(tile_size=224, tile_mpp=0.5))
    loader = DataLoader(train, batch_size=32, collate_fn=multicrop_collate)
    for batch in loader:  # {"global": (2, 32, 3, 224, 224), "local": (8, 32, 3, 98, 98)}
        ...
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from patchology import TileGenerator
from pathotrain.dataloader.augmentation import (
    BLUR_SIGMA,
    JPEG_QUALITY,
    HEDJitter,
    HSVJitter,
    RandomJPEG,
    RandomRot90,
)
from pathotrain.dataloader.base import SlideDataset, by_stem
from pathotrain.dataloader.pathogen import IMAGE_SUFFIXES, read_image

# Midnight's normalisation, which pathotrain.histossl starts from.
SSL_MEAN = (0.5, 0.5, 0.5)
SSL_STD = (0.5, 0.5, 0.5)


def _sample(tile: tv_tensors.Image, transform: Callable | None) -> object:
    return tile.float() / 255 if transform is None else transform(tile)


class UnlabelledSlideDataset(SlideDataset):
    """Lazy tiles of every slide in ``slide_dirs`` (files keyed by stem across the
    directories; two slides with one stem are an error).

    Each slide is tiled once in ``__init__``; keyword arguments go to
    :class:`patchology.TileGenerator` (``tiling``, ``tissue``, ``artifacts``,
    ``color``, ``roi``, ...). Thumbnails and masks are released after tiling.
    """

    generator_cls = TileGenerator

    def __init__(
        self,
        slide_dirs: Sequence[str | Path],
        transform: Callable | None = None,
        **generator_kwargs,
    ) -> None:
        self.transform = transform
        slides = by_stem(slide_dirs, None)
        if not slides:
            raise ValueError(f"no slide in {list(slide_dirs)}")
        self._open([(path,) for path in slides.values()], generator_kwargs)

    def __getitem__(self, index: int) -> object:
        gen, i = self.locate(index)
        _, tile = gen.get_tile(i)
        # torch.tensor copies: tiles from the reader may be read-only.
        return _sample(tv_tensors.Image(torch.tensor(tile).permute(2, 0, 1)), self.transform)


class UnlabelledTileFolderDataset(Dataset):
    """Image tiles (png, jpg, tif) in ``tile_dirs``, keyed by stem across them."""

    def __init__(self, tile_dirs: Sequence[str | Path], transform: Callable | None = None) -> None:
        self.paths = list(by_stem(tile_dirs, IMAGE_SUFFIXES).values())
        if not self.paths:
            raise ValueError(f"no image in {list(tile_dirs)}")
        self.transform = transform
        logger.info(f"{len(self.paths)} tiles from {list(map(str, tile_dirs))}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> object:
        return _sample(read_image(self.paths[index]), self.transform)


class MultiCropTransform:
    """``global_crops`` large and ``local_crops`` small random views of one uint8
    tile, as DINO and DINOv2 train on (Caron et al. 2021; Oquab et al. 2023).

    Each view is a random resized crop (``*_scale`` of the tile area), random flips
    and 90 degree turns, then the H&E colour changes of
    :mod:`~pathotrain.dataloader.augmentation` in place of DINO's colour jitter:
    HED and HSV jitter up to ``hed`` and ``hsv`` (0 turns each off), Gaussian blur
    with probability ``blur`` and JPEG compression with probability ``jpeg``; then
    ``[0, 1]`` scaling and ``mean`` / ``std`` normalisation.

    Returns ``{"global": [G x (3, global_size, global_size)], "local": [L x (3,
    local_size, local_size)]}`` float32 tensors (``"local"`` only when
    ``local_crops``). With ``mask``, a callable returning a bool mask of one view's
    patch tokens (e.g. :func:`pathotrain.histossl.masking.block_mask`), also
    ``"masks": [G x mask]``, one per global view.
    """

    def __init__(
        self,
        global_crops: int = 2,
        local_crops: int = 8,
        global_size: int = 224,
        local_size: int = 98,
        global_scale: tuple[float, float] = (0.32, 1.0),
        local_scale: tuple[float, float] = (0.05, 0.32),
        hed: float = 0.05,
        hsv: float = 0.05,
        blur: float = 0.5,
        jpeg: float = 0.0,
        mean: Sequence[float] = SSL_MEAN,
        std: Sequence[float] = SSL_STD,
        mask: Callable[[], torch.Tensor] | None = None,
    ) -> None:
        if global_crops < 1 or local_crops < 0:
            raise ValueError(
                f"need >= 1 global and >= 0 local crops, got {global_crops}, {local_crops}"
            )
        for name, p in (("blur", blur), ("jpeg", jpeg)):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} is a probability in [0, 1], got {p}")
        self.global_crops, self.local_crops, self.mask = global_crops, local_crops, mask

        def view(size: int, scale: tuple[float, float]) -> v2.Compose:
            ops: list[v2.Transform] = [
                v2.RandomResizedCrop(size, scale=scale, antialias=True),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                RandomRot90(),
            ]
            if hed:
                ops.append(HEDJitter(hed, hed))
            if hsv:
                ops.append(HSVJitter(hsv, hsv))
            if blur:  # 9 px covers 3 sigma, as in train_transform
                ops.append(v2.RandomApply([v2.GaussianBlur(9, sigma=BLUR_SIGMA)], p=blur))
            if jpeg:
                ops.append(v2.RandomApply([RandomJPEG(JPEG_QUALITY)], p=jpeg))
            ops += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(list(mean), list(std))]
            return v2.Compose(ops)

        self.global_view = view(global_size, global_scale)
        self.local_view = view(local_size, local_scale)

    def __call__(self, tile: tv_tensors.Image) -> dict[str, list[torch.Tensor]]:
        views = {"global": [self._plain(self.global_view(tile)) for _ in range(self.global_crops)]}
        if self.local_crops:
            views["local"] = [self._plain(self.local_view(tile)) for _ in range(self.local_crops)]
        if self.mask is not None:
            views["masks"] = [self.mask() for _ in range(self.global_crops)]
        return views

    @staticmethod
    def _plain(image: torch.Tensor) -> torch.Tensor:
        return image.as_subclass(torch.Tensor)


def multicrop_collate(samples: list[dict[str, list[torch.Tensor]]]) -> dict[str, torch.Tensor]:
    """``collate_fn`` for :class:`MultiCropTransform` samples: each key's views
    stacked view-first, ``(views, batch, ...)``, e.g. ``"global"`` ``(G, B, 3, H, W)``."""
    return {
        key: torch.stack([torch.stack([s[key][v] for s in samples]) for v in range(len(views))])
        for key, views in samples[0].items()
    }
