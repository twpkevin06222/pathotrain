"""Paired tiles for image-to-image translation (e.g. H&E to IHC, for
:mod:`pathotrain.pathogen`): a source and a target image of the same registered
tissue, with the same file stem."""

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import pil_to_tensor

from pathotrain.dataloader.augmentation import RandomRot90
from pathotrain.dataloader.base import by_stem

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def read_image(path: str | Path) -> tv_tensors.Image:
    """An RGB image file as a ``(3, H, W)`` uint8 tensor."""
    with Image.open(path) as image:
        return tv_tensors.Image(pil_to_tensor(image.convert("RGB")))


class PairedTileDataset(Dataset):
    """``(source, target)`` tiles paired by file stem across ``source_dirs`` and
    ``target_dirs``; tiles without a partner are skipped with a warning.

    A sample is two ``(3, H, W)`` float32 tensors in ``[0, 1]``. ``transform``
    takes and returns both uint8 ``tv_tensors.Image`` and must change them
    together (see :func:`paired_transform`).

    Raises:
        FileNotFoundError: A directory does not exist.
        ValueError: Two files in one side share a stem, or nothing pairs.
    """

    def __init__(
        self,
        source_dirs: Sequence[str | Path],
        target_dirs: Sequence[str | Path],
        transform: Callable | None = None,
    ) -> None:
        source = by_stem(source_dirs, IMAGE_SUFFIXES)
        target = by_stem(target_dirs, IMAGE_SUFFIXES)
        for name, only in (
            ("source", source.keys() - target.keys()),
            ("target", target.keys() - source.keys()),
        ):
            if only:
                logger.warning(
                    f"{len(only)} {name} tiles have no partner, skipped (e.g. {min(only)})"
                )
        self.pairs = [(source[s], target[s]) for s in sorted(source.keys() & target.keys())]
        if not self.pairs:
            raise ValueError(
                f"no tile in {list(source_dirs)} pairs with one in {list(target_dirs)}"
            )
        self.transform = transform
        logger.info(f"{len(self.pairs)} tile pairs from {list(map(str, source_dirs))}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = (read_image(p) for p in self.pairs[i])
        if source.shape != target.shape:
            raise ValueError(
                f"{self.pairs[i][0].name}: source {source.shape} != target {target.shape}"
            )
        if self.transform is not None:
            source, target = self.transform(source, target)
        return source.float() / 255, target.float() / 255


def paired_transform(train: bool, crop: int | None = None, resize: int | None = None) -> v2.Compose:
    """Geometric transforms applied identically to source and target: an optional
    resize (shorter side to ``resize`` px), then a random (``train``) or centre
    crop of ``crop`` px, then for training random flips and 90 degree turns.

    No colour changes: the target's colours are what the model learns.
    """
    ops: list[v2.Transform] = []
    if resize:
        ops.append(v2.Resize(resize, antialias=True))
    if crop:
        ops.append(v2.RandomCrop(crop) if train else v2.CenterCrop(crop))
    if train:
        ops += [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), RandomRot90()]
    return v2.Compose(ops)
