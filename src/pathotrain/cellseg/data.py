"""Tiles with nucleus instance and type masks, as HoVer-Net-style training needs them."""

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

from pathotrain.cellseg.instances import hv_maps, instances_from_contours
from pathotrain.dataloader.augmentation import SegmentationTransform, eval_transform
from pathotrain.dataloader.base import by_stem


class CellDataset(Dataset):
    """``(image, target)`` tiles from three folders paired by file stem: images, instance
    masks and type masks. Each prefix is removed before pairing (PanNuke's ``img_a.png``,
    ``inst_a.png`` and ``sem_a.png`` pair as ``a``).

    Instance masks hold one id per nucleus (0 background), or with ``contours`` the
    nucleus outlines, turned into ids by :func:`instances_from_contours`. Type masks hold
    the class per pixel (0 background, ``ignore_index`` not scored).

    ``target`` is ``{"inst": (H, W) int64, "type": (H, W) int64, "hv": (2, H, W)
    float32}``, the HV maps computed after augmentation. Pixels a geometric augmentation
    uncovers are ``ignore_index`` in ``type`` and background in ``inst``.

    Args:
        transform: A :func:`~pathotrain.dataloader.augmentation.train_transform` or
            :func:`~pathotrain.dataloader.augmentation.eval_transform`; default the
            latter (ImageNet normalisation).
    """

    def __init__(
        self,
        images: str | Path,
        instances: str | Path,
        types: str | Path,
        transform: SegmentationTransform | None = None,
        *,
        image_prefix: str = "",
        instance_prefix: str = "",
        type_prefix: str = "",
        contours: bool = False,
        ignore_index: int = 255,
    ) -> None:
        self.transform = transform or eval_transform()
        self.contours = contours
        self.ignore_index = ignore_index
        folders = [(images, image_prefix), (instances, instance_prefix), (types, type_prefix)]
        files = [
            {s.removeprefix(prefix): p for s, p in by_stem([d], None).items()}
            for d, prefix in folders
        ]
        stems = set.intersection(*(set(f) for f in files))
        for f in files:
            for stem in sorted(f.keys() - stems):
                logger.warning("{} has no partner in the other folders, skipped", f[stem])
        if not stems:
            raise ValueError(f"no file stems shared by {images}, {instances} and {types}")
        self.samples = [tuple(f[s] for f in files) for s in sorted(stems)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path, inst_path, type_path = self.samples[index]
        tile = np.asarray(Image.open(image_path).convert("RGB"))
        types = np.asarray(Image.open(type_path))
        inst = np.asarray(Image.open(inst_path)).astype(np.int32)
        if self.contours:
            inst = instances_from_contours(inst, types)
        image, inst, types = self.transform.compose(
            tv_tensors.Image(torch.tensor(tile).permute(2, 0, 1)),
            tv_tensors.Mask(torch.from_numpy(inst)),
            tv_tensors.Mask(torch.tensor(types)),
        )
        types = types.as_subclass(torch.Tensor).long()
        # Augmentation fills uncovered pixels with ignore_index in both masks.
        inst = inst.as_subclass(torch.Tensor).long().masked_fill(types == self.ignore_index, 0)
        hv = torch.from_numpy(hv_maps(inst.numpy()))
        return image.as_subclass(torch.Tensor), {"inst": inst, "type": types, "hv": hv}
