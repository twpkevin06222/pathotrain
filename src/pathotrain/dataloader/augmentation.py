"""Augmentation and preprocessing for segmentation and detection tiles, built on torchvision v2.

:func:`train_transform` and :func:`eval_transform` return a ``transform(tile, mask)``
for :class:`~pathotrain.dataloader.SegmentationDataset` and
:class:`~pathotrain.dataloader.TileFolderDataset`, or with ``boxes=True`` a
``transform(tile, target)`` for :class:`~pathotrain.dataloader.DetectionDataset`.
Both end in the same normalisation, so the splits never drift apart::

    train = SegmentationDataset(slides, annotations, transform=train_transform(), **kwargs)
    val = SegmentationDataset(val_slides, val_annotations, transform=eval_transform(), **kwargs)

Training augmentation is flips and 90 degree rotations, then H&E RandAugment
(:func:`he_randaugment`), ported from Faryna et al., "Tailoring automated data
augmentation to H&E-stained histopathology", MIDL 2021
(https://github.com/DIAGNijmegen/pathology-he-auto-augment, Apache-2.0). Their
operations and magnitude ranges are kept, so their tuning applies: they used
``n=3, m=5`` and found ``m > 15`` destroys structure.

Tiles and masks travel as ``tv_tensors.Image`` and ``tv_tensors.Mask``, so
geometric operations move the mask with the tile (nearest neighbour, so labels
stay intact) and colour operations leave it alone. Uncovered mask pixels are
``mask_fill`` (255), so train with ``CrossEntropyLoss(ignore_index=255)``.
Detection boxes travel as ``tv_tensors.BoundingBoxes`` and move the same way;
rotation and shear grow a box to enclose its rotated corners.
Randomness comes from torch's generator, which a ``DataLoader`` seeds per worker.
"""

import io
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

# preprocessing normalization is based on imagenet and subject to change
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Scanner artifacts for the optional train_transform extras (blur, noise, jpeg).
BLUR_SIGMA = (0.1, 1.5)  # pixels; slight defocus
NOISE_SIGMA = 0.03  # in [0, 1] units, about 8 grey levels
JPEG_QUALITY = (50, 95)  # WSI tiles are often JPEG compressed

# Ruifrok-Johnston H&E-DAB stain vectors, as in the reference implementation.
RGB_FROM_HED = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
HED_FROM_RGB = torch.linalg.inv(RGB_FROM_HED)


def _uniform(low: float, high: float) -> float:
    return torch.empty(()).uniform_(low, high).item()


def _to_uint8(img: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """``[0, 1]`` float to uint8, keeping ``like``'s tv_tensor type."""
    return tv_tensors.wrap((img.clamp(0, 1) * 255).to(torch.uint8), like=like)


def rgb_to_hed(rgb: torch.Tensor) -> torch.Tensor:
    """``(3, H, W)`` uint8 RGB to float32 HED stain concentrations."""
    # The reference offsets RGB by 2 before the log (instead of clipping 0);
    # its magnitude ranges are tuned to this, so it is kept.
    od = -torch.log(rgb.as_subclass(torch.Tensor).float() / 255 + 2)
    return torch.einsum("chw,cd->dhw", od, HED_FROM_RGB)


def hed_to_rgb(hed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`rgb_to_hed`, as ``[0, 1]`` float RGB."""
    return torch.exp(-torch.einsum("dhw,dc->chw", hed, RGB_FROM_HED)) - 2


def hed_jitter(
    img: torch.Tensor,
    sigma: Sequence[float],
    bias: Sequence[float],
    cutoff: tuple[float, float] = (0.15, 0.85),
) -> torch.Tensor:
    """Each HED channel ``c`` becomes ``c * (1 + sigma) + bias``; tiles with mean outside ``cutoff``
    (in ``[0, 1]``, i.e. mostly glass or mostly dark) are returned unchanged."""
    if not cutoff[0] <= img.float().mean().item() / 255 <= cutoff[1]:
        return img
    hed = rgb_to_hed(img) * (1 + torch.tensor(sigma))[:, None, None]
    return _to_uint8(hed_to_rgb(hed + torch.tensor(bias)[:, None, None]), like=img)


def hsv_jitter(img: torch.Tensor, hue: float, saturation: float) -> torch.Tensor:
    """Rotate hue (in turns) by ``hue`` and scale saturation, value unchanged.

    Saturation ``s`` is scaled by ``1 + saturation`` when ``saturation < 0`` and by
    ``1 + (1 - s) * saturation`` otherwise, so it never exceeds 1.
    """
    x = img.as_subclass(torch.Tensor).float() / 255
    v = x.amax(dim=0, keepdim=True)
    s = torch.where(v > 0, (v - x.amin(dim=0, keepdim=True)) / v.clamp_min(1e-12), 0)
    ratio = 1 + saturation if saturation < 0 else 1 + (1 - s) * saturation
    # At fixed hue and value each channel is v - (v - c) * s'/s, so this rescales s alone.
    x = v - (v - x) * ratio
    # adjust_hue takes a factor in [-0.5, 0.5] turns.
    return F.adjust_hue(_to_uint8(x, like=img), (hue + 0.5) % 1.0 - 0.5)


class _ImageOnly(v2.Transform):
    """A random colour transform: images only, masks pass through untouched."""

    _transformed_types = (tv_tensors.Image,)


class HEDJitter(_ImageOnly):
    """:func:`hed_jitter` with ``sigma`` and ``bias`` drawn per channel from
    ``[-sigma_limit, sigma_limit]`` and ``[-bias_limit, bias_limit]``."""

    def __init__(
        self,
        sigma_limit: float = 0.05,
        bias_limit: float = 0.05,
        cutoff: tuple[float, float] = (0.15, 0.85),
    ) -> None:
        super().__init__()
        self.sigma_limit, self.bias_limit, self.cutoff = sigma_limit, bias_limit, cutoff

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {
            "sigma": [_uniform(-self.sigma_limit, self.sigma_limit) for _ in range(3)],
            "bias": [_uniform(-self.bias_limit, self.bias_limit) for _ in range(3)],
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return hed_jitter(inpt, params["sigma"], params["bias"], self.cutoff)


class HSVJitter(_ImageOnly):
    """:func:`hsv_jitter` with factors drawn from ``[-hue_limit, hue_limit]`` and
    ``[-saturation_limit, saturation_limit]``."""

    def __init__(self, hue_limit: float = 0.05, saturation_limit: float = 0.05) -> None:
        super().__init__()
        self.hue_limit, self.saturation_limit = hue_limit, saturation_limit

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {
            "hue": _uniform(-self.hue_limit, self.hue_limit),
            "saturation": _uniform(-self.saturation_limit, self.saturation_limit),
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return hsv_jitter(inpt, params["hue"], params["saturation"])


ENHANCE = {
    "Brightness": F.adjust_brightness,
    "Color": F.adjust_saturation,
    "Contrast": F.adjust_contrast,
    "Sharpness": F.adjust_sharpness,
}


class Enhance(_ImageOnly):
    """PIL ``ImageEnhance``-equivalent op with a factor drawn from ``factor_range``
    (1 is unchanged); torchvision's kernels follow PIL's definitions."""

    def __init__(
        self,
        op: Literal["Brightness", "Color", "Contrast", "Sharpness"],
        factor_range: tuple[float, float],
    ) -> None:
        super().__init__()
        self.op, self.factor_range = op, factor_range

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {"factor": _uniform(*self.factor_range)}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return ENHANCE[self.op](inpt, params["factor"])


class RandomJPEG(_ImageOnly):
    """JPEG round trip at a quality drawn from ``quality`` (inclusive), via PIL.

    Replaces ``v2.JPEG``, whose torchvision encoder is deprecated since 0.29.
    """

    def __init__(self, quality: tuple[int, int] = JPEG_QUALITY) -> None:
        super().__init__()
        self.quality = quality

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {"quality": int(torch.randint(self.quality[0], self.quality[1] + 1, ()))}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        buffer = io.BytesIO()
        Image.fromarray(inpt.permute(1, 2, 0).numpy()).save(
            buffer, "JPEG", quality=params["quality"]
        )
        decoded = torch.from_numpy(np.array(Image.open(buffer))).permute(2, 0, 1)
        return tv_tensors.wrap(decoded, like=inpt)


class RandomRot90(v2.Transform):
    """Rotate image, mask and boxes together by a random multiple of 90 degrees."""

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {"k": int(torch.randint(4, ()))}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, tv_tensors.BoundingBoxes):
            # Same turn as torch.rot90 below; expand swaps the canvas of a non-square tile.
            return F.rotate(inpt, 90 * params["k"], expand=True)
        return tv_tensors.wrap(torch.rot90(inpt, params["k"], dims=(-2, -1)), like=inpt)


class RandomTranslate(v2.Transform):
    """Shift image and mask along one axis by up to ``limit`` pixels."""

    def __init__(self, axis: Literal["x", "y"], limit: int, fill: int, mask_fill: int) -> None:
        super().__init__()
        self.axis, self.limit, self.fill, self.mask_fill = axis, limit, fill, mask_fill

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        shift = int(torch.randint(-self.limit, self.limit + 1, ()))
        return {"translate": [shift, 0] if self.axis == "x" else [0, shift]}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        fill = self.mask_fill if isinstance(inpt, tv_tensors.Mask) else self.fill
        return F.affine(
            inpt,
            angle=0.0,
            translate=params["translate"],
            scale=1.0,
            shear=[0.0, 0.0],
            fill=fill,
        )


def he_randaugment(n: int = 3, m: float = 5, fill: int = 128, mask_fill: int = 255) -> v2.Compose:
    """H&E RandAugment: ``n`` operations, each drawn at random (with replacement).

    Each operation's strength is up to magnitude ``m`` (about 1-15), mapped as in
    the reference: translation up to ``m`` px, rotation up to ``3m`` degrees, shear
    up to ``atan(0.03m)``, brightness / contrast / colour / sharpness factors in
    ``[0.1, 0.1 + 0.18m]``, and HED and HSV shifts up to ``0.03m``. Equalize and
    identity have no magnitude. Pixels uncovered by a geometric operation are
    ``fill`` grey in the tile and ``mask_fill`` in the mask.
    """
    # The reference draws HSV strength from the sampled level rather than m (a bug
    # in its op dispatch); m is used for both colour spaces, as its README describes.
    shift = 0.03 * m
    factors = (0.1, 0.1 + 0.18 * m)
    shear = float(np.degrees(np.arctan(0.03 * m)))
    fills = {tv_tensors.Image: fill, tv_tensors.Mask: mask_fill}
    bilinear = v2.InterpolationMode.BILINEAR  # masks are always resampled nearest
    ops = [
        RandomTranslate("x", int(m), fill, mask_fill),
        RandomTranslate("y", int(m), fill, mask_fill),
        v2.RandomAffine(0, shear=(-shear, shear, 0, 0), interpolation=bilinear, fill=fills),
        v2.RandomAffine(0, shear=(0, 0, -shear, shear), interpolation=bilinear, fill=fills),
        v2.RandomAffine(3 * m, interpolation=bilinear, fill=fills),
        Enhance("Brightness", factors),
        Enhance("Sharpness", factors),
        Enhance("Color", factors),
        Enhance("Contrast", factors),
        v2.RandomEqualize(p=1.0),
        v2.Identity(),
        HSVJitter(shift, shift),
        HEDJitter(shift, shift),
    ]
    return v2.Compose([v2.RandomChoice(ops) for _ in range(n)])


class SegmentationTransform:
    """``transform(tile, mask)`` running ``transforms``, then normalisation and tensor conversion.

    Takes the ``(H, W, 3)`` uint8 tile and ``(H, W)`` uint8 mask from the datasets and
    returns a ``(3, H, W)`` float32 image tensor and an ``(H, W)`` int64 mask tensor.
    """

    def __init__(
        self,
        transforms: Sequence[v2.Transform] = (),
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.compose = v2.Compose(
            [
                *transforms,
                v2.ToDtype(
                    {tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64},
                    scale=True,
                ),
                v2.Normalize(mean=list(mean), std=list(std)),
            ]
        )

    def __call__(self, tile: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        # torch.tensor copies: tiles read from disk are read-only arrays.
        image = tv_tensors.Image(torch.tensor(tile).permute(2, 0, 1))
        image, mask = self.compose(image, tv_tensors.Mask(torch.tensor(mask)))
        return image.as_subclass(torch.Tensor), mask.as_subclass(torch.Tensor)


class DetectionTransform:
    """``transform(tile, target)`` running ``transforms``, then normalisation and tensor conversion.

    Takes the ``(H, W, 3)`` uint8 tile and ``{"boxes": (N, 4) [x, y, w, h], "labels": (N,)}``
    target of :class:`patchology.DetectionTileGenerator` (COCO boxes in tile pixels).
    Returns a ``(3, H, W)`` float32 image tensor and ``{"boxes": (N, 4) float32
    [x1, y1, x2, y2], "labels": (N,) int64}``, as :class:`~pathotrain.detection.DetectionTask`
    takes. Boxes are clipped to the tile and those left under 1 px dropped.
    """

    def __init__(
        self,
        transforms: Sequence[v2.Transform] = (),
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.compose = v2.Compose(
            [
                *transforms,
                v2.ConvertBoundingBoxFormat("XYXY"),
                v2.ClampBoundingBoxes(),
                v2.SanitizeBoundingBoxes(),
                v2.ToDtype({tv_tensors.Image: torch.float32, "others": None}, scale=True),
                v2.Normalize(mean=list(mean), std=list(std)),
            ]
        )

    def __call__(
        self, tile: np.ndarray, target: dict[str, np.ndarray]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image = tv_tensors.Image(torch.tensor(tile).permute(2, 0, 1))
        boxes = tv_tensors.BoundingBoxes(
            torch.tensor(target["boxes"], dtype=torch.float32).reshape(-1, 4),
            format="XYWH",
            canvas_size=tile.shape[:2],
        )
        labels = torch.tensor(target["labels"], dtype=torch.int64).reshape(-1)
        image, out = self.compose(image, {"boxes": boxes, "labels": labels})
        return image.as_subclass(torch.Tensor), {
            "boxes": out["boxes"].as_subclass(torch.Tensor),
            "labels": out["labels"],
        }


def train_transform(
    n: int = 3,
    m: float = 5,
    crop: int | None = None,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    fill: int = 128,
    mask_fill: int = 255,
    blur: float = 0.0,
    noise: float = 0.0,
    jpeg: float = 0.0,
    boxes: bool = False,
) -> SegmentationTransform | DetectionTransform:
    """Random crop (if ``crop``), flips and 90 degree rotations, H&E RandAugment,
    optional scanner artifacts, normalisation.

    Args:
        n, m: H&E RandAugment operations per tile and magnitude, see
            :func:`he_randaugment`; ``n=0`` turns it off.
        crop: Random ``crop x crop`` window; tile larger than this for shift
            augmentation, and pass the same ``crop`` to :func:`eval_transform`.
        mean, std: Normalisation, in ``[0, 1]`` units (ImageNet by default).
        fill, mask_fill: Tile and mask value for pixels uncovered by geometric ops.
        blur, noise, jpeg: Probability of Gaussian blur (sigma in :data:`BLUR_SIGMA`),
            Gaussian noise (:data:`NOISE_SIGMA`) and JPEG compression (quality in
            :data:`JPEG_QUALITY`), applied in that order, as a scanner would; 0 (the
            default) turns each off. They change the tile only, never the mask.
        boxes: Detection targets instead of masks (a :class:`DetectionTransform`).
            A crop clips the boxes it cuts; ``min_box_visibility`` is not applied
            again, so tile at the training size rather than cropping where you can.
    """
    transforms = [v2.RandomCrop(crop)] if crop else []
    transforms += [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), RandomRot90()]
    if n:
        transforms.append(he_randaugment(n, m, fill, mask_fill))
    extras = [
        (blur, v2.GaussianBlur(kernel_size=9, sigma=BLUR_SIGMA)),  # 9 px covers 3 sigma
        (noise, v2.GaussianNoise(sigma=NOISE_SIGMA)),
        (jpeg, RandomJPEG(JPEG_QUALITY)),
    ]
    for p, t in extras:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"blur, noise and jpeg are probabilities in [0, 1], got {p}")
        if p:
            transforms.append(v2.RandomApply([t], p=p))
    return (DetectionTransform if boxes else SegmentationTransform)(transforms, mean, std)


def eval_transform(
    crop: int | None = None,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    boxes: bool = False,
) -> SegmentationTransform | DetectionTransform:
    """Deterministic preprocessing: centre crop (if ``crop``) and the same normalisation as
    training. ``boxes`` as for :func:`train_transform`."""
    cls = DetectionTransform if boxes else SegmentationTransform
    return cls([v2.CenterCrop(crop)] if crop else [], mean, std)
