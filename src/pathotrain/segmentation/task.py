import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassJaccardIndex

from pathotrain.engine.schedules import warmup_cosine
from pathotrain.engine.task import Task
from pathotrain.segmentation.losses import CEDiceLoss


def gaussian_window(height: int, width: int, device: torch.device | None = None) -> Tensor:
    """``(height, width)`` Gaussian with sigma side / 8 and peak 1, zeros raised to the
    smallest positive value; pathoinfer's ``stitch.segmentation.gaussian_window`` (as in
    nnU-Net), so validation blends windows as inference does."""
    lines = []
    for size in (height, width):
        offsets = torch.arange(size, device=device) - (size - 1) / 2
        lines.append(torch.exp(-(offsets**2) / (2 * (size / 8) ** 2)))
    window = torch.outer(*lines)
    window /= window.max()
    return window.clamp_min(window[window > 0].min())


def _starts(length: int, size: int, overlap: float) -> list[int]:
    """Window starts covering ``length``; the last window is shifted back to the edge."""
    if length <= size:
        return [0]
    stride = max(1, round(size * (1 - overlap)))
    count = math.ceil((length - size) / stride) + 1
    return sorted({min(i * stride, length - size) for i in range(count)})


def input_multiple(model: nn.Module) -> int:
    """The number ``model``'s input sides must be divisible by: an SMP encoder's
    output stride (32; the U-Net and DeepLabV3+ decoders cope without, FPN, U-Net++
    and others refuse), nnU-Net's total downsampling (64 with 7 stages), a model's own
    ``output_stride`` (the ``cellseg`` networks), else 1."""
    if isinstance(getattr(model, "output_stride", None), int):
        return model.output_stride
    encoder = getattr(model, "encoder", None)
    if hasattr(encoder, "output_stride"):
        return encoder.output_stride
    if hasattr(encoder, "strides"):  # nnU-Net: a stride per stage, per axis
        return math.prod(s if isinstance(s, int) else s[0] for s in encoder.strides)
    return 1


def sliding_window(model: nn.Module, image: Tensor, size: int, overlap: float = 0.25) -> Tensor:
    """Log-probabilities ``(B, C, H, W)`` of ``image`` from ``size x size`` windows that
    overlap by ``overlap``, their softmax probabilities blended with Gaussian weights.

    Log-probabilities stand in for logits: their softmax is the blended probabilities,
    so losses and argmax work on them unchanged.
    """
    _, _, height, width = image.shape
    h, w = min(size, height), min(size, width)
    weight = gaussian_window(h, w, image.device)
    probs = total = None
    for y in _starts(height, h, overlap):
        for x in _starts(width, w, overlap):
            p = model(image[:, :, y : y + h, x : x + w]).float().softmax(1)
            if probs is None:
                probs = image.new_zeros((image.shape[0], p.shape[1], height, width), dtype=p.dtype)
                total = image.new_zeros((height, width), dtype=p.dtype)
            probs[:, :, y : y + h, x : x + w] += p * weight
            total[y : y + h, x : x + w] += weight
    return (probs / total).clamp_min(torch.finfo(probs.dtype).tiny).log()


class SegmentationTask(Task):
    """Semantic segmentation on ``(image, mask)`` batches, as produced by
    ``SegmentationDataset`` with ``train_transform`` / ``eval_transform``.

    Logs the validation loss as ``val/loss``, per-class IoU as ``val/iou_<name>``,
    their mean as ``val/miou`` and the mean without background (class 0, usually the
    largest and easiest) as ``val/miou_fg``. A class absent from both prediction and
    target over the whole validation set has no IoU and is left out of the means.

    Args:
        model: Network returning ``(B, num_classes, H, W)`` logits.
        num_classes: Classes including background.
        loss: Defaults to :class:`CEDiceLoss` with ``ignore_index``.
        lr: AdamW learning rate, decayed to 0 with a cosine schedule.
        warmup_epochs: Linear warmup to ``lr`` before the cosine decay; may be
            fractional (``0.5`` is half an epoch). ``0`` disables it.
        weight_decay: AdamW weight decay.
        ignore_index: Mask value excluded from loss and metrics.
        class_names: Names for the IoU keys, default the class indices.
        window: Training crop side. Validation tiles larger than this are scored
            whole, from overlapping ``window`` windows blended as at inference
            (see :func:`sliding_window`); None runs the model on the tile as is.
            Either way a validation tile whose side the model cannot take (see
            :func:`input_multiple`) is zero-padded (the normalised mean colour) at
            the bottom and right, and the logits cropped back, so it is scored as is.
        window_overlap: Overlap of those windows, as a fraction of ``window``.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        loss: nn.Module | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        ignore_index: int = 255,
        class_names: list[str] | None = None,
        warmup_epochs: float = 0,
        window: int | None = None,
        window_overlap: float = 0.25,
    ) -> None:
        super().__init__()
        if window is not None and window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if not 0.0 <= window_overlap < 1.0:
            raise ValueError(f"window_overlap must be in [0, 1), got {window_overlap}")
        self.window = window
        self.window_overlap = window_overlap
        self.model = model
        self.loss = loss or CEDiceLoss(ignore_index)
        self.ignore_index = ignore_index
        self.lr = lr
        self.weight_decay = weight_decay
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        self.warmup_epochs = warmup_epochs
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        if len(self.class_names) != num_classes:
            raise ValueError(f"{len(self.class_names)} class names for {num_classes} classes")
        self.iou = MulticlassJaccardIndex(
            num_classes, average="none", ignore_index=ignore_index, zero_division=float("nan")
        )
        self.val_loss = MeanMetric()

    def training_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        image, mask = batch
        loss = self.loss(self.model(image).float(), mask)
        return loss, {"loss": loss.detach()}

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> None:
        image, mask = batch
        height, width = image.shape[-2:]
        multiple = input_multiple(self.model)
        image = F.pad(image, (0, -width % multiple, 0, -height % multiple))
        if self.window is not None and max(image.shape[-2:]) > self.window:
            logits = sliding_window(self.model, image, self.window, self.window_overlap)
        else:
            logits = self.model(image).float()
        logits = logits[..., :height, :width]
        # Weight by labelled pixels, so the epoch value is the per-pixel mean.
        self.val_loss.update(self.loss(logits, mask), (mask != self.ignore_index).sum())
        self.iou.update(logits.argmax(1), mask)

    def compute_metrics(self) -> dict[str, float]:
        iou = self.iou.compute()
        self.iou.reset()
        metrics = {f"val/iou_{n}": v.item() for n, v in zip(self.class_names, iou, strict=True)}
        metrics["val/miou"] = torch.nanmean(iou).item()
        metrics["val/miou_fg"] = torch.nanmean(iou[1:]).item()
        metrics["val/loss"] = self.val_loss.compute().item()
        self.val_loss.reset()
        return metrics

    def configure_optimizers(
        self, total_steps: int, steps_per_epoch: int
    ) -> tuple[Optimizer, LRScheduler]:
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer, warmup_cosine(optimizer, total_steps, steps_per_epoch, self.warmup_epochs)
