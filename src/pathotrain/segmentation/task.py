import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torchmetrics.classification import MulticlassJaccardIndex

from pathotrain.engine.task import Task
from pathotrain.segmentation.losses import CEDiceLoss


class SegmentationTask(Task):
    """Semantic segmentation on ``(image, mask)`` batches, as produced by
    ``SegmentationDataset`` with ``train_transform`` / ``eval_transform``.

    Logs per-class IoU as ``val/iou_<name>`` and their mean as ``val/miou``. A
    class absent from both prediction and target over the whole validation set
    has no IoU and is left out of the mean.

    Args:
        model: Network returning ``(B, num_classes, H, W)`` logits.
        num_classes: Classes including background.
        loss: Defaults to :class:`CEDiceLoss` with ``ignore_index``.
        lr: AdamW learning rate, decayed to 0 with a cosine schedule.
        weight_decay: AdamW weight decay.
        ignore_index: Mask value excluded from loss and metrics.
        class_names: Names for the IoU keys, default the class indices.
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
    ) -> None:
        super().__init__()
        self.model = model
        self.loss = loss or CEDiceLoss(ignore_index)
        self.lr = lr
        self.weight_decay = weight_decay
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        if len(self.class_names) != num_classes:
            raise ValueError(f"{len(self.class_names)} class names for {num_classes} classes")
        self.iou = MulticlassJaccardIndex(
            num_classes, average="none", ignore_index=ignore_index, zero_division=float("nan")
        )

    def training_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        image, mask = batch
        loss = self.loss(self.model(image).float(), mask)
        return loss, {"loss": loss.detach()}

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> None:
        image, mask = batch
        self.iou.update(self.model(image).argmax(1), mask)

    def compute_metrics(self) -> dict[str, float]:
        iou = self.iou.compute()
        self.iou.reset()
        metrics = {f"val/iou_{n}": v.item() for n, v in zip(self.class_names, iou, strict=True)}
        metrics["val/miou"] = torch.nanmean(iou).item()
        return metrics

    def configure_optimizers(self, total_steps: int) -> tuple[Optimizer, LRScheduler]:
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer, CosineAnnealingLR(optimizer, T_max=total_steps)
