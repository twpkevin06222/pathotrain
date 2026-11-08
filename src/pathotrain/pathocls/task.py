import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassRecall,
)

from pathotrain.engine.schedules import warmup_cosine
from pathotrain.engine.task import Task


class ClassificationTask(Task):
    """Tile classification on ``(input, label)`` batches; ``model`` maps the input to
    logits (a head on cached embeddings, or a whole classifier on images).

    Logs ``val/loss`` and the linear probe's metrics (histossl's, so numbers compare):
    ``val/acc``, ``val/balanced_acc`` (mean recall per class), ``val/macro_f1``,
    ``val/auroc`` (macro one-vs-rest), and ``val/recall_<name>`` per class.

    Args:
        model: Network returning ``(B, num_classes)`` logits.
        num_classes: Number of classes.
        class_names: Names for the per-class keys, default the class indices.
        lr: AdamW learning rate, decayed to 0 with a cosine schedule.
        weight_decay: AdamW weight decay.
        warmup_epochs: Linear warmup to ``lr``; may be fractional, ``0`` disables it.
        backbone_lr: Learning rate of ``model.backbone``'s trainable parameters when
            fine-tuning, default ``lr / 10``; a frozen backbone is not optimised.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        class_names: list[str] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: float = 0,
        backbone_lr: float | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        if len(self.class_names) != num_classes:
            raise ValueError(f"{len(self.class_names)} class names for {num_classes} classes")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        self.lr, self.weight_decay, self.warmup_epochs = lr, weight_decay, warmup_epochs
        self.backbone_lr = lr / 10 if backbone_lr is None else backbone_lr
        n = {"num_classes": num_classes}
        self.metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(average="micro", **n),
                "balanced_acc": MulticlassRecall(average="macro", **n),
                "macro_f1": MulticlassF1Score(average="macro", **n),
                "auroc": MulticlassAUROC(average="macro", **n),
                "recall": MulticlassRecall(average="none", **n),
            }
        )
        self.val_loss = MeanMetric()

    def training_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        x, y = batch
        loss = F.cross_entropy(self.model(x).float(), y)
        return loss, {"loss": loss.detach()}

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> None:
        x, y = batch
        logits = self.model(x).float()
        self.val_loss.update(F.cross_entropy(logits, y), len(y))
        self.metrics.update(logits.softmax(1), y)

    def compute_metrics(self) -> dict[str, float]:
        values = self.metrics.compute()
        self.metrics.reset()
        recall = values.pop("recall")
        metrics = {f"val/{k}": v.item() for k, v in values.items()}
        metrics |= {
            f"val/recall_{n}": r.item() for n, r in zip(self.class_names, recall, strict=True)
        }
        metrics["val/loss"] = self.val_loss.compute().item()
        self.val_loss.reset()
        return metrics

    def configure_optimizers(
        self, total_steps: int, steps_per_epoch: int
    ) -> tuple[Optimizer, LRScheduler]:
        backbone = getattr(self.model, "backbone", None)
        backbone_ids = {id(p) for p in backbone.parameters()} if backbone is not None else set()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in trainable if id(p) not in backbone_ids]},
            {"params": [p for p in trainable if id(p) in backbone_ids], "lr": self.backbone_lr},
        ]
        groups = [g for g in groups if g["params"]]
        optimizer = AdamW(groups, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer, warmup_cosine(optimizer, total_steps, steps_per_epoch, self.warmup_epochs)
