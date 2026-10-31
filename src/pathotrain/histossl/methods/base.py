"""What every histossl method shares: the optimizer, validation and the probe."""

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MeanMetric

from pathotrain.engine import Task, warmup_cosine
from pathotrain.histossl.adaptation import param_groups


class SSLTask(Task):
    """A self-supervised task. ``model`` holds every trainable module (it is what
    Fabric wraps for DDP) and has a ``backbone`` attribute, the encoder being
    trained; subclasses give ``_losses(batch, generator)``, a dict with the total
    ``"loss"`` and any parts to log. :attr:`encoder` is what the probe scores and
    :func:`~pathotrain.histossl.train.load_backbone` returns (the student's
    backbone unless a method says otherwise, e.g. DINOv2's EMA teacher).

    A training batch is the ``{"global": (G, B, 3, H, W), ...}`` dict of
    :func:`pathotrain.dataloader.multicrop_collate`, or a ``(B, 3, H, W)`` tensor;
    validation batches are tensors. Validation reports ``val/loss`` on held-out
    tiles, with randomness (e.g. masks) fixed per batch so epochs compare. Every
    ``probe_every`` validations it also adds ``probe(encoder)``'s metrics (e.g.
    :func:`pathotrain.histossl.eval.evaluate_features`'s ``probe/*`` and ``knn/*``).

    The optimizer is AdamW over the trainable parameters (see
    :func:`pathotrain.histossl.adaptation.param_groups`) with linear warmup and
    cosine decay.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        layer_decay: float | None = None,
        warmup_epochs: float = 0,
        seed: int = 0,
        probe: Callable[[nn.Module], dict[str, float]] | None = None,
        probe_every: int = 1,
    ) -> None:
        super().__init__()
        if probe_every < 1:
            raise ValueError(f"probe_every must be >= 1, got {probe_every}")
        self.model = model
        # Kept apart from self.model, which Fabric replaces by its wrapper.
        self.backbone = model.backbone
        self.lr, self.weight_decay, self.layer_decay = lr, weight_decay, layer_decay
        self.warmup_epochs, self.seed = warmup_epochs, seed
        self.probe, self.probe_every = probe, probe_every
        self.val_loss = MeanMetric()
        self._val_batches = 0
        self._validations = 0

    @property
    def encoder(self) -> nn.Module:
        return self.backbone

    def _losses(self, batch, generator: torch.Generator | None) -> dict[str, Tensor]:
        raise NotImplementedError

    @staticmethod
    def _images(batch) -> Tensor:
        """The batch's global views as one ``(G * B, 3, H, W)`` tensor."""
        return batch["global"].flatten(0, 1) if isinstance(batch, dict) else batch

    def training_step(self, batch) -> tuple[Tensor, dict[str, Tensor]]:
        losses = self._losses(batch, None)
        return losses["loss"], {k: v.detach() for k, v in losses.items()}

    def validation_step(self, batch) -> None:
        generator = torch.Generator().manual_seed(self.seed + self._val_batches)
        loss = self._losses(batch, generator)["loss"]
        self.val_loss.update(loss, len(self._images(batch)))
        self._val_batches += 1

    def compute_metrics(self) -> dict[str, float]:
        metrics = {"val/loss": self.val_loss.compute().item()}
        self.val_loss.reset()
        self._val_batches = 0
        self._validations += 1
        if self.probe is not None and self._validations % self.probe_every == 0:
            metrics.update(self.probe(self.encoder))
        return metrics

    def configure_optimizers(
        self, total_steps: int, steps_per_epoch: int
    ) -> tuple[Optimizer, LRScheduler]:
        groups = param_groups(self.model, self.lr, self.weight_decay, self.layer_decay)
        optimizer = AdamW(groups, lr=self.lr)
        return optimizer, warmup_cosine(optimizer, total_steps, steps_per_epoch, self.warmup_epochs)
