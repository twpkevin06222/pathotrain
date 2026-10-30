from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class Task(nn.Module):
    """What a task (segmentation, detection, SSL, ...) gives the :class:`Trainer`.

    The trainer never looks inside a batch: it moves it to the device and hands
    it to the task, so batches can be tuples, dicts or lists of any shape.
    ``model`` is the network the trainer wraps for distributed training; the
    rest of the module (metrics, losses) is only moved to the device. Keep
    metrics as torchmetrics so they are synced across ranks.
    """

    model: nn.Module

    def training_step(self, batch) -> tuple[Tensor, dict[str, Tensor]]:
        """Return the loss to backpropagate and scalars to log (under ``train/``)."""
        raise NotImplementedError

    def validation_step(self, batch) -> None:
        """Update the validation metrics with one batch."""
        raise NotImplementedError

    def compute_metrics(self) -> dict[str, float]:
        """Return the metrics accumulated since the last call, then reset them."""
        raise NotImplementedError

    def on_optimizer_step(self) -> None:
        """Called after every optimizer step, e.g. to update an EMA of the weights."""

    def checkpoint_state(self) -> dict[str, nn.Module]:
        """Modules besides ``model`` to save in checkpoints and restore on resume
        (e.g. ``{"ema": self.ema}``); keys must not clash with the trainer's own."""
        return {}

    def configure_optimizers(
        self, total_steps: int, steps_per_epoch: int
    ) -> tuple[Optimizer, LRScheduler | None]:
        """Optimizer for ``self.model`` and an optional scheduler stepped once per
        optimizer step (``total_steps`` of them in the whole run, ``steps_per_epoch``
        per epoch)."""
        raise NotImplementedError
