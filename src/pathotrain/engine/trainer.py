"""Task-agnostic training loop on Lightning Fabric.

Fabric handles devices, the distributed strategy (``"ddp"``, ``"fsdp"``, ...),
mixed precision and checkpoint I/O; this module owns the loop. For several
GPUs, launch the script with ``fabric run`` or ``torchrun``, or set
``devices`` and let :meth:`Trainer.fit` spawn the processes.
"""

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from lightning_fabric import Fabric
from lightning_fabric.loggers import Logger
from loguru import logger
from torch.utils.data import DataLoader

from pathotrain.engine.task import Task


@dataclass
class TrainerConfig:
    """Loop and Fabric settings, logged as hyperparameters with every run."""

    max_epochs: int = 10
    accelerator: str = "auto"
    strategy: str = "auto"
    devices: int | str = "auto"
    num_nodes: int = 1
    precision: str = "bf16-mixed"
    accumulate: int = 1  # batches per optimizer step
    grad_clip: float | None = None  # max gradient norm
    # "epoch": one row per epoch with the mean train loss, the learning rate and the
    # validation metrics. "step": train loss and learning rate every `log_every`
    # optimizer steps, plus the validation metrics per epoch.
    log_interval: Literal["epoch", "step"] = "epoch"
    log_every: int = 50

    def __post_init__(self) -> None:
        if self.log_interval not in ("epoch", "step"):
            raise ValueError(f"log_interval must be 'epoch' or 'step', got {self.log_interval!r}")


class Trainer:
    """Train any :class:`Task`.

    Callbacks are objects with any of the hooks ``on_validation_end(trainer,
    metrics)``, ``on_epoch_end(trainer)`` (after all ``on_validation_end``) and
    ``on_fit_end(trainer)``; they run on every rank. Callbacks with
    ``state_dict`` / ``load_state_dict`` are saved in full checkpoints and
    restored on resume. Loggers are Fabric loggers (e.g. :class:`MLflowLogger`,
    :class:`CSVLogger`) and log from rank 0 only. Set ``trainer.should_stop =
    True`` from a callback to end training after the current epoch.
    """

    def __init__(
        self,
        config: TrainerConfig | None = None,
        callbacks: list | None = None,
        loggers: list[Logger] | None = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.callbacks = list(callbacks or [])
        c = self.config
        self.fabric = Fabric(
            accelerator=c.accelerator,
            strategy=c.strategy,
            devices=c.devices,
            num_nodes=c.num_nodes,
            precision=c.precision,
            callbacks=self.callbacks,
            loggers=loggers,
        )
        self.task: Task | None = None
        self.optimizer = None
        self.scheduler = None
        self.epoch = 0
        self.step = 0
        self.should_stop = False

    def fit(
        self,
        task: Task,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        hparams: dict | None = None,
        ckpt_path: str | Path | None = None,
    ) -> dict[str, float]:
        """Train for ``max_epochs`` (or until a callback stops it), validating
        after each epoch, and return the last validation metrics.

        Args:
            hparams: Extra hyperparameters to log next to the trainer config.
            ckpt_path: Resume model, optimizer, counters and callback state from a
                full checkpoint written by :meth:`save` (e.g. ``last.ckpt``). The
                scheduler is rebuilt for this run's length and replayed to the saved
                step, so ``max_epochs`` may change; it must depend on the step only.
        """
        fabric, c = self.fabric, self.config
        fabric.launch()
        # shortcut: moves the whole task to the device before sharding, upgrade for FSDP
        # models that do not fit on one GPU.
        task.to(fabric.device)
        train_loader = fabric.setup_dataloaders(train_loader)
        if val_loader is not None:
            val_loader = fabric.setup_dataloaders(val_loader)
        steps_per_epoch = -(-len(train_loader) // c.accumulate)
        optimizer, self.scheduler = task.configure_optimizers(
            steps_per_epoch * c.max_epochs, steps_per_epoch
        )
        task.model, self.optimizer = fabric.setup(task.model, optimizer)
        self.task = task
        if ckpt_path is not None:
            self.load(ckpt_path)
        for lg in fabric.loggers:
            lg.log_hyperparams({**asdict(c), **(hparams or {})})

        metrics: dict[str, float] = {}
        status = "failed"
        try:
            while self.epoch < c.max_epochs and not self.should_stop:
                train = self._train_epoch(train_loader)
                self.epoch += 1
                metrics = self.validate(val_loader) if val_loader is not None else {}
                logs = {**train, **metrics} if c.log_interval == "epoch" else metrics
                if logs:
                    fabric.log_dict({"epoch": self.epoch, **logs}, step=self.step)
                logger.info(f"epoch {self.epoch}/{c.max_epochs} {_fmt({**train, **metrics})}")
                if metrics:
                    fabric.call("on_validation_end", trainer=self, metrics=metrics)
                fabric.call("on_epoch_end", trainer=self)
            fabric.call("on_fit_end", trainer=self)
            status = "success"
        finally:
            for lg in fabric.loggers:
                lg.finalize(status)
        return metrics

    def _train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Train one epoch; return the train logs averaged over its steps and ranks."""
        fabric, c, task = self.fabric, self.config, self.task
        task.train()
        totals: dict[str, torch.Tensor] = {}
        steps = 0
        for i, batch in enumerate(loader):
            last = (i + 1) % c.accumulate == 0 or i + 1 == len(loader)
            # Skip the gradient all-reduce on accumulation steps.
            with fabric.no_backward_sync(task.model, enabled=not last):
                loss, logs = task.training_step(batch)
                fabric.backward(loss / c.accumulate)
            if not last:
                continue
            if c.grad_clip is not None:
                fabric.clip_gradients(task.model, self.optimizer, max_norm=c.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()
            self.step += 1
            steps += 1
            for k, v in logs.items():
                totals[k] = totals.get(k, 0) + v.detach()
            if c.log_interval == "step" and self.step % c.log_every == 0:
                logs = {f"train/{k}": v.item() for k, v in logs.items()}
                logs["train/lr"] = self.optimizer.param_groups[0]["lr"]
                fabric.log_dict({"epoch": self.epoch, **logs}, step=self.step)
                logger.debug(f"step {self.step} {_fmt(logs)}")
        means = fabric.all_reduce({k: v / steps for k, v in totals.items()}, reduce_op="mean")
        out = {f"train/{k}": v.item() for k, v in means.items()}
        return {**out, "train/lr": self.optimizer.param_groups[0]["lr"]}

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Run the task's validation over ``loader`` and return its metrics."""
        # shortcut: DistributedSampler pads the last batch by repeating samples, so DDP
        # metrics count a few tiles twice; fine for model selection, not for reporting.
        self.task.eval()
        for batch in loader:
            self.task.validation_step(batch)
        return self.task.compute_metrics()

    def evaluate(self, task: Task, loader: DataLoader) -> dict[str, float]:
        """Score ``task`` over ``loader`` with the trainer's device and precision,
        without training; use one device for metrics you report (see :meth:`validate`)."""
        self.fabric.launch()
        task.to(self.fabric.device)
        task.model = self.fabric.setup(task.model)
        self.task = task
        return self.validate(self.fabric.setup_dataloaders(loader))

    def save(self, path: str | Path, weights_only: bool = False, **extra) -> None:
        """Checkpoint ``model`` (a plain state dict), ``epoch``, ``step`` and
        ``extra``; unless ``weights_only``, also the optimizer, scheduler and
        callback state needed to resume. Call on every rank."""
        state = {"model": self.task.model, "epoch": self.epoch, "step": self.step, **extra}
        if not weights_only:
            state["optimizer"] = self.optimizer
            state["callbacks"] = {
                key: cb.state_dict() for key, cb in self._stateful_callbacks().items()
            }
        self.fabric.save(path, state)

    def load(self, path: str | Path) -> None:
        # Fabric restores the model and optimizer in place and returns the other entries.
        rest = self.fabric.load(path, {"model": self.task.model, "optimizer": self.optimizer})
        if self.scheduler is not None:
            # A saved scheduler state would carry the old run length (cosine T_max), so a
            # longer resumed run would raise the lr again; replay the new schedule instead.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "Detected call of `lr_scheduler.step", UserWarning
                )
                for _ in range(rest["step"]):
                    self.scheduler.step()
        for key, cb in self._stateful_callbacks().items():
            if key in rest["callbacks"]:
                cb.load_state_dict(rest["callbacks"][key])
        self.epoch, self.step = rest["epoch"], rest["step"]

    def _stateful_callbacks(self) -> dict:
        # Keyed by position and class so two callbacks of one class stay apart.
        return {
            f"{i}:{type(cb).__name__}": cb
            for i, cb in enumerate(self.callbacks)
            if hasattr(cb, "state_dict")
        }


def _fmt(metrics: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
