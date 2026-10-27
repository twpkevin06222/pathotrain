"""Trainer callbacks. Each runs on every rank; validation metrics are already
synced across ranks, so every rank takes the same decision."""

import math
from pathlib import Path

import optuna


def _improved(value: float, best: float | None, mode: str) -> bool:
    if math.isnan(value):
        return False
    return best is None or (value > best if mode == "max" else value < best)


class ModelCheckpoint:
    """Keep ``best.ckpt`` (weights only) at the best ``monitor`` value and
    ``last.ckpt`` (full state, for resuming) after every epoch.

    ``mode`` is ``"max"`` or ``"min"``. The best value and epoch are saved in
    ``last.ckpt``, so a resumed run keeps comparing against the true best. At
    the end of training the best value, epoch and path are logged to every
    logger; with ``log_artifact=True`` ``best.ckpt`` is also uploaded to loggers
    that store files (:class:`MLflowLogger`).
    """

    def __init__(
        self,
        dirpath: str | Path,
        monitor: str = "val/miou",
        mode: str = "max",
        log_artifact: bool = False,
    ) -> None:
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.log_artifact = log_artifact
        self.best: float | None = None
        self.best_epoch: int | None = None

    @property
    def best_path(self) -> Path:
        return self.dirpath / "best.ckpt"

    def on_validation_end(self, trainer, metrics: dict[str, float]) -> None:
        if _improved(metrics[self.monitor], self.best, self.mode):
            self.best, self.best_epoch = metrics[self.monitor], trainer.epoch
            trainer.save(self.best_path, weights_only=True, monitor=self.monitor, value=self.best)

    def on_epoch_end(self, trainer) -> None:
        # After every on_validation_end, so other callbacks' state is current too.
        trainer.save(self.dirpath / "last.ckpt")

    def on_fit_end(self, trainer) -> None:
        if self.best is None:
            return
        for lg in trainer.fabric.loggers:
            lg.log_metrics(
                {f"best/{self.monitor}": self.best, "best/epoch": self.best_epoch},
                step=trainer.step,
            )
            lg.log_hyperparams({"best_checkpoint": str(self.best_path)})
            if self.log_artifact and hasattr(lg, "log_artifact"):
                lg.log_artifact(self.best_path)

    def state_dict(self) -> dict:
        return {"best": self.best, "best_epoch": self.best_epoch}

    def load_state_dict(self, state: dict) -> None:
        self.best, self.best_epoch = state["best"], state["best_epoch"]


class EarlyStopping:
    """Stop when ``monitor`` has not improved for ``patience`` validations."""

    def __init__(self, monitor: str = "val/miou", mode: str = "max", patience: int = 10) -> None:
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.best: float | None = None
        self.bad = 0

    def on_validation_end(self, trainer, metrics: dict[str, float]) -> None:
        if _improved(metrics[self.monitor], self.best, self.mode):
            self.best, self.bad = metrics[self.monitor], 0
            return
        self.bad += 1
        trainer.should_stop = self.bad >= self.patience

    def state_dict(self) -> dict:
        return {"best": self.best, "bad": self.bad}

    def load_state_dict(self, state: dict) -> None:
        self.best, self.bad = state["best"], state["bad"]


class OptunaPruning:
    """Report ``monitor`` to an Optuna trial after each validation and abort the
    run with ``optuna.TrialPruned`` when the study's pruner says so."""

    def __init__(self, trial: optuna.Trial, monitor: str = "val/miou") -> None:
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, metrics: dict[str, float]) -> None:
        # shortcut: single-process trials only; with DDP every rank would report to the
        # trial, upgrade by reporting on rank 0 and broadcasting the decision.
        self.trial.report(metrics[self.monitor], trainer.epoch)
        if self.trial.should_prune():
            for lg in trainer.fabric.loggers:
                lg.finalize("pruned")
            raise optuna.TrialPruned(f"pruned at epoch {trainer.epoch}")
