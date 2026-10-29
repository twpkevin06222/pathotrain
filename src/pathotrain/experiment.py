"""Plumbing shared by the config-driven train and eval commands
(:mod:`pathotrain.segmentation.train`, :mod:`pathotrain.detection.train` and their
``eval`` modules)."""

from pathlib import Path

import torch
from lightning_fabric import seed_everything
from torch.utils.data import DataLoader, Dataset

from pathotrain.dataloader.augmentation import IMAGENET_MEAN, IMAGENET_STD
from pathotrain.engine import (
    CSVLogger,
    EarlyStopping,
    MLflowLogger,
    ModelCheckpoint,
    Task,
    Trainer,
    TrainerConfig,
    dump_config,
    flatten,
)
from pathotrain.pathofm import MIDNIGHT_MEAN, MIDNIGHT_STD

NORMALISATION = {
    "IMAGENET_MEAN": IMAGENET_MEAN,
    "IMAGENET_STD": IMAGENET_STD,
    "MIDNIGHT_MEAN": MIDNIGHT_MEAN,
    "MIDNIGHT_STD": MIDNIGHT_STD,
}


def normalisation(cfg: dict, key: str) -> tuple[float, float, float]:
    """``cfg[key]`` ("mean" or "std") as an RGB triple in [0, 1] units.

    A string must be ``IMAGENET_<KEY>`` or ``MIDNIGHT_<KEY>``; anything else
    must be a list or tuple of 3 numbers (custom statistics).
    """
    value = cfg.get(key)
    if isinstance(value, str):
        allowed = [f"IMAGENET_{key.upper()}", f"MIDNIGHT_{key.upper()}"]
        if value not in allowed:
            raise ValueError(f"{key} must be one of {allowed} or 3 numbers, got {value!r}")
        return NORMALISATION[value]
    if not (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    ):
        raise ValueError(f"{key} must be a preset name or 3 numbers, got {value!r}")
    value = tuple(float(v) for v in value)
    # Tiles are scaled to [0, 1] before Normalize, so 0-255 statistics are a mistake.
    if key == "mean" and not all(0 <= v <= 1 for v in value):
        raise ValueError(f"mean must be in [0, 1] units, got {value}")
    if key == "std" and not all(0 < v <= 1 for v in value):
        raise ValueError(f"std must be in (0, 1] units, got {value}")
    return value


def start(cfg: dict, resume: str | Path | None = None) -> Path:
    """Copy ``cfg`` to ``<output_dir>/config.yaml`` and seed everything; return the copy's path.

    A changed config for an existing run is refused unless resuming (see ``dump_config``).
    """
    # A resumed run may change the config (e.g. more epochs); record the new one.
    path = dump_config(cfg, Path(cfg["output_dir"]), overwrite=resume is not None)
    seed_everything(cfg["seed"], workers=True)
    return path


def fit(
    cfg: dict,
    task: Task,
    train_set: Dataset,
    val_set: Dataset | None,
    config_path: Path,
    resume: str | Path | None = None,
    monitor: str = "val/miou",
    trainer_defaults: dict | None = None,
    **loader_kwargs,
) -> dict[str, float]:
    """Train ``task`` as the ``trainer``, ``checkpoint``, ``early_stopping`` and
    ``loader`` sections of ``cfg`` say, logging to MLflow and CSV.

    ``monitor`` is the checkpoint and early-stopping default, ``trainer_defaults``
    the TrainerConfig defaults and ``loader_kwargs`` the DataLoader defaults
    (e.g. ``collate_fn``); the config overrides each.
    """
    out = Path(cfg["output_dir"])
    checkpoint = {"monitor": monitor, **cfg.get("checkpoint", {})}
    callbacks = [ModelCheckpoint(out / "checkpoints", config=cfg, **checkpoint)]
    if "early_stopping" in cfg:
        callbacks.append(EarlyStopping(**{"monitor": monitor, **cfg["early_stopping"]}))
    mlflow_logger = MLflowLogger(experiment=cfg["name"])
    mlflow_logger.log_artifact(config_path)  # starts the run that fit() logs to
    trainer = Trainer(
        TrainerConfig(**{**(trainer_defaults or {}), **cfg.get("trainer", {})}),
        callbacks=callbacks,
        loggers=[mlflow_logger, CSVLogger(out, name="logs")],
    )
    loader = _loader_kwargs(cfg, loader_kwargs)
    val_loader = None if val_set is None else DataLoader(val_set, **loader)
    return trainer.fit(
        task,
        # Training workers persist (no per-epoch restart); val ones don't, so the two sets
        # are not both held in RAM for the whole run.
        DataLoader(
            train_set,
            shuffle=True,
            **{"persistent_workers": loader.get("num_workers", 0) > 0, **loader},
        ),
        val_loader,
        hparams=flatten(cfg),
        ckpt_path=resume,
    )


def evaluate(
    cfg: dict, task: Task, dataset: Dataset, split: str, **loader_kwargs
) -> dict[str, float]:
    """Score ``task`` (holding trained weights) on ``dataset`` and return its metrics
    keyed ``<split>/...``. Runs on one device, with ``cfg``'s accelerator, precision
    and ``loader`` section; ``loader_kwargs`` are DataLoader defaults, as for :func:`fit`."""
    trainer = cfg.get("trainer", {})
    config = {k: trainer[k] for k in ("accelerator", "precision") if k in trainer}
    # One device: DDP pads the last batch with repeated tiles (see Trainer.validate).
    metrics = Trainer(TrainerConfig(**config, devices=1)).evaluate(
        task, DataLoader(dataset, **_loader_kwargs(cfg, loader_kwargs))
    )
    return {f"{split}/{k.removeprefix('val/')}": v for k, v in metrics.items()}


def _loader_kwargs(cfg: dict, defaults: dict) -> dict:
    """DataLoader keyword arguments: pathotrain's defaults, then ``defaults``, then
    the config's ``loader`` section."""
    loader = {
        # Pinned batches let the non-blocking copy to the GPU overlap compute (~2x faster).
        "pin_memory": torch.cuda.is_available(),
        # Fresh workers: forking the threaded training process can deadlock them.
        "multiprocessing_context": "forkserver",
        **defaults,
        **cfg.get("loader", {}),
    }
    if not loader.get("num_workers"):  # DataLoader refuses a context without workers
        loader.pop("multiprocessing_context", None)
    return loader
