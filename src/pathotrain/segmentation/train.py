"""Train a segmentation model from one YAML file::

    python -m pathotrain.segmentation.train config.yaml
    python -m pathotrain.segmentation.train config.yaml --resume runs/exp/checkpoints/last.ckpt

Start from ``config.example.yaml`` next to this module, which lists every
section with its defaults. The config is copied to ``<output_dir>/config.yaml``, uploaded to the MLflow run
and logged as flat params; checkpoints store it too (see :func:`load_model`).
Sections (``?`` marks optional ones)::

    name: exp01                 # MLflow experiment
    seed: 0
    output_dir: runs/exp01      # config.yaml, checkpoints/, logs/
    mean: IMAGENET_MEAN         # or MIDNIGHT_MEAN, or 3 numbers in [0, 1]
    std: IMAGENET_STD
    model: {...}                # build_model(**model)
    loss?: {name: FocalDiceLoss, ...}  # a loss class from pathotrain.segmentation
    task?: {...}                # SegmentationTask (lr, warmup_epochs, class_names, ...)
    trainer?: {...}             # TrainerConfig
    checkpoint?: {...}          # ModelCheckpoint (monitor, mode, log_artifact)
    early_stopping?: {...}      # EarlyStopping; leave out to train every epoch
    augmentation?: {...}        # train_transform (n, m, crop, blur, noise, jpeg)
    loader?: {...}              # both DataLoaders (batch_size, num_workers, ...);
                                # pin_memory defaults to true when CUDA is available,
                                # persistent_workers to true for the training loader
    data:
      train: {slides: [dir, ...], annotations: [dir, ...], tiling: {...}}
      val?: {slides: [...], annotations: [...], tiling: {...}}
      ...                       # the rest goes to SegmentationTileGenerator for both
                                # splits (class_map, roi_class, backend, ...)
"""

import argparse
from pathlib import Path

import torch
from lightning_fabric import seed_everything
from torch import nn
from torch.utils.data import DataLoader

from patchology import TilingConfig
from pathotrain import segmentation
from pathotrain.dataloader import SegmentationDataset
from pathotrain.dataloader.augmentation import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    eval_transform,
    train_transform,
)
from pathotrain.engine import (
    CSVLogger,
    EarlyStopping,
    MLflowLogger,
    ModelCheckpoint,
    Trainer,
    TrainerConfig,
    dump_config,
    flatten,
    load_config,
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


def build_loss(cfg: dict) -> nn.Module | None:
    """The ``loss`` section as a loss module, or None for the task's default."""
    if "loss" not in cfg:
        return None
    kwargs = dict(cfg["loss"])
    name = kwargs.pop("name")
    if not name.endswith("Loss") or not hasattr(segmentation, name):
        raise ValueError(f"unknown loss {name!r}, expected e.g. CEDiceLoss, FocalDiceLoss")
    return getattr(segmentation, name)(**kwargs)


def build_dataset(cfg: dict, split: str, transform) -> SegmentationDataset:
    data = {k: v for k, v in cfg["data"].items() if k not in ("train", "val")}
    s = cfg["data"][split]
    return SegmentationDataset(
        s["slides"],
        s["annotations"],
        transform=transform,
        tiling=TilingConfig(**s.get("tiling", {})),
        **data,
    )


def train(cfg: dict, resume: str | Path | None = None) -> dict[str, float]:
    """Run the experiment ``cfg`` describes and return the last validation metrics."""
    mean, std = normalisation(cfg, "mean"), normalisation(cfg, "std")
    out = Path(cfg["output_dir"])
    # A resumed run may change the config (e.g. more epochs); record the new one.
    config_path = dump_config(cfg, out, overwrite=resume is not None)
    seed_everything(cfg["seed"], workers=True)

    model = segmentation.build_model(**cfg["model"])
    task = segmentation.SegmentationTask(
        model, cfg["model"]["num_classes"], loss=build_loss(cfg), **cfg.get("task", {})
    )
    callbacks = [ModelCheckpoint(out / "checkpoints", config=cfg, **cfg.get("checkpoint", {}))]
    if "early_stopping" in cfg:
        callbacks.append(EarlyStopping(**cfg["early_stopping"]))
    mlflow_logger = MLflowLogger(experiment=cfg["name"])
    mlflow_logger.log_artifact(config_path)  # starts the run that fit() logs to
    trainer = Trainer(
        TrainerConfig(**cfg.get("trainer", {})),
        callbacks=callbacks,
        loggers=[mlflow_logger, CSVLogger(out, name="logs")],
    )

    augmentation = cfg.get("augmentation", {})
    # Pinned batches let the non-blocking copy to the GPU overlap compute (~2x faster copy).
    loader = {"pin_memory": torch.cuda.is_available(), **cfg.get("loader", {})}
    train_set = build_dataset(cfg, "train", train_transform(mean=mean, std=std, **augmentation))
    val_loader = None
    if "val" in cfg["data"]:
        val_set = build_dataset(
            cfg, "val", eval_transform(augmentation.get("crop"), mean=mean, std=std)
        )
        val_loader = DataLoader(val_set, **loader)
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


def load_model(path: str | Path, map_location: str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the model of a ``best.ckpt`` / ``last.ckpt`` written by :func:`train`
    and return it with its config (for ``mean``/``std``, ``class_map``, ...)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if "config" not in ckpt:
        raise ValueError(f"{path} has no config; it was not written by pathotrain's train")
    cfg = ckpt["config"]
    kwargs = dict(cfg["model"])
    if kwargs["name"].startswith("smp:"):
        kwargs["encoder_weights"] = None  # the checkpoint holds every weight
    model = segmentation.build_model(**kwargs)
    model.load_state_dict(ckpt["model"])
    return model, cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="experiment YAML")
    parser.add_argument("--resume", help="last.ckpt to continue from")
    args = parser.parse_args(argv)
    train(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
