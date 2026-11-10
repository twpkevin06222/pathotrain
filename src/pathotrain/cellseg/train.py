"""Train a nucleus instance segmentation model (HoVer-Net style) from one YAML file::

    python -m pathotrain.cellseg.train instance.yaml
    python -m pathotrain.cellseg.train instance.yaml --resume runs/exp/checkpoints/last.ckpt

Start from ``instance.example.yaml`` next to this module. The sections are those of
``pathotrain.segmentation.train`` (``trainer``, ``checkpoint``, ``early_stopping``,
``augmentation``, ``loader``, ...), except::

    model: {name: cellseg:hovernet, num_classes: 6, ...}  # build_model(..., instance=True)
    task?: {...}         # CellInstanceTask (lr, loss_weights, postprocess, class_names, ...)
    data:
      train: {images: dir, instances: dir, types: dir, ...}  # CellDataset
      val?: {...}
      test?: {...}       # scored only by pathotrain.cellseg.eval

The best checkpoint is chosen on ``val/mpq`` unless ``checkpoint.monitor`` says otherwise.
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from pathotrain import experiment
from pathotrain.cellseg.data import CellDataset
from pathotrain.cellseg.models import build_model
from pathotrain.cellseg.task import CellInstanceTask
from pathotrain.dataloader.augmentation import eval_transform, train_transform
from pathotrain.engine import load_config
from pathotrain.experiment import normalisation


def build_dataset(cfg: dict, split: str, train: bool = False) -> CellDataset:
    """``data[split]``, augmented as ``augmentation`` says when ``train``."""
    norm = {"mean": normalisation(cfg, "mean"), "std": normalisation(cfg, "std")}
    if train:
        transform = train_transform(**norm, **cfg.get("augmentation", {}))
    else:
        transform = eval_transform(**norm)
    return CellDataset(transform=transform, **cfg["data"][split])


def build_task(cfg: dict, pretrained: bool = True) -> CellInstanceTask:
    """The model ``cfg["model"]`` names, with its instance branches, in its task."""
    kwargs = dict(cfg["model"])
    arch = kwargs.pop("name").removeprefix("cellseg:")
    if not pretrained:
        kwargs["encoder_weights"] = None  # a checkpoint holds every weight
    model = build_model(arch, instance=True, **kwargs)
    return CellInstanceTask(model, cfg["model"]["num_classes"], **cfg.get("task", {}))


def train(cfg: dict, resume: str | Path | None = None) -> dict[str, float]:
    """Run the experiment ``cfg`` describes and return the last validation metrics."""
    config_path = experiment.start(cfg, resume)
    task = build_task(cfg)
    train_set = build_dataset(cfg, "train", train=True)
    val_set = build_dataset(cfg, "val") if "val" in cfg["data"] else None
    return experiment.fit(cfg, task, train_set, val_set, config_path, resume, monitor="val/mpq")


def load_task(path: str | Path, map_location: str = "cpu") -> tuple[CellInstanceTask, dict]:
    """The task of a checkpoint written by :func:`train`, its model trained, and its config."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if "config" not in ckpt:
        raise ValueError(f"{path} has no config; it was not written by pathotrain's train")
    task = build_task(ckpt["config"], pretrained=False)
    task.model.load_state_dict(ckpt["model"])
    return task, ckpt["config"]


def load_model(path: str | Path, map_location: str = "cpu") -> tuple[nn.Module, dict]:
    """The model (returning the ``np``, ``hv`` and ``nt`` branches) of a checkpoint
    written by :func:`train`, and its config."""
    task, cfg = load_task(path, map_location)
    return task.model, cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="experiment YAML")
    parser.add_argument("--resume", help="last.ckpt to continue from")
    args = parser.parse_args(argv)
    train(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
