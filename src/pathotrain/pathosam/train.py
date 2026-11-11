"""Fine-tune SAM 3's point and box model for interactive segmentation from one YAML file::

    python -m pathotrain.pathosam.train interactive.yaml
    python -m pathotrain.pathosam.train interactive.yaml --resume runs/x/checkpoints/last.ckpt

Start from ``configs/interactive.example.yaml`` beside this module. The image encoder stays
frozen; the prompt encoder and mask decoder train (see :class:`InteractiveTask`), and
checkpoints hold only those. Sections are those of ``pathotrain.segmentation.train``
(``trainer``, ``checkpoint``, ``early_stopping``, ``augmentation``, ``loader``,
``val_loader``, ...), except::

    model?: {revision: <facebook/sam3 commit>}   # InteractiveSam
    task?: {...}             # InteractiveTask (objects, rounds, box_share, lr, ...)
    data:
      train: {images: dir, instances: dir, types: dir, ...}  # InstanceTileFolderDataset
      val?: {...}

Tiles are normalised with mean = std = 0.5, as SAM 3 expects. ``best.ckpt`` is chosen on
``val/noc90`` (fewest clicks to 90% IoU). Score a checkpoint on a split with
``python -m pathotrain.pathosam.eval ... --checkpoint best.ckpt``.
"""

import argparse
from pathlib import Path

import torch

from pathotrain import experiment
from pathotrain.dataloader import InstanceTileFolderDataset
from pathotrain.dataloader.augmentation import eval_transform, train_transform
from pathotrain.engine import load_config, load_trainable_state
from pathotrain.pathosam.eval import SAM_MEAN, SAM_STD
from pathotrain.pathosam.model import InteractiveSam
from pathotrain.pathosam.task import InteractiveTask


def build_dataset(cfg: dict, split: str, train: bool = False) -> InstanceTileFolderDataset:
    """``data[split]``, augmented as ``augmentation`` says when ``train``."""
    norm = {"mean": SAM_MEAN, "std": SAM_STD}
    if train:
        transform = train_transform(**norm, **cfg.get("augmentation", {}))
    else:
        transform = eval_transform(**norm)
    return InstanceTileFolderDataset(transform=transform, **cfg["data"][split])


def train(cfg: dict, resume: str | Path | None = None) -> dict[str, float]:
    """Run the experiment ``cfg`` describes and return the last validation metrics."""
    config_path = experiment.start(cfg, resume)
    task = InteractiveTask(InteractiveSam(**cfg.get("model", {})), **cfg.get("task", {}))
    train_set = build_dataset(cfg, "train", train=True)
    val_set = build_dataset(cfg, "val") if "val" in cfg["data"] else None
    return experiment.fit(
        cfg, task, train_set, val_set, config_path, resume, monitor="val/noc90", mode="min"
    )


def load_model(path: str | Path, map_location: str = "cpu") -> tuple[InteractiveSam, dict]:
    """The fine-tuned model of a checkpoint written by :func:`train` (SAM 3 rebuilt from its
    pinned weights, then the trained parts loaded), and its config."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if "config" not in ckpt:
        raise ValueError(f"{path} has no config; it was not written by pathotrain's train")
    model = InteractiveSam(**ckpt["config"].get("model", {}))
    load_trainable_state(model, ckpt["model"])
    return model.eval(), ckpt["config"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="experiment YAML")
    parser.add_argument("--resume", help="last.ckpt to continue from")
    args = parser.parse_args(argv)
    train(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
