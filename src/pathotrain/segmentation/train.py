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
                                # or tiles on disk, per split (TileFolderDataset):
                                # {root: dir} from export_tiles, or {images: dir, masks: dir}
      ...                       # the rest goes to SegmentationTileGenerator for the
                                # slide splits (class_map, roi_class, backend, ...)
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from patchology import TilingConfig
from pathotrain import experiment, segmentation
from pathotrain.dataloader import SegmentationDataset, TileFolderDataset
from pathotrain.dataloader.augmentation import eval_transform, train_transform
from pathotrain.engine import load_config
from pathotrain.experiment import normalisation


def build_loss(cfg: dict) -> nn.Module | None:
    """The ``loss`` section as a loss module, or None for the task's default."""
    if "loss" not in cfg:
        return None
    kwargs = dict(cfg["loss"])
    name = kwargs.pop("name")
    if not name.endswith("Loss") or not hasattr(segmentation, name):
        raise ValueError(f"unknown loss {name!r}, expected e.g. CEDiceLoss, FocalDiceLoss")
    return getattr(segmentation, name)(**kwargs)


def build_dataset(cfg: dict, split: str, transform) -> Dataset:
    """``data[split]`` as a dataset: slides with annotations, tiled on the fly, or
    tiles on disk (``root`` from ``export_tiles``, or ``images`` and ``masks``)."""
    s = cfg["data"][split]
    if "slides" not in s:
        return TileFolderDataset(transform=transform, **s)
    data = {k: v for k, v in cfg["data"].items() if k not in ("train", "val")}
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
    config_path = experiment.start(cfg, resume)
    model = segmentation.build_model(**cfg["model"])
    task = segmentation.SegmentationTask(
        model, cfg["model"]["num_classes"], loss=build_loss(cfg), **cfg.get("task", {})
    )
    augmentation = cfg.get("augmentation", {})
    train_set = build_dataset(cfg, "train", train_transform(mean=mean, std=std, **augmentation))
    val_set = None
    if "val" in cfg["data"]:
        val_set = build_dataset(
            cfg, "val", eval_transform(augmentation.get("crop"), mean=mean, std=std)
        )
    return experiment.fit(cfg, task, train_set, val_set, config_path, resume)


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
