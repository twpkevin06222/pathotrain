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
    init_weights?: best.ckpt    # start from a checkpoint's weights (fine-tuning): one
                                # written by train for the same model section
    loss?: {name: FocalDiceLoss, ...}  # a loss class from pathotrain.segmentation
    task?: {...}                # SegmentationTask (lr, warmup_epochs, class_names, ...)
    trainer?: {...}             # TrainerConfig
    checkpoint?: {...}          # ModelCheckpoint (monitor, mode, log_artifact)
    early_stopping?: {...}      # EarlyStopping; leave out to train every epoch
    augmentation?: {...}        # train_transform (n, m, crop, blur, noise, jpeg)
    sampling?: {...}            # class-weighted tile sampling for training
                                # (power, max_ratio, background, num_samples),
                                # see pathotrain.dataloader.sampling
    loader?: {...}              # both DataLoaders (batch_size, num_workers, ...);
                                # pin_memory defaults to true when CUDA is available,
                                # persistent_workers to true for the training loader,
                                # multiprocessing_context to forkserver
    data:
      train: {slides: [dir, ...], annotations: [dir, ...], tiling: {...}, ...}
      val?: {slides: [...], annotations: [...], tiling: {...}}
      test?: {...}              # as val; scored only by the eval command
                                # or tiles on disk, per split (SegmentationTileFolderDataset):
                                # {root: dir} from export_segmentation_tiles,
                                # or {images: dir, masks: dir}
      ...                       # the rest goes to SegmentationTileGenerator for the
                                # slide splits (class_map, roi_class, backend, ...);
                                # the same keys in a split override them for it,
                                # e.g. train: {..., negative_sampling: 0.2}
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from patchology import TilingConfig
from pathotrain import experiment, segmentation
from pathotrain.dataloader import SegmentationDataset, SegmentationTileFolderDataset
from pathotrain.dataloader.augmentation import eval_transform, train_transform
from pathotrain.dataloader.sampling import build_sampler
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
    tiles on disk (``root`` from ``export_segmentation_tiles``, or ``images`` and ``masks``)."""
    s = cfg["data"][split]
    if "slides" not in s:
        return SegmentationTileFolderDataset(transform=transform, **s)
    data = {k: v for k, v in cfg["data"].items() if k not in ("train", "val", "test", split)}
    data |= {k: v for k, v in s.items() if k not in ("slides", "annotations", "tiling")}
    return SegmentationDataset(
        s["slides"],
        s["annotations"],
        transform=transform,
        tiling=TilingConfig(**s.get("tiling", {})),
        **data,
    )


def task_kwargs(cfg: dict) -> dict:
    """The ``task`` section, with ``window`` defaulting to ``augmentation.crop`` so
    validation scores whole tiles from windows of the training size."""
    return {"window": cfg.get("augmentation", {}).get("crop"), **cfg.get("task", {})}


def train(cfg: dict, resume: str | Path | None = None) -> dict[str, float]:
    """Run the experiment ``cfg`` describes and return the last validation metrics."""
    mean, std = normalisation(cfg, "mean"), normalisation(cfg, "std")
    experiment.check_foreground_crop(cfg)
    config_path = experiment.start(cfg, resume)
    model = segmentation.build_model(**cfg["model"])
    if cfg.get("init_weights") and resume is None:
        load_weights(model, cfg["init_weights"])
    augmentation = cfg.get("augmentation", {})
    task = segmentation.SegmentationTask(
        model, cfg["model"]["num_classes"], loss=build_loss(cfg), **task_kwargs(cfg)
    )
    train_set = build_dataset(cfg, "train", train_transform(mean=mean, std=std, **augmentation))
    val_set = None
    if "val" in cfg["data"]:
        val_set = build_dataset(cfg, "val", eval_transform(mean=mean, std=std))
    sampler = None
    if "sampling" in cfg:  # class_fraction without its background column
        area = getattr(train_set, "image_area", None)  # tile folders only
        sampler = build_sampler(cfg, train_set.class_fraction[:, 1:], area)
    return experiment.fit(cfg, task, train_set, val_set, config_path, resume, train_sampler=sampler)


def load_weights(model: nn.Module, path: str | Path) -> None:
    """Load the weights of a ``best.ckpt`` / ``last.ckpt`` written by :func:`train` into
    ``model``, built from the same ``model`` section (fine-tuning, unlike ``--resume``:
    the optimizer, schedule and epoch count start afresh)."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "model" not in state:
        raise ValueError(f"{path} has no model weights; it was not written by pathotrain's train")
    model.load_state_dict(state["model"])


def load_model(path: str | Path, map_location: str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the model of a ``best.ckpt`` / ``last.ckpt`` written by :func:`train`
    and return it with its config (for ``mean``/``std``, ``class_map``, ...)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if "config" not in ckpt:
        raise ValueError(f"{path} has no config; it was not written by pathotrain's train")
    cfg = ckpt["config"]
    kwargs = dict(cfg["model"])
    if kwargs["name"].startswith(("smp:", "cellseg:")):
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
