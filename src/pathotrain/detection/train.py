"""Train an RF-DETR detector from one YAML file::

    python -m pathotrain.detection.train config.yaml
    python -m pathotrain.detection.train config.yaml --resume runs/det01/checkpoints/last.ckpt

Start from ``config.example.yaml`` next to this module, which lists every
section with its defaults. The config is copied to ``<output_dir>/config.yaml``,
uploaded to the MLflow run and logged as flat params; checkpoints store it too
(see :func:`load_model`). Sections (``?`` marks optional ones)::

    name: det01                 # MLflow experiment
    seed: 0
    output_dir: runs/det01      # config.yaml, checkpoints/, logs/
    mean: IMAGENET_MEAN         # or 3 numbers in [0, 1]
    std: IMAGENET_STD
    model: {name: rfdetr:nano, num_classes: 2, ...}  # build_detector(**model)
    task?: {...}                # DetectionTask (lr, lr_encoder, warmup_epochs, class_names, ...)
    trainer?: {...}             # TrainerConfig; grad_clip defaults to 0.1 (RF-DETR's)
    checkpoint?: {...}          # ModelCheckpoint; monitor defaults to val/map
    early_stopping?: {...}      # EarlyStopping (monitor val/map); leave out to train every epoch
    augmentation?: {...}        # train_transform (n, m, crop, blur, noise, jpeg)
    loader?: {...}              # both DataLoaders, as for segmentation; collate_fn is set,
                                # multiprocessing_context defaults to forkserver
    data:
      train: {slides: [dir, ...], annotations: [dir, ...], tiling: {...}}
      val?: {slides: [...], annotations: [...], tiling: {...}}
                                # or a COCO folder per split: {root: dir} from
                                # export_detection_tiles (DetectionTileFolderDataset)
      ...                       # the rest goes to DetectionTileGenerator for the slide
                                # splits (class_map, roi_class, min_box_visibility, ...)
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset

from patchology import TilingConfig
from pathotrain import experiment
from pathotrain.dataloader import DetectionDataset, DetectionTileFolderDataset
from pathotrain.dataloader.augmentation import eval_transform, train_transform
from pathotrain.detection.models import Detector, build_detector
from pathotrain.detection.task import DetectionTask, collate
from pathotrain.engine import load_config
from pathotrain.experiment import normalisation


def build_dataset(cfg: dict, split: str, transform) -> Dataset:
    """``data[split]`` as a dataset: slides with annotations, tiled on the fly, or a
    COCO folder (``root``); its classes must match ``model.num_classes``."""
    s = cfg["data"][split]
    if "slides" not in s:
        dataset = DetectionTileFolderDataset(transform=transform, **s)
        num_classes = len(dataset.class_names)
    else:
        data = {k: v for k, v in cfg["data"].items() if k not in ("train", "val")}
        dataset = DetectionDataset(
            s["slides"],
            s["annotations"],
            transform=transform,
            tiling=TilingConfig(**s.get("tiling", {})),
            **data,
        )
        num_classes = len(set(data["class_map"].values()))
    if num_classes != cfg["model"]["num_classes"]:
        raise ValueError(
            f"data.{split} has {num_classes} classes but model.num_classes is "
            f"{cfg['model']['num_classes']}"
        )
    return dataset


def train(cfg: dict, resume: str | Path | None = None) -> dict[str, float]:
    """Run the experiment ``cfg`` describes and return the last validation metrics."""
    mean, std = normalisation(cfg, "mean"), normalisation(cfg, "std")
    config_path = experiment.start(cfg, resume)
    task = DetectionTask(build_detector(**cfg["model"]), **cfg.get("task", {}))
    augmentation = cfg.get("augmentation", {})
    train_set = build_dataset(
        cfg, "train", train_transform(mean=mean, std=std, boxes=True, **augmentation)
    )
    val_set = None
    if "val" in cfg["data"]:
        val_set = build_dataset(
            cfg, "val", eval_transform(augmentation.get("crop"), mean=mean, std=std, boxes=True)
        )
    return experiment.fit(
        cfg,
        task,
        train_set,
        val_set,
        config_path,
        resume,
        monitor="val/map",
        trainer_defaults={"grad_clip": 0.1},
        collate_fn=collate,
        # Fresh workers: forking the threaded training process can deadlock them.
        multiprocessing_context="forkserver",
    )


def load_model(path: str | Path, map_location: str = "cpu") -> tuple[Detector, dict]:
    """Rebuild the detector of a ``best.ckpt`` / ``last.ckpt`` written by :func:`train`
    and return it with its config (for ``mean``/``std``, ``class_map``, ...).

    ``detector.model`` holds the trained weights; ``detector.postprocess`` turns its
    outputs into boxes, scores and labels, as in ``DetectionTask.validation_step``.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if "config" not in ckpt:
        raise ValueError(f"{path} has no config; it was not written by pathotrain's train")
    cfg = ckpt["config"]
    # The checkpoint holds every weight, so skip the COCO download.
    detector = build_detector(**{**cfg["model"], "pretrained": False})
    detector.model.load_state_dict(ckpt["model"])
    return detector, cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="experiment YAML")
    parser.add_argument("--resume", help="last.ckpt to continue from")
    args = parser.parse_args(argv)
    train(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
