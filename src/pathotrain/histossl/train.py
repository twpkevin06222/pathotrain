"""Self-supervised training from one YAML file::

    python -m pathotrain.histossl.train config.yaml
    python -m pathotrain.histossl.train config.yaml --resume runs/jepa01/checkpoints/last.ckpt

Start from ``teacher_jepa.example.yaml`` next to this module. The config is
checked first (:func:`pathotrain.histossl.config.validate`), then copied to
``<output_dir>/config.yaml``, logged to MLflow and CSV and stored in both
checkpoints with its provenance (backbone, normalisation, input size,
adaptation, method, git commit). Sections (``?`` marks optional ones)::

    name, seed, output_dir, mlflow_db?
    method: {name: teacher_jepa, width?, depth?, heads?}  # the method's own settings
    backbone: {init: pretrained, id: midnight-12k}       # or {init: scratch, arch: vit_b14}
    adaptation?: {type: lora, rank: 16}                  # full, frozen, last_n (n), lora
    teacher?: {type: frozen}                             # frozen, ema, none
    masking?: {type: block, ratio: [0.4, 0.6]}
    task?: {lr, weight_decay, layer_decay, warmup_epochs}
    trainer?, checkpoint?, early_stopping?, loader?      # as for pathotrain
    data:
      train: {slides: [dir, ...], tiling: {tile_size: 224, tile_mpp: 0.5}}  # or {tiles: [dir, ...]}
      val?: {slides|tiles: ..., max_tiles?: 512}         # needed for checkpoints and eval
      samples_per_epoch?: 32000                          # tiles drawn per epoch (with replacement)
      crops?: {global_crops: 1, global_size: 224, ...}
      augmentation?: {hed: 0.05, hsv: 0.05, blur: 0.5, jpeg: 0.0}
    eval?: {train: mhist/train.csv, test: mhist/test.csv, every: 1}  # linear probe + kNN

``best.ckpt`` follows ``probe/balanced_acc`` with ``eval``, else ``val/loss``.
"""

import argparse
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.utils.data import Dataset, RandomSampler, Subset

from patchology import TilingConfig
from pathotrain import experiment
from pathotrain.dataloader import (
    ClassTileDataset,
    MultiCropTransform,
    UnlabelledSlideDataset,
    UnlabelledTileFolderDataset,
    multicrop_collate,
)
from pathotrain.engine import load_config
from pathotrain.histossl.backbones import normalisation
from pathotrain.histossl.config import SSLConfig, validate
from pathotrain.histossl.eval.linear_probe import evaluate_features, probe_transform
from pathotrain.histossl.methods import BUILDERS, SSLTask
from pathotrain.pathofm import FOUNDATION_MODELS


def _tiles(cfg: dict, split: str, transform) -> Dataset:
    s = dict(cfg["data"][split])
    max_tiles = s.pop("max_tiles", None)
    if "tiles" in s:
        dataset = UnlabelledTileFolderDataset(s["tiles"], transform)
    else:
        tiling = TilingConfig(**s.pop("tiling", {}))
        dataset = UnlabelledSlideDataset(s.pop("slides"), transform, tiling=tiling, **s)
    if max_tiles is not None and len(dataset) > max_tiles:
        # Evenly spaced, so a capped val split still spans every slide.
        dataset = Subset(dataset, np.linspace(0, len(dataset) - 1, max_tiles).round().astype(int))
    return dataset


def build_dataset(cfg: dict, ssl: SSLConfig, split: str) -> Dataset:
    """``data[split]``: random multi-crop views for ``train``, the centre
    ``global_size`` crop of each tile otherwise."""
    mean, std = normalisation(ssl.backbone.init, ssl.backbone.id)
    c = ssl.crops
    if split != "train":
        return _tiles(cfg, split, probe_transform(mean, std, c.global_size))
    views = MultiCropTransform(
        c.global_crops,
        c.local_crops,
        c.global_size,
        c.local_size,
        c.global_scale,
        c.local_scale,
        mean=mean,
        std=std,
        **cfg["data"].get("augmentation", {}),
    )
    return _tiles(cfg, split, views)


def build_probe(ssl: SSLConfig):
    """``probe(backbone) -> metrics`` on the ``eval`` tiles, or None without ``eval``."""
    e = ssl.eval
    if e is None:
        return None
    mean, std = normalisation(ssl.backbone.init, ssl.backbone.id)
    transform = probe_transform(mean, std, ssl.crops.global_size)

    def dataset(spec: str, class_names=None) -> ClassTileDataset:
        if Path(spec).suffix.lower() == ".csv":
            return ClassTileDataset(e.root, transform, csv=spec, class_names=class_names)
        return ClassTileDataset(spec, transform, class_names=class_names)

    train = dataset(e.train)
    test = dataset(e.test, train.class_names)
    classes = len(train.class_names)

    def probe(backbone: nn.Module) -> dict[str, float]:
        return evaluate_features(backbone, train, test, classes, e.k, e.batch_size, e.num_workers)

    return probe


def build_task(cfg: dict, ssl: SSLConfig | None = None, probe=None) -> SSLTask:
    """The task ``cfg`` describes, with pretrained weights loaded."""
    ssl = ssl or validate(cfg)
    return BUILDERS[ssl.method.name](cfg, ssl, probe)


def provenance(ssl: SSLConfig, task: SSLTask) -> dict:
    """What a checkpoint must say about its weights besides the config (with the
    LoRA targets ``task`` really adapted, the default resolved)."""
    bb, ad = ssl.backbone, ssl.adaptation
    mean, std = normalisation(bb.init, bb.id)
    package = Path(__file__).resolve().parent
    try:
        git = ["git", "-C", str(package)]
        commit = subprocess.run(
            [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            [*git, "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        git_commit = commit.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        logger.warning("no git commit found for the checkpoint provenance")
        git_commit = "unknown"
    return {
        "backbone": (
            f"{bb.id}@{FOUNDATION_MODELS[bb.id].revision}" if bb.init == "pretrained" else bb.arch
        ),
        "init": bb.init,
        "normalisation": {"mean": list(mean), "std": list(std)},
        "input_size": ssl.crops.global_size,
        "adaptation": {
            "type": ad.type,
            **({"n": ad.n} if ad.type == "last_n" else {}),
            **({"rank": ad.rank, "targets": _lora_targets(task)} if ad.type == "lora" else {}),
        },
        "method": ssl.method.name,
        "git_commit": git_commit,
        "pathotrain_version": version("pathotrain"),
    }


def _lora_targets(task: SSLTask) -> list[str]:
    return sorted(task.backbone.peft_config["default"].target_modules)


def train(cfg: dict, resume: str | Path | None = None) -> dict[str, float]:
    """Run the experiment ``cfg`` describes and return the last validation metrics."""
    ssl = validate(cfg)
    config_path = experiment.start(cfg, resume)
    task = build_task(cfg, ssl, build_probe(ssl))
    train_set = build_dataset(cfg, ssl, "train")
    val_set = build_dataset(cfg, ssl, "val") if "val" in cfg["data"] else None
    if val_set is None:
        logger.warning("no data.val: no validation, so no best.ckpt (last.ckpt is kept)")
    n = cfg["data"].get("samples_per_epoch")
    sampler = RandomSampler(train_set, replacement=True, num_samples=n) if n else None
    monitor, mode = ("probe/balanced_acc", "max") if ssl.eval else ("val/loss", "min")
    return experiment.fit(
        cfg,
        task,
        train_set,
        val_set,
        config_path,
        resume,
        monitor=monitor,
        mode=mode,
        trainer_defaults={"grad_clip": 3.0},
        checkpoint_extra={"provenance": provenance(ssl, task)},
        train_sampler=sampler,
        collate_fn=collate,
    )


def collate(samples: list) -> object:
    """Multi-crop training samples as :func:`multicrop_collate` batches them,
    validation tiles stacked to ``(B, 3, H, W)``."""
    return multicrop_collate(samples) if isinstance(samples[0], dict) else torch.stack(samples)


def load_model(path: str | Path, map_location: str = "cpu") -> tuple[SSLTask, dict]:
    """Rebuild the task of a ``best.ckpt`` / ``last.ckpt`` written by :func:`train`
    with its trained weights, in eval mode; return it with its config."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if "config" not in ckpt:
        raise ValueError(f"{path} has no config; it was not written by pathotrain.histossl.train")
    task = build_task(ckpt["config"])
    task.model.load_state_dict(ckpt["model"])
    return task.eval(), ckpt["config"]


def load_backbone(path: str | Path) -> tuple[nn.Module, dict]:
    """The trained encoder of a checkpoint as a plain ``Dinov2Model`` (LoRA merged
    into its weights), in eval mode, with its config."""
    task, cfg = load_model(path)
    backbone = task.encoder
    if hasattr(backbone, "merge_and_unload"):
        backbone = backbone.merge_and_unload()
    return backbone.eval(), cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="experiment YAML")
    parser.add_argument("--resume", help="last.ckpt to continue from")
    args = parser.parse_args(argv)
    train(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
