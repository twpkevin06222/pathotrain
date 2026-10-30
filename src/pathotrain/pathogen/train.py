"""Train a virtual-staining model from one YAML file::

    python -m pathotrain.pathotrain.pathogen.train config.yaml
    python -m pathotrain.pathotrain.pathogen.train config.yaml --resume runs/her2_ldm01/checkpoints/last.ckpt

Start from ``config.example.yaml`` next to this module. Training runs on
pathotrain's engine (Lightning Fabric): the config is copied to
``<output_dir>/config.yaml``, logged to MLflow and CSV, and stored in the
checkpoints (see :func:`load_model`). Sections (``?`` marks optional ones)::

    name: her2_ldm01            # MLflow experiment
    seed: 0
    output_dir: runs/her2_ldm01 # config.yaml, checkpoints/, logs/
    model: {name: ldm, vae: stabilityai/sd-vae-ft-mse, unet: {...}}
    task?: {...}                # LatentDiffusionTask (lr, warmup_epochs, sample_steps, ...)
    trainer?: {...}             # TrainerConfig; grad_clip defaults to 1.0
    checkpoint?: {...}          # ModelCheckpoint; monitor defaults to val/ssim
    early_stopping?: {...}      # EarlyStopping; leave out to train every epoch
    loader?: {...}              # both DataLoaders, as for pathotrain
    data:
      crop?: 512                # random (train) or centre (val) crop, px
      resize?: null             # shorter side resized first, px
      train: {source: [dir, ...], target: [dir, ...]}  # tiles paired by file stem
      val?: {source: [...], target: [...]}
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset

from pathotrain import experiment
from pathotrain.engine import load_config
from pathotrain.pathogen.data import PairedTileDataset, paired_transform
from pathotrain.pathogen.diffusion import DEFAULT_VAE, LatentDiffusionTask, build_unet, build_vae


def build_task(cfg: dict) -> LatentDiffusionTask:
    """The ``model`` and ``task`` sections as a task (the VAE with its weights, the
    UNet freshly initialised)."""
    model = dict(cfg["model"])
    name = model.pop("name", "ldm")
    if name != "ldm":
        raise ValueError(f"unknown model {name!r}, expected ldm")
    vae = build_vae(model.pop("vae", DEFAULT_VAE))
    unet = build_unet(vae.config.latent_channels, **model.pop("unet", {}))
    if model:
        raise ValueError(f"unknown model settings {sorted(model)}")
    return LatentDiffusionTask(unet, vae, **cfg.get("task", {}))


def build_dataset(cfg: dict, split: str) -> Dataset:
    """``data[split]`` as paired tiles, augmented for ``train``."""
    data = cfg["data"]
    transform = paired_transform(split == "train", data.get("crop"), data.get("resize"))
    return PairedTileDataset(data[split]["source"], data[split]["target"], transform)


def train(cfg: dict, resume: str | Path | None = None) -> dict[str, float]:
    """Run the experiment ``cfg`` describes and return the last validation metrics."""
    config_path = experiment.start(cfg, resume)
    task = build_task(cfg)
    val_set = build_dataset(cfg, "val") if "val" in cfg["data"] else None
    return experiment.fit(
        cfg,
        task,
        build_dataset(cfg, "train"),
        val_set,
        config_path,
        resume,
        monitor="val/ssim",
        trainer_defaults={"grad_clip": 1.0},
    )


def load_model(path: str | Path, map_location: str = "cpu") -> tuple[LatentDiffusionTask, dict]:
    """Rebuild the task of a ``best.ckpt`` / ``last.ckpt`` written by :func:`train`,
    in eval mode, and return it with its config. ``task.translate`` samples."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if "config" not in ckpt:
        raise ValueError(f"{path} has no config; it was not written by pathotrain.pathogen.train")
    task = build_task(ckpt["config"])
    task.model.load_state_dict(ckpt["model"])
    return task.eval(), ckpt["config"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="experiment YAML")
    parser.add_argument("--resume", help="last.ckpt to continue from")
    args = parser.parse_args(argv)
    train(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
