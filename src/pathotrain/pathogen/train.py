"""Train a virtual-staining model from one YAML file::

    python -m pathotrain.pathogen.train config.yaml
    python -m pathotrain.pathogen.train config.yaml --resume runs/her2_ldm01/checkpoints/last.ckpt

Start from ``config.example.yaml`` next to this module. Training runs on
pathotrain's engine (Lightning Fabric): the config is copied to
``<output_dir>/config.yaml``, logged to MLflow and CSV, and stored in the
checkpoints (see :func:`load_model`). Sections (``?`` marks optional ones)::

    name: her2_ldm01            # MLflow experiment
    seed: 0
    output_dir: runs/her2_ldm01 # config.yaml, checkpoints/, logs/
    model: {name: ldm, vae: stabilityai/sd-vae-ft-mse, unet: {...}}  # baseline, or
    model: {name: sit, vae: ..., size: B, patch_size: 2, repa?: {encoder: midnight-12k,
            depth: 4, weight: 0.5, image: target}}  # SiT, rectified flow, REPA
    task?: {...}                # LatentDiffusionTask or FlowMatchingTask (lr, sample_steps, ...)
    trainer?: {...}             # TrainerConfig; grad_clip defaults to 1.0
    checkpoint?: {...}          # ModelCheckpoint; monitor defaults to val/ssim
    early_stopping?: {...}      # EarlyStopping; leave out to train every epoch
    loader?: {...}              # both DataLoaders, as for pathotrain
    data:
      crop?: 512                # random (train) or centre (val) crop, px
      resize?: null             # shorter side resized first, px
      cache?: false             # true: train from cached latents (built on first use)
      cache_variants?: 2        # orientations cached, 1-8
      cache_dir?: null          # default ~/.cache/pathotrain/latents/<fingerprint>
      train: {source: [dir, ...], target: [dir, ...]}  # tiles paired by file stem
      val?: {source: [...], target: [...]}
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset

from pathotrain import experiment
from pathotrain.engine import load_config
from pathotrain.pathogen.cache import CachedLatentDataset, ensure_latent_cache
from pathotrain.pathogen.data import PairedTileDataset, paired_transform
from pathotrain.pathogen.diffusion import (
    DEFAULT_VAE,
    LatentDiffusionTask,
    LatentTranslationTask,
    build_unet,
    build_vae,
)
from pathotrain.pathogen.flow import SIT_SIZES, FlowMatchingTask, SiT, build_encoder


def build_task(cfg: dict, repa: bool = True) -> LatentTranslationTask:
    """The ``model`` and ``task`` sections as a task: the VAE (and REPA encoder) with
    their weights, the UNet or SiT freshly initialised. ``repa=False`` leaves out
    REPA's encoder and projector, which only training needs."""
    model = dict(cfg["model"])
    name = model.pop("name", "ldm")
    if name not in ("ldm", "sit"):
        raise ValueError(f"unknown model {name!r}, expected ldm or sit")
    vae = build_vae(model.pop("vae", DEFAULT_VAE))
    channels = vae.config.latent_channels
    if name == "ldm":
        unet = build_unet(channels, **model.pop("unet", {}))
        if model:
            raise ValueError(f"unknown model settings {sorted(model)}")
        return LatentDiffusionTask(unet, vae, **cfg.get("task", {}))

    repa_cfg = dict(model.pop("repa", None) or {})
    size = model.pop("size", "B")
    if size not in SIT_SIZES:
        raise ValueError(f"unknown SiT size {size!r}, expected one of {sorted(SIT_SIZES)}")
    dims = dict(zip(("hidden_size", "depth", "num_heads"), SIT_SIZES[size], strict=True))
    encoder = build_encoder(repa_cfg.pop("encoder")) if repa_cfg and repa else None
    sit = SiT(
        2 * channels,
        channels,
        repa_dim=encoder.config.hidden_size if encoder is not None else None,
        repa_depth=repa_cfg.pop("depth", 4),
        **{**dims, **model},
    )
    kwargs = {f"repa_{k}": v for k, v in repa_cfg.items()} if encoder is not None else {}
    return FlowMatchingTask(sit, vae, encoder, **kwargs, **cfg.get("task", {}))


def build_dataset(cfg: dict, split: str) -> Dataset:
    """``data[split]`` as paired tiles, augmented for ``train``. With ``data.cache``,
    the train split reads cached latents, building them first if needed (see
    :mod:`pathotrain.pathogen.cache`); validation always reads images."""
    data, s = cfg["data"], cfg["data"][split]
    if data.get("cache") and split == "train":
        if data.get("resize"):
            raise ValueError("data.resize does not apply to cached latents; leave it out")
        model = cfg["model"]
        repa = model.get("repa") if model.get("name") == "sit" else None
        return CachedLatentDataset(
            ensure_latent_cache(cfg),
            data.get("crop"),
            image=(repa.get("image", "target") if repa else None),
            vae=model.get("vae", DEFAULT_VAE),
            source_dirs=s["source"],
            target_dirs=s["target"],
        )
    transform = paired_transform(split == "train", data.get("crop"), data.get("resize"))
    return PairedTileDataset(s["source"], s["target"], transform)


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


def load_model(path: str | Path, map_location: str = "cpu") -> tuple[LatentTranslationTask, dict]:
    """Rebuild the task of a ``best.ckpt`` / ``last.ckpt`` written by :func:`train`,
    in eval mode, with the EMA weights (when the run kept an EMA) as ``task.model``
    and without REPA's training-only parts; return it with its config.
    ``task.translate`` samples."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if "config" not in ckpt:
        raise ValueError(f"{path} has no config; it was not written by pathotrain.pathogen.train")
    task = build_task(ckpt["config"], repa=False)
    # The EMA is what validation scored; drop REPA's projector, a training-only head.
    weights = {
        k: v
        for k, v in ckpt.get("ema", ckpt["model"]).items()
        if not k.startswith("projector.") and k != "ema_updates"
    }
    task.model.load_state_dict(weights)
    task.ema = None
    return task.eval(), ckpt["config"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="experiment YAML")
    parser.add_argument("--resume", help="last.ckpt to continue from")
    args = parser.parse_args(argv)
    train(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
