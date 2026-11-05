"""Contrastive fine-tuning of an image-text model on pathology captions, from one YAML::

    python -m pathotrain.pathoclip.train config.yaml
    python -m pathotrain.pathoclip.train config.yaml --resume runs/clip01/checkpoints/last.ckpt

Start from ``configs/siglip2-base.example.yaml`` next to this module. The config
is checked first (:func:`pathotrain.pathoclip.config.validate`), copied to
``<output_dir>/config.yaml``, logged to MLflow and CSV and stored in both
checkpoints with its provenance, as for every pathotrain run. Sections (``?``
marks optional ones)::

    name, seed, output_dir, mlflow_db?
    model: {name: siglip2-base-224, image?: midnight-12k}  # image: a frozen pathofm tower (LiT)
    adaptation?: {type: lora, rank: 16, alpha: 32, dropout: 0.05, targets: [q_proj, ...],
                  towers: [vision, text]}  # or type: full; towers: [text] freezes the image tower
    data: {train: arch/train.jsonl, val: arch/val.jsonl, root?: null}  # pathotrain.dataloader.caption
    eval?: {test: crc/test.csv, train?: crc/train.csv, template: "an H&E image of {}.", prompts?: {}}
    task?: {lr: 1.0e-4, weight_decay: 0.0, warmup_epochs: 0.5}
    trainer?, loader?, checkpoint?, early_stopping?   # as for pathotrain (loader batch_size: 64)

``best.ckpt`` follows ``zs/balanced_acc`` with ``eval``, else ``val/recall_mean``.

With ``model.image`` (a :data:`~pathotrain.pathofm.FOUNDATION_MODELS` name and
``adaptation.towers: [text]``), the image tower is that frozen backbone and a
trained projection (:class:`~.models.LockedImageCLIP`): its embedding of every
train and val image is computed once into ``<output_dir>/image_features_*.pt``,
so training runs the text tower only and takes large batches. Images get no
augmentation either way.
"""

import argparse
from importlib.metadata import version
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from pathotrain import experiment
from pathotrain.dataloader import ClassTileDataset
from pathotrain.dataloader.caption import CaptionDataset
from pathotrain.engine import load_config, load_trainable_state
from pathotrain.histossl.eval.linear_probe import probe_transform
from pathotrain.pathoclip.config import TOWERS, Adaptation, CLIPConfig, Eval, adaptation, validate
from pathotrain.pathoclip.data import CaptionCollator, EmbeddedCaptions
from pathotrain.pathoclip.models import (
    CLIPS,
    TEXT_LENGTH,
    LockedImageCLIP,
    image_transform,
    load_clip,
)
from pathotrain.pathoclip.task import ContrastiveTask
from pathotrain.pathoclip.zero_shot import evaluator
from pathotrain.pathofm import FOUNDATION_MODELS


def adapt(model: nn.Module, a: Adaptation) -> nn.Module:
    """``model`` with LoRA on the blocks' ``a.targets`` of ``a.towers`` (the logit
    scale and bias train too), or itself fully trainable but for the towers not in
    ``a.towers``."""
    if a.type == "full":
        for tower, module in TOWERS.items():
            if tower not in a.towers:
                getattr(model, module).requires_grad_(False)
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError(
            "LoRA needs the histossl extra: pip install 'pathotrain[histossl]'"
        ) from e
    # Blocks only: the vision tower's attention-pooling head has an out_proj too.
    towers = "|".join(TOWERS[t] for t in a.towers)
    pattern = rf".*({towers})\.encoder\.layers\.\d+\.(self_attn|mlp)\.({'|'.join(a.targets)})"
    config = LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules=pattern
    )
    model = get_peft_model(model, config)
    for name, p in model.named_parameters():
        if name.endswith(("logit_scale", "logit_bias")):
            p.requires_grad_(True)
    return model


def build_model(name: str, image: str | None, a: Adaptation) -> tuple[nn.Module, object]:
    """The adapted model a config's ``model`` and ``adaptation`` describe, and its processor."""
    model, processor = load_clip(name)
    model = adapt(model, a)
    return (LockedImageCLIP(image, model) if image else model), processor


@torch.no_grad()
def cache_image_features(
    model: LockedImageCLIP, captions: CaptionDataset, processor, path: Path, num_workers: int
) -> Tensor:
    """``model``'s backbone embedding of each of ``captions``' images ``(N, D)``
    float32 on the CPU, computed once into ``path``."""
    if path.is_file():
        features = torch.load(path, weights_only=True)
        # shortcut: the cache is checked by row count only; delete it if the images change.
        if len(features) == len(captions):
            return features
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.image.to(device)
    loader = DataLoader(
        captions,
        batch_size=64,
        num_workers=num_workers,
        collate_fn=CaptionCollator(processor, image_transform(model.image_name)),
        multiprocessing_context="forkserver" if num_workers else None,
    )
    features = []
    for batch in loader:
        with torch.autocast(device, torch.bfloat16, enabled=device == "cuda"):
            features.append(model.backbone_features(batch["pixel_values"].to(device)).cpu())
    features = torch.cat(features)
    torch.save(features, path)
    return features


def _tiles(e: Eval, spec: str, transform, class_names=None) -> ClassTileDataset:
    spec, root = Path(spec).expanduser(), e.root and Path(e.root).expanduser()
    if spec.suffix.lower() == ".csv":
        return ClassTileDataset(root, transform, csv=spec, class_names=class_names)
    return ClassTileDataset(spec, transform, class_names=class_names)


def build_evaluator(e: Eval, processor, image: str | None = None):
    """The zero-shot (and linear probe) evaluator ``eval`` describes, for a
    model whose image tower is ``image`` (a pathofm name) or the processor's."""
    ip = processor.image_processor
    if image:
        transform = image_transform(image)
    else:
        transform = probe_transform(ip.image_mean, ip.image_std, ip.size["height"])
    train = _tiles(e, e.train, transform) if e.train else None
    test = _tiles(e, e.test, transform, train.class_names if train else None)
    prompts = [e.prompts.get(name, e.template.format(name)) for name in test.class_names]
    return evaluator(processor, test, prompts, train, e.batch_size, e.num_workers)


def provenance(c: CLIPConfig, processor) -> dict:
    """What a checkpoint must say about its weights besides the config."""
    clip, a, ip = CLIPS[c.model.name], c.adaptation, processor.image_processor
    out = {
        "model": c.model.name,
        "base": f"{clip.repo}@{clip.revision}",
        "adaptation": {"type": a.type}
        | (
            {"rank": a.rank, "alpha": a.alpha, "targets": list(a.targets)}
            if a.type == "lora"
            else {}
        )
        | {"towers": list(a.towers)},
        "input_size": ip.size["height"],
        "normalisation": {"mean": list(ip.image_mean), "std": list(ip.image_std)},
        "text_length": TEXT_LENGTH,
        "git_commit": experiment.git_commit(),
        "pathotrain_version": version("pathotrain"),
    }
    if c.model.image:  # the image side is the frozen backbone, not SigLIP's tower
        fm = FOUNDATION_MODELS[c.model.image]
        out["image"] = {"model": c.model.image, "base": f"{fm.repo_id}@{fm.revision}"}
        out["normalisation"] = {"mean": list(fm.mean), "std": list(fm.std)}
    return out


def train(cfg: dict, resume: str | Path | None = None) -> dict[str, float]:
    """Run the fine-tuning ``cfg`` describes and return the last validation metrics."""
    c = validate(cfg)
    config_path = experiment.start(cfg, resume)
    model, processor = build_model(c.model.name, c.model.image, c.adaptation)
    train_set = CaptionDataset(c.data.train, c.data.root)
    val_set = CaptionDataset(c.data.val, c.data.root)
    if c.model.image:
        workers = cfg.get("loader", {}).get("num_workers", 0)
        out = config_path.parent
        train_features = cache_image_features(
            model, train_set, processor, out / "image_features_train.pt", workers
        )
        val_features = cache_image_features(
            model, val_set, processor, out / "image_features_val.pt", workers
        )
        train_set = EmbeddedCaptions(train_set, train_features)
        val_set = EmbeddedCaptions(val_set, val_features)
    t = c.task
    task = ContrastiveTask(
        model,
        t.lr,
        t.weight_decay,
        t.warmup_epochs,
        evaluators=[build_evaluator(c.eval, processor, c.model.image)] if c.eval else [],
        trainable_only=c.adaptation.type == "lora" or c.model.image is not None,
    )
    monitor = "zs/balanced_acc" if c.eval else "val/recall_mean"
    return experiment.fit(
        cfg,
        task,
        train_set,
        val_set,
        config_path,
        resume,
        monitor=monitor,
        mode="max",
        checkpoint_extra={"provenance": provenance(c, processor)},
        collate_fn=CaptionCollator(processor),
        batch_size=64,
        # Validation workers persist too: starting them every epoch took ~20 s of a ~50 s
        # epoch (forkserver); ARCH's workers hold little RAM.
        persistent_workers=True,
    )


def load_model(path: str | Path) -> tuple[nn.Module, object, dict]:
    """The model of a ``best.ckpt`` / ``last.ckpt`` written by :func:`train` with
    its trained weights (LoRA still separate), in eval mode, with its processor
    and config: a :class:`~.models.LockedImageCLIP` for a ``model.image`` run."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "config" not in ckpt:
        raise ValueError(f"{path} has no config; it was not written by pathotrain.pathoclip.train")
    cfg = ckpt["config"]
    a = adaptation(cfg.get("adaptation", {}))
    image = cfg["model"].get("image")
    model, processor = build_model(cfg["model"]["name"], image, a)
    if a.type == "lora" or image:  # checkpoints hold the trainable weights only
        load_trainable_state(model, ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])
    return model.eval(), processor, cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="experiment YAML")
    parser.add_argument("--resume", help="last.ckpt to continue from")
    args = parser.parse_args(argv)
    print(train(load_config(args.config), args.resume))


if __name__ == "__main__":
    main()
