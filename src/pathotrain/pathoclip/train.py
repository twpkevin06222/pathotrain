"""Contrastive fine-tuning of an image-text model on pathology captions, from one YAML::

    python -m pathotrain.pathoclip.train config.yaml
    python -m pathotrain.pathoclip.train config.yaml --resume runs/clip01/checkpoints/last.ckpt

Start from ``configs/siglip2-base.example.yaml`` next to this module. The config
is checked first (:func:`pathotrain.pathoclip.config.validate`), copied to
``<output_dir>/config.yaml``, logged to MLflow and CSV and stored in both
checkpoints with its provenance, as for every pathotrain run. Sections (``?``
marks optional ones)::

    name, seed, output_dir, mlflow_db?
    model: {name: siglip2-base-224}
    adaptation?: {type: lora, rank: 16, alpha: 32, dropout: 0.05, targets: [q_proj, ...]}  # or full
    data: {train: arch/train.jsonl, val: arch/val.jsonl, root?: null}  # pathotrain.dataloader.caption
    eval?: {test: crc/test.csv, train?: crc/train.csv, template: "an H&E image of {}.", prompts?: {}}
    task?: {lr: 1.0e-4, weight_decay: 0.0, warmup_epochs: 0.5}
    trainer?, loader?, checkpoint?, early_stopping?   # as for pathotrain (loader batch_size: 64)

``best.ckpt`` follows ``zs/balanced_acc`` with ``eval``, else ``val/recall_mean``.
"""

import argparse
from importlib.metadata import version
from pathlib import Path

import torch
from torch import nn

from pathotrain import experiment
from pathotrain.dataloader import ClassTileDataset
from pathotrain.dataloader.caption import CaptionDataset
from pathotrain.engine import load_config, load_trainable_state
from pathotrain.histossl.eval.linear_probe import probe_transform
from pathotrain.pathoclip.config import Adaptation, CLIPConfig, Eval, validate
from pathotrain.pathoclip.data import CaptionCollator
from pathotrain.pathoclip.models import CLIPS, TEXT_LENGTH, load_clip
from pathotrain.pathoclip.task import ContrastiveTask
from pathotrain.pathoclip.zero_shot import evaluator


def adapt(model: nn.Module, a: Adaptation) -> nn.Module:
    """``model`` with LoRA on the blocks' ``a.targets`` of both towers (the logit
    scale and bias train too), or itself for full fine-tuning."""
    if a.type == "full":
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError(
            "LoRA needs the histossl extra: pip install 'pathotrain[histossl]'"
        ) from e
    # Blocks only: the vision tower's attention-pooling head has an out_proj too.
    pattern = rf".*\.layers\.\d+\.(self_attn|mlp)\.({'|'.join(a.targets)})"
    config = LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules=pattern
    )
    model = get_peft_model(model, config)
    for name, p in model.named_parameters():
        if name.endswith(("logit_scale", "logit_bias")):
            p.requires_grad_(True)
    return model


def _tiles(e: Eval, spec: str, transform, class_names=None) -> ClassTileDataset:
    spec, root = Path(spec).expanduser(), e.root and Path(e.root).expanduser()
    if spec.suffix.lower() == ".csv":
        return ClassTileDataset(root, transform, csv=spec, class_names=class_names)
    return ClassTileDataset(spec, transform, class_names=class_names)


def build_evaluator(e: Eval, processor):
    """The zero-shot (and linear probe) evaluator ``eval`` describes."""
    ip = processor.image_processor
    transform = probe_transform(ip.image_mean, ip.image_std, ip.size["height"])
    train = _tiles(e, e.train, transform) if e.train else None
    test = _tiles(e, e.test, transform, train.class_names if train else None)
    prompts = [e.prompts.get(name, e.template.format(name)) for name in test.class_names]
    return evaluator(processor, test, prompts, train, e.batch_size, e.num_workers)


def provenance(c: CLIPConfig, processor) -> dict:
    """What a checkpoint must say about its weights besides the config."""
    clip, a, ip = CLIPS[c.model.name], c.adaptation, processor.image_processor
    return {
        "model": c.model.name,
        "base": f"{clip.repo}@{clip.revision}",
        "adaptation": {"type": a.type}
        | (
            {"rank": a.rank, "alpha": a.alpha, "targets": list(a.targets)}
            if a.type == "lora"
            else {}
        ),
        "input_size": ip.size["height"],
        "normalisation": {"mean": list(ip.image_mean), "std": list(ip.image_std)},
        "text_length": TEXT_LENGTH,
        "git_commit": experiment.git_commit(),
        "pathotrain_version": version("pathotrain"),
    }


def train(cfg: dict, resume: str | Path | None = None) -> dict[str, float]:
    """Run the fine-tuning ``cfg`` describes and return the last validation metrics."""
    c = validate(cfg)
    config_path = experiment.start(cfg, resume)
    model, processor = load_clip(c.model.name)
    model = adapt(model, c.adaptation)
    t = c.task
    task = ContrastiveTask(
        model,
        t.lr,
        t.weight_decay,
        t.warmup_epochs,
        evaluators=[build_evaluator(c.eval, processor)] if c.eval else [],
        trainable_only=c.adaptation.type == "lora",
    )
    monitor = "zs/balanced_acc" if c.eval else "val/recall_mean"
    return experiment.fit(
        cfg,
        task,
        CaptionDataset(c.data.train, c.data.root),
        CaptionDataset(c.data.val, c.data.root),
        config_path,
        resume,
        monitor=monitor,
        mode="max",
        checkpoint_extra={"provenance": provenance(c, processor)},
        collate_fn=CaptionCollator(processor),
        batch_size=64,
    )


def load_model(path: str | Path) -> tuple[nn.Module, object, dict]:
    """The model of a ``best.ckpt`` / ``last.ckpt`` written by :func:`train` with
    its trained weights (LoRA still separate), in eval mode, with its processor
    and config."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "config" not in ckpt:
        raise ValueError(f"{path} has no config; it was not written by pathotrain.pathoclip.train")
    cfg = ckpt["config"]
    a = Adaptation(**cfg.get("adaptation", {}))
    model, processor = load_clip(cfg["model"]["name"])
    model = adapt(model, Adaptation(**{**a.__dict__, "targets": tuple(a.targets)}))
    if a.type == "lora":  # checkpoints hold the trainable weights only
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
