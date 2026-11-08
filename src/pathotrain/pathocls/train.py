"""Train a tile classifier (a linear probe) from one YAML file::

    python -m pathotrain.pathocls.train config.yaml
    python -m pathotrain.pathocls.train config.yaml --resume runs/cls01/checkpoints/last.ckpt

Start from ``config.example.yaml`` next to this module. The classifier is a linear (or
MLP) head on the backbone's standardised tile embedding (for pathofm / histossl models
the CLS token and mean patch token, as the model cards and histossl's linear probe use;
for ``timm:<name>`` models timm's pooled features), trained with AdamW. Tiles are resized and
centre cropped to the backbone's input size and normalised as it expects. Two paths:

- Cached (the default: frozen backbone, no ``augmentation``): every tile is embedded
  once and the head trains on the stored embeddings; seconds per epoch.
- Online (an ``augmentation`` section, or ``model.freeze: false``): the backbone runs
  on augmented tiles every step (resized, randomly cropped to the input size, flips,
  90 degree rotations, H&E RandAugment, scanner artifacts). Frozen, checkpoints hold
  the head only; ``freeze: false`` fine-tunes the backbone at ``task.backbone_lr``
  (default ``lr / 10``) with gradient checkpointing, and checkpoints hold both. The
  head's standardisation comes from up to 4096 un-augmented train tiles.

The config is copied to ``<output_dir>/config.yaml`` with the
class order (``task.class_names``) and ``model.num_classes`` filled in; checkpoints hold
the head and the config (see :func:`load_model`). Sections (``?`` marks optional ones)::

    name: cls01                 # MLflow experiment
    seed: 0
    output_dir: runs/cls01      # config.yaml, checkpoints/, logs/
    model:
      name: midnight-12k        # a pathofm model, a histossl .ckpt or export folder, or
                                # timm:<name> (e.g. timm:convnext_tiny.in12k_ft_in1k)
      pretrained?: true         # timm only: false trains from scratch (needs freeze: false)
      num_classes?: 9           # checked against the classes found
      freeze?: true             # false fine-tunes the backbone (online)
      head?: {hidden: 512, dropout: 0.1}  # an MLP head; default linear (Head)
    task?: {...}                # ClassificationTask (lr, weight_decay, warmup_epochs,
                                # backbone_lr, class_names: the label order, default sorted)
    augmentation?: {...}        # train_transform (n, m, blur, noise, jpeg); trains online
    features?: {...}            # embedding passes (batch_size, num_workers)
    trainer?: {...}             # TrainerConfig
    sampling?: {...}            # class-weighted sampling of training tiles (power,
                                # max_ratio, num_samples), see pathotrain.dataloader.sampling
    checkpoint?: {...}          # ModelCheckpoint (default monitor val/balanced_acc)
    early_stopping?: {...}
    loader?: {...}              # training batches (embeddings, or tiles online)
    val_loader?: {...}
    data:
      train: {root: dir}        # class folders (root/<class>/*.png), or
                                # {csv: file, root?: dir} with path,label columns
      val?: {...}
      test?: {...}              # scored only by the eval command
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset, WeightedRandomSampler
from torchvision.transforms import v2

from pathotrain import experiment
from pathotrain.dataloader import ClassTileDataset
from pathotrain.dataloader.augmentation import train_transform
from pathotrain.dataloader.sampling import build_sampler
from pathotrain.engine import load_config, load_trainable_state
from pathotrain.histossl.eval.linear_probe import probe_transform
from pathotrain.pathocls import backbones
from pathotrain.pathocls.backbones import embedding_dim, extract, preprocessing
from pathotrain.pathocls.model import Classifier, Head
from pathotrain.pathocls.task import ClassificationTask

STATS_TILES = 4096  # train tiles whose embeddings set the head's standardisation online


def build_dataset(cfg: dict, split: str, transform=None) -> ClassTileDataset:
    """``data[split]`` as tiles in ``task.class_names`` order (default sorted labels)."""
    names = cfg.get("task", {}).get("class_names")
    return ClassTileDataset(transform=transform, class_names=names, **cfg["data"][split])


def embed(cfg: dict, backbone: nn.Module, split: str, limit: int | None = None) -> TensorDataset:
    """``(embedding, label)`` pairs of ``data[split]`` from the frozen ``backbone``, on
    CPU; with ``limit``, of at most that many tiles drawn at random (seeded)."""
    pp = preprocessing(backbone)
    dataset = build_dataset(cfg, split, probe_transform(pp["mean"], pp["std"], pp["input_size"]))
    if limit is not None and len(dataset) > limit:
        generator = torch.Generator().manual_seed(cfg.get("seed", 0))
        dataset = Subset(
            dataset, torch.randperm(len(dataset), generator=generator)[:limit].tolist()
        )
    x, y = extract(backbone, dataset, **cfg.get("features", {}))
    return TensorDataset(x.cpu(), y.cpu())


def complete(cfg: dict) -> dict:
    """``cfg`` with ``task.class_names`` (the train split's labels, sorted, unless set)
    and ``model.num_classes`` filled in, so the run's config records the class order.

    Raises:
        ValueError: ``model.num_classes`` disagrees with the classes.
    """
    names = list(build_dataset(cfg, "train").class_names)
    n = cfg["model"].get("num_classes", len(names))
    if n != len(names):
        raise ValueError(f"model.num_classes is {n} but data.train has classes {names}")
    return cfg | {
        "model": cfg["model"] | {"num_classes": n},
        "task": cfg.get("task", {}) | {"class_names": names},
    }


def train(cfg: dict, resume: str | Path | None = None) -> dict[str, float]:
    """Run the experiment ``cfg`` describes and return the last validation metrics."""
    cfg = complete(cfg)
    freeze = cfg["model"].get("freeze", True)
    if freeze and not cfg["model"].get("pretrained", True):
        raise ValueError("a backbone trained from scratch (pretrained: false) needs freeze: false")
    config_path = experiment.start(cfg, resume)
    backbone, head = _build(cfg)
    if freeze and "augmentation" not in cfg:
        train_set = embed(cfg, backbone, "train")
        val_set = embed(cfg, backbone, "val") if "val" in cfg["data"] else None
        del backbone  # the head trains on the embeddings alone
        torch.cuda.empty_cache()
        model = head
        _standardise(head, train_set.tensors[0])
    else:
        train_set, val_set = _online_sets(cfg, backbone)
        stats = embed(cfg, backbone, "train", STATS_TILES)
        _standardise(head, stats.tensors[0])
        if not freeze:
            backbone.requires_grad_(True)
            backbones.enable_grad_checkpointing(backbone)
        model = Classifier(backbone, head)
    task = ClassificationTask(model, cfg["model"]["num_classes"], **cfg["task"])
    task.checkpoint_trainable_only = freeze  # online, the head without the frozen backbone
    return experiment.fit(
        cfg,
        task,
        train_set,
        val_set,
        config_path,
        resume,
        monitor="val/balanced_acc",
        train_sampler=train_sampler(cfg),
    )


def train_sampler(cfg: dict) -> WeightedRandomSampler | None:
    """The class-weighted sampler over the train split that the ``sampling`` section
    asks for (each tile's class content is its one-hot label), or None without one."""
    if "sampling" not in cfg:
        return None
    labels = [label for _, label in build_dataset(cfg, "train").samples]
    return build_sampler(cfg, np.eye(cfg["model"]["num_classes"])[labels])


def _build(cfg: dict, pretrained: bool = True) -> tuple[nn.Module, Head]:
    """The frozen backbone ``model`` names and a fresh head on it."""
    model = cfg["model"]
    pretrained = pretrained and model.get("pretrained", True)
    backbone = backbones.load(model["name"], pretrained).requires_grad_(False)
    head = Head(embedding_dim(backbone), model["num_classes"], **model.get("head", {}))
    return backbone, head


def _standardise(head: Head, x: torch.Tensor) -> None:
    head.mean.copy_(x.mean(0))
    head.std.copy_(x.std(0).clamp_min(1e-6))


def _online_sets(
    cfg: dict, backbone: nn.Module
) -> tuple[ClassTileDataset, ClassTileDataset | None]:
    """Train tiles augmented and randomly cropped to the backbone's input size, and val
    tiles prepared as for embedding."""
    pp = preprocessing(backbone)
    size, norm = pp["input_size"], {"mean": pp["mean"], "std": pp["std"]}
    augment = train_transform(crop=size, **norm, **cfg.get("augmentation", {})).compose
    transform = v2.Compose([v2.Resize(size, antialias=True), augment, v2.ToPureTensor()])
    val = "val" in cfg["data"]
    return build_dataset(cfg, "train", transform), (
        build_dataset(cfg, "val", probe_transform(**norm, size=size)) if val else None
    )


def load_model(path: str | Path, map_location: str = "cpu") -> tuple[Classifier, dict]:
    """The classifier (backbone, fine-tuned if trained so, and head) of a ``best.ckpt`` /
    ``last.ckpt`` written by :func:`train`, on ``map_location``, and its config."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if "config" not in ckpt:
        raise ValueError(f"{path} has no config; it was not written by pathotrain's train")
    cfg = ckpt["config"]
    freeze = cfg["model"].get("freeze", True)
    # Fine-tuned timm weights are all in the checkpoint: skip the ImageNet download.
    timm = cfg["model"]["name"].startswith(backbones.TIMM)
    backbone, head = _build(cfg, pretrained=freeze or not timm)
    backbone.eval()
    classifier, state = Classifier(backbone, head), ckpt["model"]
    if not any(k.startswith("head.") for k in state):
        head.load_state_dict(state)  # cached path: the head alone
    elif freeze:
        load_trainable_state(classifier, state)  # frozen online: the head (and buffers)
    else:
        classifier.load_state_dict(state)  # fine-tuned: backbone and head
    return classifier.to(map_location), cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="experiment YAML")
    parser.add_argument("--resume", help="last.ckpt to continue from")
    args = parser.parse_args(argv)
    train(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
