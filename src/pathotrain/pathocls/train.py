"""Train a tile classifier (a linear probe) from one YAML file::

    python -m pathotrain.pathocls.train config.yaml
    python -m pathotrain.pathocls.train config.yaml --resume runs/cls01/checkpoints/last.ckpt

Start from ``config.example.yaml`` next to this module. The backbone stays frozen: every
tile is embedded once (CLS token and mean patch token, as the model cards and histossl's
linear probe do), and a linear head on the standardised embeddings is trained on them
with AdamW. Tiles are resized and centre cropped to the backbone's input size and
normalised as it expects. The config is copied to ``<output_dir>/config.yaml`` with the
class order (``task.class_names``) and ``model.num_classes`` filled in; checkpoints hold
the head and the config (see :func:`load_model`). Sections (``?`` marks optional ones)::

    name: cls01                 # MLflow experiment
    seed: 0
    output_dir: runs/cls01      # config.yaml, checkpoints/, logs/
    model:
      name: midnight-12k        # a pathofm model, a histossl .ckpt or a histossl export folder
      num_classes?: 9           # checked against the classes found
    task?: {...}                # ClassificationTask (lr, weight_decay, warmup_epochs,
                                # class_names: the label order, default sorted)
    features?: {...}            # embedding pass (batch_size, num_workers)
    trainer?: {...}             # TrainerConfig
    checkpoint?: {...}          # ModelCheckpoint (default monitor val/balanced_acc)
    early_stopping?: {...}
    loader?: {...}              # head training on the cached embeddings (batch_size, ...)
    val_loader?: {...}
    data:
      train: {root: dir}        # class folders (root/<class>/*.png), or
                                # {csv: file, root?: dir} with path,label columns
      val?: {...}
      test?: {...}              # scored only by the eval command
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from pathotrain import experiment
from pathotrain.dataloader import ClassTileDataset
from pathotrain.engine import load_config
from pathotrain.histossl.backbones import preprocessing
from pathotrain.histossl.eval.linear_probe import extract, load_encoder, probe_transform
from pathotrain.pathocls.model import Classifier, LinearHead, embedding_dim
from pathotrain.pathocls.task import ClassificationTask


def build_dataset(cfg: dict, split: str, transform=None) -> ClassTileDataset:
    """``data[split]`` as tiles in ``task.class_names`` order (default sorted labels)."""
    names = cfg.get("task", {}).get("class_names")
    return ClassTileDataset(transform=transform, class_names=names, **cfg["data"][split])


def embed(cfg: dict, backbone: nn.Module, split: str) -> TensorDataset:
    """``(embedding, label)`` pairs of ``data[split]`` from the frozen ``backbone``, on CPU."""
    pp = preprocessing(backbone)
    transform = probe_transform(pp["mean"], pp["std"], pp["input_size"])
    x, y = extract(backbone, build_dataset(cfg, split, transform), **cfg.get("features", {}))
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
    config_path = experiment.start(cfg, resume)
    backbone = load_encoder(cfg["model"]["name"]).requires_grad_(False)
    train_set = embed(cfg, backbone, "train")
    val_set = embed(cfg, backbone, "val") if "val" in cfg["data"] else None
    head = LinearHead(embedding_dim(backbone), cfg["model"]["num_classes"])
    del backbone  # the head trains on the embeddings alone
    torch.cuda.empty_cache()
    x = train_set.tensors[0]
    head.mean.copy_(x.mean(0))
    head.std.copy_(x.std(0).clamp_min(1e-6))
    task = ClassificationTask(head, cfg["model"]["num_classes"], **cfg["task"])
    return experiment.fit(
        cfg, task, train_set, val_set, config_path, resume, monitor="val/balanced_acc"
    )


def load_model(path: str | Path, map_location: str = "cpu") -> tuple[Classifier, dict]:
    """The classifier (frozen backbone and trained head) of a ``best.ckpt`` /
    ``last.ckpt`` written by :func:`train`, on ``map_location``, and its config."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if "config" not in ckpt:
        raise ValueError(f"{path} has no config; it was not written by pathotrain's train")
    cfg = ckpt["config"]
    backbone = load_encoder(cfg["model"]["name"]).requires_grad_(False).eval()
    head = LinearHead(embedding_dim(backbone), cfg["model"]["num_classes"])
    head.load_state_dict(ckpt["model"])
    return Classifier(backbone, head).to(map_location), cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="experiment YAML")
    parser.add_argument("--resume", help="last.ckpt to continue from")
    args = parser.parse_args(argv)
    train(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
