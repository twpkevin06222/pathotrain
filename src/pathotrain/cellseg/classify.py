"""CellViT++'s cell classifier: a linear or MLP head on each cell's foundation-model
tokens, the encoder frozen, so new cell classes need only a small head::

    python -m pathotrain.cellseg.classify classify.yaml

Start from ``classify.example.yaml`` next to this module. Each cell's embedding is the
mean of a ``pathofm`` encoder's tokens (layer ``model.layer``, default the last)
over the 16 px patches the cell covers, weighted by its area in each. The head trains on
the embeddings of the ground-truth cells of ``data.train`` (cached once, as pathocls's
linear probe) and is scored on those of ``data.val``. Classes are the nucleus types
without background: type ``t`` is class ``t - 1``.

At inference, :func:`classify_cells` types the cells of a predicted instance map, e.g.
from a ``cellseg:cellvit`` model trained with :mod:`pathotrain.cellseg.train`.

Sections as in ``pathotrain.pathocls.train`` (``task``, ``trainer``, ``checkpoint``,
``loader``, ...), except::

    model: {backbone: midnight-12k, num_classes: 5, layer?: null, head?: {hidden, dropout}}
    features?: {batch_size: 16, num_workers: 0}   # tiles per batch while embedding
    data: {train: {...}, val?: {...}}   # CellDataset (images, instances, types, ...)
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from pathotrain import experiment
from pathotrain.cellseg.data import CellDataset
from pathotrain.cellseg.instances import instance_types
from pathotrain.dataloader.augmentation import eval_transform
from pathotrain.engine import load_config
from pathotrain.experiment import normalisation
from pathotrain.pathocls.model import Head
from pathotrain.pathocls.task import ClassificationTask
from pathotrain.pathofm import load_backbone
from pathotrain.segmentation.encoders import token_grids


def cell_embeddings(
    backbone: nn.Module, image: Tensor, instances: Tensor, layer: int | None = None
) -> list[tuple[Tensor, Tensor]]:
    """Per image of the batch, ``(ids, embeddings)``: the instance ids of
    ``instances`` ``(B, H, W)`` and their ``(N, D)`` token embeddings (see the module
    docstring). ``layer`` is 1-based, default the last. Cells wholly beyond the last
    full 16 px patch are left out."""
    layer = layer or backbone.config.num_hidden_layers
    grids = token_grids(backbone, image, (layer,))[0]  # (B, D, H/16, W/16)
    h, w = grids.shape[-2:]
    out = []
    for grid, inst in zip(grids, instances[:, : h * 16, : w * 16].to(grids.device), strict=True):
        ids = inst.unique()
        ids = ids[ids > 0]
        area = F.avg_pool2d((inst == ids[:, None, None]).float()[:, None], 16)[:, 0]  # (N, h, w)
        total = area.sum((1, 2))
        keep = total > 0
        emb = torch.einsum("nhw,dhw->nd", area[keep], grid.float()) / total[keep, None]
        out.append((ids[keep], emb))
    return out


@torch.no_grad()
def classify_cells(
    backbone: nn.Module, head: nn.Module, image: Tensor, instances: Tensor, layer: int | None = None
) -> list[tuple[Tensor, Tensor]]:
    """Per image, ``(ids, types)``: each cell's predicted nucleus type (class + 1)."""
    return [
        (ids, head(emb).argmax(1) + 1)
        for ids, emb in cell_embeddings(backbone, image, instances, layer)
    ]


@torch.no_grad()
def embed_cells(cfg: dict, backbone: nn.Module, split: str) -> TensorDataset:
    """``(embedding, class)`` of every ground-truth cell of a known type in ``data[split]``,
    on CPU."""
    norm = {"mean": normalisation(cfg, "mean"), "std": normalisation(cfg, "std")}
    dataset = CellDataset(transform=eval_transform(**norm), **cfg["data"][split])
    features = {"batch_size": 16, "num_workers": 0, **cfg.get("features", {})}
    loader = DataLoader(
        dataset,
        batch_size=features["batch_size"],
        num_workers=features["num_workers"],
    )
    num_classes = cfg["model"]["num_classes"] + 1  # with background
    device = next(backbone.parameters()).device
    xs, ys = [], []
    for image, target in loader:
        cells = cell_embeddings(
            backbone, image.to(device), target["inst"], cfg["model"].get("layer")
        )
        for (ids, emb), inst, types in zip(cells, target["inst"], target["type"], strict=True):
            labels = torch.from_numpy(instance_types(inst.numpy(), types.numpy(), num_classes))
            labels = labels[ids.cpu()]
            known = labels > 0
            xs.append(emb[known.to(emb.device)].cpu())
            ys.append(labels[known] - 1)
    return TensorDataset(torch.cat(xs), torch.cat(ys))


def build_head(cfg: dict) -> Head:
    model = cfg["model"]
    dim = model["embedding_dim"]
    return Head(dim, model["num_classes"], **model.get("head", {}))


def train(cfg: dict, resume: str | Path | None = None) -> dict[str, float]:
    """Embed the cells, train the head, and return the last validation metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = load_backbone(cfg["model"]["backbone"]).eval().requires_grad_(False).to(device)
    # The run's config records the embedding width, so the head rebuilds without the backbone.
    cfg = cfg | {"model": cfg["model"] | {"embedding_dim": backbone.config.hidden_size}}
    config_path = experiment.start(cfg, resume)
    train_set = embed_cells(cfg, backbone, "train")
    val_set = embed_cells(cfg, backbone, "val") if "val" in cfg["data"] else None
    del backbone  # the head trains on the embeddings alone
    torch.cuda.empty_cache()
    head = build_head(cfg)
    x = train_set.tensors[0]
    head.mean.copy_(x.mean(0))
    head.std.copy_(x.std(0).clamp_min(1e-6))
    task = ClassificationTask(head, cfg["model"]["num_classes"], **cfg.get("task", {}))
    return experiment.fit(
        cfg, task, train_set, val_set, config_path, resume, monitor="val/balanced_acc"
    )


def load_head(path: str | Path, map_location: str = "cpu") -> tuple[Head, dict]:
    """The head of a checkpoint written by :func:`train`, and its config (whose
    ``model.backbone`` and ``model.layer`` give the encoder to embed with)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    if "config" not in ckpt:
        raise ValueError(f"{path} has no config; it was not written by pathotrain's train")
    head = build_head(ckpt["config"])
    head.load_state_dict(ckpt["model"])
    return head.eval(), ckpt["config"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="experiment YAML")
    parser.add_argument("--resume", help="last.ckpt to continue from")
    args = parser.parse_args(argv)
    train(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
