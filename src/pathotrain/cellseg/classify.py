"""CellViT++'s cell classifier: a small head on the tokens of the cells a frozen CellViT
detects, so new cell classes need no new segmentation model::

    python -m pathotrain.cellseg.classify classify.yaml

Start from ``configs/classify.example.yaml`` beside this module. As CellViT++ (Hörst et al. 2025):

1. A trained ``cellseg:cellvit`` instance model (``model.checkpoint``, written by
   :mod:`pathotrain.cellseg.train`) detects each tile's cells, and the same forward pass
   gives its encoder's deepest tokens.
2. A cell's embedding is the mean of the tokens of the 16 px patches its bounding box
   touches.
3. Detected cells are paired one to one with ground-truth cells by centroid distance
   (Hungarian, within ``match_radius`` px, 15 as in CellViT++) and take the paired cell's
   type. Unpaired detections, and ground-truth cells no detection reaches, are left out.
4. The embeddings are computed once, without augmentation, and a head trains on them.

Classes are the types of ``data``'s type masks without background: type ``t`` is class
``t - 1``. They need not be the detector's types. Normalisation is the detector's, read
from its checkpoint. At inference, :func:`classify_cells` detects and types cells in one pass.

Sections as in ``pathotrain.pathocls.train`` (``task``, ``trainer``, ``checkpoint``,
``loader``, ...), except::

    model: {checkpoint: path, num_classes: 5, head?: {hidden: 100, dropout: 0.0}}
    match_radius?: 15                             # px
    features?: {batch_size: 16, num_workers: 0}   # tiles per batch while embedding
    data: {train: {...}, val?: {...}}             # CellDataset (images, instances, types, ...)
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from scipy import ndimage as ndi
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from pathotrain import experiment
from pathotrain.cellseg import train as cell_train
from pathotrain.cellseg.data import CellDataset
from pathotrain.cellseg.instances import (
    centroids,
    instance_types,
    instances_from_outputs,
    pair_centroids,
)
from pathotrain.cellseg.models import CellViT
from pathotrain.dataloader.augmentation import eval_transform
from pathotrain.engine import load_config
from pathotrain.pathocls.model import Head
from pathotrain.pathocls.task import ClassificationTask


def cell_tokens(tokens: Tensor, instances: np.ndarray) -> tuple[np.ndarray, Tensor]:
    """The instance ids of ``instances`` ``(H, W)`` and their ``(N, D)`` embeddings: the
    mean of ``tokens`` ``(D, H/16, W/16)`` over the 16 px patches each bounding box touches."""
    ids, out = [], []
    h, w = tokens.shape[-2:]
    for i, box in enumerate(ndi.find_objects(instances), start=1):
        if box is None:
            continue
        ys, xs = box
        y0, x0 = ys.start // 16, xs.start // 16
        y1, x1 = min(math.ceil(ys.stop / 16), h), min(math.ceil(xs.stop / 16), w)
        if y0 >= y1 or x0 >= x1:  # wholly beyond the last full patch
            continue
        ids.append(i)
        out.append(tokens[:, y0:y1, x0:x1].float().mean((1, 2)))
    if not out:
        return np.zeros(0, np.int64), tokens.new_zeros(0, tokens.shape[0])
    return np.array(ids), torch.stack(out)


def detect_cells(
    model: CellViT, image: Tensor, postprocess: dict | None = None
) -> list[tuple[np.ndarray, np.ndarray, Tensor]]:
    """Per image of the batch: the detected instance map, its ids and their ``(N, D)``
    embeddings, from one forward pass of the instance ``model``."""
    with torch.no_grad():
        out, tokens = model(image, return_tokens=True)
    outputs = [out[k].float().cpu().numpy() for k in ("np", "hv", "nt")]
    cells = []
    for i in range(len(image)):
        inst, _ = instances_from_outputs(*(o[i] for o in outputs), **(postprocess or {}))
        cells.append((inst, *cell_tokens(tokens[i], inst)))
    return cells


@torch.no_grad()
def classify_cells(
    model: CellViT, head: nn.Module, image: Tensor, postprocess: dict | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """CellViT++ inference: per image, the detected instance map and the class of each
    instance id (indexed by id; 0 for ids without one, as for ``instance_types``)."""
    results = []
    for inst, ids, emb in detect_cells(model, image, postprocess):
        types = np.zeros(int(inst.max()) + 1, np.int64)
        if len(ids):
            types[ids] = head(emb.to(next(head.parameters()).device)).argmax(1).cpu().numpy()
        results.append((inst, types))
    return results


def embed_cells(cfg: dict, model: CellViT, postprocess: dict, split: str) -> TensorDataset:
    """``(embedding, class)`` of every detected cell of ``data[split]`` paired with a
    ground-truth cell of a known type, on CPU."""
    norm = {"mean": cfg["mean"], "std": cfg["std"]}
    dataset = CellDataset(transform=eval_transform(**norm), **cfg["data"][split])
    features = {"batch_size": 16, "num_workers": 0, **cfg.get("features", {})}
    loader = DataLoader(dataset, **features)
    num_types = cfg["model"]["num_classes"] + 1  # with background
    radius = cfg.get("match_radius", 15)
    device = next(model.parameters()).device
    xs, ys, counts = [], [], np.zeros(3, np.int64)  # detected, ground truth, paired
    for image, target in loader:
        cells = detect_cells(model, image.to(device), postprocess)
        for (pred, ids, emb), true, types in zip(
            cells, target["inst"], target["type"], strict=True
        ):
            true = true.numpy()
            true_ids, true_xy = centroids(true)
            true_types = instance_types(true, types.numpy(), num_types)[true_ids]
            _, pred_xy = centroids(pred, ids)  # in the order of emb
            pairs = pair_centroids(true_xy, pred_xy, radius)
            pairs = pairs[true_types[pairs[:, 0]] > 0]  # paired with a typed cell
            counts += (len(ids), len(true_ids), len(pairs))
            xs.append(emb[torch.from_numpy(pairs[:, 1])].cpu())
            ys.append(torch.from_numpy(true_types[pairs[:, 0]] - 1))
    logger.info("{}: {} cells detected, {} in the ground truth, {} paired", split, *counts)
    if not counts[2]:
        raise ValueError(f"no detected cell of {split} is within {radius} px of a typed cell")
    return TensorDataset(torch.cat(xs), torch.cat(ys))


def build_head(cfg: dict) -> Head:
    model = cfg["model"]
    return Head(model["embedding_dim"], model["num_classes"], **model.get("head", {}))


def load_detector(path: str | Path, device: str = "cpu") -> tuple[CellViT, dict]:
    """The frozen CellViT instance model of a :mod:`pathotrain.cellseg.train` checkpoint,
    in eval mode on ``device``, and its config."""
    task, cfg = cell_train.load_task(path)
    if cfg["model"]["name"].removeprefix("cellseg:") != "cellvit":
        raise ValueError(f"{path} is a {cfg['model']['name']} model; CellViT++ needs cellvit")
    return task.model.eval().requires_grad_(False).to(device), cfg


def train(cfg: dict, resume: str | Path | None = None) -> dict[str, float]:
    """Detect and embed the cells, train the head, and return the last validation metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, detector = load_detector(cfg["model"]["checkpoint"], device)
    # The run's config records the detector's normalisation and the embedding width.
    cfg = cfg | {
        "mean": experiment.normalisation(detector, "mean"),
        "std": experiment.normalisation(detector, "std"),
        "model": cfg["model"] | {"embedding_dim": model.backbone.config.hidden_size},
    }
    config_path = experiment.start(cfg, resume)
    postprocess = detector.get("task", {}).get("postprocess", {})
    train_set = embed_cells(cfg, model, postprocess, "train")
    val_set = embed_cells(cfg, model, postprocess, "val") if "val" in cfg["data"] else None
    del model  # the head trains on the embeddings alone
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
    ``model.checkpoint`` is the detector to run it with, see :func:`load_detector`)."""
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
