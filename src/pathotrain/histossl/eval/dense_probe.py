"""Dense linear probe on frozen patch tokens::

    python -m pathotrain.histossl.eval.dense_probe midnight-12k \\
        --train tiger224/train --test tiger224/val

The patch-level counterpart of :mod:`~pathotrain.histossl.eval.linear_probe`: it
checks that SSL keeps the patch tokens' spatial information, which the tile
embedding (CLS + mean patch) can hide. ``--train`` and ``--test`` are folders
written by :func:`pathotrain.dataloader.export_segmentation_tiles` (tiles a
multiple of the patch size, e.g. 224 px). The backbone is given as for the
linear probe; tiles are normalised as it expects and not resized.

- Each patch token (last layer, after the final LayerNorm, registers dropped:
  :func:`pathotrain.pathofm.segmentation_embedding`) is labelled with the
  majority class of its pixels. Pixels at ``ignore_index`` (outside the ROI) do
  not vote; a patch with fewer than half its pixels labelled is skipped.
- Logistic regression on standardised tokens (L-BFGS), its L2 strength chosen by
  mIoU on a 20% hold-out of the train *tiles*, then refit on all of them and
  scored on every test tile. Train uses ``max_tiles`` evenly spaced tiles.

Metrics, under ``dense/``: ``miou`` (mean over the classes present in test),
``acc`` and ``iou_<class>``, all over labelled patches.
"""

import argparse
import functools
import json

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2

from pathotrain.dataloader import SegmentationTileFolderDataset
from pathotrain.histossl.backbones import preprocessing
from pathotrain.histossl.eval.linear_probe import WEIGHT_DECAYS, _fit, load_encoder
from pathotrain.pathofm import segmentation_embedding


def _transform(tile: np.ndarray, mask: np.ndarray, normalize: v2.Compose):
    return normalize(np.array(tile)), torch.from_numpy(mask.astype(np.int64))  # writable copy


def dense_transform(mean, std):
    """``(tile, mask)`` arrays -> normalised float tile, int64 mask; no resizing."""
    normalize = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(list(mean), list(std)),
            v2.ToPureTensor(),
        ]
    )
    return functools.partial(_transform, normalize=normalize)  # picklable for workers


def patch_labels(masks: Tensor, patch: int, ignore_index: int = 255) -> Tensor:
    """``(B, H, W)`` pixel labels (0-255) to ``(B, H/patch * W/patch)`` patch labels,
    row-major as the patch tokens: each patch's majority label, ``ignore_index``
    where under half its pixels are labelled."""
    b, h, w = masks.shape
    if h % patch or w % patch:
        raise ValueError(f"tiles must be a multiple of the patch size {patch}, got {h} x {w}")
    cells = masks.reshape(b, h // patch, patch, w // patch, patch).permute(0, 1, 3, 2, 4)
    cells = cells.reshape(-1, patch * patch)
    n = len(cells)
    offset = torch.arange(n, device=masks.device)[:, None] * 256
    counts = torch.bincount((offset + cells).flatten(), minlength=n * 256).view(n, 256)
    counts[:, ignore_index] = 0
    labels = counts.argmax(1)
    labels[counts.sum(1) * 2 < patch * patch] = ignore_index
    return labels.view(b, -1)


@torch.no_grad()
def _batches(model: nn.Module, dataset: Dataset, batch_size: int, num_workers: int, ignore: int):
    """Per batch: labelled patch tokens ``(M, D)`` float32, their labels and tile indices."""
    device = next(model.parameters()).device
    patch = model.config.patch_size
    context = "forkserver" if num_workers else None  # as linear_probe.extract
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, multiprocessing_context=context
    )
    first = 0
    for images, masks in loader:
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            grid = segmentation_embedding(model, images.to(device))
        tokens = grid.flatten(2).transpose(1, 2).float()  # (B, N, D)
        labels = patch_labels(masks.to(device), patch, ignore)
        tiles = torch.arange(first, first + len(images), device=device)[:, None].expand_as(labels)
        first += len(images)
        keep = labels != ignore
        yield tokens[keep], labels[keep], tiles[keep]


def _confusion(pred: Tensor, target: Tensor, num_classes: int) -> Tensor:
    return torch.bincount(target * num_classes + pred, minlength=num_classes**2).view(
        num_classes, num_classes
    )


def _scores(confusion: Tensor) -> dict[str, float]:
    confusion = confusion.double()
    hit, truth = confusion.diag(), confusion.sum(1)
    iou = hit / (truth + confusion.sum(0) - hit).clamp_min(1)
    present = truth > 0
    return {
        "dense/miou": iou[present].mean().item(),
        "dense/acc": (hit.sum() / truth.sum()).item(),
        **{f"dense/iou_{c}": iou[c].item() for c in present.nonzero().flatten().tolist()},
    }


def evaluate_dense(
    model: nn.Module,
    train: Dataset,
    test: Dataset,
    max_tiles: int = 1000,
    ignore_index: int = 255,
    batch_size: int = 64,
    num_workers: int = 4,
    weight_decays: tuple[float, ...] = WEIGHT_DECAYS,
    seed: int = 0,
) -> dict[str, float]:
    """Dense probe metrics of ``model``'s frozen patch tokens (see the module
    docstring); ``train`` and ``test`` yield ``(image, mask)``, already
    normalised for the model. Also returns the chosen ``dense/weight_decay``."""
    was_training = model.training
    model.eval()
    if len(train) > max_tiles:
        train = Subset(train, np.linspace(0, len(train) - 1, max_tiles).round().astype(int))
    parts = list(_batches(model, train, batch_size, num_workers, ignore_index))
    x, y, tiles = (torch.cat(p) for p in zip(*parts, strict=True))
    del parts
    num_classes = int(y.max()) + 1
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    x = (x - mean) / std

    generator = torch.Generator().manual_seed(seed)
    n_tiles = int(tiles.max()) + 1
    held_tiles = torch.randperm(n_tiles, generator=generator)[: max(1, n_tiles // 5)]
    held = torch.isin(tiles, held_tiles.to(tiles.device))
    best = None
    for wd in weight_decays:
        linear = _fit(x[~held], y[~held], num_classes, wd)
        with torch.no_grad():
            pred = linear(x[held]).argmax(1)
        miou = _scores(_confusion(pred, y[held], num_classes))["dense/miou"]
        if best is None or miou > best[0]:
            best = (miou, wd)
    linear = _fit(x, y, num_classes, best[1])
    del x

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=y.device)
    for tokens, labels, _ in _batches(model, test, batch_size, num_workers, ignore_index):
        with torch.no_grad():
            pred = linear((tokens - mean) / std).argmax(1)
        # A test class unseen in train cannot be predicted; it still counts as missed.
        size = max(num_classes, int(labels.max()) + 1)
        if size > len(confusion):
            confusion = nn.functional.pad(confusion, (0, size - len(confusion)) * 2)
        confusion += _confusion(pred, labels, len(confusion))
    model.train(was_training)
    return {**_scores(confusion), "dense/weight_decay": best[1]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "backbone",
        help="a pathotrain.pathofm model (e.g. midnight-12k), a histossl .ckpt, "
        "or a folder written by pathotrain.histossl.export",
    )
    parser.add_argument("--train", required=True, help="folder from export_segmentation_tiles")
    parser.add_argument("--test", required=True, help="folder from export_segmentation_tiles")
    parser.add_argument("--max-tiles", type=int, default=1000, help="train tiles used")
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)
    model = load_encoder(args.backbone)
    pp = preprocessing(model)
    transform = dense_transform(pp["mean"], pp["std"])
    metrics = evaluate_dense(
        model,
        SegmentationTileFolderDataset(args.train, transform),
        SegmentationTileFolderDataset(args.test, transform),
        args.max_tiles,
        args.ignore_index,
        args.batch_size,
        args.num_workers,
    )
    print(json.dumps({"backbone": args.backbone, **metrics}, indent=2))


if __name__ == "__main__":
    main()
