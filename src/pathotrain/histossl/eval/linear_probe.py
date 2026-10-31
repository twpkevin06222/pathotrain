"""Linear probe and kNN on frozen tile embeddings::

    python -m pathotrain.histossl.eval.linear_probe midnight-12k \\
        --train mhist/train.csv --test mhist/test.csv
    python -m pathotrain.histossl.eval.linear_probe runs/jepa01/checkpoints/best.ckpt ...

``--train`` and ``--test`` are class folders (``root/<class>/*.png``) or ``path,label``
CSVs (paths relative to the CSV, or to ``--root``), read with
:class:`pathotrain.dataloader.ClassTileDataset`. Tiles are resized and centre
cropped to 224 px and normalised as the backbone expects; the embedding is
the CLS token concatenated with the mean patch token
(:func:`pathotrain.pathofm.classification_embedding`). Prints the metrics as JSON.

- Linear probe: multinomial logistic regression on standardised features, fit
  with L-BFGS. Its L2 strength is chosen from :data:`WEIGHT_DECAYS` by balanced
  accuracy on a stratified 20% of the train split, then the probe is refit on
  all of it and scored on the test split.
- kNN: the ``k`` nearest train tiles by cosine similarity vote, weighted by
  ``exp(similarity / 0.07)`` (DINO's evaluation).

Metrics, under ``probe/`` and ``knn/``: ``acc``, ``balanced_acc`` (mean recall per
class), ``macro_f1`` and ``auroc`` (macro one-vs-rest).
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_auroc,
    multiclass_f1_score,
    multiclass_recall,
)
from torchvision.transforms import v2

from pathotrain.dataloader import ClassTileDataset
from pathotrain.histossl.backbones import build_backbone, normalisation
from pathotrain.pathofm import classification_embedding

WEIGHT_DECAYS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def probe_transform(mean, std, size: int = 224) -> v2.Compose:
    """uint8 tile -> ``size`` px normalised float tensor, as the model cards prepare tiles."""
    return v2.Compose(
        [
            v2.Resize(size, antialias=True),
            v2.CenterCrop(size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(list(mean), list(std)),
            v2.ToPureTensor(),
        ]
    )


@torch.no_grad()
def extract(
    model: nn.Module, dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> tuple[Tensor, Tensor]:
    """Embeddings ``(N, D)`` float32 and labels ``(N,)`` of ``(image, label)``
    samples, on ``model``'s device (bf16 autocast on CUDA). Order is kept."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    feats, labels = [], []
    # Fresh workers: forking a threaded (training) process can deadlock them.
    context = "forkserver" if num_workers else None
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, multiprocessing_context=context
    )
    for images, y in loader:
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            feats.append(classification_embedding(model, images.to(device)).float())
        labels.append(y.to(device))
    model.train(was_training)
    return torch.cat(feats), torch.cat(labels)


def _metrics(prefix: str, scores: Tensor, labels: Tensor, num_classes: int) -> dict[str, float]:
    kw = {"num_classes": num_classes}
    return {
        f"{prefix}/acc": multiclass_accuracy(scores, labels, average="micro", **kw).item(),
        f"{prefix}/balanced_acc": multiclass_recall(scores, labels, average="macro", **kw).item(),
        f"{prefix}/macro_f1": multiclass_f1_score(scores, labels, average="macro", **kw).item(),
        f"{prefix}/auroc": multiclass_auroc(scores, labels, average="macro", **kw).item(),
    }


def _fit(x: Tensor, y: Tensor, num_classes: int, weight_decay: float) -> nn.Linear:
    linear = nn.Linear(x.shape[1], num_classes).to(x.device)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    optimizer = torch.optim.LBFGS(
        linear.parameters(), max_iter=500, history_size=20, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(linear(x), y) + weight_decay * linear.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return linear


def _holdout(labels: Tensor, fraction: float, seed: int) -> Tensor:
    """A stratified bool mask over ``labels``: about ``fraction`` of each class."""
    generator = torch.Generator().manual_seed(seed)
    held = torch.zeros(len(labels), dtype=torch.bool)
    for c in labels.unique():
        idx = (labels == c).nonzero().squeeze(1).cpu()
        idx = idx[torch.randperm(len(idx), generator=generator)]
        held[idx[: max(1, round(fraction * len(idx)))]] = True
    return held.to(labels.device)


def linear_probe(
    train_x: Tensor,
    train_y: Tensor,
    test_x: Tensor,
    test_y: Tensor,
    num_classes: int,
    weight_decays: tuple[float, ...] = WEIGHT_DECAYS,
    seed: int = 0,
) -> dict[str, float]:
    """Fit a logistic-regression probe on ``train`` and score it on ``test`` (see
    the module docstring); also returns the chosen ``probe/weight_decay``."""
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-6)
    train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
    held = _holdout(train_y, 0.2, seed)
    best = None
    for wd in weight_decays:
        linear = _fit(train_x[~held], train_y[~held], num_classes, wd)
        with torch.no_grad():
            scores = linear(train_x[held]).softmax(1)
        bacc = multiclass_recall(scores, train_y[held], num_classes, average="macro").item()
        if best is None or bacc > best[0]:
            best = (bacc, wd)
    linear = _fit(train_x, train_y, num_classes, best[1])
    with torch.no_grad():
        scores = linear(test_x).softmax(1)
    return {**_metrics("probe", scores, test_y, num_classes), "probe/weight_decay": best[1]}


def knn(
    train_x: Tensor,
    train_y: Tensor,
    test_x: Tensor,
    test_y: Tensor,
    num_classes: int,
    k: int = 20,
    temperature: float = 0.07,
) -> dict[str, float]:
    """Weighted kNN on cosine similarity (DINO's evaluation)."""
    sims = F.normalize(test_x, dim=1) @ F.normalize(train_x, dim=1).T
    top, index = sims.topk(min(k, len(train_x)), dim=1)
    votes = F.one_hot(train_y[index], num_classes).float() * (top / temperature).exp()[..., None]
    scores = votes.sum(1)
    return _metrics("knn", scores / scores.sum(1, keepdim=True), test_y, num_classes)


def evaluate_features(
    model: nn.Module,
    train: Dataset,
    test: Dataset,
    num_classes: int,
    k: int = 20,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, float]:
    """Linear probe and kNN metrics of ``model``'s frozen embeddings, trained on
    ``train`` and scored on ``test`` (``(image, label)`` datasets, already
    transformed for the model)."""
    train_x, train_y = extract(model, train, batch_size, num_workers)
    test_x, test_y = extract(model, test, batch_size, num_workers)
    return {
        **linear_probe(train_x, train_y, test_x, test_y, num_classes),
        **knn(train_x, train_y, test_x, test_y, num_classes, k),
    }


def _dataset(spec: str, root: str | None, transform, class_names=None) -> ClassTileDataset:
    if Path(spec).suffix.lower() == ".csv":
        return ClassTileDataset(root, transform, csv=spec, class_names=class_names)
    return ClassTileDataset(spec, transform, class_names=class_names)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "backbone", help="a pathotrain.pathofm model (e.g. midnight-12k) or a histossl .ckpt"
    )
    parser.add_argument("--train", required=True, help="class folder root or path,label CSV")
    parser.add_argument("--test", required=True, help="class folder root or path,label CSV")
    parser.add_argument("--root", help="folder the CSV paths are relative to")
    parser.add_argument("--k", type=int, default=20, help="kNN neighbours")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.backbone.endswith(".ckpt"):
        from pathotrain.histossl.train import load_backbone

        model, cfg = load_backbone(args.backbone)
        bb = cfg["backbone"]
        mean_std = normalisation(bb["init"], bb.get("id"))
    else:
        model = build_backbone("pretrained", id=args.backbone, grad_checkpointing=False)
        mean_std = normalisation("pretrained", args.backbone)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    transform = probe_transform(*mean_std)
    train = _dataset(args.train, args.root, transform)
    test = _dataset(args.test, args.root, transform, train.class_names)
    metrics = evaluate_features(
        model, train, test, len(train.class_names), args.k, args.batch_size, args.num_workers
    )
    print(
        json.dumps({"backbone": args.backbone, "classes": train.class_names, **metrics}, indent=2)
    )


if __name__ == "__main__":
    main()
