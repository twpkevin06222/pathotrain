"""Tile classification with an image-text model, on labelled tiles
(:class:`pathotrain.dataloader.ClassTileDataset`):

- zero-shot, under ``zs/``: each tile takes the class whose prompt (``template``
  filled with the class name, or a prompt of its own) its image embedding is
  closest to; ``acc``, ``balanced_acc`` (mean recall per class), ``macro_f1``.
- with a train split, a linear probe and kNN on the frozen image embeddings
  (histossl's, under ``probe/`` and ``knn/``).
"""

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
    multiclass_recall,
)
from transformers import ProcessorMixin

from pathotrain.histossl.eval.linear_probe import fit_linear_probe, knn
from pathotrain.pathoclip.models import TEXT_LENGTH
from pathotrain.pathoclip.task import embed_images, embed_texts

TEMPLATE = "an H&E image of {}."


def tile_loader(dataset: Dataset, batch_size: int = 64, num_workers: int = 4) -> DataLoader:
    """``dataset`` in order. Its workers persist, so an evaluation every epoch does not
    start them again (~20 s each time with forkserver)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # Fresh workers, as histossl's probe: forking the training process can deadlock.
        multiprocessing_context="forkserver" if num_workers else None,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def image_embeddings(model: nn.Module, loader: DataLoader) -> tuple[Tensor, Tensor]:
    """L2-normalised image embeddings ``(N, D)`` float32 and labels of ``loader``'s
    ``(image, label)`` batches, in order, on ``model``'s device (bf16 autocast on CUDA)."""
    device = next(model.parameters()).device
    feats, labels = [], []
    for images, y in loader:
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            feats.append(embed_images(model, images.to(device)).float())
        labels.append(y.to(device))
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def text_embeddings(model: nn.Module, processor: ProcessorMixin, texts: list[str]) -> Tensor:
    device = next(model.parameters()).device
    ids = processor.tokenizer(
        texts, padding="max_length", max_length=TEXT_LENGTH, truncation=True, return_tensors="pt"
    )["input_ids"]
    return embed_texts(model, ids.to(device)).float()


def evaluator(
    processor: ProcessorMixin,
    test: Dataset,
    prompts: list[str],
    train: Dataset | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Callable[[nn.Module], dict[str, float]]:
    """``model -> metrics`` on ``test`` (one prompt per class, in class order),
    plus the linear probe and kNN fit on ``train`` when given."""
    n = len(prompts)
    test_loader = tile_loader(test, batch_size, num_workers)
    train_loader = None if train is None else tile_loader(train, batch_size, num_workers)

    def evaluate(model: nn.Module) -> dict[str, float]:
        was_training = model.training
        model.eval()
        x, y = image_embeddings(model, test_loader)
        scores = (x @ text_embeddings(model, processor, prompts).T).softmax(1)
        metrics = {
            "zs/acc": multiclass_accuracy(scores, y, n, average="micro").item(),
            "zs/balanced_acc": multiclass_recall(scores, y, n, average="macro").item(),
            "zs/macro_f1": multiclass_f1_score(scores, y, n, average="macro").item(),
        }
        if train_loader is not None:
            tx, ty = image_embeddings(model, train_loader)
            metrics |= fit_linear_probe(tx, ty, x, y, n) | knn(tx, ty, x, y, n)
        model.train(was_training)
        return metrics

    return evaluate
