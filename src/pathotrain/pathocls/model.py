"""A tile classifier: a frozen backbone's tile embedding into a linear head."""

import torch
from torch import Tensor, nn

from pathotrain.pathofm import classification_embedding


class LinearHead(nn.Module):
    """Logits from tile embeddings, standardised first with the training split's
    ``mean`` and ``std`` (buffers, so they travel with the checkpoint and the export)."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(in_features))
        self.register_buffer("std", torch.ones(in_features))
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear((x - self.mean) / self.std)


class Classifier(nn.Module):
    """``image -> logits``: ``head`` on ``backbone``'s tile embedding (CLS token and
    mean patch token, :func:`~pathotrain.pathofm.classification_embedding`)."""

    def __init__(self, backbone: nn.Module, head: LinearHead) -> None:
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, image: Tensor) -> Tensor:
        return self.head(classification_embedding(self.backbone, image))


def embedding_dim(backbone: nn.Module) -> int:
    """Width of ``backbone``'s tile embedding: CLS and mean patch token side by side."""
    return 2 * backbone.config.hidden_size
