"""A tile classifier: a backbone's tile embedding into a linear or MLP head."""

import torch
from torch import Tensor, nn

from pathotrain.pathocls.backbones import embedding


class Head(nn.Module):
    """Logits from tile embeddings, standardised first with the training split's
    ``mean`` and ``std`` (buffers, so they travel with the checkpoint and the export).

    Linear by default; with ``hidden``, an MLP: one hidden layer of that width and GELU.
    ``dropout`` applies before the output layer (to the standardised embedding when
    linear, to the hidden layer when an MLP).
    """

    def __init__(
        self, in_features: int, num_classes: int, hidden: int | None = None, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(in_features))
        self.register_buffer("std", torch.ones(in_features))
        self.hidden = (
            nn.Sequential(nn.Linear(in_features, hidden), nn.GELU()) if hidden else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden or in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.dropout(self.hidden((x - self.mean) / self.std)))


class Classifier(nn.Module):
    """``image -> logits``: ``head`` on ``backbone``'s tile embedding
    (:func:`~pathotrain.pathocls.backbones.embedding`)."""

    def __init__(self, backbone: nn.Module, head: Head) -> None:
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, image: Tensor) -> Tensor:
        return self.head(embedding(self.backbone, image))

    def train(self, mode: bool = True) -> "Classifier":
        super().train(mode)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()  # frozen: no dropout or drop path, as when embedding
        return self
