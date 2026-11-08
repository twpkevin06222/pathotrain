"""Tile classification: a linear head on a frozen foundation-model backbone, trained on
its cached tile embeddings (a linear probe as a trainable, exportable model)."""

from pathotrain.pathocls.model import Classifier, LinearHead, embedding_dim
from pathotrain.pathocls.task import ClassificationTask

__all__ = ["ClassificationTask", "Classifier", "LinearHead", "embedding_dim"]
