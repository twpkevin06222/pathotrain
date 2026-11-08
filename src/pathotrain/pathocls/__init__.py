"""Tile classification: a linear or MLP head on a foundation-model backbone's tile
embedding; the backbone frozen (a probe as a trainable, exportable model) or fine-tuned."""

from pathotrain.pathocls.backbones import embedding_dim
from pathotrain.pathocls.model import Classifier, Head
from pathotrain.pathocls.task import ClassificationTask

__all__ = ["ClassificationTask", "Classifier", "Head", "embedding_dim"]
