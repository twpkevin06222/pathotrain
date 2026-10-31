"""Frozen-feature evaluation of histossl backbones."""

from pathotrain.histossl.eval.linear_probe import evaluate_features, extract, knn, linear_probe

__all__ = ["evaluate_features", "extract", "knn", "linear_probe"]
