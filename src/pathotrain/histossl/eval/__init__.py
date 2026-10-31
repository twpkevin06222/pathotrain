"""Frozen-feature evaluation of histossl backbones."""

# fit_linear_probe, not linear_probe: that name is the module (and its CLI).
from pathotrain.histossl.eval.linear_probe import evaluate_features, extract, fit_linear_probe, knn

__all__ = ["evaluate_features", "extract", "fit_linear_probe", "knn"]
