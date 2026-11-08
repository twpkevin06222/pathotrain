"""Class-weighted tile sampling for the training DataLoader.

:func:`tile_weights` turns per-tile class content (``class_fraction`` of the
segmentation datasets without its background column, ``class_count`` of the
detection ones) into sampling weights; :func:`build_sampler` wraps them in a
``WeightedRandomSampler`` from a config's ``sampling`` section::

    sampling:
      power: 0.5          # 0 keeps class shares as they are, 1 balances them fully
      max_ratio: 10       # a class is drawn at most this many times more per unit of
                          # content than the most common class
      background: null    # share of draws from tiles without labels; null keeps theirs
      num_samples: null   # draws per epoch; null = dataset size

Draws are with replacement and use torch's global generator, so the run's ``seed``
(``seed_everything``) fixes them.
"""

import numpy as np
import torch
from loguru import logger
from torch.utils.data import WeightedRandomSampler


def tile_weights(
    content: np.ndarray,
    power: float = 0.5,
    max_ratio: float = 10.0,
    background: float | None = None,
) -> np.ndarray:
    """Sampling weight of each tile from its ``(tiles, classes)`` class ``content``
    (pixel shares or box counts of the object classes, no background column).

    A class's weight is its share of all content to the power ``-power``, capped at
    ``max_ratio`` times the most common class's weight; a tile's weight is its classes'
    weights averaged by how much of each it holds. Tiles without content then get, in
    total, a ``background`` share of the draws, or their share of the tiles if None.

    Raises:
        ValueError: ``content`` is not 2D or negative, ``power < 0``, ``max_ratio < 1``
            or ``background`` outside ``[0, 1)``.
    """
    content = np.asarray(content, dtype=float)
    if content.ndim != 2 or (content < 0).any():
        raise ValueError(f"content must be a 2D array >= 0, got shape {content.shape}")
    if power < 0 or max_ratio < 1:
        raise ValueError(f"need power >= 0 and max_ratio >= 1, got {power} and {max_ratio}")
    if background is not None and not 0 <= background < 1:
        raise ValueError(f"background must be in [0, 1), got {background}")
    total = content.sum(axis=0)
    present = total > 0
    class_weight = np.zeros(len(total))
    class_weight[present] = (total[present] / total.sum()) ** -power
    if present.any():
        cap = max_ratio * class_weight[present].min()
        class_weight[present] = np.minimum(class_weight[present], cap)

    amount = content.sum(axis=1)
    labelled = amount > 0
    weights = np.ones(len(content))
    weights[labelled] = content[labelled] @ class_weight / amount[labelled]
    if labelled.any() and not labelled.all():
        share = (~labelled).mean() if background is None else background
        empty_total = share * weights[labelled].sum() / (1 - share)
        weights[~labelled] = empty_total / (~labelled).sum()
    return weights


def build_sampler(cfg: dict, content: np.ndarray) -> WeightedRandomSampler:
    """A sampler over the training tiles from the ``sampling`` section of ``cfg`` (see the
    module docstring) and the tiles' class ``content``, as for :func:`tile_weights`."""
    options = dict(cfg["sampling"])
    num_samples = options.pop("num_samples", None)
    data = cfg.get("data", {})
    negatives = {**data, **data.get("train", {})}.get("negative_sampling", 0)
    if options.get("background") is not None and negatives:
        logger.warning(
            "sampling.background and negative_sampling are both set: the fixed negatives "
            "are then reweighted to the background share; use one of them"
        )
    weights = tile_weights(content, **options)
    content = np.asarray(content, dtype=float)
    before = content.sum(axis=0) / max(content.sum(), 1e-12)
    after = weights @ content / max(weights @ content.sum(axis=1), 1e-12)
    empty = content.sum(axis=1) == 0
    logger.info(
        "class-weighted sampling over {} tiles: class shares {} -> {}, tiles without "
        "labels {:.1%} -> {:.1%} of draws",
        len(weights),
        np.round(before, 3).tolist(),
        np.round(after, 3).tolist(),
        empty.mean(),
        weights[empty].sum() / weights.sum(),
    )
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples or len(weights),
        replacement=True,
    )
