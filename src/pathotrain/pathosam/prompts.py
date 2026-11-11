"""Prompts simulated from ground-truth masks, as interactive segmentation benchmarks
(RITM, SAM) place them: boxes, a first click at the object's centre, and corrective clicks
in the largest error.

Points are ``(x, y)`` in tile pixels; masks are ``(H, W)`` bool.
"""

import numpy as np
from scipy import ndimage as ndi


def box(
    mask: np.ndarray, jitter: float = 0.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    """The mask's box ``[x0, y0, x1, y1]`` (``x1``, ``y1`` exclusive), each side moved by up
    to ``jitter`` times the box's width (x) or height (y), uniformly, and clipped to the tile."""
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    out = np.array([x0, y0, x1, y1], np.float64)
    if jitter:
        rng = rng or np.random.default_rng()
        size = np.array([x1 - x0, y1 - y0, x1 - x0, y1 - y0], np.float64)
        out += rng.uniform(-jitter, jitter, 4) * size
    h, w = mask.shape
    out[[0, 2]] = out[[0, 2]].clip(0, w)
    out[[1, 3]] = out[[1, 3]].clip(0, h)
    return out


def _deepest(region: np.ndarray) -> tuple[float, float, float]:
    """The pixel of ``region`` farthest from its boundary, as ``(x, y, distance)``."""
    (ys, xs), *_ = ndi.find_objects(region.astype(np.uint8))
    dist = ndi.distance_transform_edt(np.pad(region[ys, xs], 1))  # the tile edge is boundary
    y, x = np.unravel_index(dist.argmax(), dist.shape)
    return float(xs.start + x - 1), float(ys.start + y - 1), float(dist[y, x])


def center_click(mask: np.ndarray) -> tuple[float, float]:
    """The first click: the mask's pixel farthest from its boundary."""
    return _deepest(mask)[:2]


def corrective_click(true: np.ndarray, pred: np.ndarray) -> tuple[float, float, int] | None:
    """The next click ``(x, y, label)``, as RITM's NoC protocol places it: at the deepest
    pixel of the missed (``true`` and not ``pred``, label 1) or the extra (``pred`` and not
    ``true``, label 0) region, whichever is deeper. None when the prediction is exact."""
    clicks = [
        (*_deepest(region), label)
        for region, label in ((true & ~pred, 1), (pred & ~true, 0))
        if region.any()
    ]
    if not clicks:
        return None
    x, y, _, label = max(clicks, key=lambda c: c[2])
    return x, y, label
