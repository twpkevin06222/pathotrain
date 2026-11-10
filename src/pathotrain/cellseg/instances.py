"""Nucleus instance maps: targets from labels, HoVer-Net post-processing, PQ matching.

An instance map is an ``(H, W)`` integer array, 0 for background and one id per
nucleus. A type map gives each pixel its class, 0 for background.
"""

import math

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.segmentation import watershed

# HoVer-Net takes HV-map gradients with a 21 px Sobel kernel (cv2), whose smoothing is a
# binomial of variance 20 / 4 = 5: here a Gaussian derivative of that sigma. The maps
# are min-max scaled afterwards, so the kernel's scale does not matter.
SOBEL_SIGMA = math.sqrt(5)


def instances_from_contours(contours: np.ndarray, types: np.ndarray) -> np.ndarray:
    """Instance map from a nucleus outline map (nonzero on 1 px outlines, as in some
    PanNuke conversions) and the type map.

    Nucleus pixels inside no outline split into 4-connected instances; outline pixels on
    a nucleus then go to the nearest instance. Nucleus pixels are where ``types`` is in
    ``1..254`` (255 is ignored).
    """
    nuclei = (types > 0) & (types != 255)
    seeds, _ = ndi.label(nuclei & (contours == 0))
    if not seeds.any():
        return seeds
    nearest = ndi.distance_transform_edt(seeds == 0, return_distances=False, return_indices=True)
    return np.where(nuclei, seeds[tuple(nearest)], 0)


def instance_types(instances: np.ndarray, types: np.ndarray, num_classes: int) -> np.ndarray:
    """The type of each instance id, indexed by id (``out[0]`` is 0): the most common
    nucleus type (``1..num_classes-1``) under it, 0 if it has none."""
    size = int(instances.max()) + 1
    labels = np.where((types > 0) & (types < num_classes), types, 0)
    counts = np.bincount(
        instances.ravel() * num_classes + labels.ravel(), minlength=size * num_classes
    ).reshape(size, num_classes)
    counts[:, 0] = 0  # background votes only when nothing else does
    out = counts.argmax(1)
    out[0] = 0
    return out


def hv_maps(instances: np.ndarray) -> np.ndarray:
    """HoVer-Net's ``(2, H, W)`` float32 targets: each nucleus pixel's horizontal and
    vertical offset from its nucleus's centre of mass, scaled to [-1, 1] on each side
    of the centre separately; 0 on background."""
    hv = np.zeros((2, *instances.shape), np.float32)
    for i, box in enumerate(ndi.find_objects(instances), start=1):
        if box is None:
            continue
        inside = instances[box] == i
        cy, cx = ndi.center_of_mass(inside)
        ys, xs = np.nonzero(inside)
        for channel, offset in ((0, xs - cx), (1, ys - cy)):
            offset = offset.astype(np.float32)
            neg, pos = offset < 0, offset > 0
            if neg.any():
                offset[neg] /= -offset[neg].min()
            if pos.any():
                offset[pos] /= offset[pos].max()
            hv[channel][box][inside] = offset
    return hv


def _minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def _remove_small(labels: np.ndarray, min_size: int) -> np.ndarray:
    sizes = np.bincount(labels.ravel())
    small = sizes < min_size
    small[0] = False
    return np.where(small[labels], 0, labels)


def postprocess(
    foreground: np.ndarray,
    hv: np.ndarray,
    threshold: float = 0.5,
    marker_threshold: float = 0.4,
    min_size: int = 10,
) -> np.ndarray:
    """HoVer-Net's watershed: an instance map from the nucleus probability ``(H, W)``
    and the predicted HV maps ``(2, H, W)``.

    Nuclei are where ``foreground >= threshold``. Strong HV gradients mark boundaries
    between touching nuclei; the nucleus pixels away from them (gradient energy below
    ``marker_threshold``) seed a watershed over the nuclei. Blobs and seeds smaller than
    ``min_size`` pixels are dropped.
    """
    blobs, _ = ndi.label(foreground >= threshold)
    nuclei = _remove_small(blobs, min_size) > 0
    if not nuclei.any():
        return np.zeros(foreground.shape, np.int32)
    h, v = _minmax(hv[0]), _minmax(hv[1])
    # Signed, as HoVer-Net: the maps rise across a nucleus and fall steeply between two
    # touching ones, so the most negative gradients mark the boundaries (energy near 1).
    grad_h = 1 - _minmax(ndi.gaussian_filter(h, SOBEL_SIGMA, order=(0, 1)))
    grad_v = 1 - _minmax(ndi.gaussian_filter(v, SOBEL_SIGMA, order=(1, 0)))
    energy = np.clip(np.maximum(grad_h, grad_v) - ~nuclei, 0, None)
    # HoVer-Net blurs with a 3 px kernel (cv2 sigma 0.8 for that size).
    dist = -ndi.gaussian_filter((1 - energy) * nuclei, 0.8)
    markers = ndi.binary_fill_holes(nuclei & (energy < marker_threshold))
    disk = np.hypot(*np.mgrid[-2:3, -2:3]) <= 2.5  # 5 x 5 ellipse
    markers, _ = ndi.label(ndi.binary_opening(markers, disk))
    markers = _remove_small(markers, min_size).astype(np.int32)
    return watershed(dist, markers, mask=nuclei).astype(np.int32)


def predicted_types(instances: np.ndarray, type_probs: np.ndarray) -> np.ndarray:
    """The type of each predicted instance, indexed by id: the most common nucleus type
    of the per-pixel argmax of ``type_probs`` ``(C, H, W)`` (or logits) under it."""
    return instance_types(instances, type_probs.argmax(0), type_probs.shape[0])


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(0, keepdims=True))
    return e / e.sum(0, keepdims=True)


def instances_from_outputs(
    np_logits: np.ndarray, hv: np.ndarray, nt_logits: np.ndarray, **options
) -> tuple[np.ndarray, np.ndarray]:
    """One tile's nuclei from a model's ``np`` ``(2, H, W)``, ``hv`` ``(2, H, W)`` and
    ``nt`` ``(C, H, W)`` outputs (PyTorch or ONNX, as numpy): the instance map and the
    type of each instance id. ``options`` go to :func:`postprocess`."""
    inst = postprocess(_softmax(np_logits)[1], hv, **options)
    return inst, predicted_types(inst, nt_logits)


def centroids(
    instances: np.ndarray, ids: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``ids`` (default every instance id of ``instances``) and their ``(N, 2)`` centres
    of mass, ``(y, x)``."""
    if ids is None:
        ids = np.unique(instances)
        ids = ids[ids > 0]
    if not len(ids):
        return ids, np.zeros((0, 2))
    return ids, np.array(ndi.center_of_mass(np.ones_like(instances), instances, ids))


def pair_centroids(true: np.ndarray, pred: np.ndarray, radius: float) -> np.ndarray:
    """``(K, 2)`` index pairs ``[i_true, i_pred]`` of two ``(N, 2)`` point sets: the
    pairing with the least total distance (Hungarian), keeping pairs within ``radius``."""
    if not len(true) or not len(pred):
        return np.zeros((0, 2), np.int64)
    distance = cdist(true, pred)
    rows, cols = linear_sum_assignment(distance)
    near = distance[rows, cols] <= radius
    return np.stack([rows[near], cols[near]], 1)


def match(true: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, float]:
    """``(tp, fp, fn, iou_sum)`` of two instance maps: an instance pair matches when
    their IoU exceeds 0.5 (which makes the matching unique)."""
    t_ids, t = np.unique(true, return_inverse=True)
    p_ids, p = np.unique(pred, return_inverse=True)
    # Index 0 is background whenever it is present; shift so ids start at 1.
    t = t.ravel() + (t_ids[0] != 0)
    p = p.ravel() + (p_ids[0] != 0)
    n_t, n_p = int(t.max()), int(p.max())
    if n_t == 0 or n_p == 0:
        return 0, n_p, n_t, 0.0
    joint = np.bincount(t * (n_p + 1) + p, minlength=(n_t + 1) * (n_p + 1))
    joint = joint.reshape(n_t + 1, n_p + 1)
    inter = joint[1:, 1:]
    union = joint[1:].sum(1, keepdims=True) + joint[:, 1:].sum(0, keepdims=True) - inter
    iou = inter / union
    matched = iou > 0.5
    tp = int(matched.sum())
    return tp, n_p - tp, n_t - tp, float(iou[matched].sum())
