"""Register a non-H&E slide (e.g. IHC) to its H&E slide with one affine transform::

    python -m pathotrain.registration.affine HE.tiff IHC.tiff out.json --overlay out.jpg

The H&E is the fixed slide, the other the moving one. Both are read at ``mpp``
(16 um per pixel by default) and turned into stain-independent tissue images:
optical density, blurred, kept on tissue only and rank-equalised to ``(0, 1]``,
so a pale counterstain and a dark H&E look alike. Then:

1. Rigid search at ``4 * mpp``: both flips, rotations every ``rotation_step``
   degrees, translations by FFT cross-correlation (on the GPU when present).
2. ECC affine refinement of the ``top_k`` best poses at ``2 * mpp``.
3. The pose whose warped moving image has the highest normalised
   cross-correlation (NCC) with the fixed image wins, and is refined again at
   ``mpp``.

Below ``min_ncc`` (0.55) :func:`register` logs a warning: on ACROBAT validation
every misaligned pair scored below it (as did 7 others), so check those
with :func:`overlay`. The transform is affine (rotation, flips, scale,
shear), so tissue that tore or folded differently stays off by its local
deformation. It works at 16 um per pixel: enough to pair tiles over the same
tissue, not to match cells.

The result maps fixed level-0 pixels to moving level-0 pixels and is saved as
JSON; :func:`pathotrain.dataloader.registration.export_registered_tiles` tiles
the H&E and reads each matching moving tile through it.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger

from patchology import open_slide
from patchology.reader import SlideReader


@dataclass
class Registration:
    """An affine map from the fixed slide to the moving slide.

    Attributes:
        fixed: Fixed slide path (the H&E).
        moving: Moving slide path (e.g. the IHC).
        matrix: 2 x 3 affine matrix taking fixed level-0 ``(x, y)`` to moving level-0.
        ncc: Normalised cross-correlation of the aligned tissue images, -1 to 1.
    """

    fixed: str
    moving: str
    matrix: list[list[float]]
    ncc: float

    def to_moving(self, xy: np.ndarray) -> np.ndarray:
        """Map ``(N, 2)`` fixed level-0 points to moving level-0."""
        xy = np.asarray(xy, np.float64)
        a = np.asarray(self.matrix)
        return xy @ a[:, :2].T + a[:, 2]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Registration":
        return cls(**json.loads(Path(path).read_text()))


def tissue_image(rgb: np.ndarray) -> np.ndarray:
    """Stain-independent float32 image of an ``(H, W, 3)`` uint8 slide thumbnail:
    rank-equalised optical density on tissue, 0 on glass, labels and slide edges."""
    x = rgb.astype(np.float32)
    background = np.percentile(x.reshape(-1, 3), 90, axis=0)
    od = -np.log(np.clip(x / background, 1e-3, 1)).max(-1)
    od = cv2.GaussianBlur(od, (0, 0), 2)
    # Glass is ~0.002 OD with ~0.004 noise; even a pale counterstain is > 0.05.
    tissue = (od > 0.03) & (x.mean(-1) > 60)  # dark: labels, marker, slide edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tissue = cv2.morphologyEx(tissue.astype(np.uint8), cv2.MORPH_OPEN, kernel)  # thin lines
    n, labels, stats, _ = cv2.connectedComponentsWithStats(tissue)
    # Drop components lying mostly in a band along the border: slide and coverslip edges.
    h, w = tissue.shape
    b = max(3, int(0.04 * min(h, w)))
    band = np.ones((h, w), bool)
    band[b:-b, b:-b] = False
    in_band = np.bincount(labels[band], minlength=n)
    keep = (stats[:, cv2.CC_STAT_AREA] >= 50) & (in_band < 0.5 * stats[:, cv2.CC_STAT_AREA])
    keep[0] = False
    tissue = keep[labels]
    out = np.zeros_like(od)
    if tissue.any():
        values = od[tissue]
        out[tissue] = (np.argsort(np.argsort(values)) + 1) / len(values)
    return out


def _thumbnail(slide: SlideReader, mpp: float) -> tuple[np.ndarray, np.ndarray]:
    """The slide at ``mpp`` and the diagonal scale from its pixels to level-0 pixels."""
    if slide.mpp is None:
        raise ValueError(f"{slide.path} reports no microns per pixel")
    rgb = slide.read_at_downsample(mpp / slide.mpp[0])
    width, height = slide.dimensions
    return rgb, np.diag([width / rgb.shape[1], height / rgb.shape[0], 1.0])


def _rotation(shape: tuple[int, int], degrees: float, flip: bool, canvas: int) -> np.ndarray:
    """3 x 3 map from an image of ``shape`` to it flipped, turned about its centre and
    centred on a ``canvas`` square."""
    h, w = shape
    r = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0), [0, 0, 1]])
    if flip:
        r = r @ np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]])
    r[:2, 2] += (canvas / 2 - w / 2, canvas / 2 - h / 2)
    return r


@torch.no_grad()
def rigid_search(
    fixed: np.ndarray,
    moving: np.ndarray,
    factor: int = 4,
    rotation_step: float = 4.0,
    device: str | torch.device | None = None,
) -> list[tuple[float, np.ndarray]]:
    """Score every flip and rotation of ``moving`` at its best translation onto
    ``fixed``, both downsampled ``factor`` times.

    Returns ``(score, A)`` per pose, ``A`` the 2 x 3 map from ``fixed`` pixels to
    ``moving`` pixels (at the input resolution).
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def down(a: np.ndarray) -> np.ndarray:
        size = (max(1, a.shape[1] // factor), max(1, a.shape[0] // factor))
        return cv2.resize(
            cv2.GaussianBlur(a, (0, 0), factor / 2), size, interpolation=cv2.INTER_AREA
        )

    f, m = down(fixed), down(moving)
    canvas = int(np.ceil(np.hypot(*m.shape)))
    poses = [
        _rotation(m.shape, deg, flip, canvas)
        for flip in (False, True)
        for deg in np.arange(0, 360, rotation_step)
    ]
    size = (f.shape[0] + canvas, f.shape[1] + canvas)  # pad: linear, not circular, correlation
    ft = torch.from_numpy(f).to(device)
    f_spectrum = torch.fft.rfft2(ft, size)
    f_norm = ft.square().sum().sqrt()
    scale = np.diag([factor, factor, 1.0])
    out = []
    for start in range(0, len(poses), 16):
        batch = poses[start : start + 16]
        mt = torch.from_numpy(
            np.stack([cv2.warpAffine(m, r[:2], (canvas, canvas)) for r in batch])
        ).to(device)
        corr = torch.fft.irfft2(f_spectrum * torch.fft.rfft2(mt, size).conj(), size)
        corr /= (mt.square().sum((1, 2)).sqrt() * f_norm)[:, None, None] + 1e-9
        best, index = corr.flatten(1).max(1)
        for r, score, k in zip(batch, best.tolist(), index.tolist(), strict=True):
            dy, dx = np.unravel_index(k, size)
            dy -= size[0] if dy > f.shape[0] else 0
            dx -= size[1] if dx > f.shape[1] else 0
            to_fixed = r.copy()
            to_fixed[:2, 2] += (dx, dy)
            out.append((score, (scale @ np.linalg.inv(to_fixed) @ np.linalg.inv(scale))[:2]))
    return out


def _half(a: np.ndarray) -> np.ndarray:
    return cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2), interpolation=cv2.INTER_AREA)


def _rescale(a: np.ndarray, factor: float) -> np.ndarray:
    """``a`` (fixed pixels to moving pixels) for both images resized by ``factor``."""
    out = a.copy()
    out[:, 2] *= factor
    return out


def refine(fixed: np.ndarray, moving: np.ndarray, a: np.ndarray) -> np.ndarray | None:
    """ECC affine refinement of ``a`` (fixed pixels to moving pixels); None if it fails."""
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        _, out = cv2.findTransformECC(
            cv2.GaussianBlur(fixed, (0, 0), 2),
            cv2.GaussianBlur(moving, (0, 0), 2),
            a.astype(np.float32),
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5,
        )
    except cv2.error:
        return None
    return out.astype(np.float64)


def ncc(fixed: np.ndarray, moving: np.ndarray, a: np.ndarray) -> float:
    """Normalised cross-correlation of ``fixed`` and ``moving`` warped onto it by ``a``."""
    warped = cv2.warpAffine(moving, a, fixed.shape[::-1], flags=cv2.WARP_INVERSE_MAP)
    x = cv2.GaussianBlur(fixed, (0, 0), 4)
    y = cv2.GaussianBlur(warped, (0, 0), 4)
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))


def register(
    fixed: str | Path,
    moving: str | Path,
    mpp: float = 16.0,
    rotation_step: float = 4.0,
    top_k: int = 5,
    min_ncc: float = 0.55,
    device: str | torch.device | None = None,
) -> Registration:
    """Register ``moving`` (e.g. IHC) to ``fixed`` (the H&E); see the module docstring."""
    with open_slide(fixed) as f_slide, open_slide(moving) as m_slide:
        f_rgb, f_scale = _thumbnail(f_slide, mpp)
        m_rgb, m_scale = _thumbnail(m_slide, mpp)
    f_img, m_img = tissue_image(f_rgb), tissue_image(m_rgb)
    poses = sorted(rigid_search(f_img, m_img, 4, rotation_step, device), key=lambda p: -p[0])
    # ECC dominates the run time: compare the poses at half resolution, refine the winner.
    f_half, m_half = _half(f_img), _half(m_img)
    candidates = []
    for _, a in poses[:top_k]:
        a = _rescale(a, 0.5)
        candidates.append(a)
        refined = refine(f_half, m_half, a)
        if refined is not None and 0.67 < np.sqrt(abs(np.linalg.det(refined[:, :2]))) < 1.5:
            candidates.append(refined)
    best = _rescale(max(candidates, key=lambda a: ncc(f_half, m_half, a)), 2)
    refined = refine(f_img, m_img, best)
    score = ncc(f_img, m_img, best)
    if refined is not None and (refined_score := ncc(f_img, m_img, refined)) > score:
        best, score = refined, refined_score
    level0 = m_scale @ np.vstack([best, [0, 0, 1]]) @ np.linalg.inv(f_scale)
    result = Registration(str(fixed), str(moving), level0[:2].tolist(), score)
    log = logger.warning if result.ncc < min_ncc else logger.info
    log(f"registered {Path(moving).name} to {Path(fixed).name}: NCC {result.ncc:.3f}")
    return result


def overlay(registration: Registration, mpp: float = 16.0) -> np.ndarray:
    """An RGB check image at ``mpp``: the fixed tissue in green, the warped moving
    tissue in magenta, white where they overlap."""
    with open_slide(registration.fixed) as f_slide, open_slide(registration.moving) as m_slide:
        f_rgb, f_scale = _thumbnail(f_slide, mpp)
        m_rgb, m_scale = _thumbnail(m_slide, mpp)
    level0 = np.vstack([registration.matrix, [0, 0, 1]])
    a = (np.linalg.inv(m_scale) @ level0 @ f_scale)[:2]
    f_img = tissue_image(f_rgb)
    warped = cv2.warpAffine(tissue_image(m_rgb), a, f_img.shape[::-1], flags=cv2.WARP_INVERSE_MAP)
    return (np.dstack([warped, f_img, warped]) * 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("fixed", help="the H&E slide")
    parser.add_argument("moving", help="the slide to align to it, e.g. IHC")
    parser.add_argument("output", help="JSON file to write the registration to")
    parser.add_argument("--overlay", help="also write a check image (.jpg / .png) here")
    parser.add_argument("--mpp", type=float, default=16.0, help="registration resolution")
    args = parser.parse_args(argv)
    result = register(args.fixed, args.moving, args.mpp)
    result.save(args.output)
    if args.overlay:
        cv2.imwrite(args.overlay, overlay(result, args.mpp)[:, :, ::-1])
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
