import cv2
import numpy as np

from pathotrain.dataloader.registration import read_registered
from pathotrain.registration.affine import (
    Registration,
    ncc,
    refine,
    rigid_search,
    tissue_image,
    tissue_pixels,
)


def _tissue(seed=0, shape=(300, 400)):
    """A pink, textured, asymmetric blob on white glass, like an H&E thumbnail."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(mask, (150, 150), (90, 60), 20, 0, 360, 1, -1)
    cv2.circle(mask, (260, 200), 40, 1, -1)
    cv2.rectangle(mask, (80, 220), (130, 260), 1, -1)
    texture = cv2.GaussianBlur(rng.random(shape).astype(np.float32), (0, 0), 3)
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    rgb = np.full((*shape, 3), 235, np.float32)
    stain = np.array([0.35, 0.75, 0.3]) * (0.3 + texture[..., None])  # OD per channel
    rgb = np.where(mask[..., None] > 0, rgb * np.exp(-stain), rgb)
    return rgb.clip(0, 255).astype(np.uint8)


def test_tissue_image_ignores_glass_and_slide_edges():
    rgb = _tissue()
    rgb[:, :4] = 30  # dark slide edge
    rgb[2:5, :] = 200  # coverslip line along the border
    img = tissue_image(rgb)
    assert img[150, 150] > 0 and img[20, 300] == 0
    assert img[:, :4].max() == 0 and img[2:5, 200:].max() == 0
    assert 0 < img.max() <= 1


def test_search_and_refine_recover_a_flipped_turned_shifted_pair():
    rgb = _tissue()
    # moving = fixed mirrored, turned 150 degrees and shifted: m(A_true @ p) = f(p)
    to_moving = cv2.getRotationMatrix2D((200, 150), 150, 1.0) @ np.array(
        [[-1, 0, 399], [0, 1, 0], [0, 0, 1]]
    )
    to_moving[:, 2] += (30, -10)
    moving = cv2.warpAffine(rgb, to_moving, (400, 300), borderValue=(235, 235, 235))
    f, m = tissue_image(rgb), tissue_image(moving)
    poses = sorted(
        rigid_search(f, m, factor=2, rotation_step=10, device="cpu"), key=lambda p: -p[0]
    )
    candidates = [a for _, a in poses[:3]]
    candidates += [r for r in (refine(f, m, a) for a in candidates) if r is not None]
    best = max(candidates, key=lambda a: ncc(f, m, a))
    assert ncc(f, m, best) > 0.9
    points = np.array([[150, 150, 1], [260, 200, 1], [100, 240, 1]], float)
    error = np.linalg.norm(points @ best.T - points @ to_moving.T, axis=1)
    assert error.max() < 3, error


class _ArraySlide:
    """The read_region_scaled of a SlideReader over an in-memory level-0 image."""

    def __init__(self, image):
        self.image = image

    def read_region_scaled(self, location, downsample, size):
        x, y = location
        w, h = round(size[0] * downsample), round(size[1] * downsample)
        out = np.full((h, w, 3), 255, np.uint8)
        H, W = self.image.shape[:2]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x1 > x0 and y1 > y0:
            out[y0 - y : y1 - y, x0 - x : x1 - x] = self.image[y0:y1, x0:x1]
        return cv2.resize(out, size, interpolation=cv2.INTER_AREA)


def test_read_registered_warps_the_moving_slide_onto_the_fixed_tile():
    fixed = _tissue(1, (600, 800))
    moving = np.ascontiguousarray(np.rot90(fixed))  # (800, 600): fixed (x, y) -> (y, 799 - x)
    reg = Registration("f", "m", [[0, 1, 0], [-1, 0, 799]], 1.0)
    location, downsample, size = (200, 100), 2.0, 128
    tile = read_registered(_ArraySlide(moving), reg, location, downsample, size)
    expected = _ArraySlide(fixed).read_region_scaled(location, downsample, (size, size))
    assert tile.shape == (size, size, 3)
    diff = np.abs(tile[4:-4, 4:-4].astype(int) - expected[4:-4, 4:-4].astype(int))
    assert diff.mean() < 3, diff.mean()


def test_registration_round_trips_and_maps_points(tmp_path):
    reg = Registration("he.tiff", "ihc.tiff", [[0, -1, 100], [1, 0, 5]], 0.8)
    reg.save(tmp_path / "r.json")
    loaded = Registration.load(tmp_path / "r.json")
    assert loaded == reg
    np.testing.assert_allclose(loaded.to_moving([[10, 20]]), [[80, 15]])


def test_tissue_pixels_measures_a_tile_against_the_slide_glass():
    glass = np.array([233.0, 234.0, 236.0])
    blank = np.broadcast_to(glass.astype(np.uint8), (64, 64, 3)).copy()
    pale = (glass * np.exp(-np.array([0.11, 0.08, 0.06]))).astype(np.uint8)  # pale counterstain
    half = blank.copy()
    half[:, 32:] = pale
    off_slide = np.full((64, 64, 3), 255, np.uint8)  # read_registered fills with white
    assert tissue_pixels(blank, glass)[1].mean() == 0
    assert tissue_pixels(off_slide, glass)[1].mean() == 0
    assert 0.4 < tissue_pixels(half, glass)[1].mean() < 0.6


def test_pair_slides_by_key(tmp_path):
    import pytest

    from pathotrain.registration.batch import pair_slides

    he, ihc = tmp_path / "he", tmp_path / "ihc"
    he.mkdir()
    ihc.mkdir()
    for name in ("0_HE_val.tiff", "10_HE_val.tiff", "7_HE_val.tiff"):
        (he / name).touch()
    for name in ("0_KI67_val.tiff", "10_ER_val.tiff", "3_PGR_val.tiff"):
        (ihc / name).touch()
    pairs = pair_slides(he, ihc)
    assert [(f.name, m.name) for f, m in pairs] == [
        ("0_HE_val.tiff", "0_KI67_val.tiff"),
        ("10_HE_val.tiff", "10_ER_val.tiff"),
    ]
    (ihc / "0_ER_val.tiff").touch()
    with pytest.raises(ValueError, match="share key"):
        pair_slides(he, ihc)
