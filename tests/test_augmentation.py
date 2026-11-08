import io
import pickle

import cv2
import numpy as np
import pytest
import torch
from PIL import Image, ImageEnhance
from torchvision import tv_tensors

from pathotrain.dataloader.augmentation import (
    ENHANCE,
    HEDJitter,
    HSVJitter,
    RandomJPEG,
    eval_transform,
    hed_jitter,
    hed_to_rgb,
    hsv_jitter,
    rgb_to_hed,
    train_transform,
)

RNG = np.random.default_rng(0)
TILE = RNG.integers(40, 230, (64, 64, 3), dtype=np.uint8)
MASK = RNG.integers(0, 4, (64, 64), dtype=np.uint8)
IMAGE = tv_tensors.Image(torch.from_numpy(TILE).permute(2, 0, 1))


def as_hwc(img):
    return img.permute(1, 2, 0).numpy().astype(int)


def test_hed_round_trip():
    rgb = (hed_to_rgb(rgb_to_hed(IMAGE)).clamp(0, 1) * 255).round()
    assert (rgb - IMAGE.float()).abs().max() <= 1


def test_hed_jitter_matches_reference():
    # Reference formula (pathology-he-auto-augment) in numpy.
    sigma, bias = [0.1, -0.05, 0.02], [0.03, 0.01, -0.02]
    rgb_from_hed = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
    hed = -np.log(TILE / 255 + 2).reshape(-1, 3) @ np.linalg.inv(rgb_from_hed)
    hed = hed * (1 + np.array(sigma)) + np.array(bias)
    ref = np.clip(np.exp(-hed @ rgb_from_hed) - 2, 0, 1).reshape(TILE.shape) * 255
    assert np.abs(as_hwc(hed_jitter(IMAGE, sigma, bias)) - ref.astype(np.uint8)).max() <= 1


def test_hsv_jitter_matches_reference():
    # Reference HSV saturation and hue rule, via OpenCV.
    for hue, sat in [(0.1, 0.3), (-0.2, -0.4)]:
        hsv = cv2.cvtColor(TILE.astype(np.float32) / 255, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0] / 360 + hue) % 1.0 * 360
        s = hsv[..., 1]
        hsv[..., 1] = s * (1 + sat) if sat < 0 else s * (1 + (1 - s) * sat)
        ref = (np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1) * 255).astype(int)
        assert np.abs(as_hwc(hsv_jitter(IMAGE, hue, sat)) - ref).max() <= 2


def test_enhance_matches_pil():
    pil = {"Brightness": 1.7, "Color": 0.4, "Contrast": 1.5, "Sharpness": 2.0}
    for op, factor in pil.items():
        ref = np.asarray(getattr(ImageEnhance, op)(Image.fromarray(TILE)).enhance(factor))
        assert np.abs(as_hwc(ENHANCE[op](IMAGE, factor)) - ref).max() <= 2, op


def test_jitter_leaves_masks_alone():
    mask = tv_tensors.Mask(torch.from_numpy(MASK))
    for t in (HEDJitter(0.5, 0.5), HSVJitter(0.5, 0.5)):
        _, out = t(IMAGE, mask)
        assert torch.equal(out, mask)


def test_hed_cutoff_skips_glass():
    glass = tv_tensors.Image(torch.full((3, 8, 8), 250, dtype=torch.uint8))
    assert torch.equal(HEDJitter(0.5, 0.5)(glass), glass)


def test_train_transform_keeps_labels_and_tensors():
    t = train_transform(n=3, m=10, crop=48)
    for _ in range(30):
        image, mask = t(TILE, MASK)
        assert type(image) is torch.Tensor and type(mask) is torch.Tensor
        assert image.shape == (3, 48, 48) and image.dtype == torch.float32
        assert mask.shape == (48, 48) and mask.dtype == torch.int64
        # Nearest-neighbour masks: only original labels, plus the fill for uncovered pixels.
        assert set(mask.unique().tolist()) <= {0, 1, 2, 3, 255}
    pickle.loads(pickle.dumps(t))  # DataLoader workers


def test_eval_transform_is_deterministic_and_keeps_the_whole_tile():
    t = eval_transform()
    (a, ma), (b, mb) = t(TILE, MASK), t(TILE, MASK)
    assert torch.equal(a, b) and torch.equal(ma, mb)
    assert a.shape == (3, 64, 64) and torch.equal(ma, torch.from_numpy(MASK).long())


def test_scanner_extras_change_tile_not_mask():
    # Extras run after the flips, so the same seed gives the same geometry.
    torch.manual_seed(0)
    plain, plain_mask = train_transform(n=0)(TILE, MASK)
    torch.manual_seed(0)
    noisy, noisy_mask = train_transform(n=0, blur=1.0, noise=1.0, jpeg=1.0)(TILE, MASK)
    assert torch.equal(plain_mask, noisy_mask)
    assert not torch.equal(plain, noisy)


def test_scanner_extras_off_by_default():
    assert "RandomApply" not in repr(train_transform().compose)


def test_scanner_extras_reject_bad_probability():
    with pytest.raises(ValueError, match="probabilities"):
        train_transform(jpeg=1.5)


def test_jpeg_matches_pil():
    buffer = io.BytesIO()
    Image.fromarray(TILE).save(buffer, "JPEG", quality=60)
    ref = np.array(Image.open(buffer))
    assert np.array_equal(as_hwc(RandomJPEG((60, 60))(IMAGE)), ref)


def _sparse_tile():
    """128 px tile: a 10 px labelled patch in one corner, an ignored band in another."""
    tile = RNG.integers(40, 230, (128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128), np.uint8)
    mask[100:110, 100:110] = 1
    mask[:40, :] = 255
    return tile, mask


def test_foreground_crop_centres_on_labels_when_asked():
    tile, mask = _sparse_tile()
    torch.manual_seed(0)
    always = train_transform(n=0, crop=32, foreground_crop=1.0)
    never = train_transform(n=0, crop=32, foreground_crop=0.0)
    pickle.loads(pickle.dumps(always))  # DataLoader workers
    hits = [(always(tile, mask)[1] == 1).any().item() for _ in range(50)]
    assert all(hits)
    uniform = np.mean([(never(tile, mask)[1] == 1).any().item() for _ in range(200)])
    assert uniform < 0.3  # a uniform 32 px window rarely finds the 10 px patch
    # Ignored pixels are not foreground: without labels the crop falls back to uniform.
    only_ignored = np.where(mask == 255, 255, 0).astype(np.uint8)
    assert always(tile, only_ignored)[1].shape == (32, 32)


def test_foreground_crop_keeps_a_box_in_detection_crops():
    tile, _ = _sparse_tile()
    target = {"boxes": np.array([[100, 100, 10, 10]], np.float32), "labels": np.array([1])}
    t = train_transform(n=0, crop=32, foreground_crop=1.0, boxes=True)
    torch.manual_seed(0)
    for _ in range(30):
        image, out = t(tile, target)
        assert image.shape == (3, 32, 32) and out["labels"].tolist() == [1]
        assert (out["boxes"] >= 0).all() and (out["boxes"] <= 32).all()


def test_foreground_crop_validation():
    with pytest.raises(ValueError, match="needs crop"):
        train_transform(foreground_crop=0.3)
    with pytest.raises(ValueError, match="probability"):
        train_transform(crop=32, foreground_crop=1.5)


@pytest.mark.parametrize("foreground_crop", [0.0, 1.0])
def test_crop_pads_short_tiles_with_ignore_in_the_mask(foreground_crop):
    # 40 x 60 tile, 64 px crop: both sides are padded, the padding is never a label.
    tile = np.full((40, 60, 3), 200, np.uint8)
    mask = np.ones((40, 60), np.uint8)
    t = train_transform(n=0, crop=64, foreground_crop=foreground_crop, mean=(0,) * 3, std=(1,) * 3)
    torch.manual_seed(0)
    for _ in range(20):
        image, out = t(tile, mask)
        assert image.shape == (3, 64, 64) and out.shape == (64, 64)
        padded = out == 255
        assert set(out.unique().tolist()) <= {1, 255}
        assert (out == 1).sum() <= 40 * 60 and padded.sum() >= 64 * 64 - 40 * 60
        assert torch.allclose(image[:, padded], torch.tensor(128 / 255))  # fill grey
        assert torch.allclose(image[:, ~padded], torch.tensor(200 / 255))


def test_foreground_crop_pads_and_keeps_boxes_in_place():
    tile = np.zeros((40, 60, 3), np.uint8)
    tile[10:20, 30:40] = 255  # the box's pixels
    target = {"boxes": np.array([[30, 10, 10, 10]], np.float32), "labels": np.array([1])}
    t = train_transform(n=0, crop=64, foreground_crop=1.0, boxes=True, mean=(0,) * 3, std=(1,) * 3)
    torch.manual_seed(0)
    for _ in range(20):
        image, out = t(tile, target)
        x1, y1, x2, y2 = out["boxes"][0].round().int().tolist()
        assert out["labels"].tolist() == [1] and (x2 - x1, y2 - y1) == (10, 10)
        assert (image[:, y1:y2, x1:x2] == 1).all()  # the box still covers its pixels


def test_foreground_crop_is_refused_for_slide_training_splits():
    from pathotrain.experiment import check_foreground_crop

    folder = {"augmentation": {"foreground_crop": 0.3}, "data": {"train": {"root": "tiles"}}}
    check_foreground_crop(folder)
    check_foreground_crop({"data": {"train": {"slides": ["s"]}}})
    with pytest.raises(ValueError, match="tile folders"):
        check_foreground_crop({**folder, "data": {"train": {"slides": ["s"]}}})
