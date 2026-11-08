import numpy as np
import pytest
import torch

from pathotrain.dataloader import sampling
from pathotrain.dataloader.sampling import build_sampler, tile_weights

# 9 tiles of a common class, 1 of a rare one, 2 without labels.
CONTENT = np.array([[1.0, 0.0]] * 9 + [[0.0, 1.0]] + [[0.0, 0.0]] * 2)


def test_rare_class_tiles_are_upweighted_by_damped_inverse_frequency():
    w = tile_weights(CONTENT[:10], power=0.5)
    assert w[9] / w[0] == pytest.approx(3.0)  # (0.1 / 0.9) ** -0.5
    full = tile_weights(CONTENT[:10], power=1.0)
    assert full[9] / full[0] == pytest.approx(9.0)
    capped = tile_weights(CONTENT[:10], power=1.0, max_ratio=5)
    assert capped[9] / capped[0] == pytest.approx(5.0)
    assert np.ptp(tile_weights(CONTENT[:10], power=0)) == 0  # power 0: unchanged
    assert np.ptp(tile_weights(np.ones((4, 1)))) == 0  # one class: uniform


def test_mixed_tiles_average_their_class_weights():
    w = tile_weights(np.array([[1.0, 0.0]] * 9 + [[0.0, 1.0], [0.5, 0.5]]), power=1.0)
    assert w[10] == pytest.approx((w[0] + w[9]) / 2)


def test_background_sets_the_share_of_tiles_without_labels():
    empty = CONTENT.sum(axis=1) == 0
    share = lambda w: w[empty].sum() / w.sum()
    assert share(tile_weights(CONTENT)) == pytest.approx(empty.mean())  # None keeps theirs
    assert share(tile_weights(CONTENT, background=0.3)) == pytest.approx(0.3)
    assert share(tile_weights(CONTENT, background=0.0)) == 0
    assert np.ptp(tile_weights(np.zeros((3, 2)))) == 0  # nothing labelled: uniform


@pytest.mark.parametrize(
    "kwargs",
    [{"content": np.ones(3)}, {"power": -1}, {"max_ratio": 0.5}, {"background": 1.0}],
)
def test_tile_weights_validation(kwargs):
    with pytest.raises(ValueError):
        tile_weights(**{"content": CONTENT, **kwargs})


def test_sampler_draws_are_seeded_and_favour_the_rare_class(monkeypatch):
    cfg = {"sampling": {"power": 1.0, "num_samples": 4000}, "data": {"train": {}}}
    torch.manual_seed(0)
    draws = list(build_sampler(cfg, CONTENT))
    torch.manual_seed(0)
    assert list(build_sampler(cfg, CONTENT)) == draws and len(draws) == 4000
    rare = np.mean(np.array(draws) == 9)
    assert rare > 0.3  # 1 of 12 tiles, about half of the labelled draws after weighting

    warnings = []
    monkeypatch.setattr(sampling.logger, "warning", lambda msg, *a: warnings.append(msg))
    build_sampler({"sampling": {}, "data": {"train": {"negative_sampling": 0.2}}}, CONTENT)
    assert not warnings  # background left at None: no double counting
    both = {"sampling": {"background": 0.1}, "data": {"negative_sampling": 0.2, "train": {}}}
    build_sampler(both, CONTENT)
    assert len(warnings) == 1 and "negative_sampling" in warnings[0]
