import numpy as np
import pytest

from pathotrain.pathosam import box, center_click, corrective_click
from pathotrain.pathosam.eval import tracker_metrics


def _square(y0, y1, x0, x1, size=32):
    mask = np.zeros((size, size), bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_box_is_exclusive_and_jitter_stays_within_bounds():
    mask = _square(4, 14, 8, 28)  # 20 wide, 10 high
    assert box(mask).tolist() == [8, 4, 28, 14]
    rng = np.random.default_rng(0)
    for _ in range(100):
        x0, y0, x1, y1 = box(mask, 0.1, rng)
        assert abs(x0 - 8) <= 2 and abs(x1 - 28) <= 2 and abs(y0 - 4) <= 1 and abs(y1 - 14) <= 1
    edge = box(_square(0, 5, 0, 5), 0.5, np.random.default_rng(1))
    assert edge.min() >= 0 and edge.max() <= 32  # clipped to the tile


def test_center_click_is_the_deepest_pixel_and_tile_edges_count_as_boundary():
    assert center_click(_square(10, 21, 4, 15)) == (9.0, 15.0)
    x, _ = center_click(_square(0, 32, 0, 3))  # a strip along the left edge
    assert x == 1.0


def test_corrective_click_goes_to_the_deeper_error():
    true = _square(4, 24, 4, 24)
    missed = true.copy()
    missed[4:24, 14:24] = False  # half missed, nothing extra
    assert corrective_click(true, missed)[2] == 1
    extra = true | _square(4, 24, 24, 32)  # a deep extra region, nothing missed
    x, _, label = corrective_click(true, extra)
    assert label == 0 and x >= 24
    small_miss = true.copy()
    small_miss[4, 4] = False  # a 1 px miss beside a deep extra: the extra wins
    assert corrective_click(true, small_miss | _square(4, 24, 24, 32))[2] == 0
    assert corrective_click(true, true) is None


def test_tracker_metrics_count_clicks_and_failures():
    curves = np.array(
        [
            [0.80, 0.86, 0.91],  # 85% at click 2, 90% at click 3
            [0.92, 0.95, 0.96],  # both at click 1
            [0.50, 0.60, 0.70],  # never: counts as 3 clicks
        ]
    )
    m = tracker_metrics({0.0: np.array([0.9, 0.8, 0.7])}, curves)
    assert m["iou_box@0.0"] == pytest.approx(0.8)
    assert m["iou@1"] == pytest.approx(0.74)
    assert m["noc@85"] == pytest.approx((2 + 1 + 3) / 3)
    assert m["noc@90"] == pytest.approx((3 + 1 + 3) / 3)
    assert m["fail@90"] == pytest.approx(1 / 3) and m["objects"] == 3


def test_example_config_is_valid():
    import inspect
    from importlib.resources import files

    from pathotrain.dataloader import InstanceTileFolderDataset
    from pathotrain.engine import load_config

    cfg = load_config(files("pathotrain.pathosam") / "configs" / "zeroshot.example.yaml")
    args = set(inspect.signature(InstanceTileFolderDataset.__init__).parameters)
    for split in ("val", "test"):
        assert set(cfg["data"][split]) <= args
    assert len(cfg["class_names"]) == cfg["num_classes"]
