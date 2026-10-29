import json

import numpy as np
import pytest
import torch

from pathotrain.dataloader import (
    DetectionDataset,
    DetectionTileFolderDataset,
    export_detection_tiles,
)
from pathotrain.dataloader import detection as detection_module
from pathotrain.dataloader.augmentation import eval_transform, train_transform


def _tile_with_objects():
    # Bright rectangles on a dark, non-square tile; targets as patchology gives them.
    tile = np.zeros((96, 128, 3), np.uint8)
    boxes = np.array([[10, 20, 30, 12], [70, 50, 20, 40]], np.float32)  # COCO x, y, w, h
    for x, y, w, h in boxes.astype(int):
        tile[y : y + h, x : x + w] = 255
    return tile, {"boxes": boxes, "labels": np.array([0, 1])}


def test_geometric_augmentation_moves_boxes_with_the_tile():
    tile, target = _tile_with_objects()
    no_norm = {"mean": (0, 0, 0), "std": (1, 1, 1)}
    transform = train_transform(n=0, boxes=True, **no_norm)
    torch.manual_seed(0)
    for _ in range(20):  # flips and 90 degree turns
        image, out = transform(tile, target)
        assert out["boxes"].dtype == torch.float32 and out["labels"].tolist() in ([0, 1], [1, 0])
        for x1, y1, x2, y2 in out["boxes"].round().int().tolist():
            # The box covers exactly its bright rectangle.
            assert (image[0, y1:y2, x1:x2] == 1).all()
            assert image[0].sum() == sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in out["boxes"])


def test_full_train_and_eval_transforms_keep_boxes_valid():
    tile, target = _tile_with_objects()
    torch.manual_seed(0)
    for transform in (train_transform(n=3, m=10, crop=64, boxes=True), eval_transform(boxes=True)):
        for _ in range(10):
            image, out = transform(tile, target)
            h, w = image.shape[-2:]
            b = out["boxes"]
            assert len(b) == len(out["labels"])
            assert (b[:, 0] >= 0).all() and (b[:, 2] <= w).all() and (b[:, 3] <= h).all()
            assert ((b[:, 2] - b[:, 0]) >= 1).all() and ((b[:, 3] - b[:, 1]) >= 1).all()
    empty = {"boxes": np.zeros((0, 4), np.float32), "labels": np.zeros(0, np.int64)}
    image, out = train_transform(boxes=True)(tile, empty)
    assert out["boxes"].shape == (0, 4) and out["labels"].shape == (0,)


def test_dataset_samples_are_xyxy_and_class_map_is_checked():
    tile, target = _tile_with_objects()

    class FakeGenerator:
        def __len__(self):
            return 1

        def get_tile(self, i):
            return (0, 0), tile, target

    ds = DetectionDataset.__new__(DetectionDataset)
    ds.transform, ds.generators, ds.offsets = None, [FakeGenerator()], np.array([0, 1])
    image, out = ds[0]
    assert image.shape == (3, 96, 128) and image.max() == 1
    assert out["boxes"].tolist() == [[10, 20, 40, 32], [70, 50, 90, 90]]
    assert out["labels"].dtype == torch.int64

    for bad in ({}, {"a": 1}, {"a": 0, "b": 2}):
        with pytest.raises(ValueError, match="0 .. num_classes - 1"):
            DetectionDataset([], [], class_map=bad)


class _FakeGenerator:
    """Stands in for patchology's DetectionTileGenerator: one tile with objects, one empty."""

    def __init__(self, slide, ground_truth, class_map, **kwargs):
        self.tile, self.target = _tile_with_objects()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def __len__(self):
        return 2

    def __iter__(self):
        empty = {"boxes": np.zeros((0, 4), np.float32), "labels": np.zeros(0, np.int64)}
        return iter([((0, 0), self.tile, self.target), ((256, 0), self.tile, empty)])


@pytest.fixture
def exported(tmp_path, monkeypatch):
    monkeypatch.setattr(detection_module, "DetectionTileGenerator", _FakeGenerator)
    (tmp_path / "slides").mkdir()
    (tmp_path / "slides/s1.tif").touch()
    (tmp_path / "ann").mkdir()
    (tmp_path / "ann/s1.geojson").touch()
    class_map = {"class01": 0, "class01b": 0, "class02": 1}
    return export_detection_tiles(
        [tmp_path / "slides"], [tmp_path / "ann"], tmp_path / "tiles", "train", class_map
    )


def test_export_writes_coco_and_folder_dataset_reads_it_back(exported):
    coco = json.loads((exported / "_annotations.coco.json").read_text())
    assert [i["file_name"] for i in coco["images"]] == ["s1_0_0.png", "s1_256_0.png"]
    expected = {"slide": "s1.tif", "x": 256, "y": 0, "width": 128, "height": 96}
    assert coco["images"][1].items() >= expected.items()
    assert [a["bbox"] for a in coco["annotations"]] == [[10, 20, 30, 12], [70, 50, 20, 40]]
    assert [a["category_id"] for a in coco["annotations"]] == [1, 2]
    assert coco["categories"][0] == {
        "id": 1,
        "name": "class01 / class01b",
        "supercategory": "class01 / class01b",
    }

    ds = DetectionTileFolderDataset(exported)
    assert len(ds) == 2 and ds.class_names == ["class01 / class01b", "class02"]
    image, target = ds[0]
    assert image.shape == (3, 96, 128) and image.max() == 1
    assert target["boxes"].tolist() == [[10, 20, 40, 32], [70, 50, 90, 90]]
    assert target["labels"].tolist() == [0, 1]
    assert ds[1][1]["boxes"].shape == (0, 4)
    image, target = DetectionTileFolderDataset(exported, eval_transform(boxes=True))[0]
    assert target["boxes"].tolist() == [[10, 20, 40, 32], [70, 50, 90, 90]]


def test_export_is_readable_by_rfdetr(exported):
    coco_module = pytest.importorskip("rfdetr.datasets.coco")
    rf = coco_module.CocoDetection(
        exported, exported / "_annotations.coco.json", transforms=None, remap_category_ids=True
    )
    _, target = rf[0]
    assert target["boxes"].tolist() == [[10, 20, 40, 32], [70, 50, 90, 90]]
    assert target["labels"].tolist() == [0, 1]
