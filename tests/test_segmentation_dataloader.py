import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from pathotrain.dataloader import SegmentationDataset, SegmentationTileFolderDataset, find_pairs


def touch(d, *names):
    d.mkdir(parents=True, exist_ok=True)
    for name in names:
        (d / name).touch()
    return d


def test_find_pairs_matches_by_stem_across_dirs(tmp_path):
    s1 = touch(tmp_path / "a/slides", "s1.tif", "s2.tif")
    s2 = touch(tmp_path / "b/slides", "s4.svs")
    # s4's annotation sits in the other annotation dir; XML is not an annotation.
    a1 = touch(tmp_path / "a/geojson", "s1.geojson", "s3.geojson", "s4.geojson", "s2.xml")
    a2 = touch(tmp_path / "b/geojson")
    pairs = find_pairs([s1, s2], [a1, a2])
    assert [(s.name, g.name) for s, g in pairs] == [
        ("s1.tif", "s1.geojson"),
        ("s4.svs", "s4.geojson"),
    ]


def test_find_pairs_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_pairs([tmp_path / "missing"], [tmp_path])
    slides = touch(tmp_path / "slides", "s1.tif")
    with pytest.raises(ValueError, match="matches"):
        find_pairs([slides], [touch(tmp_path / "ann", "s2.geojson")])
    with pytest.raises(ValueError, match="both"):
        find_pairs([slides, touch(tmp_path / "more", "s1.svs")], [tmp_path / "ann"])


def test_locate_spans_slides():
    ds = SegmentationDataset.__new__(SegmentationDataset)
    ds.generators = ["g0", "g1", "g2"]
    ds.offsets = np.cumsum([0, 2, 0, 3])
    assert len(ds) == 5
    assert [ds.locate(i) for i in range(5)] == [
        ("g0", 0),
        ("g0", 1),
        ("g2", 0),
        ("g2", 1),
        ("g2", 2),
    ]
    assert ds.locate(-1) == ("g2", 2)
    with pytest.raises(IndexError):
        ds.locate(5)


def test_datasets_are_torch_datasets():
    assert issubclass(SegmentationDataset, Dataset)
    assert issubclass(SegmentationTileFolderDataset, Dataset)


def test_tile_folder_pairs_image_and_mask_dirs_by_stem(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for name in ("a", "b"):
        Image.fromarray(np.full((4, 6, 3), 200, np.uint8)).save(images / f"{name}.jpg")
    for name in ("a", "c"):  # b has no mask, c no image
        Image.fromarray(np.full((4, 6), 2, np.uint8)).save(masks / f"{name}.png")

    ds = SegmentationTileFolderDataset(images=images, masks=masks)
    assert ds.pairs == [(images / "a.jpg", masks / "a.png")]
    image, mask = ds[0]
    assert image.shape == (3, 4, 6) and image.dtype == torch.float32
    assert mask.shape == (4, 6) and mask.dtype == torch.int64 and mask.max() == 2

    for bad in ({}, {"root": tmp_path, "images": images}, {"images": images}):
        with pytest.raises(ValueError, match="either root"):
            SegmentationTileFolderDataset(**bad)
    with pytest.raises(ValueError, match="no image"):
        SegmentationTileFolderDataset(images=images, masks=touch(tmp_path / "empty"))
