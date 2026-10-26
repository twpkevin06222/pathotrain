import numpy as np
import pytest
from torch.utils.data import Dataset

from pathotrain.dataloader import SegmentationDataset, TileFolderDataset, find_pairs


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
    assert issubclass(TileFolderDataset, Dataset)
