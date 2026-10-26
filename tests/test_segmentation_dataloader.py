import json

import numpy as np
import pytest

from pathotrain.dataloader import SegmentationDataset, find_pairs

ASAP = """<?xml version="1.0" ?><ASAP_Annotations><Annotations>
<Annotation Name="a" Type="Polygon" PartOfGroup="tumor"><Coordinates>
<Coordinate Order="0" X="0" Y="0"/><Coordinate Order="1" X="10" Y="0"/>
<Coordinate Order="2" X="10" Y="10"/></Coordinates></Annotation>
</Annotations></ASAP_Annotations>"""


def make_dir(root, slides, annotations):
    (root / "slides").mkdir(parents=True)
    (root / "annotations").mkdir()
    for name in slides:
        (root / "slides" / name).touch()
    for name in annotations:
        (root / "annotations" / name).write_text(ASAP)
    return root


def test_find_pairs_matches_by_stem_and_converts_xml(tmp_path):
    a = make_dir(tmp_path / "a", ["s1.tif", "s2.tif"], ["s1.xml", "s3.xml"])
    b = make_dir(tmp_path / "b", ["s4.svs"], ["s4.xml"])
    pairs = find_pairs([a, b], tmp_path / "geojson")
    assert [(s.name, g.name) for s, g in pairs] == [
        ("s1.tif", "s1.geojson"),
        ("s4.svs", "s4.geojson"),
    ]
    feature = json.loads(pairs[0][1].read_text())["features"][0]
    assert feature["properties"]["classification"]["name"] == "tumor"


def test_find_pairs_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_pairs([tmp_path])
    with pytest.raises(ValueError):
        find_pairs([make_dir(tmp_path / "a", ["s1.tif"], ["s2.xml"])])


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
