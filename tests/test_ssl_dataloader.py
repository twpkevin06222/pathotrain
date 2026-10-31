import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import tv_tensors

from pathotrain.dataloader import (
    ClassTileDataset,
    MultiCropTransform,
    UnlabelledSlideDataset,
    UnlabelledTileFolderDataset,
    multicrop_collate,
)


def _tiles(d, n, size=64, seed=0):
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        Image.fromarray(rng.integers(0, 256, (size, size, 3), np.uint8)).save(d / f"t{i}.png")
    return d


class _FakeGenerator:
    """A TileGenerator stand-in: ``n`` tiles per slide, the slide's index in red."""

    def __init__(self, slide, tiling=None):
        self.n = int(slide.stem[1:]) + 1
        self.released = False

    def release_overviews(self):
        self.released = True

    def __len__(self):
        return self.n

    def get_tile(self, i):
        tile = np.zeros((32, 32, 3), np.uint8)
        tile[..., 0], tile[..., 1] = self.n, i
        tile.flags.writeable = False  # as from a reader
        return (i, 0), tile

    def close(self):
        pass


def test_unlabelled_slides_tile_every_slide(tmp_path, monkeypatch):
    monkeypatch.setattr(UnlabelledSlideDataset, "generator_cls", _FakeGenerator)
    for d, names in (("a", ["s0.tiff", "s2.tiff"]), ("b", ["s1.svs"])):
        (tmp_path / d).mkdir()
        for name in names:
            (tmp_path / d / name).touch()
    ds = UnlabelledSlideDataset([tmp_path / "a", tmp_path / "b"], tiling="kw passed through")
    assert len(ds) == 1 + 3 + 2 and all(g.released for g in ds.generators)
    tile = ds[3]  # slides in folder order, s0 (1 tile) then s2 (3): s2's last tile
    assert tile.shape == (3, 32, 32) and tile.dtype == torch.float32
    assert (tile[:2, 0, 0] * 255).round().tolist() == [3, 2]
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no slide"):
        UnlabelledSlideDataset([tmp_path / "empty"])


def test_multicrop_views_masks_and_collate(tmp_path):
    ds = UnlabelledTileFolderDataset(
        [_tiles(tmp_path / "t", 5)],
        MultiCropTransform(
            2, 3, global_size=28, local_size=14, mask=lambda: torch.ones(4, dtype=torch.bool)
        ),
    )
    views = ds[0]
    assert [v.shape for v in views["global"]] == [(3, 28, 28)] * 2
    assert [v.shape for v in views["local"]] == [(3, 14, 14)] * 3
    assert type(views["global"][0]) is torch.Tensor and views["global"][0].min() >= -1
    batch = next(iter(DataLoader(ds, batch_size=4, collate_fn=multicrop_collate)))
    assert batch["global"].shape == (2, 4, 3, 28, 28)
    assert batch["local"].shape == (3, 4, 3, 14, 14)
    assert batch["masks"].shape == (2, 4, 4) and batch["masks"].all()

    tile = tv_tensors.Image(torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8))
    assert set(MultiCropTransform(1, 0, global_size=28)(tile)) == {"global"}
    with pytest.raises(ValueError, match="probability"):
        MultiCropTransform(blur=2)


def test_class_tiles_from_folders_and_csv(tmp_path):
    _tiles(tmp_path / "train/tumour", 2)
    _tiles(tmp_path / "train/stroma", 3, seed=1)
    train = ClassTileDataset(tmp_path / "train")
    assert train.class_names == ["stroma", "tumour"] and len(train) == 5
    image, label = train[4]
    assert image.shape == (3, 64, 64) and label == 1

    (tmp_path / "val.csv").write_text("path,label\ntrain/tumour/t0.png,tumour\n")
    val = ClassTileDataset(csv=tmp_path / "val.csv", class_names=train.class_names)
    assert val.class_names == ["stroma", "tumour"] and val[0][1] == 1  # train's indices

    (tmp_path / "bad.csv").write_text("path,label\ntrain/tumour/t0.png,necrosis\n")
    with pytest.raises(ValueError, match="necrosis"):
        ClassTileDataset(csv=tmp_path / "bad.csv", class_names=train.class_names)
    (tmp_path / "cols.csv").write_text("file,class\nx.png,a\n")
    with pytest.raises(ValueError, match="path and label"):
        ClassTileDataset(csv=tmp_path / "cols.csv")
    with pytest.raises(ValueError, match="root"):
        ClassTileDataset()
