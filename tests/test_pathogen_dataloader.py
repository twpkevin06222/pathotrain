import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import tv_tensors

from pathotrain.dataloader import PairedTileDataset, paired_transform


def _tiles(root, n=6, size=32):
    rng = np.random.default_rng(0)
    for side in ("HE", "IHC"):
        (root / side).mkdir(parents=True)
    for i in range(n):
        image = rng.integers(0, 256, (size, size, 3), np.uint8)
        Image.fromarray(image).save(root / "HE" / f"t{i}.png")
        Image.fromarray(255 - image).save(root / "IHC" / f"t{i}.png")
    Image.fromarray(image).save(root / "HE" / "unpaired.png")
    return root


def test_pairs_tiles_by_stem_and_transforms_them_together(tmp_path):
    root = _tiles(tmp_path)
    same = PairedTileDataset([root / "HE"], [root / "HE"], paired_transform(True, crop=16))
    assert len(same) == 7
    for _ in range(5):  # same random crop, flip and turn on both sides
        source, target = same[0]
        assert source.shape == (3, 16, 16) and torch.equal(source, target)
    paired = PairedTileDataset([root / "HE"], [root / "IHC"])
    assert len(paired) == 6  # unpaired.png skipped
    source, target = paired[0]
    assert source.dtype == torch.float32 and source.min() >= 0 and source.max() <= 1
    assert torch.allclose(source, 1 - target, atol=1e-6)  # the IHC tile is 255 - HE
    with pytest.raises(ValueError, match="pairs"):
        PairedTileDataset([root / "HE"], [tmp_path])


def test_eval_transform_crops_the_centre_of_both():
    source = torch.arange(36, dtype=torch.uint8).view(1, 6, 6).expand(3, 6, 6)
    a, b = paired_transform(False, crop=2)(tv_tensors.Image(source), tv_tensors.Image(source))
    assert torch.equal(a, source[:, 2:4, 2:4]) and torch.equal(b, a)
