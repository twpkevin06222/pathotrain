import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

from pathotrain.dataloader import InstanceTileFolderDataset, instances_from_contours
from pathotrain.dataloader.augmentation import SegmentationTransform


def _two_touching_nuclei() -> np.ndarray:
    """Nuclei 1 and 2 share the border at x = 16; nucleus 3 stands alone."""
    inst = np.zeros((32, 32), np.int32)
    inst[4:14, 8:16] = 1
    inst[4:14, 16:24] = 2
    inst[20:28, 20:28] = 3
    return inst


def _write(root, inst_mask, types, name="a"):
    for folder, prefix, array in (
        ("images", "img_", np.full((*types.shape, 3), 128, np.uint8)),
        ("inst", "inst_", inst_mask.astype(np.uint8)),
        ("sem", "sem_", types.astype(np.uint8)),
    ):
        (root / folder).mkdir(exist_ok=True)
        Image.fromarray(array).save(root / folder / f"{prefix}{name}.png")
    return {
        "images": root / "images",
        "instances": root / "inst",
        "types": root / "sem",
        "image_prefix": "img_",
        "instance_prefix": "inst_",
        "type_prefix": "sem_",
    }


def test_contours_split_touching_nuclei_and_keep_outline_pixels():
    inst = _two_touching_nuclei()
    types = (inst > 0).astype(np.uint8) * 2
    contours = np.zeros_like(inst)
    contours[:, 16] = 1  # the outline between nuclei 1 and 2
    derived = instances_from_contours(contours, types)
    assert len(np.unique(derived)) == 4  # background and three nuclei
    assert (derived > 0).sum() == (inst > 0).sum()  # outline pixels go to a neighbour
    # Off the outline, each true nucleus is one derived instance, and they all differ.
    ids = [np.unique(derived[(inst == i) & (contours == 0)]) for i in (1, 2, 3)]
    assert all(len(i) == 1 for i in ids) and len({int(i[0]) for i in ids}) == 3
    assert instances_from_contours(contours, np.zeros_like(types)).max() == 0


def test_pairs_prefixed_folders_and_reads_instance_ids(tmp_path):
    inst = _two_touching_nuclei()
    data = _write(tmp_path, inst, (inst > 0) * 2)
    Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(tmp_path / "images" / "img_b.png")
    ds = InstanceTileFolderDataset(**data)  # b has no masks: skipped
    assert len(ds) == 1
    image, target = ds[0]
    assert image.shape == (3, 32, 32) and image.dtype == torch.float32
    assert set(target) == {"inst", "type"}  # no model-specific targets by default
    assert torch.equal(target["inst"], torch.from_numpy(inst).long())
    assert target["type"].max() == 2
    with pytest.raises(ValueError, match="no file stems"):
        InstanceTileFolderDataset(**{**data, "image_prefix": "x_"})


def test_contours_option_derives_instance_ids(tmp_path):
    inst = _two_touching_nuclei()
    contours = np.zeros_like(inst)
    contours[:, 16] = 1
    ds = InstanceTileFolderDataset(**_write(tmp_path, contours, (inst > 0) * 2), contours=True)
    assert len(ds[0][1]["inst"].unique()) == 4


def test_uncovered_pixels_are_ignored_and_target_transform_runs_after_augmentation(tmp_path):
    inst = _two_touching_nuclei()
    data = _write(tmp_path, inst, (inst > 0) * 2)
    shift = v2.RandomAffine(degrees=0, translate=(0.5, 0.5), fill={tv_tensors.Mask: 255})
    torch.manual_seed(0)
    seen = []

    def record(target):
        seen.append(target["inst"].clone())
        return {**target, "count": target["inst"].unique().numel()}

    ds = InstanceTileFolderDataset(
        transform=SegmentationTransform([shift]), target_transform=record, **data
    )
    _, target = ds[0]
    ignored = target["type"] == 255
    assert ignored.any() and (target["inst"][ignored] == 0).all()
    assert (
        torch.equal(seen[0], target["inst"]) and target["count"] == target["inst"].unique().numel()
    )
