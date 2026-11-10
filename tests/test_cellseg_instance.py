import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Dinov2Config, Dinov2Model

from pathotrain.cellseg import CellDataset, CellInstanceTask, PanopticQuality, build_model, classify
from pathotrain.cellseg.classify import cell_tokens, classify_cells
from pathotrain.cellseg.instances import (
    hv_maps,
    instance_types,
    instances_from_contours,
    match,
    pair_centroids,
    postprocess,
)
from pathotrain.segmentation.encoders import token_grids

TINY_VIT = Dinov2Config(
    hidden_size=32, num_hidden_layers=4, num_attention_heads=4, patch_size=14, image_size=224
)


def _touching_nuclei() -> np.ndarray:
    """Three nuclei on a 64 px tile, two of them touching."""
    inst = np.zeros((64, 64), np.int32)
    yy, xx = np.mgrid[:64, :64]
    inst[(yy - 20) ** 2 + (xx - 18) ** 2 < 81] = 1
    inst[(yy - 20) ** 2 + (xx - 34) ** 2 < 81] = 2  # touches 1 along x = 26
    inst[(yy - 46) ** 2 + (xx - 40) ** 2 < 64] = 3
    return inst


def test_contours_split_touching_nuclei():
    inst = _touching_nuclei()
    types = (inst > 0).astype(np.uint8) * 2
    contours = np.zeros_like(inst)
    contours[:, 26] = 1  # the outline between nuclei 1 and 2
    derived = instances_from_contours(contours, types)
    assert len(np.unique(derived)) == 4  # background and three nuclei
    assert match(inst, derived)[0] == 3
    assert (derived > 0).sum() == (inst > 0).sum()  # outline pixels are kept


def test_hv_maps_point_away_from_each_centre():
    hv = hv_maps(_touching_nuclei())
    assert hv.shape == (2, 64, 64) and hv.min() == -1 and hv.max() == 1
    assert hv[0, 20, 12] < 0 < hv[0, 20, 24]  # left and right of nucleus 1's centre
    assert hv[1, 14, 18] < 0 < hv[1, 26, 18]  # above and below it
    assert (hv[:, _touching_nuclei() == 0] == 0).all()


def test_postprocess_separates_touching_nuclei_from_ideal_outputs():
    inst = _touching_nuclei()
    pred = postprocess((inst > 0).astype(np.float32), hv_maps(inst))
    tp, fp, fn, iou = match(inst, pred)
    assert (tp, fp, fn) == (3, 0, 0) and iou / tp > 0.9


def test_match_counts_and_unmatched():
    true = _touching_nuclei()
    pred = np.where(true == 3, 0, true)  # nucleus 3 missed
    pred[60:, 60:] = 9  # a false nucleus
    assert match(true, pred)[:3] == (2, 1, 1)
    assert match(true, np.zeros_like(true)) == (0, 0, 3, 0.0)


def test_panoptic_quality_is_one_for_perfect_predictions_and_types():
    inst = _touching_nuclei()
    types = instance_types(inst, np.choose(inst, [0, 1, 2, 2]), 3)
    assert types.tolist() == [0, 1, 2, 2]
    pq = PanopticQuality(3)
    pq.update(inst, types, inst, types)
    out = pq.compute()
    assert out["bpq"] == 1 and out["mpq"] == 1 and out["sq"] == 1
    wrong = types.copy()
    wrong[1] = 2  # right instance, wrong type
    pq.reset()
    pq.update(inst, types, inst, wrong)
    assert pq.compute()["bpq"] == 1 and pq.compute()["mpq"] < 1


def _cellvit(**kwargs):
    torch.manual_seed(0)
    model = Dinov2Model(TINY_VIT)
    return build_model("cellvit", 3, backbone=model, skip_dims=(16, 8), bottleneck_dim=16, **kwargs)


@pytest.mark.parametrize(
    ("arch", "kwargs"),
    [
        ("hovernet", {"encoder_weights": None}),
        ("hovernext", {"encoder": "convnextv2_atto", "encoder_weights": None}),
        ("cellvit", None),
    ],
)
def test_instance_models_return_the_three_branches(arch, kwargs):
    if kwargs is None:
        model = _cellvit(instance=True)
    else:
        model = build_model(arch, 3, instance=True, **kwargs)
    with torch.no_grad():
        out = model.eval()(torch.rand(2, 3, 64, 64))
    assert {k: v.shape[1] for k, v in out.items()} == {"np": 2, "hv": 2, "nt": 3}
    assert all(v.shape[-2:] == (64, 64) for v in out.values())


def _write_pannuke_tile(root, name="a"):
    inst = _touching_nuclei()
    contours = np.zeros_like(inst, np.uint8)
    contours[:, 26] = 1
    for folder, prefix, array in (
        ("images", "img_", np.full((64, 64, 3), 128, np.uint8)),
        ("inst", "inst_", contours),
        ("sem", "sem_", np.choose(inst, [0, 1, 2, 2]).astype(np.uint8)),
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
        "contours": True,
    }


def test_cell_dataset_pairs_prefixed_folders_and_builds_targets(tmp_path):
    ds = CellDataset(**_write_pannuke_tile(tmp_path))
    image, target = ds[0]
    assert image.shape == (3, 64, 64)
    assert len(target["inst"].unique()) == 4 and target["type"].max() == 2
    assert target["hv"].shape == (2, 64, 64)


def test_instance_task_trains_and_scores_pq():
    inst = torch.from_numpy(_touching_nuclei()).long()[None].repeat(2, 1, 1)
    types = (
        torch.from_numpy(np.choose(_touching_nuclei(), [0, 1, 2, 2])).long()[None].repeat(2, 1, 1)
    )
    hv = torch.from_numpy(hv_maps(_touching_nuclei()))[None].repeat(2, 1, 1, 1)
    batch = (torch.rand(2, 3, 64, 64), {"inst": inst, "type": types, "hv": hv})
    task = CellInstanceTask(_cellvit(instance=True), 3, class_names=["bg", "a", "b"])
    loss, logged = task.training_step(batch)
    loss.backward()
    assert set(logged) == {"loss", "np", "hv_mse", "hv_msge", "nt"}
    with torch.no_grad():
        task.validation_step(batch)
    metrics = task.compute_metrics()
    assert {"val/bpq", "val/mpq", "val/pq_a", "val/pq_b", "val/dq", "val/sq", "val/loss"} <= set(
        metrics
    )


def test_cell_tokens_average_the_patches_each_bounding_box_touches():
    tokens = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    inst = np.zeros((64, 64), np.int32)
    inst[16:32, 32:48] = 7  # exactly patch (1, 2)
    inst[10:20, 2:6] = 3  # rows 0-1 of column 0
    ids, emb = cell_tokens(tokens, inst)
    assert ids.tolist() == [3, 7]
    torch.testing.assert_close(emb[1], tokens[:, 1, 2])
    torch.testing.assert_close(emb[0], tokens[:, 0:2, 0].mean(1))


def test_pair_centroids_is_one_to_one_within_the_radius():
    true = np.array([[10.0, 10.0], [10.0, 30.0], [50.0, 50.0]])
    pred = np.array([[11.0, 29.0], [12.0, 12.0], [10.0, 20.0], [90.0, 90.0]])
    pairs = pair_centroids(true, pred, radius=5)
    assert sorted(map(tuple, pairs.tolist())) == [(0, 1), (1, 0)]
    assert pair_centroids(true, pred[:0], 5).shape == (0, 2)


def test_cellvit_returns_its_deepest_tokens_with_the_outputs():
    model = _cellvit(instance=True).eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        out, tokens = model(image, return_tokens=True)
        expected = token_grids(model.backbone, image, (4,))[0]
    assert set(out) == {"np", "hv", "nt"}
    torch.testing.assert_close(tokens, expected)
    head = torch.nn.Linear(32, 2)
    ((inst, types),) = classify_cells(model, head, image)
    assert types.shape == (inst.max() + 1,) and set(types.tolist()) <= {0, 1}


def test_embed_cells_labels_detections_with_the_paired_ground_truth(tmp_path, monkeypatch):
    data = _write_pannuke_tile(tmp_path)
    cfg = {
        "mean": [0.5] * 3,
        "std": [0.5] * 3,
        "model": {"num_classes": 2},
        "data": {"train": data},
    }
    true = _touching_nuclei()
    pred = np.roll(true, 3, axis=1)  # every detection 3 px off its nucleus
    pred[60:, 60:] = 9  # and one false detection
    ids = np.array([1, 2, 3, 9])

    def detect(model, image, postprocess):
        return [(pred, ids, torch.tensor(ids, dtype=torch.float32)[:, None])]

    monkeypatch.setattr(classify, "detect_cells", detect)
    x, y = classify.embed_cells(cfg, torch.nn.Linear(1, 1), {}, "train").tensors
    assert x[:, 0].tolist() == [1, 2, 3]  # the false detection is unpaired
    assert y.tolist() == [0, 1, 1]  # types 1, 2, 2 as classes


def test_export_writes_three_branches_and_a_segmentation_card_on_nt(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    import json

    from pathotrain.cellseg.export import export
    from pathotrain.cellseg.instances import instances_from_outputs

    cfg = {
        "name": "cells",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "model": {
            "name": "cellseg:hovernext",
            "num_classes": 3,
            "encoder": "convnextv2_atto",
            "encoder_weights": None,
        },
        "task": {"class_names": ["background", "tumour", "immune"]},
        "data": {"train": {"images": "i", "instances": "n", "types": "t"}},
    }
    model = build_model(
        "hovernext", 3, instance=True, encoder="convnextv2_atto", encoder_weights=None
    )
    torch.save({"model": model.state_dict(), "config": cfg}, tmp_path / "m.ckpt")
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="mpp"):
        export(tmp_path / "m.ckpt", tmp_path / "out" / "m.onnx", 64)
    path = export(tmp_path / "m.ckpt", tmp_path / "out" / "m.onnx", 64, mpp=0.25, icc=False)

    card = json.loads((path.parent / "model_card.json").read_text())
    assert card["task"] == "segmentation" and card["background_id"] == 0
    assert card["output"] == {"tensor": "nt", "activation": "logits", "n_classes": 3}
    assert [c["name"] for c in card["classes"]] == ["background", "tumour", "immune"]
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [o.name for o in session.get_outputs()] == ["np", "hv", "nt"]
    np_out, hv, nt = session.run(None, {"image": np.random.rand(2, 3, 64, 64).astype(np.float32)})
    assert np_out.shape == (2, 2, 64, 64) and hv.shape == (2, 2, 64, 64)
    assert nt.shape == (2, 3, 64, 64)
    inst, types = instances_from_outputs(np_out[0], hv[0], nt[0])
    assert inst.shape == (64, 64) and len(types) == inst.max() + 1


def test_example_configs_are_valid():
    import inspect
    from importlib.resources import files

    from pathotrain.engine import ModelCheckpoint, TrainerConfig, load_config
    from pathotrain.experiment import normalisation
    from pathotrain.pathocls.model import Head
    from pathotrain.pathocls.task import ClassificationTask

    configs = files("pathotrain.cellseg") / "configs"
    instance = load_config(configs / "instance.example.yaml")
    normalisation(instance, "mean")
    model = torch.nn.Conv2d(3, 1, 1)
    CellInstanceTask(model, instance["model"]["num_classes"], **instance["task"])
    TrainerConfig(**instance["trainer"])
    ModelCheckpoint("unused", **instance["checkpoint"])
    dataset_args = set(inspect.signature(CellDataset.__init__).parameters)
    for split in ("train", "val", "test"):
        assert set(instance["data"][split]) <= dataset_args

    cls = load_config(configs / "classify.example.yaml")
    assert {"checkpoint", "num_classes"} <= set(cls["model"])
    head = Head(8, cls["model"]["num_classes"], **cls["model"]["head"])
    ClassificationTask(head, cls["model"]["num_classes"], **cls["task"])
    for split in ("train", "val"):
        assert set(cls["data"][split]) <= dataset_args

    semantic = load_config(configs / "config.example.yaml")
    assert semantic["model"]["name"].startswith("cellseg:")
