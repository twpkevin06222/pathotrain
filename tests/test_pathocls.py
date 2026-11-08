import json

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Dinov2Config, Dinov2Model

from pathotrain.dataloader import ClassTileDataset
from pathotrain.histossl.backbones import set_preprocessing
from pathotrain.histossl.eval.linear_probe import probe_transform
from pathotrain.pathocls import ClassificationTask
from pathotrain.pathocls.eval import evaluate
from pathotrain.pathocls.export import export
from pathotrain.pathocls.train import load_model, train

CPU = {"accelerator": "cpu", "devices": 1, "precision": "32-true"}


@pytest.fixture
def backbone(tmp_path):
    """A tiny DINOv2 saved as a histossl export folder (input 56 px, patch 14)."""
    torch.manual_seed(0)
    config = Dinov2Config(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4, patch_size=14, image_size=56
    )
    path = tmp_path / "backbone"
    set_preprocessing(Dinov2Model(config), (0.5,) * 3, (0.5,) * 3, 56).save_pretrained(path)
    return path


def _tiles(root, rng, n=8):
    # Red-ish vs blue-ish tiles with noise, at a size the transform resizes to 56 px.
    for name, colour in (("blue", (40, 60, 200)), ("red", (200, 50, 40))):
        (root / name).mkdir(parents=True)
        for i in range(n):
            tile = np.clip(rng.normal(colour, 25, (64, 64, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(tile).save(root / name / f"{i}.png")
    return root


def _cfg(tmp_path, backbone):
    rng = np.random.default_rng(0)
    return {
        "name": "cls",
        "seed": 0,
        "output_dir": str(tmp_path / "run"),
        "model": {"name": str(backbone)},
        "task": {"lr": 1e-2},
        "features": {"batch_size": 8, "num_workers": 0},
        "trainer": {"max_epochs": 20, **CPU},
        "loader": {"batch_size": 8},
        "data": {
            "train": {"root": str(_tiles(tmp_path / "train", rng))},
            "val": {"root": str(_tiles(tmp_path / "val", rng))},
        },
    }


def test_probe_trains_evaluates_and_exports(tmp_path, backbone):
    cfg = _cfg(tmp_path, backbone)
    metrics = train(cfg)
    assert metrics["val/balanced_acc"] == 1.0 and metrics["val/auroc"] == 1.0
    assert {"val/recall_blue", "val/recall_red", "val/macro_f1", "val/loss"} <= metrics.keys()
    best = tmp_path / "run/checkpoints/best.ckpt"

    # The class order and count are recorded in the run's config.
    classifier, saved = load_model(best)
    assert saved["task"]["class_names"] == ["blue", "red"]
    assert saved["model"]["num_classes"] == 2

    # The whole classifier (backbone + head) classifies images, as exported.
    transform = probe_transform((0.5,) * 3, (0.5,) * 3, 56)
    val = ClassTileDataset(cfg["data"]["val"]["root"], transform)
    images = torch.stack([val[i][0] for i in range(len(val))])
    labels = torch.tensor([val[i][1] for i in range(len(val))])
    with torch.no_grad():
        assert (classifier(images).argmax(1) == labels).all()

    test = {"data": cfg["data"] | {"test": cfg["data"]["val"]}}
    scored = evaluate(best, "test", test)
    assert scored["test/balanced_acc"] == 1.0
    assert scored["test/acc"] == pytest.approx(metrics["val/acc"])

    pytest.importorskip("onnxruntime")
    (tmp_path / "out").mkdir()
    path = export(best, tmp_path / "out/m.onnx", mpp=0.5, icc=True)  # checks ORT vs torch
    card = json.loads((tmp_path / "out/model_card.json").read_text())
    assert card["task"] == "classification" and card["input"]["tile_size"] == 56
    assert card["output"] == {"tensor": "logits", "activation": "logits", "n_classes": 2}
    assert [c["name"] for c in card["classes"]] == ["blue", "red"]
    assert card["input"]["normalisation"]["mean"] == [0.5, 0.5, 0.5]
    loader = pytest.importorskip("pathoinfer.registry.loader")
    loaded, _ = loader.load_model(path.parent)
    assert loaded.task == "classification" and loaded.background_id is None


def test_num_classes_must_match_the_classes_found(tmp_path, backbone):
    cfg = _cfg(tmp_path, backbone)
    cfg["model"]["num_classes"] = 3
    with pytest.raises(ValueError, match="num_classes"):
        train(cfg)


def test_task_logs_the_probe_metrics_per_class():
    task = ClassificationTask(torch.nn.Identity(), 3, class_names=["a", "b", "c"])
    logits = torch.tensor([[5.0, 0, 0], [0, 5.0, 0], [0, 5.0, 0], [0, 0, 5.0]])
    task.validation_step((logits, torch.tensor([0, 1, 2, 2])))
    metrics = task.compute_metrics()
    assert metrics["val/acc"] == pytest.approx(0.75)
    assert metrics["val/recall_c"] == pytest.approx(0.5)
    assert metrics["val/balanced_acc"] == pytest.approx((1 + 1 + 0.5) / 3)
    with pytest.raises(ValueError, match="class names"):
        ClassificationTask(torch.nn.Identity(), 3, class_names=["a"])
