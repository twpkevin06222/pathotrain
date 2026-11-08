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
from pathotrain.pathocls.train import complete, load_model, train, train_sampler

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


def test_online_probe_checkpoints_only_the_head(tmp_path, backbone):
    cfg = _cfg(tmp_path, backbone) | {"augmentation": {"n": 1, "m": 1}}
    metrics = train(cfg)
    assert metrics["val/balanced_acc"] == 1.0
    best = tmp_path / "run/checkpoints/best.ckpt"
    state = torch.load(best, weights_only=True)["model"]
    assert state and all(k.startswith("head.") for k in state)
    classifier, _ = load_model(best)
    assert not any(p.requires_grad for p in classifier.backbone.parameters())
    assert evaluate(best, "val")["val/balanced_acc"] == 1.0


def test_fine_tuning_trains_and_restores_the_backbone(tmp_path, backbone):
    cfg = _cfg(tmp_path, backbone)
    cfg["model"]["freeze"] = False
    cfg["trainer"]["max_epochs"] = 3
    train(cfg)
    last = tmp_path / "run/checkpoints/last.ckpt"
    state = torch.load(last, weights_only=True)["model"]
    pretrained = Dinov2Model.from_pretrained(backbone).state_dict()
    key = "embeddings.patch_embeddings.projection.weight"
    assert not torch.equal(state[f"backbone.{key}"], pretrained[key])  # it was trained
    classifier, _ = load_model(last)
    assert torch.equal(classifier.backbone.state_dict()[key], state[f"backbone.{key}"])


def test_backbone_trains_at_its_own_lr():
    model = torch.nn.Module()
    model.backbone, model.head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    task = ClassificationTask(model, 2, lr=1e-3)
    optimizer, _ = task.configure_optimizers(10, 1)
    assert [g["lr"] for g in optimizer.param_groups] == [1e-3, 1e-4]
    model.backbone.requires_grad_(False)  # frozen: not optimised at all
    optimizer, _ = task.configure_optimizers(10, 1)
    assert len(optimizer.param_groups) == 1 and optimizer.param_groups[0]["lr"] == 1e-3


def test_mlp_head_trains_reloads_and_exports(tmp_path, backbone):
    cfg = _cfg(tmp_path, backbone)
    cfg["model"]["head"] = {"hidden": 16, "dropout": 0.1}
    assert train(cfg)["val/balanced_acc"] == 1.0
    best = tmp_path / "run/checkpoints/best.ckpt"
    classifier, _ = load_model(best)
    assert classifier.head.hidden[0].out_features == 16
    assert classifier.head.linear.in_features == 16
    assert evaluate(best, "val")["val/balanced_acc"] == 1.0
    pytest.importorskip("onnxruntime")
    (tmp_path / "out").mkdir()
    export(best, tmp_path / "out/m.onnx", mpp=0.5, icc=True)  # checks ORT vs torch


def test_timm_backbone_from_scratch_trains_reloads_and_exports(tmp_path, backbone):
    cfg = _cfg(tmp_path, backbone)
    cfg["model"] = {"name": "timm:test_resnet", "pretrained": False, "freeze": False}
    cfg["task"]["backbone_lr"] = 1e-2  # from scratch: as fast as the head
    cfg["trainer"]["max_epochs"] = 3
    train(cfg)
    last = tmp_path / "run/checkpoints/last.ckpt"
    state = torch.load(last, weights_only=True)["model"]
    classifier, _ = load_model(last)
    weights = classifier.state_dict()
    assert all(torch.equal(weights[k], v) for k, v in state.items())  # backbone and head
    assert classifier.head.linear.in_features == 96  # timm's pooled feature width
    assert "val/balanced_acc" in evaluate(last, "val")
    pytest.importorskip("onnxruntime")
    (tmp_path / "out").mkdir()
    export(last, tmp_path / "out/m.onnx", mpp=0.5, icc=True)  # checks ORT vs torch
    card = json.loads((tmp_path / "out/model_card.json").read_text())
    assert card["input"]["tile_size"] == 160  # timm's pretrained config input size


def test_a_backbone_from_scratch_cannot_be_frozen(tmp_path, backbone):
    cfg = _cfg(tmp_path, backbone)
    cfg["model"] = {"name": "timm:test_resnet", "pretrained": False}
    with pytest.raises(ValueError, match="freeze: false"):
        train(cfg)


def test_class_weighted_sampling_balances_the_classes(tmp_path, backbone):
    cfg = _cfg(tmp_path, backbone) | {"sampling": {"power": 1.0}}
    train_dir = tmp_path / "train"
    for p in sorted((train_dir / "blue").iterdir())[4:]:
        p.unlink()  # 4 blue vs 8 red tiles
    metrics = train(cfg)
    assert metrics["val/balanced_acc"] == 1.0
    sampler = train_sampler(complete(cfg))
    labels = torch.tensor([y for _, y in ClassTileDataset(train_dir).samples])
    per_class = [sampler.weights[labels == c].sum().item() for c in (0, 1)]
    assert per_class[0] == pytest.approx(per_class[1])  # each class drawn as often
    assert sampler.num_samples == 12
    assert train_sampler(_cfg(tmp_path / "plain", backbone)) is None
