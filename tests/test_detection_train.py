import json

import mlflow
import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("rfdetr")

from test_detection import _samples

from pathotrain.detection import train as train_module
from pathotrain.detection.eval import evaluate
from pathotrain.detection.train import build_dataset, load_model, train


def _cfg(out):
    return {
        "name": "det",
        "seed": 0,
        "output_dir": str(out),
        "mean": "IMAGENET_MEAN",
        "std": "IMAGENET_STD",
        "model": {"name": "rfdetr:nano", "num_classes": 2, "pretrained": False},
        "task": {"warmup_epochs": 0.5, "class_names": ["small", "large"]},
        "trainer": {"max_epochs": 2, "accelerator": "cpu", "devices": 1, "precision": "32-true"},
        "early_stopping": {"patience": 5},
        "loader": {"batch_size": 4},
        "data": {"train": {}, "val": {}, "class_map": {"small": 0, "large": 1}},
    }


def test_train_from_config_and_rebuild_detector_from_checkpoint(tmp_path, monkeypatch):
    # Slides are out of scope here; every split gets the same toy squares.
    monkeypatch.setattr(train_module, "build_dataset", lambda cfg, split, transform: _samples())
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    cfg = _cfg(tmp_path / "run")
    metrics = train(cfg)
    assert {"val/map", "val/ap_small", "val/ap_large", "val/loss"} <= metrics.keys()
    assert (tmp_path / "run/config.yaml").exists() and (tmp_path / "run/logs").is_dir()

    detector, saved = load_model(tmp_path / "run/checkpoints/best.ckpt")
    assert saved == cfg
    best = torch.load(tmp_path / "run/checkpoints/best.ckpt", weights_only=True)["model"]
    assert all(torch.equal(v, best[k]) for k, v in detector.model.state_dict().items())


def test_evaluate_reproduces_the_last_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "build_dataset", lambda cfg, split, transform: _samples())
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    cfg = _cfg(tmp_path / "run")
    metrics = train(cfg)
    last = tmp_path / "run/checkpoints/last.ckpt"
    # Same weights, same tiles: the last epoch's validation, keyed by the split.
    assert evaluate(last, "val") == pytest.approx(metrics)
    test = evaluate(last, config={"data": {**cfg["data"], "test": {}}})
    assert test == pytest.approx({k.replace("val/", "test/"): v for k, v in metrics.items()})
    assert evaluate(last, "val", mlflow_db=tmp_path / "project/mlflow.db") == pytest.approx(metrics)
    client = mlflow.MlflowClient(f"sqlite:///{tmp_path}/project/mlflow.db")
    (run,) = client.search_runs([client.get_experiment_by_name(cfg["name"]).experiment_id])
    assert run.info.run_name == "eval-val" and run.data.metrics == pytest.approx(metrics)


def test_build_dataset_checks_classes_against_the_model(tmp_path):
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(tmp_path / "a.png")
    coco = {
        "images": [{"id": 1, "file_name": "a.png", "width": 64, "height": 64}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 10, 10]}],
        "categories": [{"id": 1, "name": "cell"}],
    }
    (tmp_path / "_annotations.coco.json").write_text(json.dumps(coco))
    cfg = {"model": {"num_classes": 1}, "data": {"train": {"root": tmp_path}}}
    assert len(build_dataset(cfg, "train", None)) == 1
    with pytest.raises(ValueError, match="1 classes but model.num_classes is 2"):
        build_dataset({**cfg, "model": {"num_classes": 2}}, "train", None)


def test_config_example_is_valid():
    import inspect
    from importlib.resources import files

    from patchology import DetectionTileGenerator, TileGenerator, TilingConfig
    from pathotrain.dataloader.augmentation import train_transform
    from pathotrain.detection import DetectionTask, build_detector
    from pathotrain.engine import EarlyStopping, ModelCheckpoint, TrainerConfig, load_config
    from pathotrain.experiment import normalisation

    cfg = load_config(files("pathotrain.detection") / "config.example.yaml")
    mean, std = normalisation(cfg, "mean"), normalisation(cfg, "std")
    detector = build_detector(**{**cfg["model"], "pretrained": False})
    DetectionTask(detector, **cfg["task"])
    TrainerConfig(**cfg["trainer"])
    ModelCheckpoint("unused", **cfg["checkpoint"])
    EarlyStopping(**cfg["early_stopping"])
    train_transform(mean=mean, std=std, boxes=True, **cfg["augmentation"])
    for split in ("train", "val"):
        TilingConfig(**cfg["data"][split]["tiling"])
    allowed = {
        *inspect.signature(DetectionTileGenerator.__init__).parameters,
        *inspect.signature(TileGenerator.__init__).parameters,
    }
    assert set(cfg["data"]) - {"train", "val"} <= allowed
    assert len(set(cfg["data"]["class_map"].values())) == cfg["model"]["num_classes"]


def test_export_writes_boxes_scores_and_labels(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from pathotrain.detection import build_detector
    from pathotrain.detection.export import export

    cfg = _cfg(tmp_path / "run")
    cfg["data"]["train"] = {"slides": [], "annotations": [], "tiling": {"tile_size": 128}}
    model = build_detector(**cfg["model"]).model
    torch.save({"model": model.state_dict(), "config": cfg}, tmp_path / "m.ckpt")
    with pytest.raises(ValueError, match="multiple of 32"):
        export(tmp_path / "m.ckpt", tmp_path / "m.onnx", size=100)
    path = export(tmp_path / "m.ckpt", tmp_path / "m.onnx")  # checks ONNX against PyTorch

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    boxes, scores, labels = session.run(None, {"image": np.zeros((3, 3, 128, 128), np.float32)})
    assert boxes.shape == (3, 300, 4) and scores.shape == labels.shape == (3, 300)
    assert (np.diff(scores, axis=1) <= 0).all() and set(np.unique(labels)) <= {0, 1}
    assert boxes.min() >= 0 and boxes.max() <= 128
