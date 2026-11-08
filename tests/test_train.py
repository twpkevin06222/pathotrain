import mlflow
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from pathotrain.segmentation import FocalDiceLoss
from pathotrain.segmentation import train as train_module
from pathotrain.segmentation.eval import evaluate
from pathotrain.segmentation.train import (
    build_dataset,
    build_loss,
    load_model,
    normalisation,
    train,
)


def _cfg(out):
    return {
        "name": "exp",
        "seed": 0,
        "output_dir": str(out),
        "mean": "IMAGENET_MEAN",
        "std": [0.5, 0.5, 0.5],
        "model": {
            "name": "smp:unet",
            "num_classes": 2,
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "encoder_depth": 3,
            "decoder_channels": [32, 16, 8],
        },
        "loss": {"name": "FocalDiceLoss", "gamma": 1.0},
        "task": {"lr": 1e-2, "warmup_epochs": 0.5},
        "trainer": {"max_epochs": 2, "accelerator": "cpu", "devices": 1, "precision": "32-true"},
        "early_stopping": {"patience": 5},
        "loader": {"batch_size": 4},
        "data": {"train": {}, "val": {}, "class_map": {"tumor": 1}},
    }


@pytest.fixture
def toy_data(monkeypatch):
    # Slides are out of scope here; every split gets the same learnable toy tiles.
    image = torch.rand(16, 3, 4, 4).repeat_interleave(8, 2).repeat_interleave(8, 3)
    mask = (image[:, 0] > 0.5).long()
    monkeypatch.setattr(
        train_module, "build_dataset", lambda cfg, split, transform: TensorDataset(image, mask)
    )


def test_sampling_section_trains_with_a_class_weighted_sampler(tmp_path, monkeypatch):
    image = torch.rand(16, 3, 4, 4).repeat_interleave(8, 2).repeat_interleave(8, 3)
    mask = (image[:, 0] > 0.5).long()

    class Tiles(TensorDataset):
        class_fraction = np.stack([(mask == 0).float().mean((1, 2)), mask.float().mean((1, 2))], 1)

    monkeypatch.setattr(
        train_module, "build_dataset", lambda cfg, split, transform: Tiles(image, mask)
    )
    seen = {}
    fit = train_module.experiment.fit

    def spy(*args, train_sampler=None, **kwargs):
        seen["sampler"] = train_sampler
        return fit(*args, train_sampler=train_sampler, **kwargs)

    monkeypatch.setattr(train_module.experiment, "fit", spy)
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    cfg = {**_cfg(tmp_path / "run"), "sampling": {"power": 0.5, "num_samples": 8}}
    assert "val/miou" in train(cfg)
    assert seen["sampler"].num_samples == 8 and len(seen["sampler"].weights) == 16


def test_mlflow_db_keeps_runs_and_artifacts_beside_it(tmp_path, monkeypatch, toy_data):
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    db = tmp_path / "project/mlflow.db"
    train({**_cfg(tmp_path / "run"), "mlflow_db": str(db)})
    client = mlflow.MlflowClient(f"sqlite:///{db}")
    (run,) = client.search_runs([client.get_experiment_by_name("exp").experiment_id])
    assert run.info.artifact_uri.startswith(str(tmp_path / "project/mlruns"))
    assert not (tmp_path / "elsewhere/mlruns").exists()
    # Both checkpoints name the run, so export can record it in the model card.
    for name in ("best.ckpt", "last.ckpt"):
        ckpt = tmp_path / "run/checkpoints" / name
        assert torch.load(ckpt, weights_only=False)["mlflow_run_id"] == run.info.run_id


def test_train_from_config_and_rebuild_model_from_checkpoint(tmp_path, monkeypatch, toy_data):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    cfg = _cfg(tmp_path / "run")
    metrics = train(cfg)
    assert {"val/miou", "val/loss"} <= metrics.keys()
    assert (tmp_path / "run/config.yaml").exists() and (tmp_path / "run/logs").is_dir()
    assert "epoch 1/" in (tmp_path / "run/train.log").read_text()

    model, saved = load_model(tmp_path / "run/checkpoints/best.ckpt")
    assert saved == cfg
    best = torch.load(tmp_path / "run/checkpoints/best.ckpt", weights_only=True)["model"]
    assert all(torch.equal(v, best[k]) for k, v in model.state_dict().items())

    # A changed config for the same run is refused, unless resuming it.
    longer = {**cfg, "trainer": {**cfg["trainer"], "max_epochs": 3}}
    with pytest.raises(FileExistsError):
        train(longer)
    train(longer, resume=tmp_path / "run/checkpoints/last.ckpt")


def test_evaluate_reproduces_the_last_validation(tmp_path, monkeypatch, toy_data):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    cfg = _cfg(tmp_path / "run")
    metrics = train(cfg)
    last = tmp_path / "run/checkpoints/last.ckpt"
    # Same weights, same tiles: the last epoch's validation, keyed by the split.
    assert evaluate(last, "val") == pytest.approx(metrics)
    with pytest.raises(ValueError, match="no 'test' split"):
        evaluate(last)
    test = evaluate(last, config={"data": {**cfg["data"], "test": {}}})
    assert test == pytest.approx({k.replace("val/", "test/"): v for k, v in metrics.items()})
    assert evaluate(last, "val", mlflow_db=tmp_path / "project/mlflow.db") == pytest.approx(metrics)
    client = mlflow.MlflowClient(f"sqlite:///{tmp_path}/project/mlflow.db")
    (run,) = client.search_runs([client.get_experiment_by_name(cfg["name"]).experiment_id])
    assert run.info.run_name == "eval-val" and run.data.metrics == pytest.approx(metrics)


def test_config_helpers_validate():
    assert normalisation({"mean": "MIDNIGHT_MEAN"}, "mean") == (0.5, 0.5, 0.5)
    with pytest.raises(ValueError, match="units"):
        normalisation({"std": [58, 57, 57]}, "std")
    with pytest.raises(ValueError, match="one of"):
        normalisation({"mean": "IMAGENET_STD"}, "mean")
    assert isinstance(build_loss({"loss": {"name": "FocalDiceLoss"}}), FocalDiceLoss)
    assert build_loss({}) is None
    with pytest.raises(ValueError, match="unknown loss"):
        build_loss({"loss": {"name": "build_model"}})


def test_config_example_is_valid():
    import inspect
    from importlib.resources import files

    from patchology import SegmentationTileGenerator, TileGenerator, TilingConfig
    from pathotrain.dataloader.augmentation import train_transform
    from pathotrain.engine import EarlyStopping, ModelCheckpoint, TrainerConfig, load_config
    from pathotrain.segmentation import SegmentationTask, build_model

    cfg = load_config(files("pathotrain.segmentation") / "config.example.yaml")
    mean, std = normalisation(cfg, "mean"), normalisation(cfg, "std")
    model = build_model(**{**cfg["model"], "encoder_weights": None})
    SegmentationTask(model, cfg["model"]["num_classes"], loss=build_loss(cfg), **cfg["task"])
    TrainerConfig(**cfg["trainer"])
    ModelCheckpoint("unused", **cfg["checkpoint"])
    EarlyStopping(**cfg["early_stopping"])
    train_transform(mean=mean, std=std, **cfg["augmentation"])
    for split in ("train", "val"):
        TilingConfig(**cfg["data"][split]["tiling"])
    allowed = {
        *inspect.signature(SegmentationTileGenerator.__init__).parameters,
        *inspect.signature(TileGenerator.__init__).parameters,
    }
    assert set(cfg["data"]) - {"train", "val"} <= allowed
    for split in ("train", "val"):
        assert set(cfg["data"][split]) - {"slides", "annotations", "tiling"} <= allowed


def test_build_dataset_reads_tile_folders(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(images / "a.png")
    Image.fromarray(np.ones((8, 8), np.uint8)).save(masks / "a.png")
    cfg = {"data": {"class_map": {"tumor": 1}, "train": {"images": images, "masks": masks}}}
    ds = build_dataset(cfg, "train", None)
    assert len(ds) == 1 and ds[0][1].shape == (8, 8)
    with pytest.raises(TypeError):
        build_dataset({"data": {"train": {"images": images, "mask": masks}}}, "train", None)


def test_build_dataset_split_options_override_shared(monkeypatch):
    seen = {}

    def capture(slides, annotations, **kwargs):
        seen[slides[0]] = kwargs

    monkeypatch.setattr(train_module, "SegmentationDataset", capture)
    data = {
        "object_mask_overlap": 0.2,
        "negative_sampling": 0.1,
        "train": {"slides": ["t"], "annotations": ["a"], "negative_sampling": 0.5},
        "val": {"slides": ["v"], "annotations": ["a"]},
    }
    for split in ("train", "val"):
        build_dataset({"data": data}, split, None)
    assert seen["t"]["negative_sampling"] == 0.5 and seen["v"]["negative_sampling"] == 0.1
    assert seen["t"]["object_mask_overlap"] == seen["v"]["object_mask_overlap"] == 0.2


def test_export_writes_an_onnx_graph_with_its_metadata(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    import json

    from pathotrain.segmentation import build_model
    from pathotrain.segmentation.export import export

    cfg = _cfg(tmp_path / "run")
    cfg["data"]["train"] = {"slides": [], "annotations": [], "tiling": {"tile_size": 64}}
    torch.save(
        {"model": build_model(**cfg["model"]).state_dict(), "config": cfg}, tmp_path / "m.ckpt"
    )
    path = export(tmp_path / "m.ckpt", tmp_path / "m.onnx")  # checks ONNX against PyTorch

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    meta = json.loads(session.get_modelmeta().custom_metadata_map["pathotrain"])
    assert meta["tile_size"] == 64 and meta["std"] == [0.5, 0.5, 0.5]
    assert meta["class_map"] == {"tumor": 1} and meta["class_names"] == ["0", "1"]
    (logits,) = session.run(None, {"image": np.zeros((3, 3, 64, 64), np.float32)})
    assert logits.shape == (3, 2, 64, 64)

    db = tmp_path / "project/mlflow.db"
    for version in (1, 2):  # each export registers a new version
        export(tmp_path / "m.ckpt", tmp_path / "m.onnx", register="seg", mlflow_db=db)
    import mlflow.onnx

    mlflow.set_tracking_uri(f"sqlite:///{db}")
    client = mlflow.MlflowClient()
    assert [v.version for v in client.search_model_versions("name='seg'")] == [2, 1]
    model = mlflow.onnx.load_model("models:/seg/2")
    assert [i.name for i in model.graph.input] == ["image"]
    run = client.get_run(client.get_model_version("seg", "2").run_id)
    assert run.info.run_name == "export" and run.data.params["checkpoint"] == str(
        tmp_path / "m.ckpt"
    )


def _export_seg(tmp_path, cfg, **kwargs):
    from pathotrain.segmentation import build_model
    from pathotrain.segmentation.export import export

    state = {"model": build_model(**cfg["model"]).state_dict(), "config": cfg}
    torch.save(state | kwargs.pop("extra", {}), tmp_path / "m.ckpt")
    (tmp_path / "out").mkdir(exist_ok=True)
    return export(tmp_path / "m.ckpt", tmp_path / "out" / "m.onnx", **kwargs)


def test_export_writes_a_pathoinfer_model_folder(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    import json

    cfg = _cfg(tmp_path / "run")
    cfg["data"]["train"] = {
        "slides": [],
        "annotations": [],
        "tiling": {"tile_size": 64, "tile_mpp": 0.5},
    }
    path = _export_seg(tmp_path, cfg, version="3", extra={"mlflow_run_id": "abc123"})
    card = json.loads((path.parent / "model_card.json").read_text())
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    assert json.loads(meta["model_card"]) == card  # the same card, embedded
    assert json.loads(meta["pathotrain"])["tile_size"] == 64  # provenance kept
    assert {k: card[k] for k in ("schema_version", "name", "version", "task", "framework")} == {
        "schema_version": 1,
        "name": "exp",
        "version": "3",
        "task": "segmentation",
        "framework": "onnx",
    }
    assert card["file"] == "m.onnx" and card["precision"] == "fp32"
    assert card["input"] == {
        "name": "image",
        "tile_size": 64,
        "mpp": 0.5,
        "channel_order": "RGB",
        "layout": "NCHW",
        "dtype": "float32",
        "icc": True,
        "normalisation": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.5, 0.5, 0.5],
            "scale": 1 / 255,
        },
    }
    assert [(c["id"], c["name"]) for c in card["classes"]] == [(0, "background"), (1, "tumor")]
    assert all(len(c["color"]) == 3 for c in card["classes"])
    assert card["background_id"] == 0
    assert card["output"] == {"tensor": "logits", "activation": "logits", "n_classes": 2}
    assert card["recommended_overlap"] == 0.25
    assert card["provenance"]["training_run_id"] == "abc123"
    assert card["provenance"]["source_metadata"]["pathotrain"]["class_map"] == {"tumor": 1}
    # pathoinfer owns the schema: when it is installed, it must load the folder and the
    # bare ONNX file (from the embedded card).
    loader = pytest.importorskip("pathoinfer.registry.loader")
    for source in (path.parent, path):
        loaded, _ = loader.load_model(source)
        assert loaded.name == "exp" and loaded.input.mpp == 0.5


def test_export_from_tile_folders_needs_mpp_and_icc(tmp_path):
    pytest.importorskip("onnxruntime")
    import json

    cfg = _cfg(tmp_path / "run")
    cfg["data"]["train"] = {"images": "unused", "masks": "unused"}
    with pytest.raises(ValueError, match="tile folders"):
        _export_seg(tmp_path, cfg, size=64)
    with pytest.raises(ValueError, match="tile folders"):
        _export_seg(tmp_path, cfg, size=64, mpp=0.5)
    path = _export_seg(tmp_path, cfg, size=64, mpp=1.0, icc=False, overlap=0.5)
    card = json.loads((path.parent / "model_card.json").read_text())
    assert (card["input"]["mpp"], card["input"]["icc"], card["recommended_overlap"]) == (
        1.0,
        False,
        0.5,
    )


def test_card_follows_the_training_config_and_refuses_contradictions(tmp_path):
    from pathotrain.experiment import model_card

    cfg = _cfg(tmp_path / "run")
    cfg["data"]["train"] = {"slides": [], "tiling": {"tile_size": 64, "tile_mpp": 0.5}}
    cfg["data"]["color"] = {"icc": False}
    output = {"tensor": "logits", "activation": "logits", "n_classes": 2}
    card = model_card(cfg, "segmentation", 64, "m.onnx", output)
    assert card["input"]["icc"] is False and card["input"]["mpp"] == 0.5
    with pytest.raises(ValueError, match="trained at 0.5 mpp"):
        model_card(cfg, "segmentation", 64, "m.onnx", output, mpp=1.0)
    with pytest.raises(ValueError, match="icc=False"):
        model_card(cfg, "segmentation", 64, "m.onnx", output, icc=True)
    with pytest.raises(ValueError, match="overlap"):
        model_card(cfg, "segmentation", 64, "m.onnx", output, overlap=1.0)


def test_classes_need_real_names_and_take_colour_overrides(tmp_path):
    from pathotrain.experiment import TAB20, card_classes

    cfg = _cfg(tmp_path / "run")
    cfg["model"]["num_classes"] = 3  # label 2 has no name in class_map {tumor: 1}
    with pytest.raises(ValueError, match=r"classes \[2\] have no name"):
        card_classes(cfg, "segmentation")
    cfg["task"]["class_names"] = ["bg", "tumor", "stroma"]
    cfg["class_colors"] = {"tumor": [255, 0, 0]}
    classes = card_classes(cfg, "segmentation")
    assert [c["name"] for c in classes] == ["bg", "tumor", "stroma"]
    assert classes[1]["color"] == [255, 0, 0] and classes[2]["color"] == list(TAB20[2])
    cfg["class_colors"] = {"necrosis": [0, 0, 0]}
    with pytest.raises(ValueError, match="unknown classes"):
        card_classes(cfg, "segmentation")
