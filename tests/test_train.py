import mlflow
import pytest
import torch
from torch.utils.data import TensorDataset

from pathotrain.segmentation import FocalDiceLoss
from pathotrain.segmentation import train as train_module
from pathotrain.segmentation.train import build_loss, load_model, normalisation, train


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


def test_train_from_config_and_rebuild_model_from_checkpoint(tmp_path, monkeypatch, toy_data):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    cfg = _cfg(tmp_path / "run")
    metrics = train(cfg)
    assert {"val/miou", "val/loss"} <= metrics.keys()
    assert (tmp_path / "run/config.yaml").exists() and (tmp_path / "run/logs").is_dir()

    model, saved = load_model(tmp_path / "run/checkpoints/best.ckpt")
    assert saved == cfg
    best = torch.load(tmp_path / "run/checkpoints/best.ckpt", weights_only=True)["model"]
    assert all(torch.equal(v, best[k]) for k, v in model.state_dict().items())

    # A changed config for the same run is refused, unless resuming it.
    longer = {**cfg, "trainer": {**cfg["trainer"], "max_epochs": 3}}
    with pytest.raises(FileExistsError):
        train(longer)
    train(longer, resume=tmp_path / "run/checkpoints/last.ckpt")


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
