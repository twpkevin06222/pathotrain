import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Dinov2Config, Dinov2Model

pytest.importorskip("peft")

from pathotrain.histossl import config as ssl_config
from pathotrain.histossl.config import ConfigError, validate
from pathotrain.histossl.methods import teacher_jepa
from pathotrain.histossl.train import build_task, load_backbone, train


def _tiny(init="pretrained", id=None, arch=None, grad_checkpointing=True):
    """Stands in for Midnight: a 2-block Dinov2, the same weights on every call."""
    torch.manual_seed(0)
    config = Dinov2Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        patch_size=14,
        image_size=56,
    )
    model = Dinov2Model(config)
    if grad_checkpointing:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    return model


def _images(d, n, colour=None, seed=0):
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        mean = colour if colour is not None else rng.integers(0, 256)
        tile = np.clip(rng.normal(mean, 30, (64, 64, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(tile).save(d / f"t{i}.png")
    return d


def _cfg(tmp_path, adaptation=None):
    tiles = _images(tmp_path / "tiles", 12)
    val = _images(tmp_path / "val", 4, seed=1)
    for split, n in (("train", 6), ("test", 4)):
        _images(tmp_path / "probe" / split / "dark", n, 40, seed=2)
        _images(tmp_path / "probe" / split / "bright", n, 220, seed=3)
    return {
        "name": "jepa",
        "seed": 0,
        "output_dir": str(tmp_path / "run"),
        "method": {"name": "teacher_jepa", "width": 16, "depth": 1, "heads": 2},
        "backbone": {"init": "pretrained", "id": "midnight-12k"},
        "adaptation": adaptation or {"type": "lora", "rank": 4},
        "teacher": {"type": "frozen"},
        "masking": {"type": "block", "ratio": [0.4, 0.6]},
        "task": {"lr": 1e-3},
        "trainer": {"max_epochs": 2, "accelerator": "cpu", "devices": 1, "precision": "32-true"},
        "loader": {"batch_size": 4},
        "data": {
            "train": {"tiles": [str(tiles)]},
            "val": {"tiles": [str(val)]},
            "samples_per_epoch": 8,
            "crops": {"global_size": 56},
        },
        "eval": {
            "train": str(tmp_path / "probe/train"),
            "test": str(tmp_path / "probe/test"),
            "num_workers": 0,
        },
    }


def test_example_config_is_valid():
    from importlib.resources import files

    from pathotrain.engine import TrainerConfig, load_config

    cfg = load_config(files("pathotrain.histossl") / "teacher_jepa.example.yaml")
    ssl = validate(cfg)
    assert ssl.method.name == "teacher_jepa" and ssl.adaptation.type == "lora"
    TrainerConfig(**cfg["trainer"])


def test_validate_lists_every_invalid_combination(tmp_path):
    cfg = _cfg(tmp_path)
    bad = {
        **cfg,
        "backbone": {"init": "scratch", "arch": "vit_b14"},  # LoRA needs pretrained weights
        "teacher": {"type": "ema", "momentum": [0.99, 1.0]},  # teacher_jepa's teacher is frozen
        "masking": None,  # teacher_jepa needs masking
        "data": {**cfg["data"], "crops": {"global_size": 50, "local_crops": 4}},
        "trainer": {**cfg["trainer"], "strategy": "fsdp"},
    }
    with pytest.raises(ConfigError) as e:
        validate(bad)
    message = str(e.value)
    for needle in (
        "needs pretrained weights",
        "backbone.init in ['pretrained']",
        "teacher.type in ['frozen']",
        "needs a masking section",
        "multiple of the 14 px patch",
        "takes no local crops",
        "fsdp",
    ):
        assert needle in message, needle
    with pytest.raises(ConfigError, match="unknown keys \\['rnak'\\]"):
        validate({**cfg, "adaptation": {"type": "lora", "rnak": 4}})
    lejepa = {
        **cfg,
        "method": {"name": "lejepa"},
        "backbone": {"init": "scratch", "arch": "vit_b14"},
        "adaptation": {"type": "full"},
        "teacher": {"type": "ema"},
    }
    with pytest.raises(ConfigError, match="teacher.type in \\['none'\\]"):  # no EMA in LeJEPA
        validate(lejepa)
    with pytest.raises(ConfigError, match="needs an eval section"):
        validate({**cfg, "eval": None, "checkpoint": {"monitor": "probe/acc"}})
    assert set(ssl_config.METHODS) == {"teacher_jepa", "dinov2", "lejepa"}


def test_teacher_jepa_learns_the_teacher_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(teacher_jepa, "build_backbone", _tiny)
    cfg = _cfg(tmp_path, {"type": "last_n", "n": 1})
    task = build_task(cfg).train()
    teacher = {k: v.clone() for k, v in task.model.teacher.state_dict().items()}
    images = torch.rand(4, 3, 56, 56)
    opt = torch.optim.AdamW([p for p in task.model.parameters() if p.requires_grad], lr=3e-3)
    losses = []
    for _ in range(30):
        loss = task._loss(images, torch.Generator().manual_seed(0))
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
    assert all(torch.equal(v, teacher[k]) for k, v in task.model.teacher.state_dict().items())
    assert not task.model.teacher.training


def test_train_probes_checkpoints_and_reloads(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.setattr(teacher_jepa, "build_backbone", _tiny)
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    cfg = _cfg(tmp_path)
    metrics = train(cfg)
    assert {"val/loss", "probe/balanced_acc", "knn/balanced_acc"} <= metrics.keys()
    best = torch.load(tmp_path / "run/checkpoints/best.ckpt", weights_only=False)
    p = best["provenance"]
    assert p["backbone"].startswith("midnight-12k@") and p["method"] == "teacher_jepa"
    assert p["adaptation"] == {"type": "lora", "rank": 4, "targets": ["q_proj", "v_proj"]}
    assert p["input_size"] == 56 and p["normalisation"]["mean"] == [0.5] * 3 and p["git_commit"]
    assert best["monitor"] == "probe/balanced_acc"

    backbone, saved = load_backbone(tmp_path / "run/checkpoints/best.ckpt")
    assert saved == cfg and type(backbone) is Dinov2Model  # LoRA merged
    task = build_task(cfg)
    task.model.load_state_dict(best["model"])
    x = torch.rand(2, 3, 56, 56)
    with torch.no_grad():
        want = task.backbone.eval()(pixel_values=x).last_hidden_state
        assert torch.allclose(backbone(pixel_values=x).last_hidden_state, want, atol=1e-5)
