import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Dinov2Config, Dinov2Model

pytest.importorskip("peft")

from pathotrain.histossl import config as ssl_config
from pathotrain.histossl import preprocessing
from pathotrain.histossl.config import ConfigError, validate
from pathotrain.histossl.methods import teacher_jepa
from pathotrain.histossl.train import build_task, load_backbone, train


def _tiny(
    init="pretrained", id=None, arch=None, grad_checkpointing=True, dtype="float32", input_size=224
):
    """Stands in for Midnight: a 2-block Dinov2, the same weights on every call."""
    from pathotrain.histossl.backbones import set_preprocessing

    torch.manual_seed(0)
    config = Dinov2Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        patch_size=14,
        image_size=56,
    )
    model = Dinov2Model(config).to(getattr(torch, dtype))
    if grad_checkpointing:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    return set_preprocessing(model, (0.5,) * 3, (0.5,) * 3, input_size)


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
    with pytest.raises(ConfigError, match="backbone.dtype must be one of"):
        validate({**cfg, "backbone": {**cfg["backbone"], "dtype": "float64"}})
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
        loss = task._losses(images, torch.Generator().manual_seed(0))["loss"]
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
    every_linear = ["fc1", "fc2", "k_proj", "o_proj", "q_proj", "v_proj"]  # the default
    assert p["adaptation"] == {"type": "lora", "rank": 4, "targets": every_linear}
    assert p["input_size"] == 56 and p["normalisation"]["mean"] == [0.5] * 3 and p["git_commit"]
    assert p["dtype"] == "float32"
    assert best["monitor"] == "probe/balanced_acc"

    backbone, saved = load_backbone(tmp_path / "run/checkpoints/best.ckpt")
    assert saved == cfg and type(backbone) is Dinov2Model  # LoRA merged
    task = build_task(cfg)
    task.model.load_state_dict(best["model"])
    x = torch.rand(2, 3, 56, 56)
    with torch.no_grad():
        want = task.encoder.eval()(pixel_values=x).last_hidden_state
        assert torch.allclose(backbone(pixel_values=x).last_hidden_state, want, atol=1e-5)

    import json

    from pathotrain.histossl.export import export

    out = export(tmp_path / "run/checkpoints/best.ckpt", tmp_path / "exported")
    meta = json.loads((out / "pathotrain.json").read_text())
    assert meta["provenance"] == p and meta["config"] == cfg and meta["lora_merged"]
    assert meta["monitor"] == "probe/balanced_acc" and meta["step"] == best["step"]
    exported = Dinov2Model.from_pretrained(out).eval()
    with torch.no_grad():
        assert torch.allclose(exported(pixel_values=x).last_hidden_state, want, atol=1e-5)
    # The training input size travels with the weights, so the probe CLI needs no config.
    assert preprocessing(exported) == {"input_size": 56, "mean": [0.5] * 3, "std": [0.5] * 3}

    import contextlib
    import io

    from pathotrain.histossl.eval.linear_probe import main as probe_main

    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        probe_main(
            [str(out), "--train", cfg["eval"]["train"], "--test", cfg["eval"]["test"]]
            + ["--num-workers", "0", "--k", "3"]
        )
    assert json.loads(printed.getvalue())["probe/acc"] == 1.0  # dark vs bright tiles


def test_sinkhorn_balances_prototypes_and_koleo_rewards_spread():
    from pathotrain.histossl.methods.dinov2 import DINOHead, koleo, sinkhorn_knopp

    scores = torch.rand(64, 8) * 2 - 1  # cosines, as DINOHead scores
    scores[:, 0] += 0.5  # one prototype favoured
    q = sinkhorn_knopp(scores, 0.1, iterations=20)
    assert torch.allclose(q.sum(1), torch.ones(64), atol=1e-4)  # each row a distribution
    assert torch.allclose(q.sum(0), torch.full((8,), 64 / 8), atol=0.1)  # prototypes used evenly
    spread, clump = torch.eye(8), torch.ones(8, 8) + 0.01 * torch.randn(8, 8)
    assert koleo(spread) < koleo(clump)
    head = DINOHead(32, prototypes=10, hidden_dim=16, bottleneck_dim=8)
    assert head(torch.randn(3, 32)).abs().max() <= 1 + 1e-5  # unit-norm prototypes: cosines
    assert not head.last_layer.parametrizations.weight.original0.requires_grad


def _dinov2_cfg(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["method"] = {
        "name": "dinov2",
        "prototypes": 32,
        "hidden_dim": 16,
        "bottleneck_dim": 8,
        "freeze_last_layer_epochs": 1,
    }
    cfg["teacher"] = {"type": "ema", "momentum": [0.9, 1.0]}
    cfg["masking"] = {"type": "block", "ratio": [0.1, 0.5], "p": 0.5}
    cfg["data"]["crops"] = {
        "global_crops": 2,
        "local_crops": 2,
        "global_size": 56,
        "local_size": 28,
    }
    return cfg


def test_dinov2_config_rules(tmp_path):
    from importlib.resources import files

    from pathotrain.engine import load_config

    assert validate(load_config(files("pathotrain.histossl") / "dinov2.example.yaml"))
    cfg = _dinov2_cfg(tmp_path)
    validate(cfg)
    with pytest.raises(ConfigError, match="global_crops >= 2"):
        validate({**cfg, "data": {**cfg["data"], "crops": {"global_crops": 1}}})
    with pytest.raises(ConfigError, match="teacher.type in \\['ema'\\]"):
        validate({**cfg, "teacher": {"type": "frozen"}})


def test_dinov2_trains_the_ema_teacher_and_resumes(tmp_path, monkeypatch):
    import mlflow

    from pathotrain.histossl.methods import dinov2

    monkeypatch.setattr(dinov2, "build_backbone", _tiny)
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    cfg = _dinov2_cfg(tmp_path)
    task = build_task(cfg)
    start = {k: v.clone() for k, v in task.model.teacher.state_dict().items()}
    head_before = task.model.dino_head.last_layer.parametrizations.weight.original1.clone()

    metrics = train(cfg)
    assert {"val/loss", "probe/balanced_acc"} <= metrics.keys()
    last = torch.load(tmp_path / "run/checkpoints/last.ckpt", weights_only=False)
    steps = last["step"]
    assert steps == 2 * 2  # 8 samples per epoch, batches of 4
    assert int(last["model"]["teacher.step"]) == steps  # schedules resume from here
    teacher = {
        k[len("teacher.") :]: v for k, v in last["model"].items() if k.startswith("teacher.")
    }
    lora = [k for k in teacher if "lora_" in k]
    base = [k for k in teacher if k.startswith("backbone.") and "lora_" not in k]
    assert lora and any(not torch.equal(teacher[k], start[k]) for k in lora)  # EMA moved
    assert all(torch.equal(teacher[k], start[k]) for k in base)  # frozen weights untouched
    # The head's last layer was held for the first epoch (2 steps) and then trained.
    head_after = last["model"]["dino_head.last_layer.parametrizations.weight.original1"]
    assert not torch.equal(head_after, head_before)

    longer = {**cfg, "trainer": {**cfg["trainer"], "max_epochs": 3}}
    train(longer, resume=tmp_path / "run/checkpoints/last.ckpt")
    resumed = torch.load(tmp_path / "run/checkpoints/last.ckpt", weights_only=False)
    assert int(resumed["model"]["teacher.step"]) == resumed["step"] == 3 * 2

    backbone, _ = load_backbone(tmp_path / "run/checkpoints/last.ckpt")  # the EMA teacher
    assert type(backbone) is Dinov2Model


def test_dinov2_freezes_the_head_last_layer_first(tmp_path, monkeypatch):
    from pathotrain.histossl.methods import dinov2

    monkeypatch.setattr(dinov2, "build_backbone", _tiny)
    task = build_task(_dinov2_cfg(tmp_path)).train()
    task.configure_optimizers(total_steps=10, steps_per_epoch=5)
    batch = {"global": torch.rand(2, 2, 3, 56, 56), "local": torch.rand(2, 2, 3, 28, 28)}
    task._losses(batch, torch.Generator().manual_seed(0))["loss"].backward()
    last = task.model.dino_head.last_layer.parametrizations.weight.original1
    assert last.grad is not None
    task.on_before_optimizer_step()  # step 0 < 5: frozen
    assert last.grad is None and task.model.dino_head.mlp[0].weight.grad is not None
    task.model.teacher.step.fill_(5)
    assert task._temperature() == pytest.approx(0.07)  # warmup (1 epoch) done
    assert dinov2._cosine(0.9, 1.0, 10, 10) == pytest.approx(1.0)


def test_sigreg_scores_gaussian_embeddings_low():
    from pathotrain.histossl.methods.lejepa import epps_pulley, sigreg

    g = torch.Generator().manual_seed(0)
    gaussian = torch.randn(512, 16, generator=g)
    stat = epps_pulley(gaussian)
    assert stat.shape == (16,) and stat.mean() < 2  # O(1) noise for a true N(0, 1)
    assert epps_pulley(gaussian * 3).min() > 50 * stat.mean()  # wrong scale
    assert epps_pulley(torch.zeros(512, 4)).min() > 50 * stat.mean()  # collapsed
    squashed = gaussian.clone()
    squashed[:, 1:] *= 0.05  # all but one dimension collapsed: not isotropic
    low = sigreg(gaussian, 256, torch.Generator().manual_seed(1))
    assert sigreg(squashed, 256, torch.Generator().manual_seed(1)) > 5 * low
    same = sigreg(gaussian, 256, torch.Generator().manual_seed(1))
    assert torch.equal(low, same)  # the same directions for one seed (and every rank)


def _lejepa_cfg(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["method"] = {"name": "lejepa", "hidden_dim": 16, "out_dim": 8, "slices": 32}
    cfg["backbone"] = {"init": "scratch", "arch": "vit_s14"}
    cfg["adaptation"] = {"type": "full"}
    cfg.pop("teacher")
    cfg["masking"] = None
    cfg["data"]["crops"] = {
        "global_crops": 2,
        "local_crops": 2,
        "global_size": 56,
        "local_size": 28,
    }
    return cfg


def test_lejepa_config_rules(tmp_path):
    from importlib.resources import files

    from pathotrain.engine import load_config

    assert validate(load_config(files("pathotrain.histossl") / "lejepa.example.yaml"))
    cfg = _lejepa_cfg(tmp_path)
    validate(cfg)
    midnight = {"backbone": {"init": "pretrained", "id": "midnight-12k"}}
    validate({**cfg, **midnight, "adaptation": {"type": "lora", "rank": 4}})  # on Midnight
    for bad, needle in (
        ({"adaptation": {"type": "lora"}}, "needs pretrained weights"),
        ({**midnight, "adaptation": {"type": "frozen"}}, "adaptation.type in"),  # nothing trains
        ({"masking": {"type": "block"}}, "takes no masking"),
        ({"teacher": {"type": "frozen"}}, "teacher.type in ['none']"),
    ):
        with pytest.raises(ConfigError) as e:
            validate({**cfg, **bad})
        assert needle in str(e.value)


def test_lejepa_trains_from_scratch_and_resumes(tmp_path, monkeypatch):
    import mlflow

    from pathotrain.histossl.methods import lejepa

    monkeypatch.setattr(lejepa, "build_backbone", _tiny)
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    cfg = _lejepa_cfg(tmp_path)

    task = build_task(cfg).train()
    batch = {"global": torch.rand(2, 4, 3, 56, 56), "local": torch.rand(2, 4, 3, 28, 28)}
    losses = task._losses(batch, None)
    assert set(losses) == {"loss", "prediction_loss", "sigreg_loss"}
    losses["loss"].backward()
    trained = task.model.backbone.named_parameters()
    # No masking in LeJEPA, so only the mask token gets no gradient.
    assert [n for n, p in trained if p.requires_grad and p.grad is None] == [
        "embeddings.mask_token"
    ]

    metrics = train(cfg)
    assert {"val/loss", "probe/balanced_acc"} <= metrics.keys()
    longer = {**cfg, "trainer": {**cfg["trainer"], "max_epochs": 3}}
    train(longer, resume=tmp_path / "run/checkpoints/last.ckpt")
    last = torch.load(tmp_path / "run/checkpoints/last.ckpt", weights_only=False)
    assert int(last["model"]["step"]) == last["step"] == 3 * 2  # seeds SIGReg's directions
    assert last["provenance"]["backbone"] == "vit_s14" and last["provenance"]["init"] == "scratch"


def test_lejepa_on_midnight_trains_only_the_adapters_and_projector(tmp_path, monkeypatch):
    from pathotrain.histossl.methods import lejepa

    monkeypatch.setattr(lejepa, "build_backbone", _tiny)
    cfg = _lejepa_cfg(tmp_path)
    cfg["backbone"] = {"init": "pretrained", "id": "midnight-12k"}
    cfg["adaptation"] = {"type": "lora", "rank": 4}
    task = build_task(cfg).train()
    trained = {n for n, p in task.model.named_parameters() if p.requires_grad}
    assert trained and all("lora_" in n or n.startswith("projector.") for n in trained)
    batch = {"global": torch.rand(2, 4, 3, 56, 56), "local": torch.rand(2, 4, 3, 28, 28)}
    task._losses(batch, None)["loss"].backward()
    assert all(p.grad is not None for n, p in task.model.named_parameters() if "lora_" in n)
