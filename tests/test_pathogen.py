from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("diffusers")

from pathotrain.pathogen.data import PairedTileDataset, paired_transform
from pathotrain.pathogen.train import build_dataset, build_task, load_model, train
from pathotrain.pathogen.translate import translate_dir

# Random weights, no download: the VAE halves the side, the UNet halves it again.
TINY_VAE = {
    "down_block_types": ["DownEncoderBlock2D"] * 2,
    "up_block_types": ["UpDecoderBlock2D"] * 2,
    "block_out_channels": [8, 16],
    "layers_per_block": 1,
    "norm_num_groups": 8,
}
TINY_UNET = {
    "block_out_channels": [16, 32],
    "layers_per_block": 1,
    "norm_num_groups": 8,
    "attention_head_dim": 8,
}


def _tiles(root, n=6, size=32):
    rng = np.random.default_rng(0)
    for side in ("HE", "IHC"):
        (root / side).mkdir(parents=True)
    for i in range(n):
        image = rng.integers(0, 256, (size, size, 3), np.uint8)
        Image.fromarray(image).save(root / "HE" / f"t{i}.png")
        Image.fromarray(255 - image).save(root / "IHC" / f"t{i}.png")  # learnable: the inverse
    Image.fromarray(image).save(root / "HE" / "unpaired.png")
    return root


def _cfg(tmp_path):
    data = _tiles(tmp_path / "data")
    split = {"source": [str(data / "HE")], "target": [str(data / "IHC")]}
    return {
        "name": "ldm",
        "seed": 0,
        "output_dir": str(tmp_path / "run"),
        "model": {"name": "ldm", "vae": TINY_VAE, "unet": TINY_UNET},
        "task": {"lr": 1e-3, "sample_steps": 2, "sample_batches": 1},
        "trainer": {"max_epochs": 2, "accelerator": "cpu", "devices": 1, "precision": "32-true"},
        "loader": {"batch_size": 4},
        "data": {"crop": 16, "train": split, "val": split},
    }


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
    assert source.min() >= -1 and source.max() <= 1
    assert torch.allclose(source, -target, atol=1e-6)  # 255 - x maps to -x in [-1, 1]
    with pytest.raises(ValueError, match="pairs"):
        PairedTileDataset([root / "HE"], [tmp_path])


def test_train_load_and_translate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    cfg = _cfg(tmp_path)
    metrics = train(cfg)
    assert {"val/loss", "val/psnr", "val/ssim"} <= metrics.keys()
    assert all(np.isfinite(v) for v in metrics.values())

    task, saved = load_model(tmp_path / "run/checkpoints/best.ckpt")
    assert saved == cfg
    best = torch.load(tmp_path / "run/checkpoints/best.ckpt", weights_only=True)
    # Inference runs the EMA, which checkpoints keep beside the trained weights.
    assert task.ema is None and best["ema"].keys() - best["model"].keys() == {"ema_updates"}
    assert all(torch.equal(v, best["ema"][k]) for k, v in task.model.state_dict().items())
    assert not all(torch.equal(v, best["model"][k]) for k, v in task.model.state_dict().items())

    out = translate_dir(
        tmp_path / "run/checkpoints/best.ckpt", tmp_path / "data/HE", tmp_path / "out", batch_size=3
    )
    assert len(out) == 7
    with Image.open(out[0]) as image:
        assert image.size == (32, 32) and image.mode == "RGB"


def test_validation_is_deterministic_and_the_vae_stays_frozen(tmp_path):
    cfg = _cfg(tmp_path)
    task = build_task(cfg)
    assert not any(p.requires_grad for p in task.vae.parameters())
    assert not task.train().vae.training
    batch = (torch.rand(2, 3, 16, 16) * 2 - 1, torch.rand(2, 3, 16, 16) * 2 - 1)
    task.eval()
    runs = []
    for _ in range(2):
        task.validation_step(batch)
        runs.append(task.compute_metrics())
    assert runs[0] == runs[1]
    with pytest.raises(ValueError, match="unknown model"):
        build_task({**cfg, "model": {"name": "gan"}})


def test_config_example_is_valid():
    import inspect
    from importlib.resources import files

    from pathotrain.engine import EarlyStopping, ModelCheckpoint, TrainerConfig, load_config
    from pathotrain.pathogen.diffusion import (
        LatentDiffusionTask,
        LatentTranslationTask,
        build_unet,
    )

    cfg = load_config(files("pathotrain.pathogen") / "config.example.yaml")
    assert set(cfg["model"]) == {"name", "vae", "unet"}
    build_unet(4, **cfg["model"]["unet"])
    allowed = {
        *inspect.signature(LatentDiffusionTask.__init__).parameters,
        *inspect.signature(LatentTranslationTask.__init__).parameters,
    }
    assert set(cfg["task"]) <= allowed
    TrainerConfig(**cfg["trainer"])
    ModelCheckpoint("unused", **cfg["checkpoint"])
    EarlyStopping(**cfg["early_stopping"])
    paired_transform(True, cfg["data"]["crop"], cfg["data"]["resize"])


TINY_ENCODER = {  # a random DINOv2 for REPA: 14 px patches, 16 features
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "patch_size": 14,
    "image_size": 56,
}


def _sit_cfg(tmp_path, repa=True):
    cfg = _cfg(tmp_path)
    cfg["model"] = {
        "name": "sit",
        "vae": TINY_VAE,
        "hidden_size": 32,
        "depth": 2,
        "num_heads": 2,
        "repa": {"encoder": TINY_ENCODER, "depth": 1, "weight": 0.5} if repa else None,
    }
    cfg["task"] = {"lr": 1e-3, "sample_steps": 2, "sample_batches": 1, "sampler": "heun"}
    return cfg


def test_sit_shapes_and_repa_features():
    from pathotrain.pathogen.flow import SiT

    sit = SiT(8, 4, hidden_size=32, depth=2, num_heads=2, repa_dim=16, repa_depth=1)
    x, t = torch.randn(3, 8, 12, 8), torch.rand(3)
    out, projected = sit.train()(x, t)
    assert out.shape == (3, 4, 12, 8) and projected.shape == (3, 6 * 4, 16)
    assert torch.equal(out, torch.zeros_like(out))  # adaLN-Zero: zero velocity at init
    assert sit.eval()(x, t)[1] is None  # REPA only in training
    with pytest.raises(ValueError, match="repa_depth"):
        SiT(8, 4, depth=2, repa_dim=16, repa_depth=3)


@pytest.mark.parametrize("sampler", ["euler", "heun"])
def test_sampling_follows_the_flow_to_the_data(tmp_path, sampler):
    # With the true velocity of x_t = (1 - t) x_0 + t noise, the ODE is a straight line:
    # every step lands on it, so the sample is x_0 whatever the step count.
    task = build_task(_sit_cfg(tmp_path, repa=False))
    task.sampler = sampler
    x0 = torch.randn(2, 4, 8, 8)

    class Exact(torch.nn.Module):
        def forward(self, x, t):
            t = t[:, None, None, None]
            return (x[:, :4] - x0) / t, None

    task.model, task.ema = Exact(), None
    sample = task._sample(torch.zeros(2, 4, 8, 8), 3, torch.Generator().manual_seed(0))
    assert torch.allclose(sample, x0, atol=1e-5)


def test_repa_aligns_with_the_encoder(tmp_path):
    task = build_task(_sit_cfg(tmp_path)).train()
    assert not any(p.requires_grad for p in task.encoder.parameters())
    batch = (torch.rand(2, 3, 16, 16) * 2 - 1, torch.rand(2, 3, 16, 16) * 2 - 1)
    loss, logs = task.training_step(batch)
    assert set(logs) == {"loss", "flow_loss", "repa_loss"}
    assert 0 <= logs["repa_loss"] <= 2
    assert torch.isclose(loss, logs["flow_loss"] + 0.5 * logs["repa_loss"])
    loss.backward()
    assert task.model.projector[0].weight.grad.abs().sum() > 0


def test_train_sit_load_and_translate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    cfg = _sit_cfg(tmp_path)
    metrics = train(cfg)
    assert {"val/loss", "val/psnr", "val/ssim"} <= metrics.keys()
    best = torch.load(tmp_path / "run/checkpoints/best.ckpt", weights_only=True)["ema"]
    assert any(k.startswith("projector.") for k in best)

    task, _ = load_model(tmp_path / "run/checkpoints/best.ckpt")
    assert task.encoder is None and task.model.projector is None  # training-only parts
    assert all(torch.equal(v, best[k]) for k, v in task.model.state_dict().items())
    out = translate_dir(
        tmp_path / "run/checkpoints/best.ckpt", tmp_path / "data/HE", tmp_path / "out"
    )
    assert len(out) == 7


def test_ema_follows_the_weights_with_warmup(tmp_path):
    cfg = _cfg(tmp_path)
    task = build_task({**cfg, "task": {"ema_decay": 0.9}})
    before = [p.clone() for p in task.ema.parameters()]
    with torch.no_grad():
        for p in task.model.parameters():
            p.add_(1.0)
    task.on_optimizer_step()  # first update: decay min(0.9, 2 / 11)
    decay = 2 / 11
    for old, ema, live in zip(before, task.ema.parameters(), task.model.parameters(), strict=True):
        assert torch.allclose(ema, decay * old + (1 - decay) * live)
    assert task.ema.ema_updates == 1 and task.checkpoint_state() == {"ema": task.ema}
    assert task.net is task.ema
    no_ema = build_task({**cfg, "task": {"ema_decay": None}})
    assert no_ema.net is no_ema.model and no_ema.checkpoint_state() == {}


def test_resume_restores_the_ema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    cfg = _cfg(tmp_path)
    train(cfg)
    last = torch.load(tmp_path / "run/checkpoints/last.ckpt", weights_only=True)
    assert last["ema"]["ema_updates"] == last["step"] == 4  # 2 epochs of 2 steps
    longer = {**cfg, "trainer": {**cfg["trainer"], "max_epochs": 3}}
    train(longer, resume=tmp_path / "run/checkpoints/last.ckpt")
    last = torch.load(tmp_path / "run/checkpoints/last.ckpt", weights_only=True)
    assert last["ema"]["ema_updates"] == last["step"] == 6


def test_dihedral_gives_the_eight_flips_and_turns():
    from pathotrain.pathogen.cache import dihedral

    x = torch.arange(6.0).view(1, 2, 3)
    assert torch.equal(dihedral(x, 0), x) and torch.equal(dihedral(x, 1), x.flip(-1))
    square = torch.arange(9.0).view(3, 3)
    assert len({tuple(dihedral(square, k).flatten().tolist()) for k in range(8)}) == 8


def test_latent_cache_matches_the_vae_and_trains(tmp_path, monkeypatch):
    from pathotrain.pathogen.cache import (
        CachedLatentDataset,
        build_latent_cache,
        dihedral,
    )
    from pathotrain.pathogen.data import read_image, to_model_range
    from pathotrain.pathogen.diffusion import build_vae

    cfg = _sit_cfg(tmp_path)
    train_dirs = cfg["data"]["train"]
    torch.manual_seed(0)  # the tiny VAE's random weights: the same in the cache and below
    cache = build_latent_cache(
        train_dirs["source"], train_dirs["target"], tmp_path / "cache", TINY_VAE, variants=2
    )
    latents = np.load(cache / "latents.npy")
    assert latents.shape == (6, 2, 2, 4, 16, 16)  # pairs, variants, sides, C, 32 px / 2
    torch.manual_seed(0)
    vae = build_vae(TINY_VAE)
    tile = to_model_range(read_image(tmp_path / "data/HE/t0.png"))[None]
    for k in range(2):  # each variant encodes the flipped tile, not a flipped latent
        want = vae.encode(dihedral(tile, k)).latent_dist.mean[0]
        assert torch.allclose(torch.from_numpy(latents[0, k, 0]).float(), want, atol=1e-2)

    ds = CachedLatentDataset(cache, crop=16, image="target", vae=TINY_VAE)
    sample = ds[0]
    assert sample["source"].shape == sample["target"].shape == (4, 8, 8)
    assert sample["image"].shape == (3, 16, 16)
    whole = CachedLatentDataset(cache, image="target", vae=TINY_VAE)
    ihc = to_model_range(read_image(tmp_path / "data/IHC/t0.png"))
    image = whole[0]["image"]  # whole tile: identity or flipped, matching its latents
    assert torch.equal(image, ihc) or torch.equal(image, ihc.flip(-1))
    with pytest.raises(ValueError, match="VAE"):
        CachedLatentDataset(cache, vae="stabilityai/sd-vae-ft-mse")
    with pytest.raises(ValueError, match="built from source"):
        CachedLatentDataset(cache, vae=TINY_VAE, source_dirs=["elsewhere"])
    with pytest.raises(ValueError, match="multiple of 2"):
        CachedLatentDataset(cache, crop=15, vae=TINY_VAE)
    with pytest.raises(FileNotFoundError, match="pathotrain.pathogen.cache"):
        CachedLatentDataset(tmp_path / "none", vae=TINY_VAE)

    # SiT with REPA trains from the cache (REPA reads the cached sample's image).
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    cfg["data"]["cache_dir"] = str(cache)
    cfg["data"]["cache"] = True
    assert {"val/loss", "val/ssim"} <= train(cfg).keys()


def test_cache_toggle_builds_once_per_fingerprint(tmp_path, monkeypatch):
    from pathotrain.pathogen.cache import cache_dir_for, ensure_latent_cache

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "home_cache"))
    cfg = _cfg(tmp_path)
    cfg["data"]["cache"] = True
    where = cache_dir_for(cfg)
    assert where.parent == tmp_path / "home_cache/pathotrain/latents"
    assert ensure_latent_cache(cfg) == where and (where / "meta.json").exists()
    built = (where / "latents.npy").stat().st_mtime_ns
    assert ensure_latent_cache(cfg) == where  # found, not rebuilt
    assert (where / "latents.npy").stat().st_mtime_ns == built
    # Anything the latents depend on gives another cache.
    assert cache_dir_for({**cfg, "data": {**cfg["data"], "cache_variants": 1}}) != where
    (tmp_path / "data/HE/t9.png").write_bytes((tmp_path / "data/HE/t0.png").read_bytes())
    (tmp_path / "data/IHC/t9.png").write_bytes((tmp_path / "data/IHC/t0.png").read_bytes())
    assert cache_dir_for(cfg) != where
    assert cache_dir_for({**cfg, "data": {**cfg["data"], "cache_dir": "x"}}) == Path("x")
    with pytest.raises(ValueError, match="resize"):
        build_dataset({**cfg, "data": {**cfg["data"], "resize": 32}}, "train")
    assert isinstance(build_dataset(cfg, "val"), PairedTileDataset)  # val reads images
