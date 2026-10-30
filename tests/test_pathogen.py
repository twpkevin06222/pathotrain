import mlflow
import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("diffusers")

from pathotrain.pathogen.data import PairedTileDataset, paired_transform
from pathotrain.pathogen.train import build_task, load_model, train
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
    best = torch.load(tmp_path / "run/checkpoints/best.ckpt", weights_only=True)["model"]
    assert all(torch.equal(v, best[k]) for k, v in task.model.state_dict().items())

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
    from pathotrain.pathogen.diffusion import LatentDiffusionTask, build_unet

    cfg = load_config(files("pathotrain.pathogen") / "config.example.yaml")
    assert set(cfg["model"]) == {"name", "vae", "unet"}
    build_unet(4, **cfg["model"]["unet"])
    assert set(cfg["task"]) <= set(inspect.signature(LatentDiffusionTask.__init__).parameters)
    TrainerConfig(**cfg["trainer"])
    ModelCheckpoint("unused", **cfg["checkpoint"])
    EarlyStopping(**cfg["early_stopping"])
    paired_transform(True, cfg["data"]["crop"], cfg["data"]["resize"])
