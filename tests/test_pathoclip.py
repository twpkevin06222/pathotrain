import csv
import json

import mlflow
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

pytest.importorskip("peft")
from transformers import AutoModel, AutoProcessor, SiglipConfig, SiglipModel

from pathotrain.dataloader.caption import CaptionDataset
from pathotrain.pathoclip import CLIPS
from pathotrain.pathoclip import eval as clip_eval
from pathotrain.pathoclip import export as clip_export
from pathotrain.pathoclip import train as clip_train
from pathotrain.pathoclip.config import ConfigError, validate
from pathotrain.pathoclip.task import retrieval, sigmoid_loss


def _tiny_siglip():
    """SigLIP 2 base's processor (its real tokenizer) with a ~9M-parameter model."""
    clip = CLIPS["siglip2-base-224"]
    try:
        processor = AutoProcessor.from_pretrained(
            clip.repo, revision=clip.revision, local_files_only=True
        )
    except OSError:
        pytest.skip("siglip2-base-224's processor is not in the Hugging Face cache")
    small = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1}
    cfg = SiglipConfig(
        text_config={**small, "num_attention_heads": 2, "vocab_size": len(processor.tokenizer)},
        vision_config={**small, "num_attention_heads": 2, "image_size": 224, "patch_size": 16},
    )
    torch.manual_seed(0)
    return SiglipModel(cfg), processor


def _captions(path, n, groups=None):
    colours = ["pink", "purple", "white", "navy"]
    for c in colours:
        Image.new("RGB", (64, 48), c).save(path.parent / f"{c}.png")
    rows = [
        {"image": f"{colours[i % 4]}.png", "caption": f"a {colours[i % 4]} field, case {i}."}
        | ({"group": groups[i]} if groups else {})
        for i in range(n)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _tiles(root):
    for label, colour in (("tumour", "purple"), ("stroma", "pink")):
        (root / label).mkdir(parents=True, exist_ok=True)
        for i in range(3):
            Image.new("RGB", (32, 32), colour).save(root / label / f"{i}.png")
    return str(root)


def _cfg(tmp_path):
    return {
        "name": "clip_test",
        "seed": 0,
        "output_dir": str(tmp_path / "run"),
        "mlflow_db": str(tmp_path / "mlflow.db"),
        "model": {"name": "siglip2-base-224"},
        "adaptation": {"rank": 4, "alpha": 8},
        "data": {
            "train": _captions(tmp_path / "train.jsonl", 8),
            "val": _captions(tmp_path / "val.jsonl", 4, groups=["f1", "f1", "f2", "f3"]),
        },
        "eval": {
            "test": _tiles(tmp_path / "tiles"),
            "train": _tiles(tmp_path / "tiles"),
            "num_workers": 0,
        },
        "task": {"warmup_epochs": 0},
        "trainer": {"max_epochs": 2, "accelerator": "cpu", "precision": "32-true"},
        "loader": {"batch_size": 4, "num_workers": 0},
    }


def test_retrieval_counts_shared_captions_as_matches():
    images = torch.eye(3)
    texts = torch.eye(3)[[0, 2, 1]]  # caption 1 and 2 swapped
    assert retrieval(images, texts, torch.tensor([0, 1, 2]))["i2t_r1"] == pytest.approx(1 / 3)
    # Pairs 1 and 2 share a caption: the swap is no longer a miss.
    out = retrieval(images, texts, torch.tensor([0, 1, 1]))
    assert out["i2t_r1"] == out["t2i_r1"] == 1.0 and out["recall_mean"] == 1.0


def test_sigmoid_loss_is_siglips_with_distinct_captions():
    model, _ = _tiny_siglip()
    pixels, ids = torch.randn(3, 3, 224, 224), torch.randint(0, 100, (3, 64))
    with torch.no_grad():
        out = model(pixel_values=pixels, input_ids=ids, return_loss=True)
    assert sigmoid_loss(out.logits_per_text, torch.arange(3)) == pytest.approx(out.loss.item())
    # A shared caption makes the off-diagonal pair a positive: a different loss.
    assert sigmoid_loss(out.logits_per_text, torch.tensor([0, 0, 1])) != pytest.approx(
        out.loss.item()
    )


def test_caption_groups_default_to_the_caption(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(
        "".join(json.dumps({"image": "x.png", "caption": c}) + "\n" for c in ["a", "b", "a"])
    )
    assert [r["group"] for r in CaptionDataset(path).rows] == [0, 1, 0]


def test_lora_run_logs_checkpoints_and_exports(tmp_path, monkeypatch):
    model, processor = _tiny_siglip()
    monkeypatch.setattr(clip_train, "load_clip", lambda name: (model, processor))
    cfg = _cfg(tmp_path)

    clip_train.train(cfg)

    run = tmp_path / "run"
    with open(run / "logs/version_0/metrics.csv", newline="") as f:
        columns = set(csv.DictReader(f).fieldnames)
    expected = {"train/loss", "train/temperature", "val/loss", "val/i2t_r1", "val/t2i_r10"}
    expected |= {"val/recall_mean", "zs/balanced_acc", "probe/balanced_acc", "knn/acc"}
    assert expected <= columns
    ckpt = torch.load(run / "checkpoints/best.ckpt", weights_only=True)
    assert ckpt["provenance"]["base"].startswith("google/siglip2-base-patch16-224@")
    assert ckpt["provenance"]["text_length"] == 64
    # LoRA on the blocks of both towers, plus the logit scale and bias; nothing else saved.
    keys = ckpt["model"].keys()
    assert any("vision_model" in k and "lora_A" in k for k in keys)
    assert any("text_model" in k and "lora_A" in k for k in keys)
    assert any(k.endswith("logit_scale") for k in keys)
    assert not any("head.attention" in k and "lora" in k for k in keys)
    client = mlflow.MlflowClient(f"sqlite:///{tmp_path}/mlflow.db")
    (mlrun,) = client.search_runs([client.get_experiment_by_name("clip_test").experiment_id])
    assert mlrun.info.status == "FINISHED"

    # The checkpoint reloads onto a fresh base, exports, and the export embeds as it does.
    base, _ = _tiny_siglip()
    monkeypatch.setattr(clip_train, "load_clip", lambda name: (base, processor))
    trained, _, _ = clip_train.load_model(run / "checkpoints/best.ckpt")
    out = clip_export.export(run / "checkpoints/best.ckpt", tmp_path / "exported")
    plain, reloaded = AutoModel.from_pretrained(out), AutoProcessor.from_pretrained(out)
    pixels = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        a = trained.get_image_features(pixel_values=pixels).pooler_output
        b = plain.get_image_features(pixel_values=pixels).pooler_output
    assert torch.allclose(F.normalize(a, dim=-1), F.normalize(b, dim=-1), atol=1e-5)
    assert reloaded.image_processor.size == processor.image_processor.size
    assert json.loads((out / "pathotrain.json").read_text())["lora_merged"] is True

    # The eval CLI scores the export without any pathotrain checkpoint.
    clip_eval.main(
        [
            str(out),
            "--captions",
            cfg["data"]["val"],
            "--test",
            cfg["eval"]["test"],
            "--num-workers",
            "0",
        ]
    )


def test_config_errors_are_collected(tmp_path):
    cfg = _cfg(tmp_path)
    assert validate(cfg).adaptation.type == "lora"
    bad = {
        **cfg,
        "model": {"name": "clip9"},
        "adaptation": {"type": "last_n", "targets": ["qkv"]},
        "data": {"train": "nope.jsonl", "val": cfg["data"]["val"]},
        "eval": {"test": "missing/", "template": "an image"},
        "task": {"lrr": 1},
    }
    with pytest.raises(ConfigError) as e:
        validate(bad)
    for part in ["model.name", "adaptation.type", "adaptation.targets", "data.train", "eval.test"]:
        assert part in str(e.value)
    assert "template" in str(e.value) and "lrr" in str(e.value)
    no_eval = {k: v for k, v in cfg.items() if k != "eval"}
    with pytest.raises(ConfigError, match="needs an eval section"):
        validate({**no_eval, "checkpoint": {"monitor": "zs/balanced_acc"}})


def test_example_config_is_valid(tmp_path):
    from importlib.resources import files

    from pathotrain.engine import load_config

    cfg = load_config(files("pathotrain.pathoclip") / "configs" / "siglip2-base.example.yaml")
    data = {"train": _captions(tmp_path / "t.jsonl", 2), "val": _captions(tmp_path / "v.jsonl", 2)}
    parsed = validate(
        {**cfg, "data": data, "eval": {**cfg["eval"], "test": _tiles(tmp_path / "x")}}
    )
    assert parsed.model.name in CLIPS and set(parsed.eval.prompts) >= {"TUM", "STR"}


def test_arch_export_keeps_a_figures_panels_in_one_split(tmp_path):
    from pathotrain.dataloader.caption import export_arch

    for subset, entries in {
        "books_set": {
            str(i): {"uuid": f"b{i}", "caption": f"fig {i // 3}", "figure_id": str(i // 3)}
            for i in range(30)
        },
        "pubmed_set": {str(i): {"uuid": f"p{i}", "caption": f"case {i}"} for i in range(20)},
    }.items():
        (tmp_path / subset / "images").mkdir(parents=True)
        for e in entries.values():
            Image.new("RGB", (8, 8)).save(tmp_path / subset / "images" / f"{e['uuid']}.png")
        (tmp_path / subset / "captions.json").write_text(json.dumps(entries))
    (tmp_path / "books_set" / "images" / "b0.png").unlink()  # a caption without its image

    stats = export_arch(tmp_path, tmp_path / "out")
    assert sum(s["images"] for s in stats.values()) == 49
    splits = {split: CaptionDataset(tmp_path / "out" / f"{split}.jsonl") for split in stats}
    groups = {
        split: {r["caption"] for r in d.rows if r["source"] == "books_set"}
        for split, d in splits.items()
    }
    assert not (groups["train"] & groups["val"]) and not (groups["train"] & groups["test"])
    assert splits["train"][0]["image"].size == (8, 8)  # paths resolve from the output folder


@pytest.mark.parametrize("kind", ["lora", "full"])
def test_text_only_adaptation_keeps_the_image_tower(tmp_path, kind, monkeypatch):
    model, processor = _tiny_siglip()
    monkeypatch.setattr(clip_train, "load_clip", lambda name: (model, processor))
    cfg = _cfg(tmp_path)
    cfg = {**cfg, "adaptation": {**cfg["adaptation"], "type": kind, "towers": ["text"]}}
    cfg = {**cfg, "trainer": {**cfg["trainer"], "max_epochs": 1}}
    cfg.pop("eval")
    before = {k: v.clone() for k, v in model.vision_model.state_dict().items()}

    clip_train.train(cfg)

    ckpt = torch.load(tmp_path / "run/checkpoints/last.ckpt", weights_only=True)
    trained = [k for k, v in ckpt["model"].items() if "lora_" in k] if kind == "lora" else []
    assert all("text_model" in k for k in trained)
    after = model.vision_model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert not any(p.requires_grad for p in model.vision_model.parameters())
    assert ckpt["provenance"]["adaptation"]["towers"] == ["text"]
    with pytest.raises(ConfigError, match="adaptation.towers"):
        validate({**cfg, "adaptation": {"towers": ["audio"]}})
