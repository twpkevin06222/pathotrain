import csv
import json

import mlflow
import pytest
import torch
from PIL import Image

pytest.importorskip("trl")
from transformers import AutoConfig, AutoModelForImageTextToText

from pathotrain.pathovlm import VLMS, load_processor
from pathotrain.pathovlm import eval as vlm_eval
from pathotrain.pathovlm import train as vlm_train
from pathotrain.pathovlm.callbacks import pathotrain_name
from pathotrain.pathovlm.config import ConfigError, validate


def _tiny_smolvlm():
    """SmolVLM2-500M's architecture shrunk to ~3M parameters, with its real processor."""
    vlm = VLMS["smolvlm2-500m"]
    try:
        processor = load_processor("smolvlm2-500m", local_files_only=True)
        cfg = AutoConfig.from_pretrained(vlm.repo, revision=vlm.revision, local_files_only=True)
    except OSError:
        pytest.skip("smolvlm2-500m's processor and config are not in the Hugging Face cache")
    small = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1}
    cfg.text_config.update({**small, "num_attention_heads": 2, "num_key_value_heads": 1})
    cfg.text_config.head_dim = 16
    cfg.vision_config.update({**small, "num_attention_heads": 2})
    torch.manual_seed(0)
    return AutoModelForImageTextToText.from_config(cfg), processor


def _rows(path, n):
    (path.parent / "img.png").exists() or Image.new("RGB", (64, 48), "pink").save(
        path.parent / "img.png"
    )
    answers = ["yes", "no", "necrosis", "lung"]
    path.write_text(
        "".join(
            json.dumps({"image": "img.png", "question": f"q{i}?", "answer": answers[i % 4]}) + "\n"
            for i in range(n)
        )
    )
    return str(path)


def _cfg(tmp_path):
    return {
        "name": "vqa_test",
        "seed": 0,
        "output_dir": str(tmp_path / "run"),
        "mlflow_db": str(tmp_path / "mlflow.db"),
        "model": {"name": "smolvlm2-500m"},
        "adaptation": {"rank": 4, "alpha": 8, "targets": ["q_proj", "v_proj"]},
        "data": {
            "train": _rows(tmp_path / "train.jsonl", 8),
            "val": _rows(tmp_path / "val.jsonl", 4),
        },
        "eval": {"max_samples": 4, "max_new_tokens": 3, "batch_size": 4},
        "trainer": {
            "epochs": 1,
            "batch_size": 2,
            "eval_steps": 2,
            "logging_steps": 1,
            "num_workers": 0,
        },
    }


def test_trl_run_logs_like_pathotrain_and_saves_adapter_checkpoints(tmp_path, monkeypatch):
    model, processor = _tiny_smolvlm()
    monkeypatch.setattr(vlm_train, "load_vlm", lambda name, **image: (model, processor))
    cfg = _cfg(tmp_path)

    result = vlm_train.train(cfg)

    run = tmp_path / "run"
    assert (run / "config.yaml").exists() and (run / "train.log").exists()
    checkpoints = sorted(run.glob("checkpoint-*"))
    assert [c.name for c in checkpoints] == ["checkpoint-2", "checkpoint-4"]  # 8 rows / 2 per step
    for c in checkpoints:
        assert (c / "adapter_model.safetensors").exists()
        prov = json.loads((c / "provenance.json").read_text())
        assert (
            prov["base"].startswith("HuggingFaceTB/SmolVLM2-500M")
            and prov["adaptation"]["rank"] == 4
        )
        assert prov["prompt"]["instruction"].startswith("Answer the question")
    adapter = json.loads((checkpoints[0] / "adapter_config.json").read_text())
    assert all("text_model" in m for m in adapter["target_modules"])  # not the vision tower
    assert result["best_checkpoint"] in {str(c) for c in checkpoints}

    with open(run / "logs/version_0/metrics.csv", newline="") as f:
        columns = set(csv.DictReader(f).fieldnames)
    assert {"train/loss", "train/lr", "val/loss", "val/closed_acc", "val/open_f1"} <= columns
    client = mlflow.MlflowClient(f"sqlite:///{tmp_path}/mlflow.db")
    (mlrun,) = client.search_runs([client.get_experiment_by_name("vqa_test").experiment_id])
    assert mlrun.info.status == "FINISHED"
    assert {"val/closed_acc", "best/val/closed_acc"} <= mlrun.data.metrics.keys()
    assert mlrun.data.params["best_checkpoint"] == result["best_checkpoint"]

    base, _ = _tiny_smolvlm()  # the trained model now holds LoRA layers
    monkeypatch.setattr(vlm_eval, "load_vlm", lambda name, **image: (base, processor))
    out = tmp_path / "eval"
    args = [cfg["data"]["val"], "--out", str(out), "--num-workers", "0", "--max-new-tokens", "3"]
    vlm_eval.main(["smolvlm2-500m", *args, "--adapter", str(checkpoints[-1])])
    report = json.loads((out / "metrics.json").read_text())
    assert report["adapter"] == str(checkpoints[-1]) and report["n_closed"] + report["n_open"] == 4
    with pytest.raises(SystemExit):  # adapter trained on another model
        vlm_eval.main(["qwen3-vl-2b", *args, "--adapter", str(checkpoints[-1])])


def test_log_keys_get_pathotrain_names():
    assert pathotrain_name("loss") == "train/loss"
    assert pathotrain_name("learning_rate") == "train/lr"
    assert pathotrain_name("eval_closed_acc") == "val/closed_acc"
    assert pathotrain_name("train_runtime") == "run/runtime"
    assert pathotrain_name("train_loss") == "run/loss"
    assert pathotrain_name("epoch") == "epoch"


def test_config_errors_are_collected(tmp_path):
    cfg = _cfg(tmp_path)
    assert validate(cfg).checkpoint.monitor == "val/closed_acc"
    bad = {
        **cfg,
        "model": {"name": "qwen9"},
        "adaptation": {"rank": 0, "targets": ["qkv"]},
        "data": {"train": "nope.jsonl", "val": cfg["data"]["val"]},
        "checkpoint": {"monitor": "val/miou"},
        "trainer": {"epochs": 1, "lrr": 1e-4},
    }
    with pytest.raises(ConfigError) as e:
        validate(bad)
    message = str(e.value)
    for part in [
        "model.name",
        "rank and alpha",
        "adaptation.targets",
        "data.train",
        "monitor",
        "lrr",
    ]:
        assert part in message
    with pytest.raises(ConfigError, match="needs eval.max_samples"):
        validate({**cfg, "eval": {"max_samples": 0}})


@pytest.mark.parametrize("name", ["qwen3-vl-2b", "qwen3-vl-4b"])
def test_example_configs_are_valid(tmp_path, name):
    from importlib.resources import files

    from pathotrain.engine import load_config

    cfg = load_config(files("pathotrain.pathovlm") / "configs" / f"{name}.example.yaml")
    data = {"train": _rows(tmp_path / "t.jsonl", 2), "val": _rows(tmp_path / "v.jsonl", 2)}
    parsed = validate({**cfg, "data": {**cfg["data"], **data}})
    assert parsed.model.name == name and parsed.checkpoint.monitor == "val/closed_acc"
