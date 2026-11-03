import csv
import json
from pathlib import Path

import mlflow
import pytest
import torch
from PIL import Image

pytest.importorskip("trl")
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from pathotrain.dataloader.vqa import VQADataset
from pathotrain.pathovlm import VLMS, VQACollator, load_processor
from pathotrain.pathovlm import eval as vlm_eval
from pathotrain.pathovlm import export as vlm_export
from pathotrain.pathovlm import sample as vlm_sample
from pathotrain.pathovlm import train as vlm_train
from pathotrain.pathovlm.callbacks import pathotrain_name
from pathotrain.pathovlm.collate import chat
from pathotrain.pathovlm.config import ConfigError, validate
from pathotrain.pathovlm.stages import (
    QwenDistillationTrainer,
    answer_reward,
    preference_pairs,
    unpaired_preferences,
)


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
    monkeypatch.setattr(vlm_export, "load_vlm", lambda name, **image: (base, processor))
    out = tmp_path / "eval"
    args = [cfg["data"]["val"], "--out", str(out), "--num-workers", "0", "--max-new-tokens", "3"]
    vlm_eval.main(["smolvlm2-500m", *args, "--adapter", str(checkpoints[-1])])
    report = json.loads((out / "metrics.json").read_text())
    assert report["adapter"] == str(checkpoints[-1]) and report["n_closed"] + report["n_open"] == 4
    with pytest.raises(SystemExit):  # adapter trained on another model
        vlm_eval.main(["qwen3-vl-2b", *args, "--adapter", str(checkpoints[-1])])

    # The merged export answers as the adapter does, from its folder alone.
    exported = vlm_export.export(checkpoints[-1], tmp_path / "exported")
    merged = AutoModelForImageTextToText.from_pretrained(exported)
    reloaded = AutoProcessor.from_pretrained(exported)
    assert reloaded.image_processor.size == processor.image_processor.size
    assert reloaded.image_processor.do_image_splitting is False
    rows = VQADataset(cfg["data"]["val"])
    answers = vlm_eval.generate_answers(merged, reloaded, rows, 4, 0, 3)
    with open(out / "predictions.csv", newline="") as f:
        assert answers == [r["prediction"] for r in csv.DictReader(f)]
    record = json.loads((exported / "pathotrain.json").read_text())
    assert record["provenance"]["model"] == "smolvlm2-500m"


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


def test_stage_args_reach_the_trl_config_and_conflicts_are_errors(tmp_path):
    cfg = _cfg(tmp_path)
    c = validate(cfg)
    assert c.stage.name == "sft"  # the default
    args = vlm_train.training_args(cfg, c)
    assert args.remove_unused_columns is False and args.learning_rate == 2e-4
    assert vlm_train.provenance(c, [])["stage"] == {"name": "sft", "preset": None}

    passed = {**cfg, "stage": {"args": {"max_grad_norm": 0.5}}}
    assert vlm_train.training_args(passed, validate(passed)).max_grad_norm == 0.5
    curated = {**cfg, "stage": {"args": {"learning_rate": 1e-3}}}  # trainer.lr sets it
    with pytest.raises(ConfigError, match="learning_rate"):
        vlm_train.training_args(curated, validate(curated))

    bad = {**cfg, "stage": {"name": "ppo"}}
    with pytest.raises(ConfigError, match="stage.name"):
        validate(bad)
    bad = {
        **cfg,
        "stage": {"preset": "mpo", "args": {"lrr": 1, "remove_unused_columns": True}},
    }
    with pytest.raises(ConfigError) as e:
        validate(bad)
    for part in ["stage.preset", "lrr", "remove_unused_columns"]:
        assert part in str(e.value)

    with pytest.raises(ConfigError, match="no provenance.json"):
        validate({**cfg, "stage": {"name": "dpo", "parent": str(tmp_path)}})
    prov = {"model": "qwen3-vl-2b", "image": {}, "prompt": {}}
    (tmp_path / "provenance.json").write_text(json.dumps(prov))
    with pytest.raises(ConfigError, match="trained on qwen3-vl-2b"):
        validate({**cfg, "stage": {"name": "dpo", "parent": str(tmp_path)}})
    with pytest.raises(ValueError, match="no samples"):  # plain VQA rows, before the run folder
        vlm_train.train({**cfg, "stage": {"name": "dpo"}})
    assert not (tmp_path / "run").exists()


def test_preference_and_rl_stages_train_from_an_sft_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_train, "load_vlm", lambda name, **image: _tiny_smolvlm())
    monkeypatch.setattr(vlm_export, "load_vlm", lambda name, **image: _tiny_smolvlm())
    cfg = {**_cfg(tmp_path), "eval": {"max_samples": 0}, "checkpoint": {"monitor": "val/loss"}}
    sft = vlm_train.train(cfg)["best_checkpoint"]

    data = {}
    for split in ("train", "val"):
        out = tmp_path / "samples" / f"{split}.jsonl"
        args = [sft, cfg["data"][split], str(out), "--n", "2", "--num-workers", "0"]
        vlm_sample.main([*args, "--max-new-tokens", "3"])
        rows = VQADataset(out).rows  # images resolve from the new folder
        assert all(len(r["samples"]) == 2 for r in rows)
        # One right and one wrong answer per question, instead of the tiny model's noise.
        lines = [json.loads(line) for line in out.read_text().splitlines()]
        out.write_text(
            "".join(json.dumps({**r, "samples": [r["answer"], "kidney"]}) + "\n" for r in lines)
        )
        data[split] = str(out)
    assert json.loads((tmp_path / "samples/train.jsonl.provenance.json").read_text())["n"] == 2

    c = validate({**cfg, "data": data, "stage": {"name": "dpo", "parent": sft}})
    rows = [{**r, "image": str(r["image"])} for r in VQADataset(data["train"]).rows]
    pairs, unpaired = preference_pairs(rows, c.prompt), unpaired_preferences(rows, c.prompt)
    assert len(pairs) == 8 and {p["rejected"][0]["content"] for p in pairs} == {"kidney"}
    assert len(unpaired) == 16 and sum(r["label"] for r in unpaired) == 8
    # TRL renders the prompt as generation (and SFT) does.
    from trl.data_utils import apply_chat_template, prepare_multimodal_messages

    _, processor = _tiny_smolvlm()
    image = Image.open(tmp_path / "img.png")
    example = {
        "prompt": prepare_multimodal_messages(pairs[0]["prompt"], images=[image]),
        "chosen": prepare_multimodal_messages(pairs[0]["chosen"]),
        "rejected": prepare_multimodal_messages(pairs[0]["rejected"]),
    }
    question = f"{rows[0]['question']}\n{c.prompt.instruction}"
    collate = VQACollator(processor, instruction=None)
    assert apply_chat_template(example, processor)["prompt"] == collate._render(
        chat(question), True
    )

    # dpo / kto on the sampled rows; grpo / rloo on plain VQA rows, 2 answers per question.
    expected = {"dpo": "val/rewards/chosen", "kto": "val/rewards/chosen"}
    expected |= {"grpo": "train/rewards/answer_reward/mean", "rloo": "train/reward"}
    expected |= {"distill": "train/loss"}  # the tiny SFT model teaches itself
    for stage, logged in expected.items():
        rl = stage in ("grpo", "rloo", "distill")  # prompts only, no val/loss
        run = {
            **cfg,
            "name": f"vqa_{stage}",
            "output_dir": str(tmp_path / stage),
            "data": cfg["data"] if rl else data,
            "eval": {"max_samples": 4, "max_new_tokens": 3, "batch_size": 4},
            "checkpoint": {"monitor": "val/closed_acc"},
            "stage": {
                "name": stage,
                "parent": sft,
                "args": {"num_generations": 2} if stage in ("grpo", "rloo") else {},
                "teacher": sft if stage == "distill" else None,
            },
        }
        best = vlm_train.train(run)["best_checkpoint"]
        prov = json.loads((Path(best) / "provenance.json").read_text())
        assert prov["stage"]["name"] == stage and prov["parent"] == str(Path(sft).resolve())
        assert prov["teacher"] == (str(Path(sft).resolve()) if stage == "distill" else None)
        with open(tmp_path / stage / "logs/version_0/metrics.csv", newline="") as f:
            columns = set(csv.DictReader(f).fieldnames)
        assert {"train/loss", logged, "val/closed_acc"} <= columns
        assert ("val/loss" in columns) is not rl  # rl stages: generated scores only
        with open(tmp_path / stage / "logs/version_0/metrics.csv", newline="") as f:
            evals = [r["step"] for r in csv.DictReader(f) if r["val/closed_acc"]]
        assert evals == sorted(set(evals))  # each evaluation once
        # Loads merged into its parent, merged into the base, e.g. for eval --adapter.
        vlm_export.load_finetuned(best)


def test_answer_reward_is_the_closed_rule_and_open_token_f1():
    completions = [
        [{"role": "assistant", "content": t}] for t in ["Yes.", "no", "lung tissue", "x"]
    ]
    rewards = answer_reward(
        completions, ["yes", "yes", "lung", "lung"], ["closed", "closed", "open", "open"]
    )
    assert rewards == [1.0, 0.0, pytest.approx(2 / 3), 0.0]


def test_rl_stages_have_no_val_loss_to_monitor(tmp_path):
    cfg = {**_cfg(tmp_path), "stage": {"name": "grpo"}, "checkpoint": {"monitor": "val/loss"}}
    with pytest.raises(ConfigError, match="grpo has no val/loss"):
        validate(cfg)


def test_distill_needs_a_teacher_with_the_students_inputs(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ConfigError, match="distill needs teacher"):
        validate({**cfg, "stage": {"name": "distill"}})
    with pytest.raises(ConfigError, match="sft takes no teacher"):
        validate({**cfg, "stage": {"teacher": str(tmp_path)}})
    # Another model is fine for a teacher, other image settings are not.
    image = {**VLMS["smolvlm2-500m"].processor, "do_image_splitting": True}
    prov = {"model": "smolvlm2-2.2b", "image": image, "prompt": validate(cfg).prompt.__dict__}
    (tmp_path / "provenance.json").write_text(json.dumps(prov))
    with pytest.raises(ConfigError) as e:
        validate({**cfg, "stage": {"name": "distill", "teacher": str(tmp_path)}})
    assert "teacher's image settings" in str(e.value) and "trained on" not in str(e.value)


def test_qwen_distillation_rebuilds_mm_token_type_ids_from_the_tokens(monkeypatch):
    from trl import DistillationTrainer

    image = 7  # the image placeholder id; the processor's ids are misaligned (left padding)
    out = {"prompt_ids": torch.tensor([[0, 0, 7, 7, 5], [7, 7, 7, 5, 5]])}
    out |= {"completion_ids": torch.tensor([[3, 2], [3, 0]])}
    out["mm_token_type_ids"] = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]])
    monkeypatch.setattr(
        DistillationTrainer, "_generate_and_score_completions", lambda s, i: {**out}
    )
    trainer = object.__new__(QwenDistillationTrainer)
    trainer._image_pad_token_id = image
    got = trainer._generate_and_score_completions(None)["mm_token_type_ids"]
    assert got.tolist() == [[0, 0, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0]]
