"""LoRA fine-tuning of a VLM on VQA rows with a TRL trainer (SFT, DPO, KTO, GRPO, RLOO,
distillation, online DPO), from one YAML::

    python -m pathotrain.pathovlm.train config.yaml
    python -m pathotrain.pathovlm.train config.yaml --resume runs/vqa01/checkpoint-1000

Start from ``configs/qwen3-vl-2b.example.yaml`` next to this module. The config is
checked first (:func:`pathotrain.pathovlm.config.validate`) and copied to
``<output_dir>/config.yaml`` with a ``train.log`` beside it, as for every
pathotrain run. Sections (``?`` marks optional ones)::

    name, seed, output_dir, mlflow_db?
    model: {name: qwen3-vl-2b, image?: {...}}    # image-processor settings over the registry's
    adaptation?: {rank: 16, alpha: 32, dropout: 0.05, targets: [q_proj, ...]}
    prompt?: {system: null, instruction: "Answer the question using a single word or phrase."}
    data: {train: pathvqa/train.jsonl, val: pathvqa/validation.jsonl, val_max_samples?: 1024}
    eval?: {max_samples: 2048, max_new_tokens: 32, batch_size: 16}
    trainer?: {epochs, lr, batch_size, accumulate, warmup, weight_decay, eval_steps, ...}
    checkpoint?: {monitor: val/closed_acc}       # or val/loss, val/open_em, val/open_f1
    stage?: {name: sft, preset: null, args: {}, parent: null, teacher: null}  # see .stages

Every ``eval_steps`` optimizer steps: ``val/loss`` on ``val_max_samples`` rows,
answers generated for ``eval.max_samples`` rows (``val/closed_acc``,
``val/open_em``, ``val/open_f1``, ``val/open_recall``) and a ``checkpoint-N/``
folder: the LoRA adapter (``adapter_model.safetensors``), trainer state and
``provenance.json``. The best one by ``checkpoint.monitor`` is kept with the
``save_total_limit`` latest and loaded at the end. Logs go to MLflow
(``mlflow_db``) and ``logs/version_0/metrics.csv`` under pathotrain names.
"""

import argparse
from dataclasses import fields
from importlib.metadata import version
from pathlib import Path

import datasets
import numpy as np
import torch
from peft import LoraConfig
from torch import nn

from pathotrain import experiment
from pathotrain.dataloader.vqa import VQADataset
from pathotrain.engine import CSVLogger, MLflowLogger, flatten, load_config
from pathotrain.engine.logger import sqlite_store
from pathotrain.pathovlm.callbacks import PathotrainLogging, Provenance
from pathotrain.pathovlm.config import MONITORS, ConfigError, VLMConfig, validate
from pathotrain.pathovlm.eval import generate_answers
from pathotrain.pathovlm.export import load_finetuned
from pathotrain.pathovlm.metrics import score
from pathotrain.pathovlm.models import VLMS, load_vlm
from pathotrain.pathovlm.stages import STAGES


def lora_targets(model: nn.Module, names: tuple[str, ...]) -> list[str]:
    """Full names of the language model's linear layers called ``names``: plain
    suffixes would also adapt the vision tower (SmolVLM's has ``q_proj`` too)."""
    language = {id(m) for m in model.get_decoder().modules()}
    return sorted(
        name
        for name, m in model.named_modules()
        if id(m) in language and isinstance(m, nn.Linear) and name.rsplit(".", 1)[-1] in names
    )


class GenerationEval:
    """Mixed into a stage's trainer class: its evaluation also answers the
    ``generation_set`` questions by generation and adds their scores
    (:func:`~pathotrain.pathovlm.metrics.score`), so they are logged and can choose
    the best checkpoint, the same for every stage. With ``own_eval=False`` the
    trainer's own evaluation (its loss) is skipped."""

    def __init__(self, *args, generation_set=None, generation_kwargs=None, own_eval=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_set, self.generation_kwargs = generation_set, generation_kwargs or {}
        self.own_eval = own_eval

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = (
            super().evaluate(eval_dataset, ignore_keys, metric_key_prefix) if self.own_eval else {}
        )
        if self.generation_set is None:
            return metrics
        training = self.model.training
        self.model.eval()
        predictions = generate_answers(
            self.model, self.processing_class, self.generation_set, **self.generation_kwargs
        )
        self.model.train(training)
        rows = self.generation_set
        scores = score(predictions, list(rows["answer"]), list(rows["type"]))
        extra = {f"{metric_key_prefix}_{k}": v for k, v in scores.items() if not k.startswith("n_")}
        self.log(extra)
        if not self.own_eval:  # what the skipped evaluate does: callbacks, and no second evaluation
            self.control = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, extra
            )
        return {**metrics, **extra}


def hf_dataset(jsonl: str, n: int | None = None, build=None) -> datasets.Dataset:
    """The rows of a VQA JSONL (checked by :class:`VQADataset`), or ``n`` evenly
    spaced ones, as a Hugging Face dataset (what TRL takes) whose ``image`` column
    decodes to an RGB PIL image on access, as :class:`VQADataset` returns it.
    ``build`` turns the rows into a stage's records first."""
    rows = VQADataset(jsonl).rows
    if n is not None and len(rows) > n:
        rows = [rows[i] for i in np.linspace(0, len(rows) - 1, n).round().astype(int)]
    records = [{**r, "image": str(r["image"])} for r in rows]
    table = datasets.Dataset.from_list(build(records) if build else records)
    return table.cast_column("image", datasets.Image(mode="RGB"))


def training_args(cfg: dict, c: VLMConfig):
    """The stage's TRL config: pathotrain's sections, then the preset, then
    ``stage.args``, then the values the stage fixes."""
    spec, t, monitor = STAGES[c.stage.name], c.trainer, c.checkpoint.monitor
    curated = {
        "output_dir": str(cfg["output_dir"]),
        "seed": cfg["seed"],
        "num_train_epochs": t.epochs,
        "learning_rate": t.lr,
        "per_device_train_batch_size": t.batch_size,
        "per_device_eval_batch_size": t.batch_size,
        "gradient_accumulation_steps": t.accumulate,
        "warmup_steps": t.warmup,
        "weight_decay": t.weight_decay,
        "lr_scheduler_type": "cosine",
        "logging_steps": t.logging_steps,
        "logging_first_step": True,
        "eval_strategy": "steps",
        "eval_steps": t.eval_steps,
        "save_strategy": "steps",
        "save_steps": t.eval_steps,
        "save_total_limit": t.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": monitor.removeprefix("val/"),
        "greater_is_better": MONITORS[monitor],
        "bf16": torch.cuda.is_available(),
        "gradient_checkpointing": t.grad_checkpointing,
        "dataloader_num_workers": t.num_workers,
        "report_to": "none",  # logged by PathotrainLogging instead
    }
    # Stages that generate while training: answers as long as evaluation's.
    for length in ("max_completion_length", "max_new_tokens"):  # online_dpo calls it max_new_tokens
        if length in {f.name for f in fields(spec.config_cls)}:
            curated[length] = c.eval.max_new_tokens
    taken = sorted(set(c.stage.args) & set(curated))
    if taken:
        raise ConfigError(f"stage.args {taken} are set by pathotrain's sections (trainer, ...)")
    preset = spec.presets.get(c.stage.preset, {})
    return spec.config_cls(**{**curated, **preset, **c.stage.args, **spec.fixed})


def provenance(c: VLMConfig, targets: list[str]) -> dict:
    """What an adapter checkpoint must say about how it was trained."""
    vlm = VLMS[c.model.name]
    return {
        "model": c.model.name,
        "base": f"{vlm.repo}@{vlm.revision}",
        "stage": {"name": c.stage.name, "preset": c.stage.preset},
        "parent": str(Path(c.stage.parent).expanduser().resolve()) if c.stage.parent else None,
        "teacher": str(Path(c.stage.teacher).expanduser().resolve()) if c.stage.teacher else None,
        "image": vlm.processor | c.model.image,
        "adaptation": {
            "type": "lora",
            "rank": c.adaptation.rank,
            "alpha": c.adaptation.alpha,
            "dropout": c.adaptation.dropout,
            "targets": list(c.adaptation.targets),
            "modules": len(targets),
        },
        "prompt": {"system": c.prompt.system, "instruction": c.prompt.instruction},
        "git_commit": experiment.git_commit(),
        "pathotrain_version": version("pathotrain"),
        "trl_version": version("trl"),
    }


def train(cfg: dict, resume: str | Path | None = None) -> dict:
    """Run the fine-tuning ``cfg`` describes; return the best metric and checkpoint."""
    c = validate(cfg)
    args = training_args(cfg, c)  # before experiment.start: bad TRL values and data fail first
    spec, monitor = STAGES[c.stage.name], c.checkpoint.monitor
    build = spec.build_dataset and (lambda rows: spec.build_dataset(rows, c.prompt))
    train_set = hf_dataset(c.data.train, build=build)
    # Unused when the stage has no val_loss, but the HF Trainer wants an eval dataset.
    val_loss = hf_dataset(c.data.val, c.data.val_max_samples, build)
    generation_set = hf_dataset(c.data.val, c.eval.max_samples) if c.eval.max_samples else None
    config_path = experiment.start(cfg, resume)
    out = Path(cfg["output_dir"])
    if c.stage.parent:  # merged: the parent is the base of the new adapter
        model, processor, _ = load_finetuned(Path(c.stage.parent).expanduser())
    else:
        model, processor = load_vlm(c.model.name, **c.model.image)
    targets = lora_targets(model, c.adaptation.targets)
    lora = LoraConfig(
        r=c.adaptation.rank,
        lora_alpha=c.adaptation.alpha,
        lora_dropout=c.adaptation.dropout,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )

    store = sqlite_store(cfg["mlflow_db"]) if "mlflow_db" in cfg else {}
    loggers = [MLflowLogger(experiment=cfg["name"], **store), CSVLogger(out, name="logs")]
    loggers[0].log_artifact(config_path)  # starts the run the callback logs to
    for lg in loggers:
        lg.log_hyperparams(flatten(cfg))

    trainer_cls = type(spec.trainer_cls.__name__, (GenerationEval, spec.trainer_cls), {})
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_loss,
        processing_class=processor,
        peft_config=lora,
        callbacks=[PathotrainLogging(loggers, monitor), Provenance(provenance(c, targets))],
        generation_set=generation_set,
        own_eval=spec.val_loss,
        generation_kwargs={
            "batch_size": c.eval.batch_size,
            "num_workers": c.trainer.num_workers,
            "max_new_tokens": c.eval.max_new_tokens,
            "system": c.prompt.system,
            "instruction": c.prompt.instruction,
        },
        **spec.trainer_kwargs(c, processor),
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    return {
        f"best/{monitor}": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="experiment YAML")
    parser.add_argument("--resume", help="checkpoint-N folder to continue from")
    args = parser.parse_args(argv)
    print(train(load_config(args.config), args.resume))


if __name__ == "__main__":
    main()
