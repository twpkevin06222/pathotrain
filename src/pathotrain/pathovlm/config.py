"""The pathovlm config schema and its checks, run before anything loads.

:func:`validate` parses the ``model``, ``adaptation``, ``prompt``, ``data``,
``eval``, ``trainer``, ``checkpoint`` and ``stage`` sections into frozen dataclasses (an
unknown key is an error, so typos fail) and raises one :class:`ConfigError`
listing every problem.
"""

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

from pathotrain.pathovlm.eval import INSTRUCTION
from pathotrain.pathovlm.models import VLMS
from pathotrain.pathovlm.stages import STAGES

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MONITORS = {"val/loss": False, "val/closed_acc": True, "val/open_em": True, "val/open_f1": True}


class ConfigError(ValueError):
    """An invalid pathovlm config; the message lists every problem found."""


@dataclass(frozen=True)
class Model:
    name: str
    image: dict = field(default_factory=dict)  # image-processor settings over the registry's


@dataclass(frozen=True)
class Adaptation:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    targets: tuple[str, ...] = LORA_TARGETS  # linear layers of the language model only


@dataclass(frozen=True)
class Prompt:
    system: str | None = None
    instruction: str | None = INSTRUCTION


@dataclass(frozen=True)
class Data:
    train: str
    val: str
    val_max_samples: int | None = 1024  # evenly spaced rows for val/loss


@dataclass(frozen=True)
class Eval:
    max_samples: int = 2048  # evenly spaced val rows answered by generation each evaluation
    max_new_tokens: int = 32
    batch_size: int = 16


@dataclass(frozen=True)
class Trainer:
    epochs: float = 3
    lr: float = 2e-4
    batch_size: int = 8
    accumulate: int = 1
    warmup: float = 0.03  # share of the steps (< 1) or a step count
    weight_decay: float = 0.0
    eval_steps: int = 500  # evaluate and checkpoint every n optimizer steps
    logging_steps: int = 10
    save_total_limit: int = 2
    num_workers: int = 4
    grad_checkpointing: bool = True


@dataclass(frozen=True)
class Checkpoint:
    monitor: str = "val/closed_acc"


@dataclass(frozen=True)
class Stage:
    name: str = "sft"  # a key of pathotrain.pathovlm.stages.STAGES
    preset: str | None = None
    args: dict = field(default_factory=dict)  # into the stage's TRL config class
    parent: str | None = None  # checkpoint-N folder to start from (merged), e.g. SFT's
    teacher: str | None = None  # distill: the checkpoint-N whose answers the student learns


@dataclass(frozen=True)
class VLMConfig:
    model: Model
    adaptation: Adaptation
    prompt: Prompt
    data: Data
    eval: Eval
    trainer: Trainer
    checkpoint: Checkpoint
    stage: Stage


def validate(cfg: dict) -> VLMConfig:
    """Parse and check ``cfg``; return the parsed sections.

    Raises:
        ConfigError: Every problem found, one per line.
    """
    errors: list[str] = []
    missing = [k for k in ("name", "seed", "output_dir") if k not in cfg]
    if missing:
        errors.append(f"missing top-level keys {missing}")
    model = _parse(Model, cfg.get("model"), "model", errors)
    adaptation = _parse(Adaptation, cfg.get("adaptation", {}), "adaptation", errors)
    prompt = _parse(Prompt, cfg.get("prompt", {}), "prompt", errors)
    data = _parse(Data, cfg.get("data"), "data", errors)
    evaluation = _parse(Eval, cfg.get("eval", {}), "eval", errors)
    trainer = _parse(Trainer, cfg.get("trainer", {}), "trainer", errors)
    checkpoint = _parse(Checkpoint, cfg.get("checkpoint", {}), "checkpoint", errors)
    stage = _parse(Stage, cfg.get("stage", {}), "stage", errors)

    if model is not None and model.name not in VLMS:
        errors.append(f"model.name must be one of {sorted(VLMS)}, got {model.name!r}")
    if adaptation is not None:
        if adaptation.rank <= 0 or adaptation.alpha <= 0:
            errors.append(f"adaptation rank and alpha must be > 0, got {adaptation}")
        unknown = sorted(set(adaptation.targets) - set(LORA_TARGETS))
        if unknown or not adaptation.targets:
            errors.append(f"adaptation.targets must be a non-empty subset of {LORA_TARGETS}")
    if data is not None:
        for split in ("train", "val"):
            if not Path(getattr(data, split)).expanduser().is_file():
                errors.append(f"data.{split} is not a file: {getattr(data, split)}")
    if evaluation is not None and evaluation.max_samples < 0:
        errors.append("eval.max_samples must be >= 0 (0: no generation metrics)")
    if checkpoint is not None and checkpoint.monitor not in MONITORS:
        errors.append(f"checkpoint.monitor must be one of {sorted(MONITORS)}")
    # Every monitor but val/loss comes from generation, which max_samples: 0 turns off.
    no_generation = evaluation is not None and evaluation.max_samples == 0
    if no_generation and checkpoint is not None and checkpoint.monitor != "val/loss":
        errors.append("checkpoint.monitor needs eval.max_samples > 0 unless it is val/loss")
    if trainer is not None and (trainer.epochs <= 0 or trainer.lr <= 0 or trainer.batch_size < 1):
        errors.append(f"trainer epochs, lr and batch_size must be positive, got {trainer}")
    spec = STAGES.get(stage.name) if stage is not None else None
    if stage is not None and spec is None:
        errors.append(f"stage.name must be one of {sorted(STAGES)}, got {stage.name!r}")
    if spec is not None:
        if stage.preset is not None and stage.preset not in spec.presets:
            errors.append(f"stage.preset for {stage.name} must be one of {sorted(spec.presets)}")
        known = {f.name for f in fields(spec.config_cls)}
        unknown = sorted(set(stage.args) - known)
        if unknown:
            errors.append(
                f"stage.args has keys {spec.config_cls.__name__} does not take: {unknown}"
            )
        if not spec.val_loss and checkpoint is not None and checkpoint.monitor == "val/loss":
            errors.append(
                f"checkpoint.monitor: {stage.name} has no val/loss, only generated scores"
            )
        fixed = sorted(set(stage.args) & set(spec.fixed))
        if fixed:
            errors.append(f"stage.args cannot set {fixed}: {stage.name} needs them as they are")
        if spec.teacher != (stage.teacher is not None):
            need = "needs" if spec.teacher else "takes no"
            errors.append(f"stage.teacher: {stage.name} {need} teacher checkpoint")
    if stage is not None and stage.parent is not None:
        errors += _checkpoint_errors("parent", stage.parent, model, prompt)
    if stage is not None and stage.teacher is not None:
        errors += _checkpoint_errors("teacher", stage.teacher, model, prompt, same_model=False)
    if errors:
        raise ConfigError("invalid pathovlm config:\n- " + "\n- ".join(errors))
    return VLMConfig(model, adaptation, prompt, data, evaluation, trainer, checkpoint, stage)


def _checkpoint_errors(
    key: str, checkpoint: str, model: Model | None, prompt: Prompt | None, same_model=True
) -> list[str]:
    """A parent (``same_model``) or teacher checkpoint must have the run's image
    settings and prompt: the run continues its training, or the teacher sees the
    student's inputs."""
    path = Path(checkpoint).expanduser() / "provenance.json"
    if not path.is_file():
        return [f"stage.{key} is not a pathovlm checkpoint (no provenance.json): {checkpoint}"]
    prov = json.loads(path.read_text())
    if model is None or model.name not in VLMS:
        return []
    if same_model and prov["model"] != model.name:
        return [f"stage.{key} was trained on {prov['model']}, not {model.name}"]
    errors = []
    if VLMS[model.name].processor | model.image != prov["image"]:
        errors.append(f"model.image must give the {key}'s image settings {prov['image']}")
    if (
        prompt is not None
        and {"system": prompt.system, "instruction": prompt.instruction} != prov["prompt"]
    ):
        errors.append(f"prompt must be the {key}'s {prov['prompt']}")
    return errors


def _parse(cls, section, name: str, errors: list[str]):
    if not isinstance(section, dict):
        errors.append(f"{name} must be a mapping, got {section!r}")
        return None
    unknown = sorted(set(section) - {f.name for f in fields(cls)})
    if unknown:
        errors.append(f"{name} has unknown keys {unknown}")
        return None
    try:
        parsed = cls(**section)
    except TypeError as e:  # a required key is missing
        errors.append(f"{name}: {e}")
        return None
    if cls is Adaptation:  # YAML gives a list
        return Adaptation(**{**section, "targets": tuple(parsed.targets)})
    return parsed
