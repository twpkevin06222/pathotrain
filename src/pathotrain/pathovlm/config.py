"""The pathovlm config schema and its checks, run before anything loads.

:func:`validate` parses the ``model``, ``adaptation``, ``prompt``, ``data``,
``eval``, ``trainer`` and ``checkpoint`` sections into frozen dataclasses (an
unknown key is an error, so typos fail) and raises one :class:`ConfigError`
listing every problem.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path

from pathotrain.pathovlm.eval import INSTRUCTION
from pathotrain.pathovlm.models import VLMS

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
    max_samples: int = 512  # evenly spaced val rows answered by generation each evaluation
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
class VLMConfig:
    model: Model
    adaptation: Adaptation
    prompt: Prompt
    data: Data
    eval: Eval
    trainer: Trainer
    checkpoint: Checkpoint


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
    if errors:
        raise ConfigError("invalid pathovlm config:\n- " + "\n- ".join(errors))
    return VLMConfig(model, adaptation, prompt, data, evaluation, trainer, checkpoint)


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
