"""The pathoclip config schema and its checks, run before anything loads.

:func:`validate` parses the ``model``, ``adaptation``, ``data``, ``eval`` and
``task`` sections into frozen dataclasses (an unknown key is an error, so typos
fail) and raises one :class:`ConfigError` listing every problem. ``trainer``,
``loader``, ``checkpoint`` and ``early_stopping`` go to pathotrain's engine as for
every task.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path

from pathotrain.pathoclip.models import CLIPS

ADAPTATIONS = ("lora", "full")
# Linear layers of both towers' transformer blocks.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")


class ConfigError(ValueError):
    """An invalid pathoclip config; the message lists every problem found."""


@dataclass(frozen=True)
class Model:
    name: str


@dataclass(frozen=True)
class Adaptation:
    type: str = "lora"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")


@dataclass(frozen=True)
class Data:
    train: str
    val: str
    root: str | None = None  # the folder image paths are relative to; default each JSONL's


@dataclass(frozen=True)
class Eval:
    test: str  # labelled tiles: class folders or a path,label CSV
    train: str | None = None  # with it, a linear probe and kNN too
    root: str | None = None  # the folder CSV paths are relative to
    template: str = "an H&E image of {}."  # filled with each class name
    prompts: dict = field(default_factory=dict)  # class name -> its own prompt, over template
    batch_size: int = 64
    num_workers: int = 4


@dataclass(frozen=True)
class TaskSection:
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_epochs: float = 0.5


@dataclass(frozen=True)
class CLIPConfig:
    model: Model
    adaptation: Adaptation
    data: Data
    eval: Eval | None
    task: TaskSection


def validate(cfg: dict) -> CLIPConfig:
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
    data = _parse(Data, cfg.get("data"), "data", errors)
    evaluation = _parse(Eval, cfg["eval"], "eval", errors) if "eval" in cfg else None
    task = _parse(TaskSection, cfg.get("task", {}), "task", errors)

    if model is not None and model.name not in CLIPS:
        errors.append(f"model.name must be one of {sorted(CLIPS)}, got {model.name!r}")
    if adaptation is not None:
        if adaptation.type not in ADAPTATIONS:
            errors.append(f"adaptation.type must be one of {ADAPTATIONS}, got {adaptation.type!r}")
        if adaptation.rank <= 0 or adaptation.alpha <= 0:
            errors.append(f"adaptation rank and alpha must be > 0, got {adaptation}")
        unknown = sorted(set(adaptation.targets) - set(LORA_TARGETS))
        if unknown or not adaptation.targets:
            errors.append(f"adaptation.targets must be a non-empty subset of {LORA_TARGETS}")
    if data is not None:
        for split in ("train", "val"):
            if not Path(getattr(data, split)).expanduser().is_file():
                errors.append(f"data.{split} is not a file: {getattr(data, split)}")
    if evaluation is not None:
        for split in ("test", "train"):
            spec = getattr(evaluation, split)
            if spec is not None and not Path(spec).expanduser().exists():
                errors.append(f"eval.{split} does not exist: {spec}")
        if "{}" not in evaluation.template:
            errors.append(f"eval.template needs a {{}} for the class name: {evaluation.template!r}")
    if task is not None and task.lr <= 0:
        errors.append(f"task.lr must be > 0, got {task.lr}")
    monitor = cfg.get("checkpoint", {}).get("monitor", "")
    if evaluation is None and monitor.startswith(("zs/", "probe/", "knn/")):
        errors.append(f"checkpoint.monitor {monitor!r} needs an eval section")
    if errors:
        raise ConfigError("invalid pathoclip config:\n- " + "\n- ".join(errors))
    return CLIPConfig(model, adaptation, data, evaluation, task)


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
