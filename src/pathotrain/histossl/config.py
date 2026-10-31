"""The histossl config schema, the method registry and the checks that reject
invalid combinations before anything loads.

:func:`validate` parses the ``method``, ``backbone``, ``adaptation``, ``teacher``,
``masking``, ``crops`` and ``eval`` sections into frozen dataclasses (an unknown
key is an error, so typos fail), checks them against the method's
:class:`MethodSpec` and each other, and raises one :class:`ConfigError` listing
every problem. Risky but valid choices only log a warning.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Literal

from loguru import logger

from pathotrain.histossl.adaptation import ADAPTATIONS, LORA_TARGETS
from pathotrain.histossl.backbones import ARCHS
from pathotrain.pathofm import FOUNDATION_MODELS

PATCH = 14  # every backbone here has 14 px patches


class ConfigError(ValueError):
    """An invalid histossl config; the message lists every problem found."""


@dataclass(frozen=True)
class MethodSpec:
    """What a method accepts. ``masking``: whether it needs, forbids or ignores a
    ``masking`` section; ``multicrop``: whether it takes local crops."""

    init: frozenset[str]
    teacher: frozenset[str]
    adaptation: frozenset[str]
    masking: Literal["required", "forbidden", "optional"]
    multicrop: bool
    implemented: bool = True


METHODS = {
    "teacher_jepa": MethodSpec(
        init=frozenset({"pretrained"}),
        teacher=frozenset({"frozen"}),
        adaptation=frozenset({"lora", "last_n", "full"}),
        masking="required",
        multicrop=False,
    ),
    "dinov2": MethodSpec(
        init=frozenset({"pretrained"}),
        teacher=frozenset({"ema"}),
        adaptation=frozenset({"lora", "last_n", "full"}),
        masking="required",
        multicrop=True,
        implemented=False,
    ),
    "lejepa": MethodSpec(
        init=frozenset({"scratch"}),
        teacher=frozenset({"none"}),
        adaptation=frozenset({"full"}),
        masking="forbidden",
        multicrop=True,
        implemented=False,
    ),
}


@dataclass(frozen=True)
class Method:
    name: str
    params: dict = field(default_factory=dict)  # the method's own settings


@dataclass(frozen=True)
class Backbone:
    init: str
    id: str | None = None
    arch: str | None = None
    grad_checkpointing: bool = True


@dataclass(frozen=True)
class Adaptation:
    type: str = "full"
    n: int | None = None
    rank: int = 16
    alpha: float | None = None
    dropout: float = 0.0
    targets: tuple[str, ...] = ("q_proj", "v_proj")


@dataclass(frozen=True)
class Teacher:
    type: str
    id: str | None = None  # frozen: default the backbone's
    momentum: tuple[float, float] | None = None  # ema: start and end


@dataclass(frozen=True)
class Masking:
    type: Literal["block", "random"] = "block"
    ratio: tuple[float, float] = (0.4, 0.6)
    min_aspect: float = 0.3
    p: float = 1.0


@dataclass(frozen=True)
class Crops:
    global_crops: int = 1
    local_crops: int = 0
    global_size: int = 224
    local_size: int = 98
    global_scale: tuple[float, float] = (0.32, 1.0)
    local_scale: tuple[float, float] = (0.05, 0.32)


@dataclass(frozen=True)
class Eval:
    """Linear probe and kNN on labelled tiles (class folders or path,label CSVs)."""

    train: str
    test: str
    root: str | None = None
    every: int = 1  # validations between probes
    k: int = 20
    batch_size: int = 64
    num_workers: int = 4


@dataclass(frozen=True)
class SSLConfig:
    method: Method
    backbone: Backbone
    adaptation: Adaptation
    teacher: Teacher
    masking: Masking | None
    crops: Crops
    eval: Eval | None


def _parse(cls, section: Any, name: str, errors: list[str]):
    if not isinstance(section, dict):
        errors.append(f"{name} must be a mapping, got {section!r}")
        return None
    known = {f.name for f in fields(cls)}
    unknown = sorted(section.keys() - known)
    if unknown:
        errors.append(f"{name} has unknown keys {unknown}; expected some of {sorted(known)}")
        return None
    try:
        values = {k: tuple(v) if isinstance(v, list) else v for k, v in section.items()}
        return cls(**values)
    except TypeError as e:  # a required key is missing
        errors.append(f"{name}: {e}")
        return None


def _range(name: str, value: tuple[float, float], errors: list[str]) -> None:
    if len(value) != 2 or not 0 <= value[0] <= value[1] <= 1:
        errors.append(f"{name} must be [low, high] in [0, 1], got {list(value)}")


def validate(cfg: dict) -> SSLConfig:
    """Parse and check ``cfg``; return the parsed sections.

    Raises:
        ConfigError: Every problem found, one per line.
    """
    errors: list[str] = []
    method_cfg = dict(cfg.get("method") or {})
    name = method_cfg.pop("name", None)
    method = Method(name, method_cfg)
    spec = METHODS.get(name)
    if spec is None:
        errors.append(f"method.name must be one of {sorted(METHODS)}, got {name!r}")
    elif not spec.implemented:
        errors.append(f"method {name!r} is not implemented yet")
    backbone = _parse(Backbone, cfg.get("backbone"), "backbone", errors)
    adaptation = _parse(Adaptation, cfg.get("adaptation", {}), "adaptation", errors)
    teacher = _parse(Teacher, cfg.get("teacher", {"type": "none"}), "teacher", errors)
    masking = None
    if cfg.get("masking") is not None:
        masking = _parse(Masking, cfg["masking"], "masking", errors)
    crops = _parse(Crops, cfg.get("data", {}).get("crops", {}), "data.crops", errors)
    evaluation = None
    if cfg.get("eval") is not None:
        evaluation = _parse(Eval, cfg["eval"], "eval", errors)

    if backbone is not None:
        if backbone.init == "pretrained":
            if backbone.id not in FOUNDATION_MODELS or backbone.arch is not None:
                errors.append(
                    f"backbone init=pretrained needs an id in {sorted(FOUNDATION_MODELS)} and "
                    f"no arch, got id={backbone.id!r}, arch={backbone.arch!r}"
                )
        elif backbone.init == "scratch":
            if backbone.arch not in ARCHS or backbone.id is not None:
                errors.append(
                    f"backbone init=scratch needs an arch in {sorted(ARCHS)} and no id, "
                    f"got arch={backbone.arch!r}, id={backbone.id!r}"
                )
        else:
            errors.append(f"backbone.init must be pretrained or scratch, got {backbone.init!r}")
    if adaptation is not None:
        if adaptation.type not in ADAPTATIONS:
            errors.append(f"adaptation.type must be one of {ADAPTATIONS}, got {adaptation.type!r}")
        if backbone is not None and backbone.init == "scratch" and adaptation.type != "full":
            errors.append(
                f"adaptation {adaptation.type!r} needs pretrained weights to adapt; "
                "init=scratch trains everything (full)"
            )
        if adaptation.type == "last_n" and (adaptation.n is None or adaptation.n < 1):
            errors.append(f"adaptation last_n needs n >= 1, got {adaptation.n}")
        if adaptation.type == "lora":
            if adaptation.rank < 1:
                errors.append(f"adaptation.rank must be >= 1, got {adaptation.rank}")
            bad = sorted(set(adaptation.targets) - set(LORA_TARGETS))
            if bad or not adaptation.targets:
                errors.append(f"adaptation.targets must be among {LORA_TARGETS}, got {bad or []}")
    if teacher is not None:
        if teacher.momentum is not None and teacher.type != "ema":
            errors.append(f"teacher.momentum is for an ema teacher, not {teacher.type!r}")
        if teacher.id is not None and teacher.type != "frozen":
            errors.append(f"teacher.id is for a frozen teacher, not {teacher.type!r}")
        if teacher.id is not None and teacher.id not in FOUNDATION_MODELS:
            errors.append(f"teacher.id must be in {sorted(FOUNDATION_MODELS)}, got {teacher.id!r}")
        if (
            teacher.type == "frozen"
            and adaptation is not None
            and adaptation.type == "lora"
            and backbone is not None
            and teacher.id not in (None, backbone.id)
        ):
            errors.append(
                "with LoRA the frozen teacher is the backbone with its adapters off, "
                f"so teacher.id must be the backbone's ({backbone.id!r}), got {teacher.id!r}"
            )
        if teacher.type == "ema" and teacher.momentum is not None:
            _range("teacher.momentum", teacher.momentum, errors)
    if masking is not None:
        if masking.type not in ("block", "random"):
            errors.append(f"masking.type must be block or random, got {masking.type!r}")
        _range("masking.ratio", masking.ratio, errors)
        if not 0 < masking.min_aspect <= 1 or not 0 <= masking.p <= 1:
            errors.append("masking.min_aspect must be in (0, 1] and masking.p in [0, 1]")
    if crops is not None:
        for size in ("global_size", "local_size"):
            if getattr(crops, size) % PATCH or getattr(crops, size) <= 0:
                errors.append(
                    f"data.crops.{size} must be a positive multiple of the {PATCH} px patch, "
                    f"got {getattr(crops, size)}"
                )
        _range("data.crops.global_scale", crops.global_scale, errors)
        _range("data.crops.local_scale", crops.local_scale, errors)
        if crops.global_crops < 1 or crops.local_crops < 0:
            errors.append("data.crops needs global_crops >= 1 and local_crops >= 0")

    if spec is not None:
        for section, value, allowed in (
            ("backbone.init", backbone and backbone.init, spec.init),
            ("teacher.type", teacher and teacher.type, spec.teacher),
            ("adaptation.type", adaptation and adaptation.type, spec.adaptation),
        ):
            if value is not None and value not in allowed:
                errors.append(f"{name} takes {section} in {sorted(allowed)}, got {value!r}")
        if spec.masking == "required" and masking is None:
            errors.append(f"{name} needs a masking section")
        if spec.masking == "forbidden" and cfg.get("masking") is not None:
            errors.append(f"{name} takes no masking section")
        if not spec.multicrop and crops is not None and crops.local_crops:
            errors.append(
                f"{name} takes no local crops, got data.crops.local_crops={crops.local_crops}"
            )

    monitor = cfg.get("checkpoint", {}).get("monitor", "")
    if monitor.startswith(("probe/", "knn/")) and evaluation is None:
        errors.append(f"checkpoint.monitor {monitor!r} needs an eval section")
    if evaluation is not None and "val" not in cfg.get("data", {}):
        errors.append("eval runs at validation, which needs a data.val split")
    strategy = str(cfg.get("trainer", {}).get("strategy", "auto"))
    if strategy.startswith(("fsdp", "deepspeed")):
        errors.append(
            f"trainer.strategy {strategy!r} is not supported (the teacher is not sharded)"
        )
    if errors:
        raise ConfigError("invalid histossl config:\n- " + "\n- ".join(errors))

    if adaptation.type == "full" and backbone.init == "pretrained":
        logger.warning(
            "full fine-tuning of a ViT-g (1.1B) is tight on 32 GB; consider lora or last_n"
        )
    if cfg.get("trainer", {}).get("precision", "bf16-mixed") != "bf16-mixed":
        logger.warning("histossl is tuned for bf16-mixed precision")
    return SSLConfig(method, backbone, adaptation, teacher, masking, crops, evaluation)
