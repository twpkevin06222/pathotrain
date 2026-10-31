"""Which backbone weights train, and their optimizer groups.

- ``full``: every weight.
- ``frozen``: none (feature extraction, probe baselines).
- ``last_n``: the last ``n`` transformer blocks and the final LayerNorm.
- ``lora``: low-rank adapters on ``targets`` (peft, Hu et al. 2021); the base
  weights stay frozen, and ``model.disable_adapter()`` runs the base model.

Dinov2 linear layers (transformers 5) per block: ``q_proj``, ``k_proj``,
``v_proj`` and ``o_proj`` in attention; in the MLP ``fc1`` and ``fc2``, or with a
SwiGLU MLP (Midnight, DINOv2 ViT-g) ``gate_proj``, ``up_proj`` and ``down_proj``.
LoRA targets are checked against the layers the model really has.
"""

import copy
import json
import re
from collections.abc import Sequence
from pathlib import Path

from torch import nn
from transformers import Dinov2Model

ADAPTATIONS = ("full", "frozen", "last_n", "lora")
# Every linear layer a Dinov2 block may have, plain or SwiGLU MLP; which exist depends on
# the model (see block_linears).
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "fc1",
    "fc2",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def block_linears(model: nn.Module) -> list[str]:
    """The names of the linear layers in the model's transformer blocks, e.g.
    ``["down_proj", "gate_proj", ..., "v_proj"]`` for Midnight."""
    return sorted(
        {
            name.rsplit(".", 1)[-1]
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and re.search(r"encoder\.layer\.\d+\.", name)
        }
    )


def apply_adaptation(
    model: Dinov2Model,
    type: str,
    n: int | None = None,
    rank: int = 16,
    alpha: float | None = None,
    dropout: float = 0.0,
    targets: Sequence[str] = ("q_proj", "v_proj"),
) -> nn.Module:
    """Set which weights of ``model`` train, as ``type`` says, and return the model
    to train: ``model`` itself, or for ``lora`` a peft ``PeftModel`` around it.
    ``alpha`` defaults to ``rank`` (adapter scale 1).

    Raises:
        ValueError: Unknown ``type``, ``n`` outside 1..depth, ``rank < 1`` or a
            target that is not a Dinov2 linear layer.
    """
    if type not in ADAPTATIONS:
        raise ValueError(f"adaptation must be one of {ADAPTATIONS}, got {type!r}")
    if type == "lora":
        return _lora(model, rank, alpha, dropout, targets)
    model.requires_grad_(type == "full")
    if type == "last_n":
        depth = len(model.encoder.layer)
        if n is None or not 1 <= n <= depth:
            raise ValueError(f"last_n needs n in 1..{depth}, got {n}")
        for module in (*model.encoder.layer[-n:], model.layernorm):
            module.requires_grad_(True)
    return model


def _lora(model, rank, alpha, dropout, targets) -> nn.Module:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError(
            "LoRA needs the histossl extra: pip install 'pathotrain[histossl]'"
        ) from e
    if rank < 1:
        raise ValueError(f"LoRA rank must be >= 1, got {rank}")
    present = block_linears(model)
    # peft only fails when no target matches at all, so a missing one would be skipped.
    missing = sorted(set(targets) - set(present))
    if missing or not targets:
        raise ValueError(
            f"LoRA targets {missing or list(targets)} are not linear layers of this model; "
            f"its blocks have {present}"
        )
    config = LoraConfig(
        r=rank, lora_alpha=alpha or rank, lora_dropout=dropout, target_modules=list(targets)
    )
    return get_peft_model(model, config)


def merge_and_export(model: nn.Module, path: str | Path, metadata: dict | None = None) -> Path:
    """Save ``model`` as a plain Hugging Face model in the folder ``path``: LoRA
    adapters, if any, merged into the weights (of a copy, so ``model`` keeps them),
    ``config.json`` and ``model.safetensors`` loadable with
    ``Dinov2Model.from_pretrained(path)`` (no peft needed), and ``metadata`` (e.g. a
    checkpoint's provenance) in ``pathotrain.json`` beside them. Returns ``path``."""
    path = Path(path)
    merged = hasattr(model, "merge_and_unload")
    plain = copy.deepcopy(model).merge_and_unload() if merged else model
    plain.save_pretrained(path)
    record = {**(metadata or {}), "lora_merged": merged}
    (path / "pathotrain.json").write_text(json.dumps(record, indent=2) + "\n")
    return path


def param_groups(
    model: nn.Module, lr: float, weight_decay: float, layer_decay: float | None = None
) -> list[dict]:
    """AdamW groups over the trainable parameters of ``model`` (and any heads in
    it): no weight decay on biases, norms, LayerScale and the CLS / mask / position
    tokens. With ``layer_decay``, block ``i`` of ``L`` trains at ``lr *
    layer_decay ** (L - i)``, the embeddings at ``layer_decay ** (L + 1)``, the rest
    (final norm, heads) at ``lr``. Each group keeps its ``weight_decay`` under
    ``"base_weight_decay"`` for weight-decay schedules."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    depth = 1 + max(
        (int(m.group(1)) for n, _ in named if (m := re.search(r"encoder\.layer\.(\d+)\.", n))),
        default=-1,
    )
    groups: dict[tuple[float, bool], list] = {}
    for name, p in named:
        decay = p.ndim > 1 and not re.search(r"(cls_token|mask_token|position_embeddings)", name)
        scale = 1.0
        if layer_decay is not None:
            block = re.search(r"encoder\.layer\.(\d+)\.", name)
            if block:
                scale = layer_decay ** (depth - int(block.group(1)))
            elif "embeddings." in name:
                scale = layer_decay ** (depth + 1)
        groups.setdefault((scale, decay), []).append(p)
    return [
        {
            "params": params,
            "lr": lr * scale,
            "weight_decay": weight_decay if decay else 0.0,
            "base_weight_decay": weight_decay if decay else 0.0,
        }
        for (scale, decay), params in groups.items()
    ]
