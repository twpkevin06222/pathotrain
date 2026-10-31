"""Which backbone weights train, and their optimizer groups.

- ``full``: every weight.
- ``frozen``: none (feature extraction, probe baselines).
- ``last_n``: the last ``n`` transformer blocks and the final LayerNorm.
- ``lora``: low-rank adapters on ``targets`` (peft, Hu et al. 2021); the base
  weights stay frozen, and ``model.disable_adapter()`` runs the base model.

Dinov2 module names (transformers 5): ``q_proj``, ``k_proj``, ``v_proj`` and
``o_proj`` in attention, ``fc1`` and ``fc2`` in the MLP.
"""

import re
from collections.abc import Sequence

from torch import nn
from transformers import Dinov2Model

ADAPTATIONS = ("full", "frozen", "last_n", "lora")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2")


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
    unknown = sorted(set(targets) - set(LORA_TARGETS))
    if unknown or not targets:
        raise ValueError(f"LoRA targets must be among {LORA_TARGETS}, got {list(targets)}")
    config = LoraConfig(
        r=rank, lora_alpha=alpha or rank, lora_dropout=dropout, target_modules=list(targets)
    )
    return get_peft_model(model, config)


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
