"""ViT backbones for histossl, all ``transformers.Dinov2Model``.

- ``init="pretrained"``: a :mod:`pathotrain.pathofm` model (e.g. ``midnight-12k``,
  ViT-g/14) at its pinned revision.
- ``init="scratch"``: random weights in one of :data:`ARCHS`. Patch 14, as
  Midnight, so a 224 px tile is a 16 x 16 token grid either way.

Masked tokens use Dinov2's own ``bool_masked_pos``: the patch embeddings at masked
positions are replaced by the model's ``mask_token`` (pretrained in Midnight).
"""

import torch
from torch import Tensor
from transformers import Dinov2Config, Dinov2Model

from pathotrain.pathofm import FOUNDATION_MODELS, MIDNIGHT_MEAN, MIDNIGHT_STD, load_backbone

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

# (hidden size, depth, heads) of the DINOv2 ViT sizes with 14 px patches.
ARCHS = {"vit_s14": (384, 12, 6), "vit_b14": (768, 12, 12), "vit_l14": (1024, 24, 16)}


def build_backbone(
    init: str,
    id: str | None = None,
    arch: str | None = None,
    grad_checkpointing: bool = True,
    dtype: str | torch.dtype = "float32",
) -> Dinov2Model:
    """A pretrained (``id``) or randomly initialised (``arch``) backbone, its
    weights in ``dtype`` (a name in :data:`DTYPES` or a torch dtype), with gradient
    checkpointing unless ``grad_checkpointing=False``.

    ``"bfloat16"`` halves the memory of frozen weights, e.g. Midnight's trunk under
    LoRA (peft keeps the adapters in fp32); weights that train should stay fp32.
    Compute precision is the trainer's (``bf16-mixed`` autocast) either way.

    Raises:
        ValueError: ``init`` is not ``pretrained`` or ``scratch``, its name (``id`` /
            ``arch``) is missing, unknown or given for the other init, or
            ``dtype`` is unknown.
    """
    if isinstance(dtype, str):
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
        dtype = DTYPES[dtype]
    if init == "pretrained":
        if arch is not None or id not in FOUNDATION_MODELS:
            raise ValueError(
                f"init=pretrained takes an id in {sorted(FOUNDATION_MODELS)} and no arch, "
                f"got id={id!r}, arch={arch!r}"
            )
        model = load_backbone(id, dtype)
    elif init == "scratch":
        if id is not None or arch not in ARCHS:
            raise ValueError(
                f"init=scratch takes an arch in {sorted(ARCHS)} and no id, "
                f"got arch={arch!r}, id={id!r}"
            )
        hidden, depth, heads = ARCHS[arch]
        # DINOv2's from-scratch recipe starts its LayerScale at 1e-5.
        config = Dinov2Config(
            hidden_size=hidden,
            num_hidden_layers=depth,
            num_attention_heads=heads,
            intermediate_size=4 * hidden,
            patch_size=14,
            image_size=224,
            layerscale_value=1e-5,
        )
        model = Dinov2Model(config).to(dtype)
    else:
        raise ValueError(f"init must be 'pretrained' or 'scratch', got {init!r}")
    if grad_checkpointing:
        # Non-reentrant: with frozen embeddings (LoRA, last-N), the reentrant variant
        # would see no input that needs gradients and drop the adapters' gradients.
        model.gradient_checkpointing_enable({"use_reentrant": False})
    return model


def normalisation(init: str, id: str | None = None) -> tuple[tuple, tuple]:
    """The ``(mean, std)`` the backbone expects: the pretrained model's, or
    Midnight's 0.5 for one trained from scratch."""
    if init == "pretrained":
        fm = FOUNDATION_MODELS[id]
        return fm.mean, fm.std
    return MIDNIGHT_MEAN, MIDNIGHT_STD


def tokens(model, pixel_values: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """The final (LayerNorm'd) CLS token ``(B, D)`` and patch tokens ``(B, N, D)``
    of ``pixel_values``, with patches where ``mask`` (bool ``(B, N)``) is True
    replaced by the mask token. ``model`` may be wrapped by peft."""
    hidden = model(pixel_values=pixel_values, bool_masked_pos=mask).last_hidden_state
    return hidden[:, 0], hidden[:, 1:]
