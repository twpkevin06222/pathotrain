"""ViT backbones for histossl, all ``transformers.Dinov2Model``.

- ``init="pretrained"``: a :mod:`pathotrain.pathofm` model (e.g. ``midnight-12k``,
  ViT-g/14) at its pinned revision.
- ``init="scratch"``: random weights in one of :data:`ARCHS`, with 14 px patches
  (as Midnight: a 224 px tile is 16 x 16 tokens) or 16 px (14 x 14).

Masked tokens use Dinov2's own ``bool_masked_pos``: the patch embeddings at masked
positions are replaced by the model's ``mask_token`` (pretrained in Midnight).

Each backbone carries its preprocessing on its config (``config.preprocessing``:
``input_size`` and ``mean`` / ``std`` on ``[0, 1]`` pixels), so it survives peft
wrapping and ``save_pretrained``; read it with :func:`preprocessing`.
"""

import torch
from torch import Tensor
from transformers import Dinov2Config, Dinov2Model

from pathotrain.pathofm import FOUNDATION_MODELS, MIDNIGHT_MEAN, MIDNIGHT_STD, load_backbone

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

# (hidden size, depth, heads, patch size) of the ViT sizes, as DINOv2 / ViT define them.
ARCHS = {
    "vit_s14": (384, 12, 6, 14),
    "vit_b14": (768, 12, 12, 14),
    "vit_l14": (1024, 24, 16, 14),
    "vit_s16": (384, 12, 6, 16),
    "vit_b16": (768, 12, 12, 16),
}
PRETRAINED_PATCH = 14  # every pathotrain.pathofm model is a ViT-g/14


def patch_size(init: str, arch: str | None = None) -> int:
    """The patch side of a backbone: 14 for the pretrained models, else its arch's."""
    return PRETRAINED_PATCH if init == "pretrained" else ARCHS[arch][3]


def build_backbone(
    init: str,
    id: str | None = None,
    arch: str | None = None,
    grad_checkpointing: bool = True,
    dtype: str | torch.dtype = "float32",
    input_size: int = 224,
) -> Dinov2Model:
    """A pretrained (``id``) or randomly initialised (``arch``) backbone, its
    weights in ``dtype`` (a name in :data:`DTYPES` or a torch dtype), with gradient
    checkpointing unless ``grad_checkpointing=False``, and its preprocessing
    (``input_size`` px, the :func:`normalisation` of ``init`` / ``id``) stored on it.

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
        hidden, depth, heads, patch = ARCHS[arch]
        # DINOv2's from-scratch recipe starts its LayerScale at 1e-5.
        config = Dinov2Config(
            hidden_size=hidden,
            num_hidden_layers=depth,
            num_attention_heads=heads,
            intermediate_size=4 * hidden,
            patch_size=patch,
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
    return set_preprocessing(model, *normalisation(init, id), input_size)


def set_preprocessing(model: Dinov2Model, mean, std, input_size: int = 224) -> Dinov2Model:
    """Store the tiles ``model`` expects on its config: ``input_size`` px, normalised
    with ``mean`` / ``std`` (``[0, 1]`` units); returns ``model``."""
    model.config.preprocessing = {
        "input_size": int(input_size),
        "mean": [float(m) for m in mean],
        "std": [float(s) for s in std],
    }
    return model


def preprocessing(model) -> dict:
    """``{"input_size", "mean", "std"}`` stored on ``model`` (or its peft wrapper, or a
    model reloaded with ``from_pretrained`` after :func:`~pathotrain.histossl.adaptation.merge_and_export`).

    Raises:
        ValueError: The model has none (it was not built by :func:`build_backbone`).
    """
    values = getattr(model.config, "preprocessing", None)
    if values is None:
        raise ValueError("this backbone has no preprocessing; build it with build_backbone")
    return dict(values)


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
