"""Open pathology foundation models: MIT or Apache-2.0 licensed, not gated, no
Hugging Face login.

All are DINOv2 ViT-g/14 (40 layers, 1536-dim tokens):

- ``midnight-12k``: Midnight trained on 12k slides (``kaiko-ai/midnight``).
- ``spectra-midnight-12k``: Midnight-12k adapted with SPECTRA for robustness to
  scanner, stain and centre (``medarc/Midnight-12k-SPECTRA``, LoRA already
  merged into the weights).
- ``openmidnight-1.1``: OpenMidnight-1.1 (``medarc/OpenMidnight-1.1``), with 4
  register tokens after the CLS token and ImageNet normalisation. Published as a
  timm checkpoint; converted on load to ``transformers.Dinov2WithRegistersModel``.

The Midnight models load as ``transformers.Dinov2Model`` (one CLS token, no
registers). Patch tokens are the last ``(H/14) * (W/14)`` tokens of either.

Revisions are pinned so the weights never change under a run.
"""

from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn
from torchvision import transforms
from transformers import Dinov2Model, Dinov2WithRegistersConfig, Dinov2WithRegistersModel

from pathotrain.dataloader.augmentation import IMAGENET_MEAN, IMAGENET_STD

# The Midnight models normalise with mean = std = 0.5, not ImageNet statistics.
MIDNIGHT_MEAN = (0.5, 0.5, 0.5)
MIDNIGHT_STD = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class FoundationModel:
    repo_id: str
    revision: str
    mean: tuple[float, float, float] = MIDNIGHT_MEAN
    std: tuple[float, float, float] = MIDNIGHT_STD
    timm: bool = False  # a timm checkpoint, converted on load
    # Model-card tile preprocessing: resize the short side, then centre crop 224.
    resize: int = 224
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR


FOUNDATION_MODELS = {
    "midnight-12k": FoundationModel(
        "kaiko-ai/midnight", "adc6b15679c981cce6f9b018bbad09d16eeeda9f"
    ),
    "spectra-midnight-12k": FoundationModel(
        "medarc/Midnight-12k-SPECTRA", "49d0fe82391949964e7248abc0596a50b57a9247"
    ),
    "openmidnight-1.1": FoundationModel(
        "medarc/OpenMidnight-1.1",
        "a9ed8ab814124bcec8ab46c746b2bd3cc7e35b0e",
        IMAGENET_MEAN,
        IMAGENET_STD,
        timm=True,
        resize=256,  # timm's crop_pct 0.875
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
}


def load_backbone(
    name: str, dtype: torch.dtype = torch.float32
) -> Dinov2Model | Dinov2WithRegistersModel:
    """Download (once, into the Hugging Face cache) and return the backbone, its
    weights in ``dtype`` (fp32 by default)."""
    if name not in FOUNDATION_MODELS:
        raise ValueError(
            f"unknown foundation model {name!r}, expected one of {sorted(FOUNDATION_MODELS)}"
        )
    fm = FOUNDATION_MODELS[name]
    if fm.timm:
        path = hf_hub_download(fm.repo_id, "model.safetensors", revision=fm.revision)
        return from_timm_dinov2(load_file(path)).to(dtype)
    return Dinov2Model.from_pretrained(fm.repo_id, revision=fm.revision, dtype=dtype)


def from_timm_dinov2(state: dict[str, Tensor]) -> Dinov2WithRegistersModel:
    """A ``Dinov2WithRegistersModel`` holding the weights of a timm DINOv2 ViT with
    registers and a SwiGLU MLP (e.g. ``vit_giant_patch14_reg4_dinov2``), giving the
    same tokens. ``state`` is consumed (its tensors become the model's).

    timm adds no position embedding to the CLS token, so its slot in the HF
    position table is zero. timm has no mask token: it starts at zero.
    """
    dim = state["cls_token"].shape[-1]
    depth = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("blocks."))
    hidden = state["blocks.0.mlp.fc2.weight"].shape[1]
    patch = state["patch_embed.proj.weight"].shape[-1]
    config = Dinov2WithRegistersConfig(
        hidden_size=dim,
        num_hidden_layers=depth,
        num_attention_heads=dim // 64,  # DINOv2 heads are 64 wide
        mlp_ratio=4,
        use_swiglu_ffn=True,
        num_register_tokens=state["reg_token"].shape[1],
        patch_size=patch,
        image_size=round(state["pos_embed"].shape[1] ** 0.5) * patch,
    )
    if hidden != (int(dim * 4 * 2 / 3) + 7) // 8 * 8:  # HF's SwiGLU width for mlp_ratio 4
        raise ValueError(f"SwiGLU width {hidden} is not DINOv2's for {dim}-dim tokens")
    pos = state.pop("pos_embed")
    hf = {
        "embeddings.cls_token": state.pop("cls_token"),
        "embeddings.register_tokens": state.pop("reg_token"),
        "embeddings.mask_token": torch.zeros(1, dim),
        "embeddings.position_embeddings": torch.cat([pos.new_zeros(1, 1, dim), pos], 1),
        "embeddings.patch_embeddings.projection.weight": state.pop("patch_embed.proj.weight"),
        "embeddings.patch_embeddings.projection.bias": state.pop("patch_embed.proj.bias"),
        "layernorm.weight": state.pop("norm.weight"),
        "layernorm.bias": state.pop("norm.bias"),
    }
    for i in range(depth):
        t, h = f"blocks.{i}.", f"encoder.layer.{i}."
        for kind in ("weight", "bias"):
            q, k, v = state.pop(f"{t}attn.qkv.{kind}").chunk(3)
            gate, up = state.pop(f"{t}mlp.fc1.{kind}").chunk(
                2
            )  # SwiGLUPacked: silu(first) * second
            hf |= {
                f"{h}norm1.{kind}": state.pop(f"{t}norm1.{kind}"),
                f"{h}norm2.{kind}": state.pop(f"{t}norm2.{kind}"),
                f"{h}attention.q_proj.{kind}": q,
                f"{h}attention.k_proj.{kind}": k,
                f"{h}attention.v_proj.{kind}": v,
                f"{h}attention.o_proj.{kind}": state.pop(f"{t}attn.proj.{kind}"),
                f"{h}mlp.gate_proj.{kind}": gate,
                f"{h}mlp.up_proj.{kind}": up,
                f"{h}mlp.down_proj.{kind}": state.pop(f"{t}mlp.fc2.{kind}"),
            }
        hf[f"{h}layer_scale1.lambda1"] = state.pop(f"{t}ls1.gamma")
        hf[f"{h}layer_scale2.lambda1"] = state.pop(f"{t}ls2.gamma")
    if state:
        raise ValueError(f"unexpected timm weights: {sorted(state)[:5]}")
    with torch.device("meta"):
        model = Dinov2WithRegistersModel(config)
    model.load_state_dict(hf, strict=True, assign=True)
    return model.eval()


def num_prefix_tokens(model: nn.Module) -> int:
    """Tokens before the patch tokens: the CLS token and any registers."""
    return 1 + getattr(model.config, "num_register_tokens", 0)


def build_encoder(name: str) -> tuple[nn.Module, transforms.Compose]:
    """Backbone and the model card's tile preprocessing for embedding extraction:
    resize (Midnight: bilinear to 224; OpenMidnight: bicubic to 256), centre crop
    224, normalise with the model's mean/std. Use
    :func:`classification_embedding` or :func:`segmentation_embedding` on the
    output. For training a segmentation model, use
    ``pathotrain.segmentation.ViTPyramidEncoder`` instead."""
    model = load_backbone(name)
    fm = FOUNDATION_MODELS[name]
    transform = transforms.Compose(
        [
            transforms.Resize(fm.resize, interpolation=fm.interpolation),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=fm.mean, std=fm.std),
        ]
    )
    return model, transform


def classification_embedding(model: Dinov2Model, pixel_values: Tensor) -> Tensor:
    """Tile embedding of the model cards: CLS token concatenated with the mean
    patch token (registers left out), ``(B, 3072)``. No spatial information; for
    classification."""
    tokens = model(pixel_values=pixel_values).last_hidden_state
    return torch.cat([tokens[:, 0], tokens[:, num_prefix_tokens(model) :].mean(1)], dim=-1)


def segmentation_embedding(model: Dinov2Model, pixel_values: Tensor) -> Tensor:
    """Dense embedding of the model card: last-layer patch tokens (after the
    final LayerNorm, CLS and registers dropped) as a ``(B, 1536, H/14, W/14)`` map, e.g.
    16 x 16 for a 224 px tile."""
    patch = model.config.patch_size
    h, w = pixel_values.shape[-2] // patch, pixel_values.shape[-1] // patch
    tokens = model(pixel_values=pixel_values).last_hidden_state[:, -h * w :]
    return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, h, w)
