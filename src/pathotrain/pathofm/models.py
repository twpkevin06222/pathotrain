"""Open pathology foundation models: MIT licensed, not gated, no Hugging Face login.

Both are DINOv2 ViT-g/14 (40 layers, 1536-dim tokens, one CLS token, no
registers) loaded as ``transformers.Dinov2Model``:

- ``midnight-12k``: Midnight trained on 12k slides (``kaiko-ai/midnight``).
- ``spectra-midnight-12k``: Midnight-12k adapted with SPECTRA for robustness to
  scanner, stain and centre (``medarc/Midnight-12k-SPECTRA``, LoRA already
  merged into the weights).

Revisions are pinned so the weights never change under a run.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torchvision import transforms
from transformers import Dinov2Model

# Both models normalise with mean = std = 0.5, not ImageNet statistics.
MIDNIGHT_MEAN = (0.5, 0.5, 0.5)
MIDNIGHT_STD = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class FoundationModel:
    repo_id: str
    revision: str
    mean: tuple[float, float, float] = MIDNIGHT_MEAN
    std: tuple[float, float, float] = MIDNIGHT_STD


FOUNDATION_MODELS = {
    "midnight-12k": FoundationModel(
        "kaiko-ai/midnight", "adc6b15679c981cce6f9b018bbad09d16eeeda9f"
    ),
    "spectra-midnight-12k": FoundationModel(
        "medarc/Midnight-12k-SPECTRA", "49d0fe82391949964e7248abc0596a50b57a9247"
    ),
}


def load_backbone(name: str, dtype: torch.dtype = torch.float32) -> Dinov2Model:
    """Download (once, into the Hugging Face cache) and return the backbone, its
    weights in ``dtype`` (fp32 by default)."""
    if name not in FOUNDATION_MODELS:
        raise ValueError(
            f"unknown foundation model {name!r}, expected one of {sorted(FOUNDATION_MODELS)}"
        )
    fm = FOUNDATION_MODELS[name]
    return Dinov2Model.from_pretrained(fm.repo_id, revision=fm.revision, dtype=dtype)


def build_encoder(name: str) -> tuple[Dinov2Model, transforms.Compose]:
    """Backbone and the model card's tile preprocessing for embedding extraction:
    bilinear resize to 224, centre crop 224, normalise with 0.5 mean/std. Use
    :func:`classification_embedding` or :func:`segmentation_embedding` on the
    output. For training a segmentation model, use
    ``pathotrain.segmentation.ViTPyramidEncoder`` instead."""
    model = load_backbone(name)
    fm = FOUNDATION_MODELS[name]
    transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=fm.mean, std=fm.std),
        ]
    )
    return model, transform


def classification_embedding(model: Dinov2Model, pixel_values: Tensor) -> Tensor:
    """Tile embedding of the model cards: CLS token concatenated with the mean
    patch token, ``(B, 3072)``. No spatial information; for classification."""
    tokens = model(pixel_values=pixel_values).last_hidden_state
    return torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=-1)


def segmentation_embedding(model: Dinov2Model, pixel_values: Tensor) -> Tensor:
    """Dense embedding of the model card: last-layer patch tokens (after the
    final LayerNorm, CLS dropped) as a ``(B, 1536, H/14, W/14)`` map, e.g.
    16 x 16 for a 224 px tile."""
    patch = model.config.patch_size
    h, w = pixel_values.shape[-2] // patch, pixel_values.shape[-1] // patch
    tokens = model(pixel_values=pixel_values).last_hidden_state[:, -h * w :]
    return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, h, w)
