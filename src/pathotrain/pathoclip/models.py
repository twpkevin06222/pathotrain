"""Image-text dual encoders for pathology, loaded with their processors.

Google's SigLIP 2 (Apache-2.0) at a fixed 224 px, the tile size of the rest of
pathotrain, and :class:`LockedImageCLIP`: a frozen pathology foundation model
with SigLIP 2's text tower. Revisions are pinned, so weights and tokenizers
never change under a run.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.transforms import v2
from transformers import AutoModel, AutoProcessor, ProcessorMixin
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.siglip.modeling_siglip import SiglipOutput

from pathotrain.pathofm import FOUNDATION_MODELS, classification_embedding, load_backbone

# SigLIP's text tower pools the last position: captions are padded to its full length.
TEXT_LENGTH = 64


@dataclass(frozen=True)
class CLIP:
    repo: str
    revision: str


CLIPS: dict[str, CLIP] = {
    "siglip2-base-224": CLIP(
        "google/siglip2-base-patch16-224", "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    ),
    "siglip2-so400m-224": CLIP(
        "google/siglip2-so400m-patch14-224", "78e403963a4f6a3640d07803284752326fdf4edf"
    ),
}


def load_clip(
    name: str, dtype: torch.dtype = torch.float32, local_files_only: bool = False
) -> tuple[nn.Module, ProcessorMixin]:
    """Download (once, into the Hugging Face cache) and return ``CLIPS[name]``
    and its processor."""
    if name not in CLIPS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(CLIPS)}")
    clip = CLIPS[name]
    kwargs = {"revision": clip.revision, "local_files_only": local_files_only}
    model = AutoModel.from_pretrained(clip.repo, dtype=dtype, **kwargs)
    return model, AutoProcessor.from_pretrained(clip.repo, **kwargs)


def image_transform(name: str) -> v2.Compose:
    """A :mod:`pathotrain.pathofm` model's input for a whole image (PIL or uint8
    tensor): squashed to 224 x 224 (as SigLIP's processor does, so figures keep
    every panel), then normalised with the model's mean / std."""
    fm = FOUNDATION_MODELS[name]
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((224, 224), interpolation=fm.interpolation, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(list(fm.mean), list(fm.std)),
            v2.ToPureTensor(),
        ]
    )


class LockedImageCLIP(nn.Module):
    """LiT (Zhai et al. 2022): a frozen :mod:`pathotrain.pathofm` backbone as the
    image tower and ``clip``'s text tower (and logit scale / bias). The image
    embedding is the backbone's CLS + mean patch token through a trainable linear
    projection to the text width. ``forward`` takes ``image_features`` (that
    backbone embedding, cached) or ``pixel_values``; ``clip``'s own vision tower is
    unused."""

    def __init__(self, image: str, clip: nn.Module) -> None:
        super().__init__()
        self.image_name = image
        self.image = load_backbone(image).requires_grad_(False).eval()
        self.clip = clip
        text = clip.config.text_config
        self.proj = nn.Linear(2 * self.image.config.hidden_size, text.projection_size)

    def train(self, mode: bool = True):
        super().train(mode)
        self.image.eval()  # frozen: never in training mode
        return self

    @torch.no_grad()
    def backbone_features(self, pixel_values: Tensor) -> Tensor:
        return classification_embedding(self.image, pixel_values).float()

    def get_image_features(
        self, pixel_values: Tensor | None = None, image_features: Tensor | None = None
    ) -> BaseModelOutputWithPooling:
        if image_features is None:
            image_features = self.backbone_features(pixel_values)
        return BaseModelOutputWithPooling(pooler_output=self.proj(image_features))

    def get_text_features(self, input_ids: Tensor) -> BaseModelOutputWithPooling:
        return self.clip.get_text_features(input_ids=input_ids)

    def forward(
        self,
        input_ids: Tensor,
        pixel_values: Tensor | None = None,
        image_features: Tensor | None = None,
    ) -> SiglipOutput:
        image = self.get_image_features(pixel_values, image_features).pooler_output
        image = F.normalize(image, dim=-1)
        text = F.normalize(self.get_text_features(input_ids).pooler_output, dim=-1)
        params = dict(self.clip.named_parameters())
        scale = next(p for n, p in params.items() if n.endswith("logit_scale"))
        bias = next(p for n, p in params.items() if n.endswith("logit_bias"))
        logits_per_text = text @ image.T.to(text.dtype) * scale.exp() + bias
        return SiglipOutput(
            logits_per_image=logits_per_text.T,
            logits_per_text=logits_per_text,
            text_embeds=text,
            image_embeds=image,
        )
