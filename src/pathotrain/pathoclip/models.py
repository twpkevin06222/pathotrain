"""Image-text dual encoders for pathology, loaded with their processors.

Google's SigLIP 2 (Apache-2.0) at a fixed 224 px, the tile size of the rest of
pathotrain. Revisions are pinned, so weights and tokenizers never change under a
run.
"""

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, AutoProcessor, ProcessorMixin

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
