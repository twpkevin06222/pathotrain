"""Vision-language models for pathology VQA, loaded with their processors.

Only open models (Apache-2.0, not gated). Revisions are pinned, so the weights
and chat templates never change under a run.
"""

from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin

# Qwen3-VL: one visual token per 32 x 32 px (patch 16, 2 x 2 merge); at most 1024
# tokens per image, so PathVQA figures (up to ~1070 x 1007) keep their resolution.
QWEN_IMAGE = {"size": {"longest_edge": 1024 * 32 * 32, "shortest_edge": 256 * 256}}
# SmolVLM: one image at the model's size (512 px for 500M, 384 px for 2.2B) instead of
# 12 sub-images plus the whole (~880-1100 tokens for a PathVQA figure, against ~80-100).
SMOLVLM_IMAGE = {"do_image_splitting": False}


@dataclass(frozen=True)
class VLM:
    repo: str
    revision: str
    processor: dict = field(default_factory=dict)  # image-processor settings, see above


VLMS: dict[str, VLM] = {
    "qwen3-vl-2b": VLM(
        "Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203", QWEN_IMAGE
    ),
    "qwen3-vl-4b": VLM(
        "Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17", QWEN_IMAGE
    ),
    "smolvlm2-2.2b": VLM(
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "482adb537c021c86670beed01cd58990d01e72e4",
        SMOLVLM_IMAGE,
    ),
    "smolvlm2-500m": VLM(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
        SMOLVLM_IMAGE,
    ),
}


def load_processor(name: str, local_files_only: bool = False, **image) -> ProcessorMixin:
    """The processor of ``VLMS[name]`` with its image settings (``image`` replaces
    them, e.g. ``do_image_splitting=True``)."""
    vlm = _vlm(name)
    processor = AutoProcessor.from_pretrained(
        vlm.repo, revision=vlm.revision, local_files_only=local_files_only
    )
    # Set on the image processor, not passed to from_pretrained: SmolVLM's video
    # processor would receive them too and warn.
    for key, value in (vlm.processor | image).items():
        setattr(processor.image_processor, key, value)
    return processor


def load_vlm(
    name: str, dtype: torch.dtype = torch.bfloat16, **image
) -> tuple[nn.Module, ProcessorMixin]:
    """Download (once, into the Hugging Face cache) and return the model and its
    processor; the model in ``dtype``, in eval mode."""
    vlm = _vlm(name)
    model = AutoModelForImageTextToText.from_pretrained(
        vlm.repo, revision=vlm.revision, dtype=dtype
    )
    return model.eval(), load_processor(name, **image)


def _vlm(name: str) -> VLM:
    if name not in VLMS:
        raise ValueError(f"unknown VLM {name!r}; choose from {sorted(VLMS)}")
    return VLMS[name]
