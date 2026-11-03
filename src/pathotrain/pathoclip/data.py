"""Batches of :class:`~pathotrain.dataloader.caption.CaptionDataset` samples for a
SigLIP model: the processor's ``pixel_values`` (resized to the model's input,
normalised), captions tokenised and padded to :data:`~.models.TEXT_LENGTH`
(truncated beyond it), and the ``groups`` that say which pairs share a caption."""

import torch
from transformers import ProcessorMixin

from pathotrain.pathoclip.models import TEXT_LENGTH


class CaptionCollator:
    def __init__(self, processor: ProcessorMixin) -> None:
        self.processor = processor

    def __call__(self, samples: list[dict]) -> dict:
        batch = self.processor(
            images=[s["image"] for s in samples],
            text=[s["caption"] for s in samples],
            padding="max_length",
            max_length=TEXT_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        groups = torch.tensor([s["group"] for s in samples])
        return {
            "pixel_values": batch["pixel_values"],
            "input_ids": batch["input_ids"],
            "groups": groups,
        }
