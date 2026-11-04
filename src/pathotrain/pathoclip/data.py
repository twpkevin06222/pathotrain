"""Batches of :class:`~pathotrain.dataloader.caption.CaptionDataset` samples for a
SigLIP model: the processor's ``pixel_values`` (resized to the model's input,
normalised) or ``transform``'s, captions tokenised and padded to
:data:`~.models.TEXT_LENGTH` (truncated beyond it), and the ``groups`` that say
which pairs share a caption. Samples of :class:`EmbeddedCaptions` give
``image_features`` instead of pixels."""

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import ProcessorMixin

from pathotrain.dataloader.caption import CaptionDataset
from pathotrain.pathoclip.models import TEXT_LENGTH


class CaptionCollator:
    def __init__(self, processor: ProcessorMixin, transform=None) -> None:
        self.processor, self.transform = processor, transform

    def __call__(self, samples: list[dict]) -> dict:
        texts = [s["caption"] for s in samples]
        kwargs = {"padding": "max_length", "max_length": TEXT_LENGTH, "truncation": True}
        batch = {"groups": torch.tensor([s["group"] for s in samples])}
        if "image_features" in samples[0]:
            batch["image_features"] = torch.stack([s["image_features"] for s in samples])
        elif self.transform is not None:
            batch["pixel_values"] = torch.stack([self.transform(s["image"]) for s in samples])
        else:
            out = self.processor(
                images=[s["image"] for s in samples], text=texts, **kwargs, return_tensors="pt"
            )
            return batch | {"pixel_values": out["pixel_values"], "input_ids": out["input_ids"]}
        ids = self.processor.tokenizer(texts, **kwargs, return_tensors="pt")["input_ids"]
        return batch | {"input_ids": ids}


class EmbeddedCaptions(Dataset):
    """``captions``' rows with ``features[i]`` (row ``i``'s image, embedded once)
    in place of the image."""

    def __init__(self, captions: CaptionDataset, features: Tensor) -> None:
        if len(features) != len(captions):
            raise ValueError(f"{len(features)} image features for {len(captions)} captions")
        self.rows, self.features = captions.rows, features

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {**row, "image_features": self.features[index]}
