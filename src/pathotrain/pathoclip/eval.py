"""Retrieval and zero-shot scores of an image-text model::

    python -m pathotrain.pathoclip.eval siglip2-base-224 --captions arch/test.jsonl \\
        --test crc/test.csv --train crc/train.csv
    python -m pathotrain.pathoclip.eval runs/clip01/checkpoints/best.ckpt --captions arch/test.jsonl
    python -m pathotrain.pathoclip.eval exported/clip01 --test crc/test   # pathoclip.export

The model is a :data:`~pathotrain.pathoclip.models.CLIPS` name (zero-shot, the
baseline), a checkpoint of :mod:`pathotrain.pathoclip.train` or a folder written
by :mod:`pathotrain.pathoclip.export`. ``--captions``: image-text recall@k
(:func:`~pathotrain.pathoclip.task.retrieval`) on a caption JSONL. ``--test``
(and ``--train``): zero-shot classification (and a linear probe and kNN) on
labelled tiles (:mod:`pathotrain.pathoclip.zero_shot`); ``--config`` takes them
from a training config's ``eval`` section instead, with its per-class prompts, so
a baseline is scored as the run's validation is. Prints the metrics as JSON.
"""

import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor

from pathotrain.dataloader.caption import CaptionDataset
from pathotrain.pathoclip.config import Eval
from pathotrain.pathoclip.data import CaptionCollator
from pathotrain.pathoclip.models import CLIPS, image_transform, load_clip
from pathotrain.pathoclip.task import embed_images, embed_texts, retrieval
from pathotrain.pathoclip.train import build_evaluator


def load(spec: str):
    """``(model, processor)`` for a CLIPS name, a ``.ckpt`` or an exported folder."""
    if spec in CLIPS:
        return load_clip(spec)
    if spec.endswith(".ckpt"):
        from pathotrain.pathoclip.train import load_model  # needs peft for LoRA checkpoints

        model, processor, _ = load_model(spec)
        return model, processor
    return AutoModel.from_pretrained(spec), AutoProcessor.from_pretrained(spec)


@torch.no_grad()
def caption_retrieval(model, processor, jsonl: str, batch_size: int, num_workers: int) -> dict:
    device = next(model.parameters()).device
    image = getattr(model, "image_name", None)  # a LockedImageCLIP's own preprocessing
    loader = DataLoader(
        CaptionDataset(jsonl),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=CaptionCollator(processor, image and image_transform(image)),
        multiprocessing_context="forkserver" if num_workers else None,
    )
    images, texts, groups = [], [], []
    for batch in loader:
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            images.append(embed_images(model, batch["pixel_values"].to(device)).float())
            texts.append(embed_texts(model, batch["input_ids"].to(device)).float())
        groups.append(batch["groups"])
    images, texts = F.normalize(torch.cat(images), dim=-1), F.normalize(torch.cat(texts), dim=-1)
    return retrieval(images, texts, torch.cat(groups).to(device))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", help="a CLIPS name, a pathoclip .ckpt or an exported folder")
    parser.add_argument("--captions", help="caption JSONL for image-text retrieval")
    parser.add_argument("--test", help="labelled tiles for zero-shot: class folders or CSV")
    parser.add_argument("--train", help="labelled tiles for a linear probe and kNN")
    parser.add_argument("--root", help="folder the CSV paths are relative to")
    parser.add_argument("--template", default=Eval.template)
    parser.add_argument("--config", help="a pathoclip config whose eval section to use")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not (args.captions or args.test or args.config):
        parser.error("give --captions, --test (or --config) or both")
    model, processor = load(args.model)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    metrics = {"model": args.model}
    if args.captions:
        metrics |= caption_retrieval(
            model, processor, args.captions, args.batch_size, args.num_workers
        )
    if args.config:
        from pathotrain.engine import load_config

        e = Eval(**load_config(args.config)["eval"])
    elif args.test:
        e = Eval(args.test, args.train, args.root, args.template)
    if args.config or args.test:
        e = Eval(**{**e.__dict__, "batch_size": args.batch_size, "num_workers": args.num_workers})
        metrics |= build_evaluator(e, processor, getattr(model, "image_name", None))(model)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
