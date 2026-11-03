"""Export a checkpoint written by :mod:`pathotrain.pathoclip.train` as a plain
Hugging Face model::

    python -m pathotrain.pathoclip.export runs/clip01/checkpoints/best.ckpt exported/clip01

Writes the model with LoRA merged into its weights (``config.json``,
``model.safetensors``), its processor, and ``pathotrain.json``: the checkpoint's
provenance (base model and revision, adaptation, input size, normalisation, text
length, git commit), epoch, step and monitored metric, and the config. Load it
with transformers alone::

    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained("exported/clip01")
    processor = AutoProcessor.from_pretrained("exported/clip01")

Pad captions to ``max_length=64`` as in training (``padding="max_length"``).
"""

import argparse
from pathlib import Path

import torch

from pathotrain.histossl.adaptation import merge_and_export
from pathotrain.pathoclip.train import load_model


def export(checkpoint: str | Path, out_dir: str | Path) -> Path:
    """Write ``checkpoint``'s model and processor to the folder ``out_dir``; return it."""
    model, processor, cfg = load_model(checkpoint)
    # mmap: only the small entries are read; load_model already holds the weights.
    ckpt = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    metadata = {
        "provenance": ckpt.get("provenance"),
        "checkpoint": str(Path(checkpoint).resolve()),
        "epoch": ckpt.get("epoch"),
        "step": ckpt.get("step"),
        **({"monitor": ckpt["monitor"], "value": ckpt["value"]} if "monitor" in ckpt else {}),
        "config": cfg,
    }
    out = merge_and_export(model, out_dir, metadata)
    processor.save_pretrained(out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="best.ckpt or last.ckpt written by pathoclip.train")
    parser.add_argument("output", help="the folder to write the model to")
    args = parser.parse_args(argv)
    print(export(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
