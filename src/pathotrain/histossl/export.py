"""Export the encoder of a checkpoint written by :mod:`pathotrain.histossl.train`
as a plain Hugging Face model::

    python -m pathotrain.histossl.export runs/jepa01/checkpoints/best.ckpt exported/jepa01

Writes ``config.json`` and ``model.safetensors`` (LoRA merged into the weights; the
EMA teacher for dinov2, the student otherwise) and ``pathotrain.json``: the
checkpoint's provenance (backbone and revision, normalisation, input size,
adaptation, method, git commit), its epoch, step and monitored metric, and the
experiment config. Load it anywhere with transformers alone::

    from transformers import Dinov2Model
    encoder = Dinov2Model.from_pretrained("exported/jepa01")

Preprocess tiles as ``pathotrain.json`` says: ``provenance.input_size`` px,
``provenance.normalisation`` mean and std on ``[0, 1]`` pixels.
"""

import argparse
from pathlib import Path

import torch

from pathotrain.histossl.adaptation import merge_and_export
from pathotrain.histossl.train import load_model


def export(checkpoint: str | Path, out_dir: str | Path) -> Path:
    """Write ``checkpoint``'s encoder to the folder ``out_dir`` and return it."""
    task, cfg = load_model(checkpoint)
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
    return merge_and_export(task.encoder.eval(), out_dir, metadata)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="best.ckpt or last.ckpt written by histossl.train")
    parser.add_argument("output", help="the folder to write the model to")
    args = parser.parse_args(argv)
    print(export(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
