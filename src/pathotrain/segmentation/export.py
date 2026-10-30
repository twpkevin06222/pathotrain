"""Export a checkpoint written by :mod:`pathotrain.segmentation.train` to ONNX::

    python -m pathotrain.segmentation.export runs/exp01/checkpoints/best.ckpt model.onnx
    python -m pathotrain.segmentation.export best.ckpt model.onnx --size 1024

Needs the export extra (``pip install 'pathotrain[export]'``). The graph takes
``image`` (batch, 3, size, size) float32, normalised as in training, and returns
``logits`` (batch, num_classes, size, size); argmax over classes gives the mask.
The batch is dynamic; the side is the training tile (or crop) unless ``--size``.
Weights are fp32: convert to fp16 or quantise to int8 from this file.

The ``pathotrain`` metadata key holds JSON with the mean/std, ``tile_size``,
``tile_mpp``, class names and ``class_map`` (see :func:`pathotrain.experiment.onnx_metadata`).
Export checks ONNX Runtime against PyTorch on random images before returning.
"""

import argparse
from pathlib import Path

from pathotrain import experiment
from pathotrain.segmentation import train


def export(checkpoint: str | Path, path: str | Path, size: int | None = None) -> Path:
    """Write ``checkpoint``'s model to ``path`` as ONNX at ``size`` px (default the
    training tile side) and return the path."""
    model, cfg = train.load_model(checkpoint)
    size = size or experiment.input_size(cfg)
    if size is None:
        raise ValueError("the config trained on tile folders without a crop; pass a size")
    outputs = {"logits": "(batch, num_classes, size, size) float32; argmax over classes"}
    metadata = experiment.onnx_metadata(cfg, "segmentation", size, outputs)
    return experiment.export_onnx(model, path, size, list(outputs), metadata)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="best.ckpt or last.ckpt written by train")
    parser.add_argument("output", help="the .onnx file to write")
    parser.add_argument("--size", type=int, help="input side in px (default: training tile)")
    args = parser.parse_args(argv)
    export(args.checkpoint, args.output, args.size)


if __name__ == "__main__":
    main()
