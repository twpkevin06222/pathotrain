"""Export a checkpoint written by :mod:`pathotrain.segmentation.train` to ONNX::

    python -m pathotrain.segmentation.export runs/exp01/checkpoints/best.ckpt model.onnx
    python -m pathotrain.segmentation.export best.ckpt model.onnx --size 1024

Needs the export extra (``pip install 'pathotrain[export]'``). The graph takes
``image`` (batch, 3, size, size) float32, normalised as in training, and returns
``logits`` (batch, num_classes, size, size); argmax over classes gives the mask.
The batch is dynamic; the side is the training tile (or crop) unless ``--size``.
Weights are fp32: convert to fp16 or quantise to int8 from this file.

Export writes a pathoinfer model folder: ``model_card.json`` beside the ONNX file, and
the same card under the graph's ``model_card`` metadata key, so pathoinfer runs the
file as is (see :func:`pathotrain.experiment.model_card`). Export into an empty folder:
the card is named ``model_card.json`` whatever the ONNX file is called. The card takes
the resolution and ICC setting from the training config; a model trained on tile
folders needs ``--mpp`` and ``--icc``/``--no-icc``, and export refuses without them.
Classes need real names (``task.class_names``, or ``data.class_map`` plus
``background`` for label 0). The ``pathotrain`` metadata key keeps the training
details (see :func:`pathotrain.experiment.onnx_metadata`).
Export checks ONNX Runtime against PyTorch on random images before returning.
With ``--register NAME``, the graph is also logged to MLflow (``--mlflow-db``,
else the config's ``mlflow_db``, else ``MLFLOW_TRACKING_URI``) and registered as
a new version of the model ``NAME``; load it with ``models:/NAME/<version>``.
"""

import argparse
from pathlib import Path

from pathotrain import experiment
from pathotrain.segmentation import train


def export(
    checkpoint: str | Path,
    path: str | Path,
    size: int | None = None,
    register: str | None = None,
    mlflow_db: str | Path | None = None,
    *,
    mpp: float | None = None,
    icc: bool | None = None,
    name: str | None = None,
    version: str = "1",
    overlap: float = 0.25,
) -> Path:
    """Write ``checkpoint``'s model to ``path`` as ONNX at ``size`` px (default the
    training tile side), with ``model_card.json`` beside it, and return the path.
    ``mpp``, ``icc``, ``name``, ``version``, ``overlap``: see
    :func:`pathotrain.experiment.model_card`. ``register``: also register it in MLflow
    under this name (see :func:`pathotrain.experiment.register_onnx`)."""
    model, cfg = train.load_model(checkpoint)
    size = size or experiment.input_size(cfg)
    if size is None:
        raise ValueError("the config trained on tile folders without a crop; pass a size")
    path = Path(path)
    outputs = {"logits": "(batch, num_classes, size, size) float32; argmax over classes"}
    metadata = experiment.onnx_metadata(cfg, "segmentation", size, outputs)
    output = {"tensor": "logits", "activation": "logits", "n_classes": cfg["model"]["num_classes"]}
    card = experiment.model_card(
        cfg,
        "segmentation",
        size,
        path.name,
        output,
        mpp=mpp,
        icc=icc,
        name=name,
        version=version,
        overlap=overlap,
        run_id=experiment.checkpoint_run_id(checkpoint),
    )
    experiment.export_onnx(model, path, size, list(outputs), metadata, card=card)
    experiment.write_card(card, path)
    if register is not None:
        experiment.register_onnx(path, register, cfg, checkpoint, mlflow_db)
    return Path(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="best.ckpt or last.ckpt written by train")
    parser.add_argument("output", help="the .onnx file to write")
    parser.add_argument("--size", type=int, help="input side in px (default: training tile)")
    parser.add_argument("--mpp", type=float, help="tile resolution; required for tile folders")
    parser.add_argument(
        "--icc",
        action=argparse.BooleanOptionalAction,
        help="whether training tiles were ICC-converted; required for tile folders",
    )
    parser.add_argument("--name", help="card name (default: the experiment name)")
    parser.add_argument("--version", default="1", help="card version (default: 1)")
    parser.add_argument(
        "--overlap", type=float, default=0.25, help="recommended tile overlap (default 0.25)"
    )
    parser.add_argument("--register", help="also register the graph in MLflow under this name")
    parser.add_argument(
        "--mlflow-db", help="MLflow SQLite file for --register (default: the config's mlflow_db)"
    )
    args = parser.parse_args(argv)
    export(
        args.checkpoint,
        args.output,
        args.size,
        args.register,
        args.mlflow_db,
        mpp=args.mpp,
        icc=args.icc,
        name=args.name,
        version=args.version,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
