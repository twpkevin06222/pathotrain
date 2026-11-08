"""Export a checkpoint written by :mod:`pathotrain.pathocls.train` to ONNX::

    python -m pathotrain.pathocls.export runs/cls01/checkpoints/best.ckpt out/model.onnx \\
        --mpp 0.5 --icc

Needs the export extra (``pip install 'pathotrain[export]'``). The graph is the whole
classifier (backbone and head): it takes ``image`` (batch, 3, size, size) float32 at the
backbone's input size, normalised with the backbone's mean and std, and returns
``logits`` (batch, num_classes); softmax gives class probabilities.

Export writes a pathoinfer model folder (``model_card.json`` beside the ONNX file, and
the card under the graph's ``model_card`` key; see
:func:`pathotrain.experiment.model_card`), so export into an empty folder. Tile folders
record no resolution or colour handling, so ``--mpp`` and ``--icc``/``--no-icc`` are
required. ``--mpp`` is the resolution of the ``size`` px input the model sees: tiles
of 448 px at 0.25 um/px resized to 224 px give 0.5. ``--register NAME`` also registers
the graph in MLflow, as for segmentation.
"""

import argparse
from pathlib import Path

from pathotrain import experiment
from pathotrain.pathocls import train
from pathotrain.pathocls.backbones import preprocessing


def export(
    checkpoint: str | Path,
    path: str | Path,
    register: str | None = None,
    mlflow_db: str | Path | None = None,
    *,
    mpp: float | None = None,
    icc: bool | None = None,
    name: str | None = None,
    version: str = "1",
) -> Path:
    """Write ``checkpoint``'s classifier to ``path`` as ONNX, with ``model_card.json``
    beside it, and return the path. ``mpp``, ``icc``, ``name``, ``version``: see
    :func:`pathotrain.experiment.model_card`; ``register``: as
    :func:`pathotrain.experiment.register_onnx`."""
    classifier, cfg = train.load_model(checkpoint)
    pp = preprocessing(classifier.backbone)
    size = pp["input_size"]
    cfg = cfg | {"mean": pp["mean"], "std": pp["std"]}  # the backbone's, for the card
    path = Path(path)
    outputs = {"logits": "(batch, num_classes) float32; softmax for probabilities"}
    metadata = experiment.onnx_metadata(cfg, "classification", size, outputs)
    n = cfg["model"]["num_classes"]
    card = experiment.model_card(
        cfg,
        "classification",
        size,
        path.name,
        {"tensor": "logits", "activation": "logits", "n_classes": n},
        mpp=mpp,
        icc=icc,
        name=name,
        version=version,
        overlap=0.0,
        run_id=experiment.checkpoint_run_id(checkpoint),
    )
    experiment.export_onnx(classifier.eval(), path, size, list(outputs), metadata, card=card)
    experiment.write_card(card, path)
    if register is not None:
        experiment.register_onnx(path, register, cfg, checkpoint, mlflow_db)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="best.ckpt or last.ckpt written by train")
    parser.add_argument("output", help="the .onnx file to write")
    parser.add_argument("--mpp", type=float, help="um/px of the model's input tile (required)")
    parser.add_argument(
        "--icc",
        action=argparse.BooleanOptionalAction,
        help="whether training tiles were ICC-converted (required)",
    )
    parser.add_argument("--name", help="card name (default: the experiment name)")
    parser.add_argument("--version", default="1", help="card version (default: 1)")
    parser.add_argument("--register", help="also register the graph in MLflow under this name")
    parser.add_argument(
        "--mlflow-db", help="MLflow SQLite file for --register (default: the config's mlflow_db)"
    )
    args = parser.parse_args(argv)
    export(
        args.checkpoint,
        args.output,
        args.register,
        args.mlflow_db,
        mpp=args.mpp,
        icc=args.icc,
        name=args.name,
        version=args.version,
    )


if __name__ == "__main__":
    main()
