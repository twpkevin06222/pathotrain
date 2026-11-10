"""Export a checkpoint written by :mod:`pathotrain.cellseg.train` to ONNX::

    python -m pathotrain.cellseg.export best.ckpt model.onnx --size 256 --mpp 0.25 --icc

Needs the export extra (``pip install 'pathotrain[export]'``). The graph takes ``image``
(batch, 3, size, size) float32, normalised as in training, and returns the network's
three branches, before any activation:

- ``np``: (batch, 2, size, size) logits, background and nucleus.
- ``hv``: (batch, 2, size, size) horizontal and vertical offsets from the nucleus centre.
- ``nt``: (batch, num_classes, size, size) nucleus-type logits.

The watershed runs outside the graph: on one tile's outputs,
``pathotrain.cellseg.instances.instances_from_outputs(np[i], hv[i], nt[i])`` returns the
instance map and each instance's type. Tiles must overlap and their nuclei be merged
across tile borders when inferring a slide.

Export writes ``model_card.json`` beside the ONNX file and embeds it, as
``pathotrain.segmentation.export`` does. pathoinfer has no instance task, so the card is
a **segmentation** card on ``nt``: pathoinfer runs the file as a per-pixel nucleus-type
map. Classes need real names (``task.class_names``). Tile folders record no resolution,
so ``--mpp`` and ``--icc``/``--no-icc`` are required, and ``--size`` without a training crop.
"""

import argparse
from pathlib import Path

from torch import Tensor, nn

from pathotrain import experiment
from pathotrain.cellseg import train
from pathotrain.cellseg.models import BRANCHES
from pathotrain.segmentation.task import input_multiple


class _Branches(nn.Module):
    """The model's branches as a tuple, in :data:`BRANCHES` order, for tracing."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        out = self.model(image)
        return tuple(out[k] for k in BRANCHES)


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
    """Write ``checkpoint``'s model to ``path`` as ONNX at ``size`` px (default the training
    crop), with ``model_card.json`` beside it, and return the path. Options as for
    :func:`pathotrain.segmentation.export.export`."""
    model, cfg = train.load_model(checkpoint)
    size = size or experiment.input_size(cfg)
    if size is None:
        raise ValueError("the config trained on tile folders without a crop; pass a size")
    multiple = input_multiple(model)
    if size % multiple:
        raise ValueError(f"size must be divisible by {multiple} for {cfg['model']['name']}")
    path = Path(path)
    outputs = {
        "np": "(batch, 2, size, size) float32 logits: background, nucleus",
        "hv": "(batch, 2, size, size) float32: horizontal, vertical offset from the centre",
        "nt": "(batch, num_classes, size, size) float32 nucleus-type logits",
    }
    metadata = experiment.onnx_metadata(cfg, "instance_segmentation", size, outputs)
    output = {"tensor": "nt", "activation": "logits", "n_classes": cfg["model"]["num_classes"]}
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
    experiment.export_onnx(_Branches(model), path, size, list(BRANCHES), metadata, card=card)
    experiment.write_card(card, path)
    if register is not None:
        experiment.register_onnx(path, register, cfg, checkpoint, mlflow_db)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="best.ckpt or last.ckpt written by train")
    parser.add_argument("output", help="the .onnx file to write")
    parser.add_argument("--size", type=int, help="input side in px (default: training crop)")
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
