"""Export a checkpoint written by :mod:`pathotrain.detection.train` to ONNX::

    python -m pathotrain.detection.export runs/det01/checkpoints/best.ckpt model.onnx

Needs the export extra (``pip install 'pathotrain[export]'``). The graph takes
``image`` (batch, 3, size, size) float32, normalised as in training, and returns
the top 300 class/query pairs per image, best first, as in validation:

- ``boxes`` (batch, 300, 4) float32, ``[x1, y1, x2, y2]`` tile pixels
- ``scores`` (batch, 300) float32 sigmoid probabilities; threshold these
- ``labels`` (batch, 300) int64, ``0 .. num_classes - 1``

The batch is dynamic; the side is fixed (the backbone's position embeddings are
frozen to it): the training tile (or crop) unless ``--size``, a multiple of 32.
Weights are fp32: convert to fp16 or quantise to int8 from this file. Export writes a
pathoinfer model folder as segmentation export does (``model_card.json`` beside the
graph and under its ``model_card`` metadata key; ``--mpp`` and ``--icc`` for tile
folders): boxes are ``xyxy`` tile pixels, and ``--score-threshold`` (default 0.5) is
the card's default threshold. Classes are ``data.class_map``'s labels, named from
``task.class_names`` or the map. The ``pathotrain`` key keeps the training details.

With ``--register NAME``, the graph is also logged to MLflow (``--mlflow-db``,
else the config's ``mlflow_db``, else ``MLFLOW_TRACKING_URI``) and registered as
a new version of the model ``NAME``; load it with ``models:/NAME/<version>``.
"""

import argparse
import copy
from pathlib import Path

import torch
from torch import Tensor, nn

from pathotrain import experiment
from pathotrain.detection import train

try:
    from rfdetr.export._backend import _switch_to_export_mode  # rfdetr is pinned
    from rfdetr.export.prepare import prepare_export_graph
except ImportError as e:
    raise ImportError(
        "RF-DETR needs the detection extra: pip install 'pathotrain[detection]'"
    ) from e


class _Decoded(nn.Module):
    """RF-DETR's export forward plus its post-processing, so the graph returns boxes."""

    def __init__(self, model: nn.Module, num_classes: int, num_select: int, size: int):
        super().__init__()
        self.model, self.num_classes, self.num_select = model, num_classes, num_select
        self.size = size

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        boxes, logits = self.model(image)[:2]  # normalised cxcywh, (batch, queries, classes + 1)
        # The head has one spare logit past the last class; it is never a target.
        prob = logits[..., : self.num_classes].sigmoid().flatten(1)
        scores, index = prob.topk(min(self.num_select, prob.shape[1]), dim=1)
        query, labels = index // self.num_classes, index % self.num_classes
        cx, cy, w, h = boxes.unbind(-1)
        xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
        xyxy = xyxy.gather(1, query.unsqueeze(-1).expand(-1, -1, 4))
        return (xyxy * self.size).clamp(0, self.size), scores, labels


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
    score_threshold: float = 0.5,
) -> Path:
    """Write ``checkpoint``'s detector to ``path`` as ONNX at ``size`` px (default the
    training tile side), with ``model_card.json`` beside it, and return the path.
    ``mpp``, ``icc``, ``name``, ``version``, ``overlap``: see
    :func:`pathotrain.experiment.model_card`; ``score_threshold`` is the card's default.
    ``register``: also register it in MLflow under this name (see
    :func:`pathotrain.experiment.register_onnx`)."""
    if not 0 <= score_threshold <= 1:
        raise ValueError(f"score_threshold must be in [0, 1], got {score_threshold}")
    detector, cfg = train.load_model(checkpoint)
    size = size or experiment.input_size(cfg) or detector.config.resolution
    if size % 32:
        raise ValueError(f"size must be a multiple of 32 for RF-DETR, got {size}")
    model = copy.deepcopy(detector.model)
    graph = prepare_export_graph(model, detector.config, shape=(size, size), device="cpu")
    _switch_to_export_mode(graph.model)
    decoded = _Decoded(
        graph.model, cfg["model"]["num_classes"], detector.postprocess.num_select, size
    )
    outputs = {
        "boxes": "(batch, 300, 4) float32 [x1, y1, x2, y2] tile pixels, best first",
        "scores": "(batch, 300) float32 sigmoid probabilities",
        "labels": "(batch, 300) int64 class labels",
    }
    path = Path(path)
    metadata = experiment.onnx_metadata(cfg, "detection", size, outputs)
    output = {
        "boxes": "boxes",
        "scores": "scores",
        "labels": "labels",
        "box_format": "xyxy",
        "box_units": "pixel",
        "score_threshold": score_threshold,
    }
    card = experiment.model_card(
        cfg,
        "detection",
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
    experiment.export_onnx(decoded, path, size, list(outputs), metadata, card=card)
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
    parser.add_argument(
        "--score-threshold", type=float, default=0.5, help="the card's default (0.5)"
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
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
