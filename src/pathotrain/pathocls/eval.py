"""Score a checkpoint written by :mod:`pathotrain.pathocls.train` on one split::

    python -m pathotrain.pathocls.eval runs/cls01/checkpoints/best.ckpt
    python -m pathotrain.pathocls.eval best.ckpt --split val --config config.yaml

The backbone, head and class order come from the checkpoint. So do ``data``,
``features``, ``loader``, ``val_loader`` and ``trainer``, unless ``--config`` names a
YAML with those sections, e.g. the experiment config with a ``test`` split added after
training. Prints the metrics as JSON keyed ``<split>/...`` (``test/acc``,
``test/balanced_acc``, ``test/macro_f1``, ``test/auroc``, ``test/recall_<class>``,
``test/loss``). With ``mlflow_db`` in the config or ``--mlflow-db``, they are also
logged as a run ``eval-<split>`` of the experiment.

Report test metrics: ``best.ckpt`` was chosen on ``val``, so its val score is optimistic.
"""

import argparse
import json
from pathlib import Path

import torch

from pathotrain import experiment
from pathotrain.engine import load_config
from pathotrain.pathocls import ClassificationTask, train

OVERRIDES = ("data", "features", "loader", "val_loader", "trainer", "mlflow_db")


def evaluate(
    checkpoint: str | Path,
    split: str = "test",
    config: dict | None = None,
    mlflow_db: str | Path | None = None,
) -> dict[str, float]:
    """Score ``checkpoint`` on ``data[split]`` and return the metrics keyed ``<split>/...``.

    ``config``: an experiment config whose sections in :data:`OVERRIDES` replace the
    checkpoint's. ``mlflow_db``: log the metrics to this MLflow store (default the
    config's ``mlflow_db``; neither logs nothing).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier, cfg = train.load_model(checkpoint, device)
    if config is not None:
        cfg = {**cfg, **{k: config[k] for k in OVERRIDES if k in config}}
    if split not in cfg["data"]:
        raise ValueError(f"data has no {split!r} split; pass --config with one")
    dataset = train.embed(cfg, classifier.backbone, split)
    task = ClassificationTask(classifier.head, cfg["model"]["num_classes"], **cfg["task"])
    metrics = experiment.evaluate(cfg, task, dataset, split)
    experiment.log_eval(cfg, checkpoint, split, metrics, mlflow_db)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="best.ckpt or last.ckpt written by train")
    parser.add_argument("--split", default="test", help="data split to score (default: test)")
    parser.add_argument("--config", help="YAML whose data, features and loader sections apply")
    parser.add_argument("--mlflow-db", help="also log the metrics to this MLflow SQLite file")
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else None
    print(json.dumps(evaluate(args.checkpoint, args.split, config, args.mlflow_db), indent=2))


if __name__ == "__main__":
    main()
