"""Score a checkpoint written by :mod:`pathotrain.cellseg.train` on one split::

    python -m pathotrain.cellseg.eval runs/exp/checkpoints/best.ckpt
    python -m pathotrain.cellseg.eval best.ckpt --split val --config instance.yaml

Prints the metrics as JSON, keyed ``<split>/...`` (``test/bpq``, ``test/mpq``,
``test/pq_<type>``, ``test/dq``, ``test/sq``, ``test/loss``). ``--config`` replaces the
checkpoint's ``data``, ``loader``, ``val_loader``, ``trainer`` and ``mlflow_db``, as in
``pathotrain.segmentation.eval``; with an MLflow store, the metrics are logged there too.
"""

import argparse
import json
from pathlib import Path

from pathotrain import experiment
from pathotrain.cellseg import train
from pathotrain.engine import load_config
from pathotrain.segmentation.eval import OVERRIDES


def evaluate(
    checkpoint: str | Path,
    split: str = "test",
    config: dict | None = None,
    mlflow_db: str | Path | None = None,
) -> dict[str, float]:
    """Score ``checkpoint`` on ``data[split]`` and return the metrics keyed ``<split>/...``."""
    task, cfg = train.load_task(checkpoint)
    if config is not None:
        cfg = {**cfg, **{k: config[k] for k in OVERRIDES if k in config}}
    if split not in cfg["data"]:
        raise ValueError(f"data has no {split!r} split; pass --config with one")
    metrics = experiment.evaluate(cfg, task, train.build_dataset(cfg, split), split)
    experiment.log_eval(cfg, checkpoint, split, metrics, mlflow_db)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="best.ckpt or last.ckpt written by train")
    parser.add_argument("--split", default="test", help="data split to score (default: test)")
    parser.add_argument(
        "--config", help="YAML whose data, loader, val_loader and trainer sections apply"
    )
    parser.add_argument("--mlflow-db", help="also log the metrics to this MLflow SQLite file")
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else None
    print(json.dumps(evaluate(args.checkpoint, args.split, config, args.mlflow_db), indent=2))


if __name__ == "__main__":
    main()
