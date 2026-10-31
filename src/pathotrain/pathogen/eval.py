"""Score a checkpoint written by :mod:`pathotrain.pathogen.train` on one split::

    python -m pathotrain.pathogen.eval runs/her2_sit01/checkpoints/best.ckpt --config config.yaml
    python -m pathotrain.pathogen.eval best.ckpt --split val --steps 50

The model and task settings (and its EMA weights) come from the checkpoint's
config. So do ``data``, ``loader`` and ``trainer``, unless ``--config`` names a
YAML with those sections, e.g. the experiment config with a ``test`` split
(``source`` and ``target`` folders) added after training. Tiles are prepared as
for validation (``data.crop`` centre crop).

Unlike validation, which samples only its first ``sample_batches`` batches,
every tile of the split is translated and scored. Runs on one device and prints
the metrics as JSON, keyed ``<split>/...``: ``psnr`` and ``ssim`` of the samples
against the targets, and ``loss``, the model's loss with fixed noise.

With ``mlflow_db`` in the config (the checkpoint's, or ``--config``'s) or
``--mlflow-db``, the metrics are also logged to that MLflow store as a run
``eval-<split>`` of the experiment, with the checkpoint path and steps as params.

Report test metrics: ``best.ckpt`` was chosen on ``val``, so its val score is
optimistic. Sampling every tile takes a while: ``--steps`` trades accuracy for time.
"""

import argparse
import json
from pathlib import Path

from pathotrain import experiment
from pathotrain.engine import load_config
from pathotrain.pathogen import train

OVERRIDES = ("data", "loader", "trainer", "mlflow_db")


def evaluate(
    checkpoint: str | Path,
    split: str = "test",
    config: dict | None = None,
    steps: int | None = None,
    mlflow_db: str | Path | None = None,
) -> dict[str, float]:
    """Score ``checkpoint`` on ``data[split]`` and return the metrics keyed ``<split>/...``.

    ``config``: an experiment config whose ``data``, ``loader``, ``trainer`` and
    ``mlflow_db`` replace the checkpoint's. ``steps``: sampling steps (default the
    task's ``sample_steps``). ``mlflow_db``: log the metrics to this MLflow store
    (default the config's ``mlflow_db``; neither logs nothing).
    """
    task, cfg = train.load_model(checkpoint)
    if config is not None:
        cfg = {**cfg, **{k: config[k] for k in OVERRIDES if k in config}}
    if split not in cfg["data"]:
        raise ValueError(f"data has no {split!r} split; pass --config with one")
    dataset = train.build_dataset(cfg, split)
    task.sample_batches = len(dataset)  # sample every batch, not validation's first few
    task.sample_steps = steps or task.sample_steps
    metrics = experiment.evaluate(cfg, task, dataset, split)
    experiment.log_eval(cfg, checkpoint, split, metrics, mlflow_db, steps=task.sample_steps)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="best.ckpt or last.ckpt written by train")
    parser.add_argument("--split", default="test", help="data split to score (default: test)")
    parser.add_argument("--config", help="YAML whose data, loader and trainer sections apply")
    parser.add_argument("--steps", type=int, help="sampling steps (default: the task's)")
    parser.add_argument("--mlflow-db", help="also log the metrics to this MLflow SQLite file")
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else None
    print(
        json.dumps(
            evaluate(args.checkpoint, args.split, config, args.steps, args.mlflow_db), indent=2
        )
    )


if __name__ == "__main__":
    main()
