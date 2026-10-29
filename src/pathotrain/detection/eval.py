"""Score a checkpoint written by :mod:`pathotrain.detection.train` on one split::

    python -m pathotrain.detection.eval runs/det01/checkpoints/best.ckpt
    python -m pathotrain.detection.eval best.ckpt --split val --config config.yaml

The model, normalisation and task settings come from the checkpoint's config.
So do ``data``, ``loader`` and ``trainer``, unless ``--config`` names a YAML with
those sections, e.g. the experiment config with a ``test`` split added after
training. Runs on one device and prints the metrics as JSON, keyed
``<split>/...`` (``test/map``, ``test/map_50``, ``test/ap_<class>``, ...).

Report test metrics: ``best.ckpt`` was chosen on ``val``, so its val score is
optimistic. Metrics are per tile; tile the split with ``tile_overlap: 0`` so no
object is scored twice.
"""

import argparse
import json
from pathlib import Path

from pathotrain import experiment
from pathotrain.dataloader.augmentation import eval_transform
from pathotrain.detection import train
from pathotrain.detection.task import DetectionTask, collate
from pathotrain.engine import load_config
from pathotrain.experiment import normalisation

OVERRIDES = ("data", "loader", "trainer")


def evaluate(
    checkpoint: str | Path, split: str = "test", config: dict | None = None
) -> dict[str, float]:
    """Score ``checkpoint`` on ``data[split]`` and return the metrics keyed ``<split>/...``.

    ``config``: an experiment config whose ``data``, ``loader`` and ``trainer``
    sections replace the checkpoint's.
    """
    detector, cfg = train.load_model(checkpoint)
    if config is not None:
        cfg = {**cfg, **{k: config[k] for k in OVERRIDES if k in config}}
    if split not in cfg["data"]:
        raise ValueError(f"data has no {split!r} split; pass --config with one")
    mean, std = normalisation(cfg, "mean"), normalisation(cfg, "std")
    crop = cfg.get("augmentation", {}).get("crop")
    transform = eval_transform(crop, mean=mean, std=std, boxes=True)
    dataset = train.build_dataset(cfg, split, transform)
    task = DetectionTask(detector, **cfg.get("task", {}))
    return experiment.evaluate(cfg, task, dataset, split, collate_fn=collate)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="best.ckpt or last.ckpt written by train")
    parser.add_argument("--split", default="test", help="data split to score (default: test)")
    parser.add_argument("--config", help="YAML whose data, loader and trainer sections apply")
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else None
    print(json.dumps(evaluate(args.checkpoint, args.split, config), indent=2))


if __name__ == "__main__":
    main()
