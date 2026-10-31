"""Plumbing shared by the config-driven train, eval and export commands
(:mod:`pathotrain.segmentation.train`, :mod:`pathotrain.detection.train` and their
``eval`` and ``export`` modules)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from lightning_fabric import seed_everything
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset

from patchology import TilingConfig
from pathotrain.dataloader.augmentation import IMAGENET_MEAN, IMAGENET_STD
from pathotrain.engine import (
    CSVLogger,
    EarlyStopping,
    MLflowLogger,
    ModelCheckpoint,
    Task,
    Trainer,
    TrainerConfig,
    dump_config,
    flatten,
)
from pathotrain.engine.logger import sqlite_store
from pathotrain.pathofm import MIDNIGHT_MEAN, MIDNIGHT_STD

NORMALISATION = {
    "IMAGENET_MEAN": IMAGENET_MEAN,
    "IMAGENET_STD": IMAGENET_STD,
    "MIDNIGHT_MEAN": MIDNIGHT_MEAN,
    "MIDNIGHT_STD": MIDNIGHT_STD,
}


def normalisation(cfg: dict, key: str) -> tuple[float, float, float]:
    """``cfg[key]`` ("mean" or "std") as an RGB triple in [0, 1] units.

    A string must be ``IMAGENET_<KEY>`` or ``MIDNIGHT_<KEY>``; anything else
    must be a list or tuple of 3 numbers (custom statistics).
    """
    value = cfg.get(key)
    if isinstance(value, str):
        allowed = [f"IMAGENET_{key.upper()}", f"MIDNIGHT_{key.upper()}"]
        if value not in allowed:
            raise ValueError(f"{key} must be one of {allowed} or 3 numbers, got {value!r}")
        return NORMALISATION[value]
    if not (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    ):
        raise ValueError(f"{key} must be a preset name or 3 numbers, got {value!r}")
    value = tuple(float(v) for v in value)
    # Tiles are scaled to [0, 1] before Normalize, so 0-255 statistics are a mistake.
    if key == "mean" and not all(0 <= v <= 1 for v in value):
        raise ValueError(f"mean must be in [0, 1] units, got {value}")
    if key == "std" and not all(0 < v <= 1 for v in value):
        raise ValueError(f"std must be in (0, 1] units, got {value}")
    return value


def start(cfg: dict, resume: str | Path | None = None) -> Path:
    """Copy ``cfg`` to ``<output_dir>/config.yaml`` and seed everything; return the copy's path.

    A changed config for an existing run is refused unless resuming (see ``dump_config``).
    """
    # A resumed run may change the config (e.g. more epochs); record the new one.
    path = dump_config(cfg, Path(cfg["output_dir"]), overwrite=resume is not None)
    seed_everything(cfg["seed"], workers=True)
    return path


def fit(
    cfg: dict,
    task: Task,
    train_set: Dataset,
    val_set: Dataset | None,
    config_path: Path,
    resume: str | Path | None = None,
    monitor: str = "val/miou",
    trainer_defaults: dict | None = None,
    **loader_kwargs,
) -> dict[str, float]:
    """Train ``task`` as the ``trainer``, ``checkpoint``, ``early_stopping`` and
    ``loader`` sections of ``cfg`` say, logging to MLflow and CSV. MLflow writes
    to ``cfg["mlflow_db"]`` (artifacts in ``mlruns/`` beside it) when set, else to
    ``MLFLOW_TRACKING_URI``.

    ``monitor`` is the checkpoint and early-stopping default, ``trainer_defaults``
    the TrainerConfig defaults and ``loader_kwargs`` the DataLoader defaults
    (e.g. ``collate_fn``); the config overrides each.
    """
    out = Path(cfg["output_dir"])
    checkpoint = {"monitor": monitor, **cfg.get("checkpoint", {})}
    callbacks = [ModelCheckpoint(out / "checkpoints", config=cfg, **checkpoint)]
    if "early_stopping" in cfg:
        callbacks.append(EarlyStopping(**{"monitor": monitor, **cfg["early_stopping"]}))
    store = sqlite_store(cfg["mlflow_db"]) if "mlflow_db" in cfg else {}
    mlflow_logger = MLflowLogger(experiment=cfg["name"], **store)
    mlflow_logger.log_artifact(config_path)  # starts the run that fit() logs to
    trainer = Trainer(
        TrainerConfig(**{**(trainer_defaults or {}), **cfg.get("trainer", {})}),
        callbacks=callbacks,
        loggers=[mlflow_logger, CSVLogger(out, name="logs")],
    )
    loader = _loader_kwargs(cfg, loader_kwargs)
    val_loader = None if val_set is None else DataLoader(val_set, **loader)
    return trainer.fit(
        task,
        # Training workers persist (no per-epoch restart); val ones don't, so the two sets
        # are not both held in RAM for the whole run.
        DataLoader(
            train_set,
            shuffle=True,
            **{"persistent_workers": loader.get("num_workers", 0) > 0, **loader},
        ),
        val_loader,
        hparams=flatten(cfg),
        ckpt_path=resume,
    )


def evaluate(
    cfg: dict, task: Task, dataset: Dataset, split: str, **loader_kwargs
) -> dict[str, float]:
    """Score ``task`` (holding trained weights) on ``dataset`` and return its metrics
    keyed ``<split>/...``. Runs on one device, with ``cfg``'s accelerator, precision
    and ``loader`` section; ``loader_kwargs`` are DataLoader defaults, as for :func:`fit`."""
    trainer = cfg.get("trainer", {})
    config = {k: trainer[k] for k in ("accelerator", "precision") if k in trainer}
    # One device: DDP pads the last batch with repeated tiles (see Trainer.validate).
    metrics = Trainer(TrainerConfig(**config, devices=1)).evaluate(
        task, DataLoader(dataset, **_loader_kwargs(cfg, loader_kwargs))
    )
    return {f"{split}/{k.removeprefix('val/')}": v for k, v in metrics.items()}


def log_eval(
    cfg: dict,
    checkpoint: str | Path,
    split: str,
    metrics: dict[str, float],
    mlflow_db: str | Path | None = None,
    **params,
) -> None:
    """Log eval ``metrics`` to ``mlflow_db`` (default ``cfg["mlflow_db"]``; neither
    logs nothing) as a run ``eval-<split>`` of the experiment, with the checkpoint
    path and ``params`` as params."""
    mlflow_db = mlflow_db or cfg.get("mlflow_db")
    if mlflow_db is None:
        return
    run = MLflowLogger(cfg["name"], f"eval-{split}", **sqlite_store(mlflow_db))
    run.log_hyperparams({"checkpoint": str(Path(checkpoint).resolve()), **params})
    run.log_metrics(metrics)
    run.finalize("success")


def register_onnx(
    path: str | Path,
    name: str,
    cfg: dict,
    checkpoint: str | Path,
    mlflow_db: str | Path | None = None,
) -> int:
    """Log the ONNX graph at ``path`` (written by :func:`export_onnx`) to MLflow as
    the model of a run ``export`` of the experiment, register it as a new version
    of the registered model ``name`` and return the version number. The store is
    ``mlflow_db`` (default ``cfg["mlflow_db"]``, else ``MLFLOW_TRACKING_URI``); the
    graph's ``pathotrain`` metadata becomes the model's metadata."""
    import mlflow.onnx
    import onnx

    mlflow_db = mlflow_db or cfg.get("mlflow_db")
    run = MLflowLogger(cfg["name"], "export", **(sqlite_store(mlflow_db) if mlflow_db else {}))
    run.log_hyperparams({"checkpoint": str(Path(checkpoint).resolve())})  # starts the run
    graph = onnx.load(str(path))
    metadata = {p.key: p.value for p in graph.metadata_props}
    info = mlflow.onnx.log_model(
        graph,
        name="model",
        registered_model_name=name,
        metadata=json.loads(metadata.get("pathotrain", "{}")),
    )
    run.finalize("success")
    version = int(info.registered_model_version)
    logger.info(f"registered {path} as {name} version {version}")
    return version


def _loader_kwargs(cfg: dict, defaults: dict) -> dict:
    """DataLoader keyword arguments: pathotrain's defaults, then ``defaults``, then
    the config's ``loader`` section."""
    loader = {
        # Pinned batches let the non-blocking copy to the GPU overlap compute (~2x faster).
        "pin_memory": torch.cuda.is_available(),
        # Fresh workers: forking the threaded training process can deadlock them.
        "multiprocessing_context": "forkserver",
        **defaults,
        **cfg.get("loader", {}),
    }
    if not loader.get("num_workers"):  # DataLoader refuses a context without workers
        loader.pop("multiprocessing_context", None)
    return loader


def input_size(cfg: dict) -> int | None:
    """The side of the tiles the model trained on: ``augmentation.crop``, else the
    train split's ``tile_size``; None for tile folders without a crop."""
    crop = cfg.get("augmentation", {}).get("crop")
    train = cfg["data"]["train"]
    if crop is None and "slides" not in train:
        return None
    return crop or TilingConfig(**train.get("tiling", {})).tile_size


def onnx_metadata(cfg: dict, task: str, size: int, outputs: dict[str, str]) -> dict:
    """What an inference runtime needs besides the graph: preprocessing, the tiling
    the model trained on and the classes. ``outputs`` describes each graph output."""
    train = cfg["data"]["train"]
    num_classes = cfg["model"]["num_classes"]
    return {
        "task": task,
        "model": cfg["model"]["name"],
        "input": f"image: (batch, 3, {size}, {size}) float32 RGB; tile / 255, then (x - mean) / std",
        "mean": normalisation(cfg, "mean"),
        "std": normalisation(cfg, "std"),
        "tile_size": size,
        "tile_mpp": TilingConfig(**train.get("tiling", {})).tile_mpp if "slides" in train else None,
        "num_classes": num_classes,
        "class_names": cfg.get("task", {}).get("class_names")
        or [str(i) for i in range(num_classes)],
        "class_map": cfg["data"].get("class_map"),
        "outputs": outputs,
    }


def export_onnx(
    model: nn.Module,
    path: str | Path,
    size: int,
    output_names: list[str],
    metadata: dict,
    opset: int = 17,
    atol: float = 1e-3,
    inputs: dict[str, tuple[int, ...]] | None = None,
) -> Path:
    """Write ``model`` (images -> ``output_names`` tensors) to ``path`` as fp32 ONNX
    with a dynamic batch, store ``metadata`` as JSON under the ``pathotrain`` metadata
    key, and check ONNX Runtime against PyTorch on random images. Weights over 2 GB
    go to ``<path>.data``, which must stay beside the graph. ``inputs``: the graph's
    inputs and their shapes without the batch, in ``model``'s argument order
    (default ``image`` of ``(3, size, size)``).

    Raises:
        AssertionError: An output differs from PyTorch by more than ``atol`` (relative
            to its largest magnitude when that is above 1).
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "ONNX export needs the export extra: pip install 'pathotrain[export]'"
        ) from e
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu().float()
    inputs = inputs or {"image": (3, size, size)}
    names = [*inputs, *output_names]
    # Over 2 GB the exporter writes one file per weight; trace into a scratch folder and
    # save the weights as a single <name>.data beside the graph instead.
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp, torch.no_grad():
        traced = Path(tmp) / path.name
        torch.onnx.export(
            model,
            tuple(torch.randn(1, *shape) for shape in inputs.values()),
            str(traced),
            input_names=list(inputs),
            output_names=output_names,
            dynamic_axes={n: {0: "batch"} for n in names},
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,  # the dynamo exporter needs onnxscript and misses RF-DETR's ops
        )
        external = len(list(Path(tmp).iterdir())) > 1
        graph = onnx.load(str(traced))
    graph.metadata_props.add(key="pathotrain", value=json.dumps(metadata))
    onnx.save(
        graph,
        str(path),
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location=f"{path.name}.data",
    )

    # Another batch size than traced: checks the axis.
    feeds = {name: torch.randn(2, *shape) for name, shape in inputs.items()}
    with torch.no_grad():
        expected = model(*feeds.values())
    expected = [expected] if isinstance(expected, torch.Tensor) else list(expected)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(output_names, {k: v.numpy() for k, v in feeds.items()})
    for name, want, got in zip(output_names, expected, actual, strict=True):
        want = want.numpy()
        diff = float(np.abs(want.astype(np.float64) - got).max()) if want.size else 0.0
        tol = atol * max(1.0, float(np.abs(want).max()) if want.size else 1.0)
        assert diff <= tol, f"ONNX output {name!r} differs from PyTorch by {diff:.3g} (> {tol:.3g})"
        logger.info(f"{name}: {tuple(got.shape)}, max |ONNX - PyTorch| = {diff:.2g}")
    logger.info(f"wrote {path}")
    return path
