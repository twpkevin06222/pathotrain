"""Plumbing shared by the config-driven train, eval and export commands
(:mod:`pathotrain.segmentation.train`, :mod:`pathotrain.detection.train` and their
``eval`` and ``export`` modules)."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from lightning_fabric import seed_everything
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

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


_log_sink: int | None = None


def git_commit() -> str:
    """HEAD of the pathotrain checkout, with ``-dirty`` for uncommitted changes, or
    ``"unknown"`` (with a warning) outside a git checkout: checkpoint provenance."""
    git = ["git", "-C", str(Path(__file__).resolve().parent)]
    try:
        commit = subprocess.run(
            [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            [*git, "status", "--porcelain"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        logger.warning("no git commit found for the checkpoint provenance")
        return "unknown"
    return commit.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def start(cfg: dict, resume: str | Path | None = None) -> Path:
    """Copy ``cfg`` to ``<output_dir>/config.yaml``, log to ``<output_dir>/train.log``
    (appended on resume) and seed everything; return the config copy's path.

    A changed config for an existing run is refused unless resuming (see ``dump_config``).
    """
    global _log_sink
    # A resumed run may change the config (e.g. more epochs); record the new one.
    path = dump_config(cfg, Path(cfg["output_dir"]), overwrite=resume is not None)
    if _log_sink is not None:  # one run per process at a time (sweeps, tests call train again)
        logger.remove(_log_sink)
    _log_sink = logger.add(path.parent / "train.log")
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
    mode: str = "max",
    trainer_defaults: dict | None = None,
    checkpoint_extra: dict | None = None,
    train_sampler: Sampler | None = None,
    **loader_kwargs,
) -> dict[str, float]:
    """Train ``task`` as the ``trainer``, ``checkpoint``, ``early_stopping`` and
    ``loader`` sections of ``cfg`` say, logging to MLflow and CSV. MLflow writes
    to ``cfg["mlflow_db"]`` (artifacts in ``mlruns/`` beside it) when set, else to
    ``MLFLOW_TRACKING_URI``.

    ``monitor`` (and its ``mode``) is the checkpoint and early-stopping default, ``trainer_defaults``
    the TrainerConfig defaults and ``loader_kwargs`` the DataLoader defaults
    (e.g. ``collate_fn``); the config overrides each. ``checkpoint_extra`` is stored
    in both checkpoints beside the config (see :class:`ModelCheckpoint`).
    ``train_sampler`` replaces shuffling, e.g. ``RandomSampler(train_set,
    num_samples=n)`` for epochs of ``n`` tiles.
    """
    out = Path(cfg["output_dir"])
    store = sqlite_store(cfg["mlflow_db"]) if "mlflow_db" in cfg else {}
    mlflow_logger = MLflowLogger(experiment=cfg["name"], **store)
    mlflow_logger.log_artifact(config_path)  # starts the run that fit() logs to
    # The run id goes into the checkpoints, so export can record it in the model card.
    extra = {**(checkpoint_extra or {}), "mlflow_run_id": mlflow_logger.run_id}
    checkpoint = {"monitor": monitor, "mode": mode, **cfg.get("checkpoint", {})}
    callbacks = [ModelCheckpoint(out / "checkpoints", config=cfg, extra=extra, **checkpoint)]
    if "early_stopping" in cfg:
        callbacks.append(
            EarlyStopping(**{"monitor": monitor, "mode": mode, **cfg["early_stopping"]})
        )
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
            **({"shuffle": True} if train_sampler is None else {"sampler": train_sampler}),
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
    if not loader.get("num_workers"):  # DataLoader refuses these without workers
        loader.pop("multiprocessing_context", None)
        loader.pop("persistent_workers", None)
    return loader


def check_foreground_crop(cfg: dict) -> None:
    """Refuse ``augmentation.foreground_crop`` when the training split is tiled from
    slides: their tiles are already selected for ground truth (``object_mask_overlap``,
    ``min_objects``), and pulling crops towards it as well would under-represent
    unlabelled tissue and undo ``negative_sampling``.

    Raises:
        ValueError: ``foreground_crop`` is set and ``data.train`` has ``slides``.
    """
    if cfg.get("augmentation", {}).get("foreground_crop") and "slides" in cfg["data"]["train"]:
        raise ValueError(
            "augmentation.foreground_crop is for tile folders; the slide training split "
            "already selects tiles by ground truth (use sampling or object_mask_overlap)"
        )


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


# Matplotlib's tab20, the default class colours of pathoinfer cards (in class order).
TAB20 = [
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229),
]
CARD_FILE = "model_card.json"


def model_card(
    cfg: dict,
    task: str,
    size: int,
    file: str,
    output: dict,
    *,
    mpp: float | None = None,
    icc: bool | None = None,
    name: str | None = None,
    version: str = "1",
    overlap: float = 0.25,
    run_id: str | None = None,
) -> dict:
    """The pathoinfer ModelCard (schema version 1) for an exported ONNX ``file``, as JSON.

    pathoinfer owns the schema (its ``docs/model_contract.md``); pathotrain writes it
    without importing pathoinfer. ``output`` is the task's output object. ``mpp`` and
    ``icc`` come from the training config when it tiled slides; a model trained on tile
    folders must be given both, as nothing records them. ``overlap`` is the tile overlap
    to infer with (``recommended_overlap``); ``run_id`` the MLflow training run.

    Raises:
        ValueError: mpp or icc unknown, a class has no name, or ``mpp`` contradicts the
            training resolution.
    """
    train = cfg["data"]["train"]
    if "slides" in train:
        trained_mpp = TilingConfig(**train.get("tiling", {})).tile_mpp
        if mpp is not None and mpp != trained_mpp:
            raise ValueError(f"the model trained at {trained_mpp} mpp; got mpp={mpp}")
        mpp = trained_mpp
        color = cfg["data"].get("color") or {}
        trained_icc = color.get("icc", True) if isinstance(color, dict) else color.icc
        if icc is not None and icc != trained_icc:
            raise ValueError(f"the model trained with icc={trained_icc}; got icc={icc}")
        icc = trained_icc
    if mpp is None or icc is None:
        raise ValueError(
            "the model trained on tile folders, which do not record their resolution or "
            "colour handling: pass mpp (--mpp) and icc (--icc or --no-icc)"
        )
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    mean, std = normalisation(cfg, "mean"), normalisation(cfg, "std")
    card = {
        "schema_version": 1,
        "name": name or cfg["name"],
        "version": version,
        "task": task,
        "framework": "onnx",
        "file": file,
        "precision": "fp32",
        "input": {
            "name": "image",
            "tile_size": size,
            "mpp": mpp,
            "channel_order": "RGB",
            "layout": "NCHW",
            "dtype": "float32",
            "normalisation": {"mean": list(mean), "std": list(std), "scale": 1 / 255},
            "icc": icc,
        },
        "classes": card_classes(cfg, task),
        "output": output,
        "recommended_overlap": overlap,
        "provenance": {
            "training_run_id": run_id,
            "source_metadata": {"pathotrain": onnx_metadata(cfg, task, size, {})},
        },
    }
    if task == "segmentation":
        card["background_id"] = 0
    return card


def card_classes(cfg: dict, task: str) -> list[dict]:
    """``{id, name, color}`` per output index. Names come from ``task.class_names``, else
    from ``data.class_map`` (inverted; label 0 is ``background`` for segmentation);
    colours from the top-level ``class_colors`` (by name), else tab20.

    Raises:
        ValueError: a class has no name, or ``class_colors`` names an unknown class.
    """
    n = cfg["model"]["num_classes"]
    names = cfg.get("task", {}).get("class_names")
    if not names:
        by_label = {label: name for name, label in (cfg["data"].get("class_map") or {}).items()}
        if task == "segmentation":
            by_label.setdefault(0, "background")
        missing = [i for i in range(n) if i not in by_label]
        if missing:
            raise ValueError(
                f"classes {missing} have no name: set task.class_names (one per class)"
            )
        names = [by_label[i] for i in range(n)]
    if len(names) != n:
        raise ValueError(f"{len(names)} class names for {n} classes")
    colors = cfg.get("class_colors") or {}
    unknown = set(colors) - set(names)
    if unknown:
        raise ValueError(f"class_colors names unknown classes {sorted(unknown)}")
    return [
        {"id": i, "name": name, "color": list(colors.get(name, TAB20[i % len(TAB20)]))}
        for i, name in enumerate(names)
    ]


def checkpoint_run_id(checkpoint: str | Path) -> str | None:
    """The MLflow run id :func:`fit` stored in ``checkpoint``, if any."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    run_id = state.get("mlflow_run_id")
    return str(run_id) if run_id else None


def write_card(card: dict, onnx_path: Path) -> Path:
    """Write ``card`` as ``model_card.json`` beside ``onnx_path``."""
    path = onnx_path.parent / CARD_FILE
    path.write_text(json.dumps(card, indent=2) + "\n")
    return path


def export_onnx(
    model: nn.Module,
    path: str | Path,
    size: int,
    output_names: list[str],
    metadata: dict,
    opset: int = 17,
    atol: float = 1e-3,
    inputs: dict[str, tuple[int, ...]] | None = None,
    card: dict | None = None,
) -> Path:
    """Write ``model`` (images -> ``output_names`` tensors) to ``path`` as fp32 ONNX
    with a dynamic batch, store ``metadata`` as JSON under the ``pathotrain`` metadata
    key (and ``card``, a pathoinfer ModelCard, under ``model_card``), and check ONNX
    Runtime against PyTorch on random images. Weights over 2 GB
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
    if card is not None:
        graph.metadata_props.add(key="model_card", value=json.dumps(card))
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
