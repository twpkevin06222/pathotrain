"""Hugging Face Trainer callbacks that make a TRL run look like any pathotrain run:
the same MLflow / CSV loggers and metric names, and provenance in every checkpoint."""

import json
from pathlib import Path

import torch
from lightning_fabric.loggers import Logger
from transformers import TrainerCallback

from pathotrain.engine.trainer import peak_gpu_memory

# The end-of-training summary of the HF Trainer (logged once, as its own keys).
SUMMARY = {"train_runtime", "train_samples_per_second", "train_steps_per_second", "total_flos"}


def pathotrain_name(key: str) -> str:
    """HF Trainer log key -> pathotrain metric name: ``loss`` -> ``train/loss``,
    ``learning_rate`` -> ``train/lr``, ``eval_x`` -> ``val/x``, summary -> ``run/x``;
    ``system/x`` (resources, throughput) unchanged."""
    if key == "epoch" or key.startswith("system/"):
        return key
    if key == "learning_rate":
        return "train/lr"
    if key.startswith("eval_"):
        return "val/" + key.removeprefix("eval_")
    if key in SUMMARY or key == "train_loss":  # train_loss: the mean over the run
        return "run/" + key.removeprefix("train_")
    return "train/" + key


class PathotrainLogging(TrainerCallback):
    """Forwards the trainer's logs to pathotrain's loggers (``MLflowLogger``,
    ``CSVLogger``) under pathotrain names, from the main process only, with
    PyTorch's peak GPU memory since the previous log (``system/gpu_peak_*_gb``);
    at the end, logs the best checkpoint as :class:`~pathotrain.engine.ModelCheckpoint`
    does."""

    def __init__(self, loggers: list[Logger], monitor: str) -> None:
        self.loggers, self.monitor = loggers, monitor

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        metrics = {
            pathotrain_name(k): float(v) for k, v in logs.items() if isinstance(v, int | float)
        }
        if torch.cuda.is_available():
            metrics |= peak_gpu_memory(torch.device("cuda", torch.cuda.current_device()))
        for lg in self.loggers:
            lg.log_metrics(metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        for lg in self.loggers:
            if state.best_metric is not None:
                lg.log_metrics({f"best/{self.monitor}": state.best_metric}, step=state.global_step)
                lg.log_hyperparams({"best_checkpoint": str(state.best_model_checkpoint)})
            lg.finalize("success")


class Provenance(TrainerCallback):
    """Writes ``provenance.json`` (what the adapter was trained on) into every
    ``checkpoint-N/`` folder the trainer saves."""

    def __init__(self, provenance: dict) -> None:
        self.provenance = provenance

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            folder = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            (folder / "provenance.json").write_text(json.dumps(self.provenance, indent=2))
