import csv
import os
from collections.abc import Callable
from pathlib import Path

import mlflow
import optuna

from pathotrain.engine.callbacks import ModelCheckpoint, OptunaPruning
from pathotrain.engine.logger import MLflowLogger


def sweep(
    objective: Callable[[optuna.Trial, list, list], float],
    n_trials: int,
    study_name: str,
    storage: str | None = None,
    direction: str = "maximize",
    monitor: str = "val/miou",
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    experiment: str = "pathotrain",
    checkpoint_dir: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> optuna.Study:
    """Run ``n_trials`` of ``objective`` under one MLflow parent run per study.

    ``objective(trial, callbacks, loggers)`` suggests hyperparameters, builds a
    :class:`Trainer` with the given ``callbacks`` and ``loggers`` and returns
    the value to optimise. Per trial the sweep provides:

    - :class:`OptunaPruning` on ``monitor``, so the study's ``pruner`` (default
      median) can stop trials that fall behind;
    - an :class:`MLflowLogger`, logged as a child run ``trial-<n>``;
    - a :class:`ModelCheckpoint` in ``checkpoint_dir/trial-<n>``, only when
      ``checkpoint_dir`` is set. By default no weights are written to disk.

    Args:
        storage: Optuna storage URL (e.g. ``"sqlite:///sweep.db"`` or Postgres).
            With a shared one, several processes, one per GPU, can run the same
            study in parallel.
        direction: ``"maximize"`` or ``"minimize"`` ``monitor``.
        sampler: Default TPE; ``optuna.samplers.GPSampler()`` for
            Gaussian-process Bayesian optimisation.
        csv_path: Rewrite this CSV after every trial, one row per trial of the
            study: number, state, value, duration and each hyperparameter.
    """
    mode = "max" if direction == "maximize" else "min"

    def run(trial: optuna.Trial) -> float:
        callbacks = [OptunaPruning(trial, monitor)]
        if checkpoint_dir is not None:
            callbacks.append(
                ModelCheckpoint(Path(checkpoint_dir) / f"trial-{trial.number}", monitor, mode)
            )
        return objective(trial, callbacks, [MLflowLogger(experiment, f"trial-{trial.number}")])

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    on_trial_end = [] if csv_path is None else [lambda s, _: write_trials_csv(s, csv_path)]
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=study_name):
        study.optimize(run, n_trials=n_trials, callbacks=on_trial_end)
        if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            mlflow.log_params({f"best/{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_value", study.best_value)
    return study


def write_trials_csv(study: optuna.Study, path: str | Path) -> None:
    """Write one row per trial: number, state, value, duration_s, then the
    hyperparameters (blank where a trial did not suggest one)."""
    trials = study.get_trials(deepcopy=False)
    params = sorted({k for t in trials for k in t.params})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["number", "state", "value", "duration_s", *params])
        for t in trials:
            duration = t.duration.total_seconds() if t.duration is not None else None
            writer.writerow(
                [t.number, t.state.name, t.value, duration, *(t.params.get(k) for k in params)]
            )
    # Atomic swap, so parallel workers sharing a study never leave a half-written file.
    os.replace(tmp, path)
