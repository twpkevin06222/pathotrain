import csv
from pathlib import Path
from typing import Any

import mlflow
from lightning_fabric import loggers
from lightning_fabric.loggers import Logger
from lightning_fabric.utilities.rank_zero import rank_zero_only

# Fabric finalize status -> MLflow run status.
_STATUS = {"success": "FINISHED", "pruned": "KILLED"}


class MLflowLogger(Logger):
    """Fabric logger writing to an MLflow run, from rank 0 only.

    The run starts on the first log call. If another run is active (e.g. the
    parent run of :func:`pathotrain.engine.sweep`), it becomes a child of it.
    The tracking server comes from ``tracking_uri`` or ``MLFLOW_TRACKING_URI``.
    """

    def __init__(
        self,
        experiment: str = "pathotrain",
        run_name: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        self.experiment = experiment
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.run_id: str | None = None

    @property
    def name(self) -> str:
        return self.experiment

    @property
    def version(self) -> str | None:
        return self.run_id

    def _start(self) -> None:
        if self.run_id is not None:
            return
        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)
        parent = mlflow.active_run()
        if parent is None:
            mlflow.set_experiment(self.experiment)
        run = mlflow.start_run(run_name=self.run_name, nested=parent is not None)
        self.run_id = run.info.run_id

    @rank_zero_only
    def log_hyperparams(self, params: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        self._start()
        mlflow.log_params(params)

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._start()
        mlflow.log_metrics(metrics, step=step)

    @rank_zero_only
    def log_artifact(self, path: str | Path) -> None:
        """Upload a file (e.g. ``best.ckpt``) to the run."""
        self._start()
        mlflow.log_artifact(str(path))

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self.run_id is None:
            return
        mlflow.end_run(_STATUS.get(status, "FAILED"))
        self.run_id = None


class CSVLogger(loggers.CSVLogger):
    """Fabric's CSV logger, plus hyperparameters.

    Metrics go to ``<root_dir>/<name>/version_<n>/metrics.csv`` (one row per
    logged step; choose epoch or step rows with ``TrainerConfig.log_interval``),
    hyperparameters to ``hparams.csv`` next to it.
    """

    def __init__(self, root_dir: str | Path = "logs", name: str = "pathotrain", **kwargs) -> None:
        super().__init__(root_dir, name, **kwargs)
        self._hparams: dict[str, Any] = {}

    @rank_zero_only
    def log_hyperparams(self, params: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        self._hparams.update(params)
        # Create the metrics writer first; it warns about a log dir that already has files.
        _ = self.experiment
        with (Path(self.log_dir) / "hparams.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "value"])
            writer.writerows(self._hparams.items())
