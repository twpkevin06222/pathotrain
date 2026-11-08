import os

from patchology.log import configure_logging

# Every MLflow run in a process that imports pathotrain records system metrics (CPU, RAM,
# GPU, disk, network); MLflow's own default is off. Export the variable as 0 to turn it off.
os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "1")

# INFO by default; export PATHOTRAIN_LOG_LEVEL=DEBUG (or ALL) for more, see patchology.log.
configure_logging("pathotrain")
