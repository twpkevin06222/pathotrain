"""Experiment settings as one YAML file, saved next to the run's outputs."""

from pathlib import Path
from typing import Any

import yaml
from lightning_fabric.utilities.rank_zero import rank_zero_only


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


@rank_zero_only
def dump_config(cfg: dict[str, Any], output_dir: str | Path) -> Path:
    """Write ``cfg`` to ``<output_dir>/config.yaml`` and return the path.

    Re-running the same config (e.g. to resume) is fine; a different config
    for an existing run raises instead of overwriting the record of that run.
    """
    path = Path(output_dir) / "config.yaml"
    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"{path} holds a different config; use a new output_dir")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def flatten(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """``{"loss": {"gamma": 2}}`` -> ``{"loss.gamma": 2}``, for MLflow params."""
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            out[key] = v
    return out
