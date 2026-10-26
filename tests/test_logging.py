import os
import subprocess
import sys

# Log from inside each package's namespace (loguru names records by the calling module).
SCRIPT = """
import sys
from loguru import logger
for name in sys.argv[1].split(","):
    __import__(name)
for package in ("pathotrain", "patchology"):
    for lv in ("TRACE", "DEBUG", "INFO"):
        exec("logger.log(lv, msg)", {"logger": logger, "__name__": package + ".m",
             "lv": lv, "msg": f"{package}:{lv}"})
"""


def run(env_levels, imports):
    env = {k: v for k, v in os.environ.items() if not k.endswith("_LOG_LEVEL")}
    env.update(env_levels)
    return subprocess.run(
        [sys.executable, "-c", SCRIPT, imports],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    ).stderr


def shown(err, package):
    return [lv for lv in ("TRACE", "DEBUG", "INFO") if f"{package}:{lv}" in err]


def test_default_is_info():
    err = run({}, "pathotrain")
    assert shown(err, "pathotrain") == shown(err, "patchology") == ["INFO"]


def test_each_variable_controls_its_own_package_in_any_import_order():
    env = {"PATHOTRAIN_LOG_LEVEL": "ALL", "PATCHOLOGY_LOG_LEVEL": "DEBUG"}
    for imports in ("pathotrain", "patchology,pathotrain", "pathotrain.dataloader"):
        err = run(env, imports)
        assert shown(err, "pathotrain") == ["TRACE", "DEBUG", "INFO"], imports
        assert shown(err, "patchology") == ["DEBUG", "INFO"], imports


def test_unknown_level_fails_at_import():
    assert "does not exist" in run({"PATHOTRAIN_LOG_LEVEL": "LOUD"}, "pathotrain")
