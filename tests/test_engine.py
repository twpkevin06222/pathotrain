import csv
import itertools
from types import SimpleNamespace

import mlflow
import optuna
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pathotrain.engine import (
    CSVLogger,
    EarlyStopping,
    MLflowLogger,
    ModelCheckpoint,
    Trainer,
    TrainerConfig,
    dump_config,
    flatten,
    load_config,
    sweep,
)
from pathotrain.segmentation import SegmentationTask, build_model

CPU = {"accelerator": "cpu", "devices": 1, "precision": "32-true", "log_every": 1}


def _loader():
    # Learnable toy task: class 1 wherever the red channel is bright, 255 border ignored.
    # 8x8 px blocks: the U-Net's first stage downsamples, so pixel noise is not learnable.
    image = torch.rand(16, 3, 4, 4).repeat_interleave(8, 2).repeat_interleave(8, 3)
    mask = (image[:, 0] > 0.5).long()
    mask[:, 0] = 255
    return DataLoader(TensorDataset(image, mask), batch_size=4)


def _task(lr=1e-2):
    model = build_model(
        "smp:unet",
        2,
        encoder_name="resnet18",
        encoder_weights=None,
        encoder_depth=3,
        decoder_channels=(32, 16, 8),
    )
    return SegmentationTask(model, num_classes=2, lr=lr, class_names=["background", "tumor"])


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MLflow stores artifacts in ./mlruns
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    return mlflow.MlflowClient(uri)


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_fit_logs_checkpoints_and_resumes(tmp_path, tracking):
    torch.manual_seed(0)
    checkpoint = ModelCheckpoint(tmp_path, log_artifact=True)
    trainer = Trainer(
        TrainerConfig(max_epochs=3, accumulate=2, grad_clip=1.0, **CPU),
        callbacks=[checkpoint, EarlyStopping(patience=5)],
        loggers=[MLflowLogger(run_name="fit")],
    )
    metrics = trainer.fit(_task(), _loader(), _loader(), hparams={"arch": "smp:unet"})

    assert set(metrics) == {"val/iou_background", "val/iou_tumor", "val/miou", "val/loss"}
    assert trainer.step == 3 * 2  # 4 batches, 2 per optimizer step

    best = torch.load(tmp_path / "best.ckpt", weights_only=False)
    last = torch.load(tmp_path / "last.ckpt", weights_only=False)
    assert "optimizer" not in best and best["value"] == checkpoint.best
    assert best["epoch"] == checkpoint.best_epoch and best["monitor"] == "val/miou"
    assert {"optimizer", "callbacks"} <= last.keys()
    _task().model.load_state_dict(best["model"])  # plain keys, loads without Fabric

    (run,) = tracking.search_runs([tracking.get_experiment_by_name("pathotrain").experiment_id])
    assert run.info.status == "FINISHED"
    assert run.data.params["arch"] == "smp:unet" and "val/miou" in run.data.metrics
    assert "val/loss" in run.data.metrics
    assert run.data.metrics["best/val/miou"] == checkpoint.best
    assert run.data.params["best_checkpoint"] == str(tmp_path / "best.ckpt")
    assert [a.path for a in tracking.list_artifacts(run.info.run_id)] == ["best.ckpt"]

    # Resuming restores the best value, so the first new epoch must beat it to replace best.ckpt.
    restored = ModelCheckpoint(tmp_path)
    resumed = Trainer(TrainerConfig(max_epochs=3, **CPU), callbacks=[restored, EarlyStopping()])
    resumed.fit(_task(), _loader(), _loader(), ckpt_path=tmp_path / "last.ckpt")
    assert (restored.best, restored.best_epoch) == (checkpoint.best, checkpoint.best_epoch)
    assert resumed.callbacks[1].state_dict() == trainer.callbacks[1].state_dict()

    longer = Trainer(TrainerConfig(max_epochs=4, **CPU))
    longer.fit(_task(), _loader(), _loader(), ckpt_path=tmp_path / "last.ckpt")
    assert (longer.epoch, longer.step) == (4, 3 * 2 + 4)
    # The lr follows the 4-epoch schedule from the resumed step, not the old 3-epoch one.
    fresh, schedule = _task().configure_optimizers(longer.step, 4)
    for _ in range(longer.step):
        fresh.step()
        schedule.step()
    assert longer.optimizer.param_groups[0]["lr"] == pytest.approx(fresh.param_groups[0]["lr"])


@pytest.mark.parametrize(("interval", "rows"), [("epoch", 2), ("step", 2 * 4 + 2)])
def test_csv_logger_writes_epoch_or_step_rows(tmp_path, interval, rows):
    csv_logger = CSVLogger(tmp_path)
    cfg = TrainerConfig(max_epochs=2, **{**CPU, "log_interval": interval})
    Trainer(cfg, callbacks=[ModelCheckpoint(tmp_path / "ckpt")], loggers=[csv_logger]).fit(
        _task(), _loader(), _loader()
    )
    metrics = _rows(f"{csv_logger.log_dir}/metrics.csv")
    assert len(metrics) == rows + 1  # plus the best/ row at the end
    epoch_rows = [r for r in metrics if r["val/miou"]]
    step_rows = [r for r in metrics if not r["val/miou"] and not r["best/val/miou"]]
    assert [r["epoch"] for r in epoch_rows] == ["1", "2"]
    assert all(r["val/loss"] for r in epoch_rows)
    assert len(step_rows) == rows - 2
    # Train loss and lr are logged once: per epoch, or per step.
    train_rows = epoch_rows if interval == "epoch" else step_rows
    assert all(r["train/loss"] and r["train/lr"] for r in train_rows)
    assert not any(r["train/loss"] for r in metrics if r not in train_rows)
    hparams = {r["name"]: r["value"] for r in _rows(f"{csv_logger.log_dir}/hparams.csv")}
    assert hparams["log_interval"] == interval and hparams["best_checkpoint"].endswith("best.ckpt")


def test_training_learns_toy_task():
    torch.manual_seed(0)
    metrics = Trainer(TrainerConfig(max_epochs=10, **CPU)).fit(_task(), _loader(), _loader())
    assert metrics["val/miou"] > 0.7


def test_early_stopping_waits_patience_validations():
    trainer = SimpleNamespace(should_stop=False)
    stop = EarlyStopping(monitor="val/miou", patience=2)
    for miou, expected in [(0.5, False), (0.4, False), (0.6, False), (0.6, False), (0.1, True)]:
        stop.on_validation_end(trainer, {"val/miou": miou})
        assert trainer.should_stop is expected


def test_bad_log_interval_rejected():
    with pytest.raises(ValueError, match="log_interval"):
        TrainerConfig(log_interval="batch")


def _objective(trial, callbacks, loggers):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    trainer = Trainer(TrainerConfig(max_epochs=2, **CPU), callbacks=callbacks, loggers=loggers)
    return trainer.fit(_task(lr), _loader(), _loader(), hparams=trial.params)["val/miou"]


def test_sweep_runs_trials_as_child_runs_and_prunes(tmp_path, tracking):
    done = sweep(
        _objective, 2, "toy", pruner=optuna.pruners.NopPruner(), csv_path=tmp_path / "trials.csv"
    )
    # mIoU never reaches 2, so this pruner stops every trial at its first report.
    pruned = sweep(_objective, 1, "toy-pruned", pruner=optuna.pruners.ThresholdPruner(lower=2.0))
    assert [t.state.name for t in done.trials] == ["COMPLETE", "COMPLETE"]
    assert [t.state.name for t in pruned.trials] == ["PRUNED"]
    assert not list(tmp_path.rglob("*.ckpt"))  # no weights by default

    trials = _rows(tmp_path / "trials.csv")
    assert [r["number"] for r in trials] == ["0", "1"]
    assert [float(r["lr"]) for r in trials] == [t.params["lr"] for t in done.trials]
    assert [float(r["value"]) for r in trials] == [t.value for t in done.trials]

    exp = tracking.get_experiment_by_name("pathotrain").experiment_id
    runs = tracking.search_runs([exp])
    parents = {
        r.info.run_id: r.info.run_name for r in runs if "mlflow.parentRunId" not in r.data.tags
    }
    children = sorted(
        (parents[r.data.tags["mlflow.parentRunId"]], r.info.status)
        for r in runs
        if "mlflow.parentRunId" in r.data.tags
    )
    assert children == [("toy", "FINISHED"), ("toy", "FINISHED"), ("toy-pruned", "KILLED")]
    assert all(r.info.status == "FINISHED" for r in runs if r.info.run_id in parents)


def test_sweep_checkpoints_each_trial_when_asked(tmp_path, tracking):
    sweep(_objective, 2, "toy", checkpoint_dir=tmp_path / "ckpt")
    assert sorted(p.parent.name for p in (tmp_path / "ckpt").rglob("best.ckpt")) == [
        "trial-0",
        "trial-1",
    ]


def test_config_round_trips_flattens_and_guards_existing_runs(tmp_path):
    cfg = {"name": "exp", "loss": {"name": "FocalLoss", "gamma": 2.0}, "classes": {"a b": 1}}
    path = dump_config(cfg, tmp_path / "run")
    assert load_config(path) == cfg
    assert dump_config(cfg, tmp_path / "run") == path  # same config again, e.g. resume
    with pytest.raises(FileExistsError):
        dump_config({**cfg, "name": "other"}, tmp_path / "run")
    assert flatten(cfg) == {
        "name": "exp",
        "loss.name": "FocalLoss",
        "loss.gamma": 2.0,
        "classes.a b": 1,
    }


@pytest.mark.parametrize("warmup_epochs", [0, 0.5, 2])
def test_warmup_ramps_to_lr_then_cosine_decays(warmup_epochs):
    task = SegmentationTask(torch.nn.Conv2d(3, 2, 1), 2, lr=1.0, warmup_epochs=warmup_epochs)
    optimizer, scheduler = task.configure_optimizers(total_steps=20, steps_per_epoch=2)
    lrs = []
    for _ in range(20):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    warmup = round(warmup_epochs * 2)
    expected_warmup = [(i + 1) / warmup for i in range(warmup)]
    assert lrs[:warmup] == pytest.approx(expected_warmup)
    decay = lrs[warmup:]
    assert decay[0] == pytest.approx(1.0) and all(a > b for a, b in itertools.pairwise(decay))
    assert lrs[-1] < 0.05


def test_warmup_must_leave_steps_to_decay():
    task = SegmentationTask(torch.nn.Conv2d(3, 2, 1), 2, warmup_epochs=10)
    with pytest.raises(ValueError, match="warmup"):
        task.configure_optimizers(total_steps=20, steps_per_epoch=2)
