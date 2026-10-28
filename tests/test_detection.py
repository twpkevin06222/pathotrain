import math

import pytest
import torch
from torch.utils.data import DataLoader

pytest.importorskip("rfdetr")

from pathotrain.detection import DetectionTask, build_detector, collate
from pathotrain.engine import ModelCheckpoint, Trainer, TrainerConfig

CPU = {"accelerator": "cpu", "devices": 1, "precision": "32-true"}


def _samples(n=8, size=128):
    # A bright square per object on a dark tile; class 1 squares are larger.
    torch.manual_seed(0)
    samples = []
    for i in range(n):
        image = torch.zeros(3, size, size)
        boxes, labels = [], []
        for k in range(i % 3):  # 0, 1 or 2 objects, so one image per three has none
            side = 16 if k == 0 else 32
            x, y = 8 + 48 * k, 8 + 40 * k
            image[:, y : y + side, x : x + side] = 1
            boxes.append([x, y, x + side, y + side])
            labels.append(k)
        # Plain torch.tensor on purpose: an empty list gives float32 of shape (0,).
        target = {"boxes": torch.tensor(boxes), "labels": torch.tensor(labels)}
        samples.append((image, target))
    return samples


def _task():
    return DetectionTask(
        build_detector("rfdetr:nano", num_classes=2, pretrained=False),
        warmup_epochs=0.5,
        class_names=["small", "large"],
    )


def test_collate_keeps_per_image_targets():
    images, targets = collate(_samples(3))
    assert images.shape == (3, 3, 128, 128)
    assert [len(t["labels"]) for t in targets] == [0, 1, 2]


def test_fit_logs_detection_metrics_and_checkpoints(tmp_path):
    loader = DataLoader(_samples(), batch_size=4, collate_fn=collate)
    checkpoint = ModelCheckpoint(tmp_path, monitor="val/map")
    trainer = Trainer(TrainerConfig(max_epochs=2, grad_clip=0.1, **CPU), callbacks=[checkpoint])
    metrics = trainer.fit(_task(), loader, loader)

    assert {"val/map", "val/map_50", "val/map_75", "val/mar_100", "val/loss"} <= metrics.keys()
    assert {"val/ap_small", "val/ap_large"} <= metrics.keys()
    assert math.isfinite(metrics["val/loss"]) and 0 <= metrics["val/map"] <= 1
    assert (tmp_path / "best.ckpt").exists() and (tmp_path / "last.ckpt").exists()


def test_optimizer_covers_every_parameter_once_with_rfdetr_rates():
    task = _task()
    optimizer, _ = task.configure_optimizers(total_steps=10, steps_per_epoch=2)
    grouped = [id(p) for g in optimizer.param_groups for p in g["params"]]
    trainable = [id(p) for p in task.model.parameters() if p.requires_grad]
    assert sorted(grouped) == sorted(trainable)
    heads, decoder, *backbone = optimizer.param_groups
    assert heads["lr"] == pytest.approx(1e-4) and decoder["lr"] == pytest.approx(1e-4 * 0.7)
    # Backbone: lr_encoder * lr_component_decay**2, smaller still towards the input.
    assert all(0 < g["lr"] <= 1.5e-4 * 0.7**2 + 1e-12 for g in backbone)
    assert len({g["lr"] for g in backbone}) > 1
    assert any(g["weight_decay"] == 0 for g in backbone)  # norms, biases, embeddings


def test_build_detector_rejects_unknown_names_and_settings():
    with pytest.raises(ValueError, match="unknown detector"):
        build_detector("rfdetr:huge", 2, pretrained=False)
    with pytest.raises(ValueError, match="unknown RF-DETR settings"):
        build_detector("rfdetr:nano", 2, pretrained=False, not_a_setting=1)
