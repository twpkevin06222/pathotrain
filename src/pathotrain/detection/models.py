"""RF-DETR (Roboflow, https://github.com/roboflow/rf-detr) detectors for the engine.

Uses rfdetr's model, loss and post-processing directly, not its Lightning
trainer. Only the Apache-2.0 sizes are offered; the PML-licensed ones (Atto,
Femto, Pico, XL, 2XL) need ``rfdetr[plus]`` and a separate licence.
"""

import os
from pathlib import Path
from typing import NamedTuple

from torch import nn

try:
    from rfdetr.config import (
        ModelConfig,
        RFDETRLargeConfig,
        RFDETRMediumConfig,
        RFDETRNanoConfig,
        RFDETRSmallConfig,
        TrainConfig,
    )
    from rfdetr.models import (
        build_criterion_from_config,
        build_model_from_config,
        load_pretrain_weights,
    )
except ImportError as e:  # an optional extra, so segmentation users do not need it
    raise ImportError(
        "RF-DETR needs the detection extra: pip install 'pathotrain[detection]'"
    ) from e

RFDETR_SIZES = {
    "nano": RFDETRNanoConfig,  # 384 px
    "small": RFDETRSmallConfig,  # 512 px
    "medium": RFDETRMediumConfig,  # 576 px
    "large": RFDETRLargeConfig,  # 704 px
}


class Detector(NamedTuple):
    """An RF-DETR network with the loss and post-processing that go with it."""

    model: nn.Module  # images -> {"pred_logits", "pred_boxes" (normalised cxcywh), ...}
    criterion: nn.Module  # Hungarian matching; focal, L1 and GIoU losses (plus aux layers)
    postprocess: nn.Module  # outputs -> per-image boxes (xyxy pixels), scores, labels
    config: ModelConfig


def default_weights_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "rfdetr"


def build_detector(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    weights_dir: str | Path | None = None,
    **kwargs,
) -> Detector:
    """Build ``"rfdetr:<size>"`` (nano, small, medium, large) for ``num_classes``
    classes, labelled ``0 .. num_classes - 1``.

    Args:
        pretrained: Start from the COCO checkpoint (downloaded once to
            ``weights_dir``, default ``~/.cache/rfdetr``); the class head is
            resized to ``num_classes``. False starts from random weights.
        **kwargs: rfdetr ``ModelConfig`` fields (e.g. ``resolution``,
            ``num_queries``) or loss settings from its ``TrainConfig`` (e.g.
            ``cls_loss_coef``, ``bbox_loss_coef``, ``giou_loss_coef``).
    """
    library, _, size = name.partition(":")
    if library != "rfdetr" or size not in RFDETR_SIZES:
        raise ValueError(f"unknown detector {name!r}, expected rfdetr:{'|'.join(RFDETR_SIZES)}")
    model_cls = RFDETR_SIZES[size]
    model_kwargs = {k: v for k, v in kwargs.items() if k in model_cls.model_fields}
    train_kwargs = {k: v for k, v in kwargs.items() if k not in model_kwargs}
    unknown = set(train_kwargs) - set(TrainConfig.model_fields)
    if unknown:
        raise ValueError(f"unknown RF-DETR settings {sorted(unknown)}")
    weights = None
    if pretrained:
        filename = model_cls.model_fields["pretrain_weights"].default
        weights = str(Path(weights_dir or default_weights_dir()) / filename)
        Path(weights).parent.mkdir(parents=True, exist_ok=True)
    config = model_cls(num_classes=num_classes, pretrain_weights=weights, **model_kwargs)
    train_config = TrainConfig(dataset_dir=".", output_dir=".", **train_kwargs)
    model = build_model_from_config(config, train_config)
    if weights is not None:
        load_pretrain_weights(model, config)  # resizes the class head to num_classes
    criterion, postprocess = build_criterion_from_config(config, train_config)
    return Detector(model, criterion, postprocess, config)
