from types import SimpleNamespace

import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MeanMetric
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_convert

from pathotrain.detection.models import Detector
from pathotrain.engine.schedules import warmup_cosine
from pathotrain.engine.task import Task

Targets = list[dict[str, Tensor]]


def _clean(targets: Targets) -> Targets:
    # torch.tensor([]) is float32, so an image without objects easily gets float
    # labels or (0,) boxes; the loss and mAP need int64 labels and (N, 4) boxes.
    return [
        {"boxes": t["boxes"].float().reshape(-1, 4), "labels": t["labels"].long()} for t in targets
    ]


def collate(batch: list[tuple[Tensor, dict[str, Tensor]]]) -> tuple[Tensor, Targets]:
    """``collate_fn`` for a DataLoader of ``(image, target)`` samples: stacks the
    images and keeps the targets (different box counts per image) as a list."""
    images, targets = zip(*batch, strict=True)
    return torch.stack(images), list(targets)


class DetectionTask(Task):
    """Object detection with an RF-DETR :class:`Detector`.

    A batch is ``(images, targets)``: ``images`` a ``(B, 3, H, W)`` float tensor,
    ImageNet-normalised, with ``H`` and ``W`` divisible by 32; ``targets`` one dict
    per image with ``"boxes"``, ``(N, 4)`` float ``x1, y1, x2, y2`` in pixels, and
    ``"labels"``, ``(N,)`` int64 in ``0 .. num_classes - 1`` (``N`` may be 0).
    :func:`collate` builds it from ``(image, target)`` samples.

    Logs ``train/loss`` and its classification, L1 and GIoU parts, and per
    epoch ``val/loss``, COCO ``val/map`` (IoU 0.5:0.95), ``val/map_50``,
    ``val/map_75``, ``val/mar_100`` and ``val/ap_<class>``. Pass
    ``monitor="val/map"`` to ``ModelCheckpoint`` and ``EarlyStopping``.

    The optimizer follows RF-DETR's recipe: AdamW with ``lr`` for the heads,
    ``lr * lr_component_decay`` for the decoder and ``lr_encoder`` for the
    DINOv2 backbone, decayed by ``lr_vit_layer_decay`` per layer towards the
    input (no weight decay on norms, biases and embeddings). RF-DETR also clips
    gradients at 0.1: ``TrainerConfig(grad_clip=0.1)``.
    """

    def __init__(
        self,
        detector: Detector,
        lr: float = 1e-4,
        lr_encoder: float = 1.5e-4,
        lr_vit_layer_decay: float = 0.8,
        lr_component_decay: float = 0.7,
        weight_decay: float = 1e-4,
        warmup_epochs: float = 0,
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.model = detector.model
        self.criterion = detector.criterion
        self.postprocess = detector.postprocess
        self.num_classes = detector.config.num_classes
        self.out_feature_indexes = detector.config.out_feature_indexes
        self.lr, self.lr_encoder, self.weight_decay = lr, lr_encoder, weight_decay
        self.lr_vit_layer_decay, self.lr_component_decay = lr_vit_layer_decay, lr_component_decay
        self.warmup_epochs = warmup_epochs
        self.class_names = class_names or [str(i) for i in range(self.num_classes)]
        if len(self.class_names) != self.num_classes:
            raise ValueError(f"{len(self.class_names)} class names for {self.num_classes} classes")
        self.map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
        self.val_loss = MeanMetric()

    def _loss(self, outputs: dict, images: Tensor, targets: Targets) -> tuple[Tensor, dict]:
        h, w = images.shape[-2:]
        scale = images.new_tensor([w, h, w, h])
        # RF-DETR's loss takes boxes as cxcywh normalised to the image size.
        normalised = [
            {"labels": t["labels"], "boxes": box_convert(t["boxes"], "xyxy", "cxcywh") / scale}
            for t in targets
        ]
        # The loss runs in fp32 even under mixed precision (focal loss, GIoU).
        outputs = apply_to_collection(outputs, Tensor, lambda x: x.float())
        losses = self.criterion(outputs, normalised)
        weights = self.criterion.weight_dict
        loss = sum(losses[k] * weights[k] for k in losses if k in weights)
        parts = {k: losses[k].detach() for k in ("loss_ce", "loss_bbox", "loss_giou")}
        return loss, parts

    def training_step(self, batch: tuple[Tensor, Targets]) -> tuple[Tensor, dict[str, Tensor]]:
        images, targets = batch[0], _clean(batch[1])
        loss, parts = self._loss(self.model(images, targets), images, targets)
        return loss, {"loss": loss.detach(), **parts}

    def validation_step(self, batch: tuple[Tensor, Targets]) -> None:
        images, targets = batch[0], _clean(batch[1])
        outputs = self.model(images)
        self.val_loss.update(self._loss(outputs, images, targets)[0], len(targets))
        sizes = images.new_tensor([images.shape[-2:]] * len(targets), dtype=torch.long)
        # The head has one spare logit past the last class (RF-DETR's convention); drop it.
        preds = [
            {k: v[p["labels"] < self.num_classes] for k, v in p.items()}
            for p in self.postprocess(
                apply_to_collection(outputs, Tensor, lambda x: x.float()), sizes
            )
        ]
        self.map.update(preds, targets)

    def compute_metrics(self) -> dict[str, float]:
        result = self.map.compute()
        self.map.reset()
        metrics = {f"val/{k}": result[k].item() for k in ("map", "map_50", "map_75", "mar_100")}
        # A class absent from targets has AP -1 in torchmetrics; report it as NaN.
        classes = result["classes"].reshape(-1).tolist()
        for c, ap in zip(classes, result["map_per_class"].reshape(-1).tolist(), strict=True):
            if 0 <= c < self.num_classes:
                metrics[f"val/ap_{self.class_names[c]}"] = ap if ap >= 0 else float("nan")
        metrics["val/loss"] = self.val_loss.compute().item()
        self.val_loss.reset()
        return metrics

    def configure_optimizers(
        self, total_steps: int, steps_per_epoch: int
    ) -> tuple[Optimizer, LRScheduler]:
        args = SimpleNamespace(
            lr_encoder=self.lr_encoder,
            lr_vit_layer_decay=self.lr_vit_layer_decay,
            lr_component_decay=self.lr_component_decay,
            weight_decay=self.weight_decay,
            out_feature_indexes=self.out_feature_indexes,
        )
        # One (lr, weight_decay) per backbone parameter; merged so AdamW gets a few groups.
        backbone = self.model.backbone[0].get_named_param_lr_pairs(args, prefix="backbone.0")
        groups: dict[tuple[float, float], list] = {}
        for g in backbone.values():
            groups.setdefault((g["lr"], g["weight_decay"]), []).append(g["params"])
        rest = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        decoder = [p for n, p in rest if n not in backbone and "transformer.decoder" in n]
        heads = [p for n, p in rest if n not in backbone and "transformer.decoder" not in n]
        param_groups = [
            {"params": heads, "lr": self.lr},
            {"params": decoder, "lr": self.lr * self.lr_component_decay},
            *({"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()),
        ]
        optimizer = AdamW(param_groups, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer, warmup_cosine(optimizer, total_steps, steps_per_epoch, self.warmup_epochs)
