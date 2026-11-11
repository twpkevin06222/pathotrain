"""Fine-tuning SAM 3's point and box model on simulated interactive sessions."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision.ops import sigmoid_focal_loss

from pathotrain.engine.schedules import warmup_cosine
from pathotrain.engine.task import Task
from pathotrain.pathosam.eval import ClickMetrics, Tracker, nuclei, track_tile
from pathotrain.pathosam.model import PAD, InteractiveSam


def _boxes(masks: Tensor, jitter: float) -> Tensor:
    """``(N, 4)`` boxes ``(x0, y0, x1, y1)`` of ``(N, H, W)`` masks, each side moved by up
    to ``jitter`` times the box's width or height, uniformly (as ``prompts.box``)."""
    h, w = masks.shape[-2:]
    rows, cols = masks.any(2), masks.any(1)
    ar_h, ar_w = torch.arange(h, device=masks.device), torch.arange(w, device=masks.device)
    y0 = torch.where(rows, ar_h, h).amin(1)
    y1 = torch.where(rows, ar_h, -1).amax(1) + 1
    x0 = torch.where(cols, ar_w, w).amin(1)
    x1 = torch.where(cols, ar_w, -1).amax(1) + 1
    out = torch.stack([x0, y0, x1, y1], 1).float()
    size = torch.stack([x1 - x0, y1 - y0, x1 - x0, y1 - y0], 1).float()
    out = out + (torch.rand_like(out) * 2 - 1) * jitter * size
    return torch.stack(
        [
            out[:, 0].clamp(0, w),
            out[:, 1].clamp(0, h),
            out[:, 2].clamp(0, w),
            out[:, 3].clamp(0, h),
        ],
        1,
    )


def _sample(region: Tensor) -> tuple[Tensor, Tensor]:
    """A uniformly random pixel ``(x, y)`` of each of the ``(N, H, W)`` regions, and
    whether the region has one (an empty region gives ``(0, 0)``)."""
    flat = region.flatten(1).float()
    some = flat.sum(1) > 0
    idx = torch.multinomial(flat + (~some)[:, None].float(), 1)[:, 0]  # empty: any pixel
    w = region.shape[-1]
    return torch.stack([idx % w, idx // w], 1).float(), some


def _dice(logits: Tensor, true: Tensor) -> Tensor:
    """Soft Dice loss per mask over the last two dims."""
    p = logits.sigmoid().flatten(-2)
    t = true.flatten(-2)
    return 1 - (2 * (p * t).sum(-1) + 1) / (p.sum(-1) + t.sum(-1) + 1)


class InteractiveTask(Task):
    """Fine-tunes an :class:`InteractiveSam`'s prompt encoder and mask decoder, SAM-style.

    Each step takes up to ``objects`` nuclei per tile (of at least ``min_size`` px). Their
    first prompt is a box (each side jittered by up to ``box_jitter``) with probability
    ``box_share``, else one click sampled inside the nucleus. Then ``rounds - 1``
    corrective clicks follow, each sampled uniformly from the error of the previous
    prediction (positive where missed, negative where extra), with the previous mask's
    logits fed back. Every round adds to the loss: ``focal`` times focal loss plus ``dice``
    times Dice loss on the mask, and ``iou`` times the MSE between the predicted and the
    true IoU. A single first click predicts three masks: the mask loss trains only the best
    one, the IoU loss all three (inference picks the mask with the best predicted IoU).

    Validation runs :func:`pathotrain.pathosam.eval.track_tile` (a box without jitter, a
    centre click and ``val_clicks - 1`` corrective clicks) and logs ``val/iou_box_0.0``,
    ``val/iou_click<k>``, ``val/noc85``, ``val/noc90``, ``val/fail85``, ``val/fail90``.
    """

    checkpoint_trainable_only = True  # the frozen image encoder comes from SAM 3

    def __init__(
        self,
        model: InteractiveSam,
        objects: int = 16,
        rounds: int = 4,
        box_share: float = 0.5,
        box_jitter: float = 0.1,
        focal: float = 20.0,
        dice: float = 1.0,
        iou: float = 1.0,
        min_size: int = 10,
        val_clicks: int = 10,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: float = 0,
    ) -> None:
        super().__init__()
        if rounds < 1 or objects < 1:
            raise ValueError(f"rounds and objects must be >= 1, got {rounds} and {objects}")
        self.model = model
        self.objects, self.rounds, self.min_size = objects, rounds, min_size
        self.box_share, self.box_jitter = box_share, box_jitter
        self.weights = {"focal": focal, "dice": dice, "iou": iou}
        self.lr, self.weight_decay, self.warmup_epochs = lr, weight_decay, warmup_epochs
        self.val_clicks = val_clicks
        self.clicks = ClickMetrics([0.0], val_clicks)

    def _objects(self, inst: Tensor) -> tuple[Tensor, Tensor]:
        """Masks ``(N, H, W)`` of up to ``objects`` random nuclei per tile, and the tile
        index of each."""
        masks, tiles = [], []
        for b, tile in enumerate(inst):
            ids = nuclei(tile.cpu().numpy(), self.min_size)
            if not ids:
                continue
            pick = torch.tensor(ids, device=tile.device)[torch.randperm(len(ids))[: self.objects]]
            masks.append(tile[None] == pick[:, None, None])
            tiles += [b] * len(pick)
        if not masks:
            return inst.new_zeros((0, *inst.shape[1:]), dtype=torch.bool), inst.new_zeros(0)
        return torch.cat(masks), torch.tensor(tiles, device=inst.device)

    def training_step(self, batch) -> tuple[Tensor, dict[str, Tensor]]:
        images, target = batch
        true, tiles = self._objects(target["inst"])
        if not len(true):  # no nucleus in the batch: a zero loss that keeps the graph
            zero = sum(p.sum() for p in self.model.parameters() if p.requires_grad) * 0
            return zero, {"loss": zero.detach()}
        size = tuple(true.shape[-2:])
        emb = [e[tiles] for e in self.model(images=images)]
        n = len(true)
        boxes = points = labels = masks = None
        if torch.rand(()) < self.box_share:
            boxes = _boxes(true, self.box_jitter)
        else:
            points = _sample(true)[0][:, None]
            labels = torch.ones(n, 1, dtype=torch.long, device=true.device)
        target_f = true.float()
        parts = {"focal": 0.0, "dice": 0.0, "iou": 0.0}
        for r in range(self.rounds):
            multimask = r == 0 and boxes is None  # one click is ambiguous
            logits, iou_pred = self.model(
                embeddings=emb,
                size=size,
                points=points,
                labels=labels,
                boxes=boxes,
                masks=masks,
                multimask=multimask,
            )
            up = F.interpolate(logits.float(), size, mode="bilinear")  # (N, M, H, W)
            t = target_f[:, None].expand_as(up)
            focal = sigmoid_focal_loss(up, t, reduction="none").mean((-2, -1))
            dice = _dice(up, t)
            best = (self.weights["focal"] * focal + self.weights["dice"] * dice).argmin(1)
            pick = torch.arange(n, device=true.device), best
            pred_all = up > 0
            inter = (pred_all & true[:, None]).sum((-2, -1))
            true_iou = inter / (pred_all | true[:, None]).sum((-2, -1)).clamp_min(1)
            pred = pred_all[pick]
            parts["focal"] += focal[pick].mean()
            parts["dice"] += dice[pick].mean()
            # Every mask's predicted IoU trains: inference picks the mask it scores best,
            # and untrained scores drift (a quarter to half of first clicks picked wrong).
            parts["iou"] += F.mse_loss(iou_pred.float(), true_iou.float())
            if r == self.rounds - 1:
                break
            masks = logits[pick][:, None].detach()
            missed, extra = true & ~pred, pred & ~true
            xy, some = _sample(missed | extra)
            x, y = xy[:, 0].long(), xy[:, 1].long()
            label = torch.where(some, true[torch.arange(n), y, x].long(), PAD)
            new_points = xy[:, None]
            points = new_points if points is None else torch.cat([points, new_points], 1)
            labels = label[:, None] if labels is None else torch.cat([labels, label[:, None]], 1)
        parts = {k: v / self.rounds for k, v in parts.items()}
        loss = sum(self.weights[k] * v for k, v in parts.items())
        return loss, {"loss": loss.detach(), **{k: v.detach() for k, v in parts.items()}}

    def validation_step(self, batch) -> None:
        images, target = batch
        tracker = Tracker(self.model)
        for image, inst in zip(images, target["inst"].cpu().numpy(), strict=True):
            ids = nuclei(inst, self.min_size)
            if ids:
                true = np.stack([inst == k for k in ids])
                rng = np.random.default_rng(0)  # unused: no jitter
                self.clicks.update(*track_tile(tracker, image, true, self.val_clicks, [0.0], rng))

    def compute_metrics(self) -> dict[str, float]:
        metrics = {f"val/{k}": v for k, v in self.clicks.compute().items() if k != "objects"}
        self.clicks.reset()
        return metrics

    def configure_optimizers(
        self, total_steps: int, steps_per_epoch: int
    ) -> tuple[Optimizer, LRScheduler]:
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer, warmup_cosine(optimizer, total_steps, steps_per_epoch, self.warmup_epochs)
