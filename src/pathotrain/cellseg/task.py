import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MeanMetric, Metric

from pathotrain.cellseg.instances import instance_types, instances_from_outputs, match
from pathotrain.engine.schedules import warmup_cosine
from pathotrain.engine.task import Task
from pathotrain.segmentation.losses import CEDiceLoss
from pathotrain.segmentation.task import input_multiple

# HoVer-Net's loss weights: NP and NT cross-entropy + Dice, HV MSE and gradient MSE.
LOSS_WEIGHTS = {"np": 1.0, "hv_mse": 1.0, "hv_msge": 1.0, "nt": 1.0}


def _sobel(size: int = 5) -> Tensor:
    """HoVer-Net's ``(2, 1, size, size)`` gradient kernels along x and along y."""
    r = torch.arange(size) - size // 2
    y, x = torch.meshgrid(r, r, indexing="ij")
    norm = x * x + y * y + 1e-15
    return torch.stack([x / norm, y / norm]).unsqueeze(1).float()


def hv_losses(pred: Tensor, target: Tensor, nuclei: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """HoVer-Net's HV losses on ``(B, 2, H, W)`` maps: MSE over ``valid`` pixels, and MSE of
    the maps' gradients (the horizontal map along x, the vertical along y) over ``nuclei``."""
    v = valid.unsqueeze(1).expand_as(pred)
    mse = ((pred - target) ** 2)[v].mean() if v.any() else pred.sum() * 0
    kernel = _sobel().to(pred)

    def grad(hv: Tensor) -> Tensor:
        return F.conv2d(hv, kernel, padding=2, groups=2)

    focus = nuclei.unsqueeze(1).expand_as(pred)
    msge = ((grad(pred) - grad(target)) ** 2)[focus].mean() if focus.any() else pred.sum() * 0
    return mse, msge


class PanopticQuality(Metric):
    """Panoptic quality of nucleus instances, summed over images (and ranks).

    Row 0 of ``stats`` counts all nuclei as one class, row ``c`` nuclei of type ``c``;
    columns are true positives, false positives, false negatives and the IoU sum of
    matches. ``bpq`` is PanNuke's binary PQ, the mean of each image's; ``mpq`` CoNIC's
    multi-class PQ+, each type's PQ over the whole set, averaged over types present.
    """

    full_state_update = False

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.add_state("stats", torch.zeros(num_classes, 4, dtype=torch.float64), "sum")
        self.add_state("bpq_sum", torch.zeros((), dtype=torch.float64), "sum")
        self.add_state("images", torch.zeros((), dtype=torch.float64), "sum")

    def update(self, true, true_types, pred, pred_types) -> None:  # numpy arrays
        rows = [match(true, pred)]
        t_type, p_type = true_types[true], pred_types[pred]
        for c in range(1, self.num_classes):
            rows.append(match(true * (t_type == c), pred * (p_type == c)))
        rows = torch.tensor(rows, dtype=torch.float64, device=self.stats.device)
        self.stats += rows
        tp, fp, fn, iou = rows[0]
        if tp + fp + fn:  # an image without nuclei, true or predicted, has no PQ
            self.bpq_sum += iou / (tp + 0.5 * fp + 0.5 * fn)
            self.images += 1

    def compute(self) -> dict[str, Tensor]:
        tp, fp, fn, iou = self.stats.T
        pq = iou / (tp + 0.5 * fp + 0.5 * fn)  # NaN for a class with no nuclei
        return {
            "bpq": self.bpq_sum / self.images,
            "mpq": torch.nanmean(pq[1:]),
            "pq": pq,
            "dq": tp[0] / (tp[0] + 0.5 * fp[0] + 0.5 * fn[0]),
            "sq": iou[0] / tp[0],
        }


class CellInstanceTask(Task):
    """Nucleus instance segmentation and typing, HoVer-Net style, on the
    ``(image, {"inst", "type", "hv"})`` batches of :class:`~pathotrain.cellseg.data.CellDataset`.

    ``model`` returns the ``np``, ``hv`` and ``nt`` branches (a ``cellseg`` model with
    ``instance=True``). The loss is HoVer-Net's: cross-entropy + Dice on NP and NT, MSE
    and gradient MSE on HV, weighted by ``loss_weights`` (default :data:`LOSS_WEIGHTS`).

    Validation runs :func:`~pathotrain.cellseg.instances.postprocess` (``postprocess``
    holds its options) on each tile and logs ``val/loss``, ``val/bpq``, ``val/mpq``,
    ``val/pq_<name>`` per type, and the binary ``val/dq`` and ``val/sq``.

    Args:
        num_classes: Nucleus types plus background (type 0).
        class_names: Names for the per-type keys, default the type indices.
        lr, weight_decay, warmup_epochs: AdamW, linear warmup, then cosine decay to 0.
        ignore_index: Type-mask value left out of the losses.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        class_names: list[str] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: float = 0,
        ignore_index: int = 255,
        loss_weights: dict[str, float] | None = None,
        postprocess: dict | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        if len(self.class_names) != num_classes:
            raise ValueError(f"{len(self.class_names)} class names for {num_classes} classes")
        unknown = set(loss_weights or {}) - set(LOSS_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown loss weights {sorted(unknown)}, expected {LOSS_WEIGHTS}")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        self.loss_weights = {**LOSS_WEIGHTS, **(loss_weights or {})}
        self.lr, self.weight_decay, self.warmup_epochs = lr, weight_decay, warmup_epochs
        self.ignore_index = ignore_index
        self.postprocess = postprocess or {}
        self.ce_dice = CEDiceLoss(ignore_index)
        self.pq = PanopticQuality(num_classes)
        self.val_loss = MeanMetric()

    def losses(self, out: dict[str, Tensor], target: dict[str, Tensor]) -> dict[str, Tensor]:
        types, inst = target["type"], target["inst"]
        valid = types != self.ignore_index
        nuclei = (inst > 0).long().masked_fill(~valid, self.ignore_index)
        mse, msge = hv_losses(out["hv"].float(), target["hv"], nuclei == 1, valid)
        return {
            "np": self.ce_dice(out["np"].float(), nuclei),
            "hv_mse": mse,
            "hv_msge": msge,
            "nt": self.ce_dice(out["nt"].float(), types),
        }

    def _total(self, losses: dict[str, Tensor]) -> Tensor:
        return sum(self.loss_weights[k] * v for k, v in losses.items())

    def training_step(self, batch) -> tuple[Tensor, dict[str, Tensor]]:
        image, target = batch
        losses = self.losses(self.model(image), target)
        loss = self._total(losses)
        return loss, {"loss": loss.detach(), **{k: v.detach() for k, v in losses.items()}}

    def validation_step(self, batch) -> None:
        image, target = batch
        height, width = image.shape[-2:]
        multiple = input_multiple(self.model)
        out = self.model(F.pad(image, (0, -width % multiple, 0, -height % multiple)))
        out = {k: v[..., :height, :width].float() for k, v in out.items()}
        self.val_loss.update(self._total(self.losses(out, target)), len(image))
        outputs = [out[k].cpu().numpy() for k in ("np", "hv", "nt")]
        true_inst, true_type = target["inst"].cpu().numpy(), target["type"].cpu().numpy()
        for i in range(len(image)):
            pred, pred_types = instances_from_outputs(*(o[i] for o in outputs), **self.postprocess)
            self.pq.update(
                true_inst[i],
                instance_types(true_inst[i], true_type[i], self.num_classes),
                pred,
                pred_types,
            )

    def compute_metrics(self) -> dict[str, float]:
        pq = self.pq.compute()
        self.pq.reset()
        metrics = {f"val/{k}": pq[k].item() for k in ("bpq", "mpq", "dq", "sq")}
        names = self.class_names[1:]
        metrics |= {f"val/pq_{n}": v.item() for n, v in zip(names, pq["pq"][1:], strict=True)}
        metrics["val/loss"] = self.val_loss.compute().item()
        self.val_loss.reset()
        return metrics

    def configure_optimizers(
        self, total_steps: int, steps_per_epoch: int
    ) -> tuple[Optimizer, LRScheduler]:
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer, warmup_cosine(optimizer, total_steps, steps_per_epoch, self.warmup_epochs)
