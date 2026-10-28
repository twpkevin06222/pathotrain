import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CEDiceLoss(nn.Module):
    """Cross-entropy plus ``dice_weight`` times soft Dice, both skipping ``ignore_index``.

    Takes ``(B, C, H, W)`` logits and a ``(B, H, W)`` int64 target. Pixels set
    to ``ignore_index`` (outside the ROI, or uncovered by geometric
    augmentation) add nothing to either term.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        dice_weight: float = 1.0,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        weight = None if class_weights is None else torch.tensor(class_weights)
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice = smp.losses.DiceLoss("multiclass", ignore_index=ignore_index)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        # Cross-entropy averages over zero pixels (NaN) when a batch is all ignore.
        if not (target != self.ignore_index).any():
            return logits.sum() * 0
        return self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


class FocalLoss(nn.Module):
    """Softmax focal loss (Lin et al., 2017), averaged over pixels not set to ``ignore_index``.

    ``-alpha_t * (1 - p_t) ** gamma * log(p_t)``: ``gamma`` down-weights pixels
    the model already gets right; ``gamma=0`` is (weighted) cross-entropy.
    ``class_weights`` are the per-class ``alpha``. Same shapes as :class:`CEDiceLoss`.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        gamma: float = 2.0,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.register_buffer(
            "alpha", None if class_weights is None else torch.tensor(class_weights)
        )

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0
        logp = F.log_softmax(logits, 1).gather(1, target.where(valid, 0).unsqueeze(1)).squeeze(1)
        loss = (-((1 - logp.exp()) ** self.gamma) * logp)[valid]
        if self.alpha is not None:
            loss = loss * self.alpha[target[valid]]
        return loss.mean()


class FocalDiceLoss(FocalLoss):
    """:class:`FocalLoss` plus ``dice_weight`` times soft Dice, both skipping ``ignore_index``."""

    def __init__(
        self,
        ignore_index: int = 255,
        gamma: float = 2.0,
        dice_weight: float = 1.0,
        class_weights: list[float] | None = None,
    ) -> None:
        super().__init__(ignore_index, gamma, class_weights)
        self.dice_weight = dice_weight
        self.dice = smp.losses.DiceLoss("multiclass", ignore_index=ignore_index)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if not (target != self.ignore_index).any():
            return logits.sum() * 0
        return super().forward(logits, target) + self.dice_weight * self.dice(logits, target)
