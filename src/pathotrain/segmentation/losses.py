import segmentation_models_pytorch as smp
import torch
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
