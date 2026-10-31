"""Patch-token masks for one view: bool ``(grid * grid,)`` tensors, True where the
student sees the mask token instead of the patch.

Pass one to :class:`pathotrain.dataloader.MultiCropTransform` as
``mask=functools.partial(block_mask, ratio=(0.1, 0.5))``; a partial (unlike a
lambda) pickles, so DataLoader workers can run it. Randomness comes from
``generator``, else torch's default generator, which a DataLoader seeds per worker.
"""

import math

import torch
from torch import Tensor


def _count(grid: int, ratio: float | tuple[float, float], generator) -> int:
    low, high = (ratio, ratio) if isinstance(ratio, (int, float)) else ratio
    if not 0 <= low <= high <= 1:
        raise ValueError(f"mask ratio must be in [0, 1], got {ratio}")
    r = low + (high - low) * torch.rand((), generator=generator).item()
    return round(r * grid * grid)


def random_mask(
    grid: int = 16,
    ratio: float | tuple[float, float] = 0.5,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Exactly ``ratio`` of the patches (drawn uniformly in a ``(low, high)`` range),
    chosen at random."""
    mask = torch.zeros(grid * grid, dtype=torch.bool)
    mask[torch.randperm(grid * grid, generator=generator)[: _count(grid, ratio, generator)]] = True
    return mask


def block_mask(
    grid: int = 16,
    ratio: float | tuple[float, float] = (0.1, 0.5),
    min_aspect: float = 0.3,
    min_block: int = 4,
    p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Rectangles of patches, as BEiT and iBOT mask (Bao et al. 2022; Zhou et al.
    2022), covering exactly ``ratio`` of the grid (drawn in a ``(low, high)``
    range). Each block holds ``min_block`` patches or more and has an aspect ratio
    in ``[min_aspect, 1 / min_aspect]``; a few random patches make up the count
    when no block fits. With probability ``1 - p``, nothing is masked (DINOv2
    masks half of the views, ``p=0.5``)."""
    if not 0 < min_aspect <= 1:
        raise ValueError(f"min_aspect must be in (0, 1], got {min_aspect}")
    mask = torch.zeros(grid, grid, dtype=torch.bool)
    if torch.rand((), generator=generator).item() >= p:
        return mask.flatten()
    target = _count(grid, ratio, generator)

    def uniform(low: float, high: float) -> float:
        return low + (high - low) * torch.rand((), generator=generator).item()

    for _ in range(100):  # attempts; small grids or high ratios may leave a remainder
        left = target - int(mask.sum())
        if left < min_block:
            break
        area = uniform(min_block, left)
        aspect = math.exp(uniform(math.log(min_aspect), -math.log(min_aspect)))
        h, w = round(math.sqrt(area * aspect)), round(math.sqrt(area / aspect))
        if not (0 < h <= grid and 0 < w <= grid):
            continue
        top = int(torch.randint(grid - h + 1, (), generator=generator))
        lft = int(torch.randint(grid - w + 1, (), generator=generator))
        block = mask[top : top + h, lft : lft + w]
        if int((~block).sum()) <= left:  # never overshoot the target
            block[:] = True
    mask = mask.flatten()
    left = target - int(mask.sum())
    if left > 0:
        free = (~mask).nonzero().squeeze(1)
        mask[free[torch.randperm(len(free), generator=generator)[:left]]] = True
    return mask
