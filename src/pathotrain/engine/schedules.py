from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR


def warmup_cosine(
    optimizer: Optimizer, total_steps: int, steps_per_epoch: int, warmup_epochs: float = 0
) -> LRScheduler:
    """Linear warmup over ``warmup_epochs`` (fractions work), then cosine decay to 0,
    stepped once per optimizer step. Scales every parameter group's own lr."""
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
    warmup = round(warmup_epochs * steps_per_epoch)
    if warmup >= total_steps:
        raise ValueError(f"{warmup} warmup steps leave none of the {total_steps} to decay")
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup)
    if warmup == 0:
        return cosine
    # The first step runs at lr / warmup, step `warmup` at the full lr.
    linear = LinearLR(optimizer, start_factor=1 / warmup, total_iters=warmup - 1)
    return SequentialLR(optimizer, [linear, cosine], milestones=[warmup])
