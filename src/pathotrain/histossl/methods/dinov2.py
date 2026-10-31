# DINOHead, Sinkhorn-Knopp centring and the KoLeo loss are adapted from
# facebookresearch/dinov2 (Apache-2.0, Copyright (c) Meta Platforms, Inc. and
# affiliates): dinov2/layers/dino_head.py, dinov2/loss/dino_clstoken_loss.py,
# dinov2/loss/ibot_patch_loss.py and dinov2/loss/koleo_loss.py.
"""dinov2: continued DINOv2-style pretraining with an EMA teacher (Oquab et al. 2023).

The student sees every view (``G`` global crops, iBOT-masked, and ``L`` local
crops); the teacher, an exponential moving average of the student, sees the
unmasked global crops. Three losses, as DINOv2:

- DINO: the student's CLS prototype scores of every view against the teacher's of
  each other global view (cross-entropy, teacher targets Sinkhorn-Knopp centred).
- iBOT: the student's patch prototype scores at masked patches against the
  teacher's at the same patches.
- KoLeo: spreads the student's global CLS tokens over the sphere.

Midnight ships no DINO heads, so the heads start from scratch: the head's last
layer is frozen for ``freeze_last_layer_epochs`` and, optionally, the backbone
for ``head_warmup_epochs``, so random heads do not erode the pretrained weights.
Schedules follow DINOv2: teacher momentum cosine to 1, teacher temperature
linear warmup, weight decay cosine. The EMA covers the trained weights only
(with LoRA, the adapters and heads); frozen weights are the same in both.
"""

import copy
import functools
import itertools
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

from pathotrain.histossl.adaptation import apply_adaptation
from pathotrain.histossl.backbones import build_backbone, tokens
from pathotrain.histossl.masking import block_mask, random_mask
from pathotrain.histossl.methods.base import SSLTask


class DINOHead(nn.Module):
    """MLP to a ``bottleneck_dim`` L2-normalised code, then a weight-normalised
    linear layer to ``prototypes`` scores (its norm fixed at 1, so scores are
    cosines)."""

    def __init__(
        self,
        in_dim: int,
        prototypes: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (max(nlayers, 1) - 1) + [bottleneck_dim]
        layers: list[nn.Module] = []
        for i, (a, b) in enumerate(itertools.pairwise(dims)):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, prototypes, bias=False))
        g = self.last_layer.parametrizations.weight.original0
        g.data.fill_(1)
        g.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.last_layer(F.normalize(self.mlp(x), dim=-1, eps=1e-6))


def _all_reduce(x: Tensor) -> Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x)
    return x


@torch.no_grad()
def sinkhorn_knopp(scores: Tensor, temperature: float, iterations: int = 3) -> Tensor:
    """Teacher ``(n, K)`` scores to targets whose prototypes are used equally
    often over the batch (summed over ranks); rows sum to 1."""
    # DINOHead scores are cosines, so exp(score / temperature) stays in float32 range.
    q = torch.exp(scores.float() / temperature).T  # (K, n)
    n = torch.tensor(q.shape[1], device=q.device)
    n = int(_all_reduce(n).item())
    prototypes = q.shape[0]
    q /= _all_reduce(q.sum())
    for _ in range(iterations):
        q /= _all_reduce(q.sum(dim=1, keepdim=True))
        q /= prototypes
        q /= q.sum(dim=0, keepdim=True)
        q /= n
    return (q * n).T


def koleo(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Kozachenko-Leonenko entropy term (Sablayrolles et al. 2019): minus the
    mean log distance of each L2-normalised row to its nearest other row."""
    x = F.normalize(x.float(), dim=-1, eps=eps)
    dots = x @ x.T
    dots.fill_diagonal_(-1)
    nearest = dots.argmax(dim=1)
    return -torch.log(F.pairwise_distance(x, x[nearest], eps=eps) + eps).mean()


def _cosine(start: float, end: float, step: int, total: int) -> float:
    return end + (start - end) * (1 + math.cos(math.pi * min(step / max(total, 1), 1))) / 2


class _Teacher(nn.Module):
    """The EMA copies, and the optimizer steps taken (for the schedules on resume)."""

    def __init__(self, backbone: nn.Module, dino_head: DINOHead, ibot_head: DINOHead | None):
        super().__init__()
        self.backbone, self.dino_head, self.ibot_head = backbone, dino_head, ibot_head
        self.register_buffer("step", torch.zeros((), dtype=torch.long))


class DINOv2Model(nn.Module):
    """Student backbone and heads, and their EMA teacher.

    ``forward(global_views, local_views, masks)`` returns the student's DINO
    scores of every view (globals first), its iBOT scores at masked patches and
    its global CLS tokens, and the teacher's DINO and iBOT scores (no gradients).
    ``ibot_head=None`` shares the DINO head for patches.
    """

    def __init__(
        self, backbone: nn.Module, dino_head: DINOHead, ibot_head: DINOHead | None = None
    ) -> None:
        super().__init__()
        self.backbone, self.dino_head, self.ibot_head = backbone, dino_head, ibot_head
        # shortcut: the teacher copies the frozen weights too (4.4 GB fp32 for ViT-g); with
        # LoRA or last_n, share them with the student if memory gets tight.
        self.teacher = _Teacher(
            copy.deepcopy(backbone), copy.deepcopy(dino_head), copy.deepcopy(ibot_head)
        ).requires_grad_(False)
        # (student, teacher) pairs the EMA updates: the trained weights.
        teacher = dict(self.teacher.named_parameters())
        self.ema_pairs = [
            (p, teacher[name]) for name, p in self._student_parameters() if p.requires_grad
        ]

    def _student_parameters(self):
        for prefix in ("backbone", "dino_head", "ibot_head"):
            module = getattr(self, prefix)
            if module is not None:
                for name, p in module.named_parameters():
                    yield f"{prefix}.{name}", p

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        student = [s.detach() for s, _ in self.ema_pairs]
        teacher = [t for _, t in self.ema_pairs]
        torch._foreach_lerp_(teacher, student, 1 - momentum)

    def forward(
        self, global_views: Tensor, local_views: Tensor | None, masks: Tensor
    ) -> dict[str, Tensor]:
        t = self.teacher
        ibot_t = t.ibot_head or t.dino_head
        with torch.no_grad():
            t_cls, t_patches = tokens(t.backbone, global_views)
            teacher_dino = t.dino_head(t_cls)
            teacher_ibot = ibot_t(t_patches[masks])
        ibot = self.ibot_head or self.dino_head
        cls, patches = tokens(self.backbone, global_views, masks)
        student_dino = [self.dino_head(cls)]
        if local_views is not None:
            student_dino.append(self.dino_head(tokens(self.backbone, local_views)[0]))
        return {
            "student_dino": torch.cat(student_dino),
            "student_ibot": ibot(patches[masks]),
            "student_cls": cls,
            "teacher_dino": teacher_dino,
            "teacher_ibot": teacher_ibot,
        }


class DINOv2Task(SSLTask):
    """Train a :class:`DINOv2Model` (see the module docstring).

    Args:
        momentum: Teacher EMA momentum, cosine from ``[0]`` to ``[1]`` over the run.
        teacher_temp: Teacher temperature, linear from ``[0]`` to ``[1]`` over
            ``teacher_temp_warmup_epochs``, then constant.
        weight_decay: A float, or ``[start, end]`` cosine over the run.
        masking, ratio, min_aspect, p: iBOT masks of each global view (``p``: the
            share of views masked at all).
        Other arguments as for :class:`SSLTask`.
    """

    def __init__(
        self,
        model: DINOv2Model,
        momentum: tuple[float, float] = (0.994, 1.0),
        teacher_temp: tuple[float, float] = (0.04, 0.07),
        teacher_temp_warmup_epochs: float = 1,
        student_temp: float = 0.1,
        dino_weight: float = 1.0,
        ibot_weight: float = 1.0,
        koleo_weight: float = 0.1,
        freeze_last_layer_epochs: float = 1,
        head_warmup_epochs: float = 0,
        masking: str = "block",
        ratio: tuple[float, float] = (0.1, 0.5),
        min_aspect: float = 0.3,
        p: float = 0.5,
        weight_decay: float | tuple[float, float] = (0.04, 0.2),
        **kwargs,
    ) -> None:
        wd = tuple(weight_decay) if isinstance(weight_decay, (list, tuple)) else (weight_decay,) * 2
        super().__init__(model, weight_decay=wd[0], **kwargs)
        self.wd_schedule = wd
        self.momentum, self.teacher_temp = tuple(momentum), tuple(teacher_temp)
        self.teacher_temp_warmup_epochs, self.student_temp = (
            teacher_temp_warmup_epochs,
            student_temp,
        )
        self.weights = {"dino": dino_weight, "ibot": ibot_weight, "koleo": koleo_weight}
        self.freeze_last_layer_epochs = freeze_last_layer_epochs
        self.head_warmup_epochs = head_warmup_epochs
        if masking == "block":
            self.mask_fn = functools.partial(block_mask, ratio=ratio, min_aspect=min_aspect, p=p)
        else:
            self.mask_fn = functools.partial(random_mask, ratio=ratio)
        self.patch = model.backbone.config.patch_size
        self.teacher_module = model.teacher
        self.net = model  # the plain module: Fabric replaces self.model by its wrapper
        self.total_steps, self.steps_per_epoch, self.optimizer = 1, 1, None

    @property
    def encoder(self) -> nn.Module:
        return self.teacher_module.backbone  # DINOv2 evaluates and releases the teacher

    @property
    def step(self) -> int:
        return int(self.teacher_module.step)

    def _temperature(self) -> float:
        warmup = self.teacher_temp_warmup_epochs * self.steps_per_epoch
        start, end = self.teacher_temp
        return end if self.step >= warmup else start + (end - start) * self.step / warmup

    def _losses(self, batch, generator: torch.Generator | None) -> dict[str, Tensor]:
        if isinstance(batch, dict):
            views = batch["global"]  # (G, B, 3, H, W)
            local = batch["local"].flatten(0, 1) if "local" in batch else None
        else:  # a validation tile and its mirror image as the two global views
            views, local = torch.stack([batch, batch.flip(-1)]), None
        g, b = views.shape[:2]
        images = views.flatten(0, 1)
        grid = images.shape[-1] // self.patch
        masks = torch.stack([self.mask_fn(grid, generator=generator) for _ in images])
        masks = masks.to(images.device)
        out = self.model(images, local, masks)

        temp = self._temperature()
        targets = sinkhorn_knopp(out["teacher_dino"], temp).chunk(g)  # per global view
        student = (out["student_dino"].float() / self.student_temp).log_softmax(-1)
        student = student.split(b)  # globals then locals, B rows each
        terms = [
            -(t * s).sum(-1).mean()
            for i, t in enumerate(targets)
            for j, s in enumerate(student)
            if i != j  # a global view is not its own target
        ]
        dino = torch.stack(terms).mean()

        if masks.any():
            target = sinkhorn_knopp(out["teacher_ibot"], temp)
            s = (out["student_ibot"].float() / self.student_temp).log_softmax(-1)
            # Each view's masked patches weigh 1 / their count, so views count equally.
            weight = (1 / masks.sum(1).clamp_min(1)).repeat_interleave(masks.sum(1))
            ibot = (-(target * s).sum(-1) * weight).sum() / masks.any(1).sum()
        else:
            ibot = out["student_dino"].new_zeros(())
        spread = torch.stack([koleo(c) for c in out["student_cls"].chunk(g)]).mean()
        w = self.weights
        loss = w["dino"] * dino + w["ibot"] * ibot + w["koleo"] * spread
        return {"loss": loss, "dino_loss": dino, "ibot_loss": ibot, "koleo_loss": spread}

    def on_before_optimizer_step(self) -> None:
        model = self.net
        frozen = []
        if self.step < self.freeze_last_layer_epochs * self.steps_per_epoch:
            frozen += [model.dino_head.last_layer]
            if model.ibot_head is not None:
                frozen.append(model.ibot_head.last_layer)
        if self.step < self.head_warmup_epochs * self.steps_per_epoch:
            frozen.append(model.backbone)
        for module in frozen:
            for p in module.parameters():
                p.grad = None  # AdamW skips them: no update, no weight decay

    def on_optimizer_step(self) -> None:
        momentum = _cosine(*self.momentum, self.step, self.total_steps)
        self.net.update_teacher(momentum)
        self.teacher_module.step += 1
        wd = _cosine(*self.wd_schedule, self.step, self.total_steps)
        for group in self.optimizer.param_groups:
            group["weight_decay"] = wd if group["base_weight_decay"] else 0.0

    def configure_optimizers(self, total_steps: int, steps_per_epoch: int):
        self.total_steps, self.steps_per_epoch = total_steps, steps_per_epoch
        optimizer, scheduler = super().configure_optimizers(total_steps, steps_per_epoch)
        self.optimizer = optimizer
        return optimizer, scheduler


HEAD_PARAMS = ("prototypes", "hidden_dim", "bottleneck_dim", "nlayers")


def build(cfg, ssl, probe=None) -> DINOv2Task:
    """The task of a validated config. Method params: the heads' ``prototypes``,
    ``hidden_dim``, ``bottleneck_dim``, ``nlayers`` and ``separate_ibot_head``; the
    rest go to :class:`DINOv2Task` (temperatures, loss weights, warmups)."""
    params = dict(ssl.method.params)
    head = {k: params.pop(k) for k in HEAD_PARAMS if k in params}
    separate = params.pop("separate_ibot_head", False)
    bb, ad = ssl.backbone, ssl.adaptation
    backbone = build_backbone(bb.init, bb.id, bb.arch, bb.grad_checkpointing)
    student = apply_adaptation(backbone, ad.type, ad.n, ad.rank, ad.alpha, ad.dropout, ad.targets)
    dim = backbone.config.hidden_size
    model = DINOv2Model(student, DINOHead(dim, **head), DINOHead(dim, **head) if separate else None)
    m = ssl.masking
    return DINOv2Task(
        model,
        momentum=ssl.teacher.momentum or (0.994, 1.0),
        masking=m.type,
        ratio=m.ratio,
        min_aspect=m.min_aspect,
        p=m.p,
        seed=cfg.get("seed", 0),
        probe=probe,
        probe_every=ssl.eval.every if ssl.eval is not None else 1,
        **params,
        **cfg.get("task", {}),
    )
