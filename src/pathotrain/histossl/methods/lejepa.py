"""lejepa: LeJEPA (Balestriero & LeCun 2025, "LeJEPA: Provable and Scalable
Self-Supervised Learning Without the Heuristics"), from scratch as in the paper,
or from Midnight's weights (``init: pretrained``, with LoRA or ``last_n``): the
objective needs no teacher, so it applies to a pretrained encoder unchanged.

Written from the paper; the reference code (rbalestr-lab/lejepa) is CC-BY-NC
and none of it is used. One network, no teacher, EMA or stop-gradient:

- Prediction: every view's projected embedding moves toward the mean of the
  global views' embeddings of the same tile (squared distance).
- SIGReg (Sketched Isotropic Gaussian Regularisation): the embeddings of each
  view, projected on random unit directions, should each look like N(0, 1),
  scored by the Epps-Pulley statistic (a weighted distance between the empirical
  and the Gaussian characteristic function). Directions are redrawn every step,
  from the step number, so every rank draws the same ones.

The loss is ``(1 - lambda) * prediction + lambda * sigreg``.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from pathotrain.histossl.adaptation import apply_adaptation
from pathotrain.histossl.backbones import build_backbone, tokens
from pathotrain.histossl.methods.base import SSLTask


def epps_pulley(x: Tensor, t_max: float = 5.0, knots: int = 17) -> Tensor:
    """Epps-Pulley statistic of each column of ``x`` ``(N, M)`` against N(0, 1):
    ``N * integral |ecf(t) - exp(-t^2 / 2)|^2 exp(-t^2 / 2) dt`` over ``[-t_max,
    t_max]`` (trapezoid rule on ``knots`` points); ``(M,)``. 0 for a perfect fit.
    Summed over ranks' samples under DDP."""
    t = torch.linspace(-t_max, t_max, knots, device=x.device)
    gauss = torch.exp(-0.5 * t**2)
    xt = x.float().unsqueeze(-1) * t  # (N, M, knots)
    n = torch.tensor(float(x.shape[0]), device=x.device)
    parts = torch.stack([xt.cos().sum(0), xt.sin().sum(0)])  # (2, M, knots)
    if dist.is_available() and dist.is_initialized():
        from torch.distributed.nn.functional import all_reduce  # keeps gradients

        parts = all_reduce(torch.cat([parts.flatten(), n[None]]))
        n, parts = parts[-1], parts[:-1].view(2, x.shape[1], knots)
    real, imag = parts[0] / n, parts[1] / n
    error = ((real - gauss) ** 2 + imag**2) * gauss
    return n * torch.trapezoid(error, t, dim=-1)


def sigreg(z: Tensor, slices: int, generator: torch.Generator, **kwargs) -> Tensor:
    """Mean Epps-Pulley statistic of ``z`` ``(N, D)`` along ``slices`` random unit
    directions drawn from the CPU ``generator``; ``kwargs`` go to :func:`epps_pulley`."""
    directions = torch.randn(z.shape[1], slices, generator=generator)
    directions = F.normalize(directions.to(z.device), dim=0)
    return epps_pulley(z.float() @ directions, **kwargs).mean()


class Projector(nn.Module):
    """Linear-BatchNorm-GELU blocks, then a linear layer to ``out_dim``."""

    def __init__(self, dim: int, hidden_dim: int = 2048, out_dim: int = 256, layers: int = 3):
        super().__init__()
        blocks: list[nn.Module] = []
        for _ in range(layers - 1):
            blocks += [nn.Linear(dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU()]
            dim = hidden_dim
        self.mlp = nn.Sequential(*blocks, nn.Linear(dim, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class LeJEPAModel(nn.Module):
    """``forward(global_views, local_views)`` -> projected CLS embeddings of every
    view, globals first, ``(views * B, out_dim)``. ``step`` counts optimizer steps
    (it seeds SIGReg's directions, and is saved with the weights)."""

    def __init__(self, backbone: nn.Module, projector: Projector) -> None:
        super().__init__()
        self.backbone, self.projector = backbone, projector
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def forward(self, global_views: Tensor, local_views: Tensor | None) -> Tensor:
        cls = [tokens(self.backbone, global_views)[0]]
        if local_views is not None:
            cls.append(tokens(self.backbone, local_views)[0])
        return self.projector(torch.cat(cls))


class LeJEPATask(SSLTask):
    """Train a :class:`LeJEPAModel` (see the module docstring); logs
    ``prediction_loss`` and ``sigreg_loss``.

    Args:
        lamb: SIGReg's weight, lambda.
        slices: Random directions per step.
        t_max, knots: Epps-Pulley's integration range and points.
        Other arguments as for :class:`SSLTask`.
    """

    def __init__(
        self,
        model: LeJEPAModel,
        lamb: float = 0.05,
        slices: int = 1024,
        t_max: float = 5.0,
        knots: int = 17,
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)
        if not 0 <= lamb <= 1:
            raise ValueError(f"lamb must be in [0, 1], got {lamb}")
        self.lamb, self.slices, self.t_max, self.knots = lamb, slices, t_max, knots
        self.net = model  # the plain module: Fabric replaces self.model by its wrapper

    def _losses(self, batch, generator: torch.Generator | None) -> dict[str, Tensor]:
        if isinstance(batch, dict):
            views = batch["global"]  # (G, B, 3, H, W)
            local = batch["local"].flatten(0, 1) if "local" in batch else None
        else:  # a validation tile and its mirror image as the two global views
            views, local = torch.stack([batch, batch.flip(-1)]), None
        g, b = views.shape[:2]
        z = self.model(views.flatten(0, 1), local).float()
        z = z.view(-1, b, z.shape[-1])  # (views, B, D)
        centre = z[:g].mean(0)
        prediction = (z - centre).square().mean()
        if generator is None:  # training: the same directions on every rank
            generator = torch.Generator().manual_seed(int(self.net.step))
        spread = torch.stack(
            [sigreg(v, self.slices, generator, t_max=self.t_max, knots=self.knots) for v in z]
        ).mean()
        loss = (1 - self.lamb) * prediction + self.lamb * spread
        return {"loss": loss, "prediction_loss": prediction, "sigreg_loss": spread}

    def on_optimizer_step(self) -> None:
        self.net.step += 1


def build(cfg, ssl, probe=None) -> LeJEPATask:
    """The task of a validated config. Method params: the projector's
    ``hidden_dim``, ``out_dim`` and ``layers``; the rest go to :class:`LeJEPATask`
    (``lamb``, ``slices``, ``t_max``, ``knots``)."""
    params = dict(ssl.method.params)
    head = {k: params.pop(k) for k in ("hidden_dim", "out_dim", "layers") if k in params}
    bb, ad = ssl.backbone, ssl.adaptation
    backbone = build_backbone(bb.init, bb.id, bb.arch, bb.grad_checkpointing, bb.dtype)
    backbone = apply_adaptation(backbone, ad.type, ad.n, ad.rank, ad.alpha, ad.dropout, ad.targets)
    model = LeJEPAModel(backbone, Projector(backbone.config.hidden_size, **head))
    return LeJEPATask(
        model,
        seed=cfg.get("seed", 0),
        probe=probe,
        probe_every=ssl.eval.every if ssl.eval is not None else 1,
        **params,
        **cfg.get("task", {}),
    )
