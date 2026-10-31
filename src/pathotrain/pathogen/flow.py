"""SiT with rectified flow and REPA for paired image translation.

- SiT (Ma et al., 2024): a DiT-style transformer over 2 x 2 patches of the latent,
  conditioned on the time with adaLN-Zero. As in the baseline, the source latent
  is concatenated to the noisy target latent on the channel axis.
- Rectified flow: ``x_t = (1 - t) x_0 + t noise`` for ``t`` in ``[0, 1]``; the model
  predicts the velocity ``noise - x_0``. Sampling integrates that ODE from noise
  (``t = 1``) to data with Euler or Heun steps, far fewer than DDPM needs.
- REPA (Yu et al., 2025): an MLP projects the tokens of one early block onto a
  frozen encoder's patch features of the clean image (Midnight by default), and
  the cosine distance is added to the loss. Faster training, better samples.
"""

import itertools
import math
from typing import Literal, Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import Dinov2Config, Dinov2Model

from pathotrain.pathofm import load_backbone, segmentation_embedding
from pathotrain.pathogen.diffusion import AutoencoderKL, LatentTranslationTask

# (hidden size, depth, heads) of the DiT / SiT sizes.
SIT_SIZES = {"S": (384, 12, 6), "B": (768, 12, 12), "L": (1024, 24, 16), "XL": (1152, 28, 16)}


def _sincos(dim: int, positions: Tensor) -> Tensor:
    freqs = torch.exp(
        -math.log(10000) * torch.arange(dim // 2, device=positions.device) / (dim // 2)
    )
    angles = positions[:, None].float() * freqs[None]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


def _sincos_2d(dim: int, h: int, w: int, device: torch.device) -> Tensor:
    """DiT's fixed 2D sin-cos position embedding for an ``h x w`` grid, ``(h * w, dim)``;
    computed per call, so any latent size works."""
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    return torch.cat([_sincos(dim // 2, ys.flatten()), _sincos(dim // 2, xs.flatten())], dim=1)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class SiTBlock(nn.Module):
    """Pre-norm transformer block whose norms are shifted, scaled and gated by the
    time embedding (adaLN-Zero)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp), nn.GELU(approximate="tanh"), nn.Linear(mlp, hidden_size)
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def _attention(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.num_heads, d // self.num_heads).unbind(2)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        return self.proj(out.transpose(1, 2).reshape(b, n, d))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c).chunk(6, dim=1)
        x = x + gate1[:, None] * self._attention(_modulate(self.norm1(x), shift1, scale1))
        return x + gate2[:, None] * self.mlp(_modulate(self.norm2(x), shift2, scale2))


class SiT(nn.Module):
    """Scalable interpolant transformer over ``patch_size`` patches of a latent.

    ``forward(x, t)`` returns the predicted velocity, shaped like ``x`` with
    ``out_channels`` channels, and, in training mode with ``repa_dim`` set, the
    tokens of block ``repa_depth`` projected to ``repa_dim`` (else None).

    Args:
        in_channels: Latent channels in (the noisy target and the source latents).
        repa_dim: The REPA encoder's feature size; None builds no projector.
        repa_depth: Which block's tokens REPA aligns (1-based).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: int = 2,
        mlp_ratio: float = 4.0,
        repa_dim: int | None = None,
        repa_depth: int = 4,
        projector_dim: int = 2048,
    ) -> None:
        super().__init__()
        if repa_dim is not None and not 1 <= repa_depth <= depth:
            raise ValueError(f"repa_depth must be in 1..{depth}, got {repa_depth}")
        self.out_channels, self.patch_size, self.repa_depth = out_channels, patch_size, repa_depth
        self.x_embed = nn.Conv2d(in_channels, hidden_size, patch_size, stride=patch_size)
        self.t_embed = nn.Sequential(
            nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.blocks = nn.ModuleList(
            SiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.final = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.projector = None
        if repa_dim is not None:
            self.projector = nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, repa_dim),
            )
        self._init_weights()

    def _init_weights(self) -> None:
        # DiT's: Xavier linears, and zero adaLN and output layers so every block starts
        # as the identity and the model predicts zero velocity.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.x_embed.weight.view(self.x_embed.weight.shape[0], -1))
        nn.init.normal_(self.t_embed[0].weight, std=0.02)
        nn.init.normal_(self.t_embed[2].weight, std=0.02)
        for ada in [b.ada for b in self.blocks] + [self.final_ada]:
            nn.init.zeros_(ada[-1].weight)
            nn.init.zeros_(ada[-1].bias)
        nn.init.zeros_(self.final.weight)

    def forward(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor | None]:
        b, _, height, width = x.shape
        p = self.patch_size
        h, w = height // p, width // p
        tokens = self.x_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + _sincos_2d(tokens.shape[-1], h, w, x.device).to(tokens.dtype)
        # t in [0, 1], spread over the sinusoid's range as for 1000 diffusion steps.
        c = self.t_embed(_sincos(256, t * 1000).to(tokens.dtype))
        projected = None
        for i, block in enumerate(self.blocks, 1):
            tokens = block(tokens, c)
            if i == self.repa_depth and self.projector is not None and self.training:
                projected = self.projector(tokens)
        shift, scale = self.final_ada(c).chunk(2, dim=1)
        out = self.final(_modulate(self.final_norm(tokens), shift, scale))
        out = out.view(b, h, w, p, p, self.out_channels).permute(0, 5, 1, 3, 2, 4)
        return out.reshape(b, self.out_channels, height, width), projected


def build_encoder(encoder: str | dict) -> Dinov2Model:
    """A frozen REPA encoder: a :mod:`pathotrain.pathofm` model name (e.g.
    ``midnight-12k``), or a dict of ``Dinov2Config`` arguments (random weights,
    e.g. for tests). Kept in bf16: it only gives targets."""
    model = (
        load_backbone(encoder) if isinstance(encoder, str) else Dinov2Model(Dinov2Config(**encoder))
    )
    return model.to(torch.bfloat16).eval().requires_grad_(False)


class FlowMatchingTask(LatentTranslationTask):
    """Train a :class:`SiT` with rectified flow, plus REPA when ``encoder`` is given
    (see :class:`~pathotrain.pathogen.diffusion.LatentTranslationTask` for validation).

    Training logs ``flow_loss`` and, with REPA, ``repa_loss`` (1 - mean cosine
    similarity); ``loss`` is ``flow_loss + repa_weight * repa_loss``. ``val/loss``
    is the flow loss alone.

    Args:
        encoder: Frozen REPA encoder (see :func:`build_encoder`); None trains without REPA.
        repa_image: Align with the encoder's features of the ``"target"`` tile (REPA's
            recipe) or of the ``"source"`` tile (when the encoder does not know the
            target stain; Midnight was trained mostly on H&E).
        time_sampling: ``"logit_normal"`` (SD3's, favouring mid times) or ``"uniform"``.
        sampler: ``"euler"`` (one model call per step) or ``"heun"`` (two, more accurate).
    """

    def __init__(
        self,
        model: SiT,
        vae: AutoencoderKL,
        encoder: Dinov2Model | None = None,
        repa_weight: float = 0.5,
        repa_image: Literal["target", "source"] = "target",
        time_sampling: Literal["logit_normal", "uniform"] = "logit_normal",
        sampler: Literal["euler", "heun"] = "euler",
        sample_steps: int = 25,
        **kwargs,
    ) -> None:
        super().__init__(model, vae, sample_steps=sample_steps, **kwargs)
        if (encoder is None) != (model.projector is None):
            raise ValueError("REPA needs both an encoder and a SiT built with repa_dim")
        for name, value, allowed in (
            ("repa_image", repa_image, ("target", "source")),
            ("time_sampling", time_sampling, ("logit_normal", "uniform")),
            ("sampler", sampler, ("euler", "heun")),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        self.encoder, self.repa_weight, self.repa_image = encoder, repa_weight, repa_image
        self.time_sampling, self.sampler = time_sampling, sampler
        self.patch_size = model.patch_size

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if self.encoder is not None:
            self.encoder.eval()  # frozen
        return self

    def _objective(
        self,
        condition: Tensor,
        latents: Tensor,
        model: nn.Module,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        noise = torch.randn(latents.shape, generator=generator, device=latents.device)
        draw = torch.randn if self.time_sampling == "logit_normal" else torch.rand
        t = draw(len(latents), generator=generator, device=latents.device)
        if self.time_sampling == "logit_normal":
            t = t.sigmoid()
        tt = t[:, None, None, None]
        pred, projected = model(torch.cat([(1 - tt) * latents + tt * noise, condition], 1), t)
        return F.mse_loss(pred.float(), noise - latents), projected

    @torch.no_grad()
    def _encoder_features(self, images: Tensor, grid: tuple[int, int]) -> Tensor:
        """The encoder's patch features of ``[0, 1]`` images, resized so its patches
        match the SiT's ``grid`` of tokens one to one; ``(B, tokens, dim)``."""
        patch = self.encoder.config.patch_size
        # Midnight normalises with mean = std = 0.5, i.e. to [-1, 1].
        x = F.interpolate(
            images * 2 - 1, size=(grid[0] * patch, grid[1] * patch), mode="bilinear", antialias=True
        )
        features = segmentation_embedding(self.encoder, x.to(torch.bfloat16))
        return features.flatten(2).transpose(1, 2).float()

    def training_step(self, batch) -> tuple[Tensor, dict[str, Tensor]]:
        condition, latents = self._latents(batch, sample=True)
        loss, projected = self._objective(condition, latents, self.model)
        logs = {"flow_loss": loss.detach()}
        if self.encoder is not None:
            if isinstance(batch, dict):  # cached latents bring the image REPA reads
                image = batch["image"]
            else:
                image = batch[1] if self.repa_image == "target" else batch[0]
            p = self.patch_size
            features = self._encoder_features(
                image, (latents.shape[-2] // p, latents.shape[-1] // p)
            )
            repa = 1 - F.cosine_similarity(projected.float(), features, dim=-1).mean()
            logs["repa_loss"] = repa.detach()
            loss = loss + self.repa_weight * repa
        logs["loss"] = loss.detach()
        return loss, logs

    def _velocity(self, latents: Tensor, condition: Tensor, t: Tensor) -> Tensor:
        return self.net(torch.cat([latents, condition], 1), t.expand(latents.shape[0]))[0].float()

    def _denoise(self, latents: Tensor, condition: Tensor, steps: int) -> Tensor:
        times = torch.linspace(1, 0, steps + 1, device=condition.device)
        for t, t_next in itertools.pairwise(times):
            v = self._velocity(latents, condition, t)
            if self.sampler == "heun" and t_next > 0:
                guess = latents + (t_next - t) * v
                v = (v + self._velocity(guess, condition, t_next)) / 2
            latents = latents + (t_next - t) * v
        return latents
