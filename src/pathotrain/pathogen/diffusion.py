"""Baseline latent diffusion (Stable Diffusion style) for paired image translation.

A frozen Stable Diffusion VAE maps ``(3, H, W)`` tiles in ``[0, 1]`` to
``(4, H/8, W/8)`` latents. A UNet, trained from scratch, denoises the target
latent given the source latent concatenated on the channel axis (as
InstructPix2Pix conditions on its input image), under Stable Diffusion's DDPM
noise schedule. Sampling uses DDIM.
"""

import copy
from typing import Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MeanMetric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from pathotrain.engine import Task, warmup_cosine

try:
    from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DModel
except ImportError as e:  # an optional extra, so other users do not need it
    raise ImportError(
        "pathogen needs the pathogen extra: pip install 'pathotrain[pathogen]'"
    ) from e

DEFAULT_VAE = "stabilityai/sd-vae-ft-mse"  # Stable Diffusion's VAE, fine-tuned; MIT licence


def build_vae(vae: str | dict = DEFAULT_VAE) -> AutoencoderKL:
    """A frozen ``AutoencoderKL``: a Hugging Face Hub id or local path, or a dict
    of ``AutoencoderKL`` arguments (random weights, e.g. for tests)."""
    model = AutoencoderKL.from_pretrained(vae) if isinstance(vae, str) else AutoencoderKL(**vae)
    return model.eval().requires_grad_(False)


def build_unet(latent_channels: int, **kwargs) -> UNet2DModel:
    """A ``UNet2DModel`` taking the noisy target and the source latents (``2 *
    latent_channels`` in) and predicting ``latent_channels``. ``kwargs`` are
    ``UNet2DModel`` arguments; by default 4 levels of 128-512 channels with
    self-attention on the two lowest resolutions."""
    levels = len(kwargs.get("block_out_channels", (128, 256, 512, 512)))
    plain, attn = max(levels - 2, 0), min(levels, 2)
    defaults = {
        "block_out_channels": (128, 256, 512, 512),
        "layers_per_block": 2,
        "down_block_types": ("DownBlock2D",) * plain + ("AttnDownBlock2D",) * attn,
        "up_block_types": ("AttnUpBlock2D",) * attn + ("UpBlock2D",) * plain,
    }
    return UNet2DModel(
        in_channels=2 * latent_channels, out_channels=latent_channels, **{**defaults, **kwargs}
    )


class LatentTranslationTask(Task):
    """What every latent model here shares: a frozen VAE, the source latent as the
    condition, an EMA of the weights, validation and AdamW with warmup and cosine
    decay. Subclasses give ``_objective`` (the training loss; deterministic given
    ``generator``) and ``_sample`` (latents from noise).

    A batch is ``(source, target)`` images in ``[0, 1]``, or, for training from a
    latent cache (:class:`~pathotrain.pathogen.cache.CachedLatentDataset`), a dict
    with unscaled ``"source"`` and ``"target"`` latents (and ``"image"`` for REPA).

    Validation runs the EMA weights (``ema_decay=None``: the trained ones) and
    reports ``val/loss`` (the loss with fixed noise, so epochs compare) and, on the
    first ``sample_batches`` batches, ``val/psnr`` and ``val/ssim`` of images sampled
    with ``sample_steps`` steps against the targets (in ``[0, 1]``);
    ``sample_batches=0`` skips sampling (then monitor ``val/loss``, mode min).

    Args:
        ema_decay: Decay of the EMA, updated after every optimizer step and warmed
            up as ``min(ema_decay, (1 + n) / (10 + n))`` over the first steps ``n``.
    """

    def __init__(
        self,
        model: nn.Module,
        vae: AutoencoderKL,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_epochs: float = 0,
        ema_decay: float | None = 0.9999,
        sample_steps: int = 50,
        sample_batches: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if ema_decay is not None and not 0 < ema_decay < 1:
            raise ValueError(f"ema_decay must be in (0, 1) or None, got {ema_decay}")
        self.model, self.vae = model, vae
        self.ema, self.ema_decay = None, ema_decay
        if ema_decay is not None:
            self.ema = copy.deepcopy(model).requires_grad_(False)
            # In the EMA's state, so a resumed run continues its warmup.
            self.ema.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))
        self.lr, self.weight_decay, self.warmup_epochs = lr, weight_decay, warmup_epochs
        self.sample_steps, self.sample_batches, self.seed = sample_steps, sample_batches, seed
        self.val_loss = MeanMetric()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self._val_batches = 0

    @property
    def net(self) -> nn.Module:
        """The weights to sample with: the EMA, or the trained model without one."""
        return self.model if self.ema is None else self.ema

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.vae.eval()  # frozen
        if self.ema is not None:
            self.ema.eval()
        return self

    @torch.no_grad()
    def on_optimizer_step(self) -> None:
        if self.ema is None:
            return
        self.ema.ema_updates += 1
        n = self.ema.ema_updates.item()
        decay = min(self.ema_decay, (1 + n) / (10 + n))
        model = getattr(self.model, "module", self.model)  # unwrap Fabric / DDP
        ema, live = list(self.ema.parameters()), list(model.parameters())
        torch._foreach_lerp_(ema, live, 1 - decay)

    def checkpoint_state(self) -> dict[str, nn.Module]:
        return {} if self.ema is None else {"ema": self.ema}

    @torch.no_grad()
    def encode(self, images: Tensor, sample: bool = False) -> Tensor:
        """``[0, 1]`` images to scaled latents (the posterior mean unless ``sample``)."""
        posterior = self.vae.encode(images * 2 - 1).latent_dist  # the VAE takes [-1, 1]
        return (posterior.sample() if sample else posterior.mode()) * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        """Scaled latents to images in ``[0, 1]``."""
        images = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        return ((images + 1) / 2).clamp(0, 1)

    def _latents(self, batch, sample: bool) -> tuple[Tensor, Tensor]:
        """Scaled ``(condition, target)`` latents of an image or a cached batch."""
        if isinstance(batch, dict):
            scale = self.vae.config.scaling_factor
            return batch["source"].float() * scale, batch["target"].float() * scale
        source, target = batch
        return self.encode(source), self.encode(target, sample=sample)

    def _objective(
        self,
        condition: Tensor,
        latents: Tensor,
        model: nn.Module,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """``model``'s loss on ``latents`` given ``condition``, and the REPA features
        (None for models without them)."""
        raise NotImplementedError

    def _sample(self, condition: Tensor, steps: int, generator: torch.Generator | None) -> Tensor:
        raise NotImplementedError

    def training_step(self, batch) -> tuple[Tensor, dict[str, Tensor]]:
        loss, _ = self._objective(*self._latents(batch, sample=True), self.model)
        return loss, {"loss": loss.detach()}

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> None:
        source, target = batch
        # Same noise every epoch, so the loss and samples compare across epochs.
        generator = torch.Generator(source.device).manual_seed(self.seed + self._val_batches)
        loss, _ = self._objective(*self._latents(batch, sample=False), self.net, generator)
        self.val_loss.update(loss, len(source))
        if self._val_batches < self.sample_batches:
            fake = self.translate(source, generator=generator)
            self.psnr.update(fake, target)
            self.ssim.update(fake, target)
        self._val_batches += 1

    def compute_metrics(self) -> dict[str, float]:
        metrics = {"val/loss": self.val_loss.compute().item()}
        if self.sample_batches and self._val_batches:
            metrics["val/psnr"] = self.psnr.compute().item()
            metrics["val/ssim"] = self.ssim.compute().item()
        for m in (self.val_loss, self.psnr, self.ssim):
            m.reset()
        self._val_batches = 0
        return metrics

    @torch.no_grad()
    def translate(
        self, source: Tensor, steps: int | None = None, generator: torch.Generator | None = None
    ) -> Tensor:
        """Sample targets with :attr:`net` for ``(B, 3, H, W)`` source images in
        ``[0, 1]`` (sides a multiple of the VAE's and the model's downsampling);
        return them in ``[0, 1]``."""
        condition = self.encode(source)
        return self.decode(self._sample(condition, steps or self.sample_steps, generator))

    def configure_optimizers(
        self, total_steps: int, steps_per_epoch: int
    ) -> tuple[Optimizer, LRScheduler]:
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer, warmup_cosine(optimizer, total_steps, steps_per_epoch, self.warmup_epochs)


class LatentDiffusionTask(LatentTranslationTask):
    """Train ``unet`` to denoise target latents conditioned on source latents, under
    Stable Diffusion's DDPM schedule; sample with DDIM (see :class:`LatentTranslationTask`).

    Args:
        prediction_type: ``"v_prediction"`` (default) or ``"epsilon"``.
    """

    def __init__(
        self,
        unet: UNet2DModel,
        vae: AutoencoderKL,
        prediction_type: str = "v_prediction",
        num_train_timesteps: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__(unet, vae, **kwargs)
        # Stable Diffusion's schedule.
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type=prediction_type,
            clip_sample=False,
        )

    def _objective(
        self,
        condition: Tensor,
        latents: Tensor,
        model: nn.Module,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, None]:
        noise = torch.randn(latents.shape, generator=generator, device=latents.device)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (len(latents),),
            generator=generator,
            device=latents.device,
        )
        noisy = self.scheduler.add_noise(latents, noise, timesteps)
        pred = model(torch.cat([noisy, condition], 1), timesteps).sample
        if self.scheduler.config.prediction_type == "epsilon":
            goal = noise
        else:
            goal = self.scheduler.get_velocity(latents, noise, timesteps)
        return F.mse_loss(pred.float(), goal), None

    def _sample(self, condition: Tensor, steps: int, generator: torch.Generator | None) -> Tensor:
        scheduler = DDIMScheduler.from_config(self.scheduler.config)
        scheduler.set_timesteps(steps, device=condition.device)
        latents = torch.randn(condition.shape, generator=generator, device=condition.device)
        for t in scheduler.timesteps:
            pred = self.net(torch.cat([latents, condition], 1), t.expand(len(latents))).sample
            latents = scheduler.step(pred.float(), t, latents).prev_sample
        return latents
