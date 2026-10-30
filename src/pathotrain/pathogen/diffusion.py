"""Baseline latent diffusion (Stable Diffusion style) for paired image translation.

A frozen Stable Diffusion VAE maps ``(3, H, W)`` tiles in ``[-1, 1]`` to
``(4, H/8, W/8)`` latents. A UNet, trained from scratch, denoises the target
latent given the source latent concatenated on the channel axis (as
InstructPix2Pix conditions on its input image), under Stable Diffusion's DDPM
noise schedule. Sampling uses DDIM.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
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


class LatentDiffusionTask(Task):
    """Train ``unet`` to denoise target latents conditioned on source latents.

    Validation reports ``val/loss`` (the denoising loss with fixed noise, so epochs
    compare) and, on the first ``sample_batches`` batches, ``val/psnr`` and
    ``val/ssim`` of images sampled with ``sample_steps`` DDIM steps against the
    targets (in ``[0, 1]``).

    Args:
        prediction_type: ``"v_prediction"`` (default) or ``"epsilon"``.
        sample_batches: 0 skips sampling (then monitor ``val/loss``, mode min).
    """

    def __init__(
        self,
        unet: UNet2DModel,
        vae: AutoencoderKL,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_epochs: float = 0,
        prediction_type: str = "v_prediction",
        num_train_timesteps: int = 1000,
        sample_steps: int = 50,
        sample_batches: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.model, self.vae = unet, vae
        self.lr, self.weight_decay, self.warmup_epochs = lr, weight_decay, warmup_epochs
        self.sample_steps, self.sample_batches, self.seed = sample_steps, sample_batches, seed
        # Stable Diffusion's schedule.
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type=prediction_type,
            clip_sample=False,
        )
        self.val_loss = MeanMetric()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self._val_batches = 0

    def train(self, mode: bool = True) -> "LatentDiffusionTask":
        super().train(mode)
        self.vae.eval()  # frozen
        return self

    @torch.no_grad()
    def encode(self, images: Tensor, sample: bool = False) -> Tensor:
        """``[-1, 1]`` images to scaled latents (the posterior mean unless ``sample``)."""
        posterior = self.vae.encode(images).latent_dist
        return (posterior.sample() if sample else posterior.mode()) * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        """Scaled latents to images in ``[0, 1]``."""
        images = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        return ((images + 1) / 2).clamp(0, 1)

    def _loss(
        self, source: Tensor, target: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        condition = self.encode(source)
        latents = self.encode(target, sample=generator is None)
        noise = torch.randn(latents.shape, generator=generator, device=latents.device)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (len(latents),),
            generator=generator,
            device=latents.device,
        )
        noisy = self.scheduler.add_noise(latents, noise, timesteps)
        pred = self.model(torch.cat([noisy, condition], 1), timesteps).sample
        if self.scheduler.config.prediction_type == "epsilon":
            goal = noise
        else:
            goal = self.scheduler.get_velocity(latents, noise, timesteps)
        return F.mse_loss(pred.float(), goal)

    def training_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        loss = self._loss(*batch)
        return loss, {"loss": loss.detach()}

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> None:
        source, target = batch
        # Same noise every epoch, so the loss and samples compare across epochs.
        generator = torch.Generator(source.device).manual_seed(self.seed + self._val_batches)
        self.val_loss.update(self._loss(source, target, generator), len(source))
        if self._val_batches < self.sample_batches:
            fake = self.translate(source, generator=generator)
            real = (target + 1) / 2
            self.psnr.update(fake, real)
            self.ssim.update(fake, real)
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
        """Sample targets for ``(B, 3, H, W)`` source images in ``[-1, 1]`` (sides
        multiples of 8 times the UNet's downsampling); return them in ``[0, 1]``."""
        condition = self.encode(source)
        scheduler = DDIMScheduler.from_config(self.scheduler.config)
        scheduler.set_timesteps(steps or self.sample_steps, device=source.device)
        latents = torch.randn(condition.shape, generator=generator, device=condition.device)
        for t in scheduler.timesteps:
            pred = self.model(torch.cat([latents, condition], 1), t.expand(len(latents))).sample
            latents = scheduler.step(pred.float(), t, latents).prev_sample
        return self.decode(latents)

    def configure_optimizers(
        self, total_steps: int, steps_per_epoch: int
    ) -> tuple[Optimizer, LRScheduler]:
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer, warmup_cosine(optimizer, total_steps, steps_per_epoch, self.warmup_epochs)
