"""Export a checkpoint written by :mod:`pathotrain.pathogen.train` to ONNX::

    python -m pathotrain.pathogen.export runs/her2_sit01/checkpoints/best.ckpt model.onnx
    python -m pathotrain.pathogen.export best.ckpt model.onnx --size 1024 --steps 10

Needs the export extra (``pip install 'pathotrain[export]'``). One graph does the
whole translation: VAE encoder, the sampler unrolled for ``--steps`` model steps
(default the task's ``sample_steps``) with the EMA weights, and VAE decoder. It
takes

- ``image`` (batch, 3, size, size) float32 source tiles in [0, 1] (tile / 255)
- ``noise`` (batch, 4, size / 8, size / 8) float32 for the SD VAE, standard
  normal: the starting point of sampling, so the runtime picks the seed

and returns ``translated`` (batch, 3, size, size) float32 in [0, 1]. The batch is
dynamic; the side is fixed: ``data.crop`` unless ``--size``, a multiple of the VAE
and model downsampling (64 for the default UNet, 16 for SiT with 2 x 2 patches).
More steps make a larger graph only in nodes; the weights are stored once. The
``pathotrain`` metadata key holds JSON with the shapes, steps and sampler. Export
checks ONNX Runtime against PyTorch on random inputs before returning.
"""

import argparse
from pathlib import Path

from torch import Tensor, nn

from pathotrain import experiment
from pathotrain.pathogen import train
from pathotrain.pathogen.diffusion import LatentTranslationTask
from pathotrain.pathogen.flow import SiT


class _Translate(nn.Module):
    """Source tiles and noise -> translated tiles, sampling with ``steps`` steps."""

    def __init__(self, task: LatentTranslationTask, steps: int) -> None:
        super().__init__()
        self.task, self.steps = task, steps

    def forward(self, image: Tensor, noise: Tensor) -> Tensor:
        condition = self.task.encode(image)
        return self.task.decode(self.task._denoise(noise, condition, self.steps))


def export(
    checkpoint: str | Path, path: str | Path, size: int | None = None, steps: int | None = None
) -> Path:
    """Write ``checkpoint``'s translation, sampled in ``steps`` steps (default the
    task's ``sample_steps``) at ``size`` px (default ``data.crop``), to ``path`` as
    ONNX and return the path."""
    task, cfg = train.load_model(checkpoint)
    size = size or cfg["data"].get("crop")
    if size is None:
        raise ValueError("the config has no data.crop; pass a size")
    vae = task.vae.config
    factor = 2 ** (len(vae.block_out_channels) - 1)
    if isinstance(task.model, SiT):
        factor *= task.model.patch_size
    else:
        factor *= 2 ** (len(task.model.config.block_out_channels) - 1)
    if size % factor:
        raise ValueError(f"size must be a multiple of {factor} for this model, got {size}")
    steps = steps or task.sample_steps
    side = size // 2 ** (len(vae.block_out_channels) - 1)
    latent = (vae.latent_channels, side, side)
    outputs = {"translated": f"(batch, 3, {size}, {size}) float32 in [0, 1]"}
    metadata = {
        "task": "pathogen",
        "model": cfg["model"].get("name", "ldm"),
        "inputs": {
            "image": f"(batch, 3, {size}, {size}) float32 RGB in [0, 1]; tile / 255",
            "noise": f"(batch, {latent[0]}, {side}, {side}) float32, standard normal",
        },
        "tile_size": size,
        "steps": steps,
        "sampler": getattr(task, "sampler", "ddim"),
        "outputs": outputs,
    }
    return experiment.export_onnx(
        _Translate(task, steps),
        path,
        size,
        list(outputs),
        metadata,
        inputs={"image": (3, size, size), "noise": latent},
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="best.ckpt or last.ckpt written by train")
    parser.add_argument("output", help="the .onnx file to write")
    parser.add_argument("--size", type=int, help="input side in px (default: data.crop)")
    parser.add_argument("--steps", type=int, help="sampling steps (default: the task's)")
    args = parser.parse_args(argv)
    export(args.checkpoint, args.output, args.size, args.steps)


if __name__ == "__main__":
    main()
