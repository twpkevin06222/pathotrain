"""Translate a folder of source tiles (e.g. H&E) with a checkpoint written by
:mod:`pathotrain.pathogen.train`::

    python -m pathotrain.pathogen.translate runs/her2_ldm01/checkpoints/best.ckpt HE/test out/

Writes ``<out>/<stem>.png`` per tile. Tiles keep their size, so one batch needs
one size, with sides a multiple of 8 times the model's downsampling (64 for the
default UNet, 16 for SiT with 2 x 2 patches). Runs on the GPU in bf16 when one is
present.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from pathotrain.dataloader.base import by_stem
from pathotrain.dataloader.pathogen import IMAGE_SUFFIXES, read_image
from pathotrain.pathogen.train import load_model


def translate_dir(
    checkpoint: str | Path,
    source_dir: str | Path,
    out_dir: str | Path,
    steps: int | None = None,
    batch_size: int = 4,
    seed: int = 0,
) -> list[Path]:
    """Translate every image in ``source_dir``; return the written paths."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task, _ = load_model(checkpoint)
    task.to(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = list(by_stem([source_dir], IMAGE_SUFFIXES).values())
    written = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        source = torch.stack([read_image(p).float() / 255 for p in chunk]).to(device)
        generator = torch.Generator(device).manual_seed(seed + i)
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            images = task.translate(source, steps, generator)
        for p, image in zip(chunk, images, strict=True):
            array = (image.float() * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(array).save(out_dir / f"{p.stem}.png")
            written.append(out_dir / f"{p.stem}.png")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="best.ckpt or last.ckpt written by pathogen.train")
    parser.add_argument("source", help="folder of source tiles")
    parser.add_argument("output", help="folder to write the translated tiles to")
    parser.add_argument("--steps", type=int, help="DDIM steps (default: the task's sample_steps)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    translate_dir(args.checkpoint, args.source, args.output, args.steps, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
