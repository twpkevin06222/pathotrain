"""A latent cache for training: encode every tile pair with the VAE once, instead
of at every step (the VAE takes ~90% of a baseline training step). Turn it on in
the config; the first run builds it, later runs with the same data reuse it::

    data:
      cache: true           # default false: encode tiles at every step
      cache_variants: 2     # orientations to encode, 1-8
      cache_dir: null       # default ~/.cache/pathotrain/latents/<fingerprint>

or build it ahead of training with ``python -m pathotrain.pathogen.cache config.yaml``.
The default directory is named after a fingerprint of the VAE, the resolved tile
directories, the paired tile names and ``cache_variants``, so changing any of
them builds a new cache instead of reading a stale one.

Whole tiles are encoded, and training crops their latents (``data.crop / 8``).
The VAE is not equivariant to flips and 90 degree turns (a flipped latent is ~60%
off the latent of the flipped tile), so each orientation that augmentation should
see is encoded on its own: ``variants`` of the 8 dihedral ones, in the order
identity, horizontal flip, then quarter turns of both. 2 (the default) doubles
the encoding time, 8 gives every flip and turn. Latents are the posterior means
(sd-vae-ft-mse's posterior has ~0 variance), unscaled, in fp16: 2 x 4 x 128 x 128
x 2 bytes = 256 KB per pair and variant for 1024 px tiles.

Validation keeps encoding images, so its metrics compare with uncached runs.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from pathotrain.dataloader.base import by_stem
from pathotrain.engine import load_config
from pathotrain.pathogen.data import IMAGE_SUFFIXES, PairedTileDataset, read_image, to_model_range
from pathotrain.pathogen.diffusion import DEFAULT_VAE, build_vae

LATENTS, META = "latents.npy", "meta.json"


def dihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    """The ``k``-th of the 8 flips and turns of the last two axes (0 is identity,
    1 the horizontal flip, then quarter turns of both)."""
    x = torch.rot90(x, k // 2, dims=(-2, -1))
    return x.flip(-1) if k % 2 else x


@torch.no_grad()
def build_latent_cache(
    source_dirs: list[str],
    target_dirs: list[str],
    out_dir: str | Path,
    vae: str | dict = DEFAULT_VAE,
    variants: int = 2,
    batch_size: int = 4,
    num_workers: int = 4,
) -> Path:
    """Encode every pair of ``source_dirs`` / ``target_dirs`` (tiles of one size)
    into ``out_dir/latents.npy``, ``(pairs, variants, 2, C, h, w)`` fp16, with
    ``meta.json`` (written last: a cache without it is incomplete)."""
    if not 1 <= variants <= 8:
        raise ValueError(f"variants must be in 1..8, got {variants}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / META).unlink(missing_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vae(vae).to(device)
    pairs = PairedTileDataset(source_dirs, target_dirs)
    height, width = pairs[0][0].shape[-2:]
    if variants > 2 and height != width:
        raise ValueError(f"quarter turns (variants > 2) need square tiles, got {height} x {width}")
    factor = 2 ** (len(model.config.block_out_channels) - 1)
    shape = (
        len(pairs),
        variants,
        2,
        model.config.latent_channels,
        height // factor,
        width // factor,
    )
    latents = np.lib.format.open_memmap(out_dir / LATENTS, "w+", np.float16, shape)
    loader = DataLoader(pairs, batch_size=batch_size, num_workers=num_workers)
    i = 0
    for n, (source, target) in enumerate(loader):
        if source.shape[-2:] != (height, width):
            raise ValueError(f"tiles must share one size: {source.shape[-2:]} != {(height, width)}")
        both = torch.cat([source, target]).to(device)
        for k in range(variants):
            z = model.encode(dihedral(both, k)).latent_dist.mean
            latents[i : i + len(source), k] = (
                z.view(2, len(source), *z.shape[1:]).transpose(0, 1).half().cpu().numpy()
            )
        i += len(source)
        if n % 100 == 0:
            logger.info(f"cached {i} / {len(pairs)} pairs")
    latents.flush()
    meta = {
        "source_dirs": _resolved(source_dirs),
        "target_dirs": _resolved(target_dirs),
        "vae": vae,
        "variants": variants,
        "factor": factor,
        "pairs": [[str(s), str(t)] for s, t in pairs.pairs],
    }
    (out_dir / META).write_text(json.dumps(meta))
    logger.info(f"cached {len(pairs)} pairs x {variants} variants in {out_dir}")
    return out_dir


class CachedLatentDataset(Dataset):
    """Training samples from a cache written by :func:`build_latent_cache`: a random
    variant (flip / turn) and a random ``crop`` px window of each pair's latents.

    A sample is a dict of unscaled ``"source"`` and ``"target"`` latents, ``(C,
    crop / 8, crop / 8)`` float32, plus with ``image`` the same window of that
    tile (``"source"`` or ``"target"``) as ``"image"`` in ``[-1, 1]``, for REPA.

    Raises:
        FileNotFoundError: No complete cache in ``cache_dir``.
        ValueError: ``vae``, ``source_dirs`` or ``target_dirs`` (when given) are not
            what the cache was built from, or ``crop`` is not a multiple of the
            VAE's downsampling.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        crop: int | None = None,
        image: Literal["source", "target"] | None = None,
        vae: str | dict = DEFAULT_VAE,
        source_dirs: list[str] | None = None,
        target_dirs: list[str] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        if not (self.cache_dir / META).exists():
            raise FileNotFoundError(
                f"no complete latent cache in {cache_dir}; build it with "
                "python -m pathotrain.pathogen.cache config.yaml"
            )
        meta = json.loads((self.cache_dir / META).read_text())
        if meta["vae"] != vae:
            raise ValueError(f"{cache_dir} holds latents of VAE {meta['vae']!r}, not {vae!r}")
        for side, dirs in (("source", source_dirs), ("target", target_dirs)):
            if dirs is not None and _resolved(dirs) != meta[f"{side}_dirs"]:
                raise ValueError(
                    f"{cache_dir} was built from {side} {meta[f'{side}_dirs']}, not {dirs}; "
                    "rebuild it"
                )
        self.pairs, self.variants, self.factor = meta["pairs"], meta["variants"], meta["factor"]
        if crop is not None and crop % self.factor:
            raise ValueError(f"crop must be a multiple of {self.factor}, got {crop}")
        self.crop = None if crop is None else crop // self.factor
        self.image = image
        self._latents = None  # opened in each worker
        logger.info(f"{len(self.pairs)} cached pairs x {self.variants} variants from {cache_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        if self._latents is None:
            self._latents = np.load(self.cache_dir / LATENTS, mmap_mode="r")
        k = int(torch.randint(self.variants, ()))
        h, w = self._latents.shape[-2:]
        size = self.crop or min(h, w)
        y, x = int(torch.randint(h - size + 1, ())), int(torch.randint(w - size + 1, ()))
        window = torch.from_numpy(np.array(self._latents[i, k, :, :, y : y + size, x : x + size]))
        sample = {"source": window[0].float(), "target": window[1].float()}
        if self.image is not None:
            tile = dihedral(read_image(self.pairs[i][0 if self.image == "source" else 1]), k)
            f = self.factor
            sample["image"] = to_model_range(
                tile[:, y * f : (y + size) * f, x * f : (x + size) * f]
            )
        return sample


def _resolved(dirs: list[str | Path]) -> list[str]:
    return [str(Path(d).resolve()) for d in dirs]


def cache_dir_for(cfg: dict) -> Path:
    """Where ``cfg``'s training latents are cached: ``data.cache_dir``, else a
    directory under ``$XDG_CACHE_HOME`` (``~/.cache``) ``/pathotrain/latents``
    named after a fingerprint of everything the latents depend on."""
    data = cfg["data"]
    if data.get("cache_dir"):
        return Path(data["cache_dir"])
    train = data["train"]
    source = by_stem(train["source"], IMAGE_SUFFIXES)
    target = by_stem(train["target"], IMAGE_SUFFIXES)
    key = {
        "vae": cfg["model"].get("vae", DEFAULT_VAE),
        "source_dirs": _resolved(train["source"]),
        "target_dirs": _resolved(train["target"]),
        "pairs": sorted(source.keys() & target.keys()),
        "variants": data.get("cache_variants", 2),
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "pathotrain" / "latents" / digest


def ensure_latent_cache(cfg: dict, batch_size: int = 4) -> Path:
    """Return :func:`cache_dir_for` ``cfg``, building the cache first unless it is
    complete. Under ``torchrun`` only rank 0 builds; the other ranks wait for it."""
    out_dir = cache_dir_for(cfg)
    if (out_dir / META).exists():
        logger.info(f"latent cache {out_dir} found")
        return out_dir
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank > 0:
        logger.info(f"waiting for rank 0 to build the latent cache {out_dir}")
        while not (out_dir / META).exists():
            time.sleep(10)
        return out_dir
    train = cfg["data"]["train"]
    logger.info(f"building the latent cache {out_dir}; later runs on this data reuse it")
    return build_latent_cache(
        train["source"],
        train["target"],
        out_dir,
        cfg["model"].get("vae", DEFAULT_VAE),
        cfg["data"].get("cache_variants", 2),
        batch_size,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "config", help="experiment YAML (its data.train, cache_variants, cache_dir)"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="tile pairs encoded at once")
    args = parser.parse_args(argv)
    print(ensure_latent_cache(load_config(args.config), args.batch_size))


if __name__ == "__main__":
    main()
