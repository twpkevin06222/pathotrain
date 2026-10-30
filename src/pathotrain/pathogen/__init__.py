"""Generative models for pathology: virtual staining from paired tiles (e.g. H&E to IHC).
Needs the ``pathogen`` extra (``pip install 'pathotrain[pathogen]'``)."""

from pathotrain.pathogen.data import PairedTileDataset, paired_transform
from pathotrain.pathogen.diffusion import LatentDiffusionTask, build_unet, build_vae

__all__ = [
    "LatentDiffusionTask",
    "PairedTileDataset",
    "build_unet",
    "build_vae",
    "paired_transform",
]
