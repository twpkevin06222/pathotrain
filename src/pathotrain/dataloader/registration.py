"""Paired tiles from a registered slide pair, for :class:`PairedTileDataset`.

:func:`export_registered_tiles` tiles the fixed slide (the H&E) with patchology
(tissue detection, artifact removal, ICC calibration) and, for each tile, reads
the moving slide's (e.g. IHC) matching window through the registration's affine
map, warped onto the H&E tile's pixel grid. Run
:func:`pathotrain.registration.affine.register` first.

Layout, ready for ``data.<split>`` of a pathogen config::

    out_dir/source/<fixed slide>_<x>_<y>.png   the H&E tile
    out_dir/target/<fixed slide>_<x>_<y>.png   the moving slide over the same tissue
    out_dir/tiles.csv                          name, x, y (fixed level-0)

Example::

    from patchology import TilingConfig
    from pathotrain.registration import register

    reg = register("HE/0_HE.tiff", "IHC/0_KI67.tiff")
    export_registered_tiles(reg, "tiles/train", tiling=TilingConfig(tile_size=1024, tile_mpp=1.0))
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from patchology import TileGenerator, open_slide
from patchology.reader import SlideReader
from pathotrain.registration.affine import Registration, tissue_pixels


def read_registered(
    slide: SlideReader,
    registration: Registration,
    location: tuple[int, int],
    downsample: float,
    size: int,
) -> np.ndarray:
    """Read the moving ``slide`` over the ``size`` px fixed tile at level-0
    ``location`` and ``downsample``, warped onto that tile's grid: ``(size, size, 3)``
    uint8 RGB, white outside the moving slide."""
    a = np.vstack([registration.matrix, [0, 0, 1]])
    # tile pixel -> fixed level 0 -> moving level 0
    tile = a @ np.array([[downsample, 0, location[0]], [0, downsample, location[1]], [0, 0, 1]])
    corners = np.array([[0, 0, 1], [size, 0, 1], [0, size, 1], [size, size, 1]], float) @ tile.T
    m_downsample = downsample * np.sqrt(abs(np.linalg.det(a[:2, :2])))
    x0, y0 = np.floor(corners[:, :2].min(0)).astype(int) - 2 * int(np.ceil(m_downsample))
    x1, y1 = np.ceil(corners[:, :2].max(0)).astype(int) + 2 * int(np.ceil(m_downsample))
    read = (max(1, round((x1 - x0) / m_downsample)), max(1, round((y1 - y0) / m_downsample)))
    region = slide.read_region_scaled((int(x0), int(y0)), m_downsample, read)
    # tile pixel -> region pixel, as warpAffine's inverse map
    to_region = np.diag([read[0] / (x1 - x0), read[1] / (y1 - y0), 1.0]) @ np.array(
        [[1, 0, -x0], [0, 1, -y0], [0, 0, 1]]
    )
    m = (to_region @ tile)[:2]
    return cv2.warpAffine(
        region,
        m,
        (size, size),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderValue=(255, 255, 255),
    )


def glass_colour(slide: SlideReader) -> np.ndarray:
    """The slide's glass RGB: the 90th percentile of a ~1024 px thumbnail."""
    thumbnail = slide.read_at_downsample(max(slide.dimensions) / 1024)
    return np.percentile(thumbnail.reshape(-1, 3).astype(np.float32), 90, axis=0)


def export_registered_tiles(
    registration: Registration | str | Path,
    out_dir: str | Path,
    fmt: str = "png",
    min_target_tissue: float = 0.1,
    **generator_kwargs,
) -> Path:
    """Write the paired tiles of ``registration`` (or its saved JSON) under
    ``out_dir`` and return it. Keyword arguments go to
    :class:`patchology.TileGenerator` for the fixed slide (``tiling``, ``tissue``,
    ``artifacts``, ``color``, ...); the moving tile is ICC calibrated when the
    fixed one is, never stain normalised.

    A pair is dropped when less than ``min_target_tissue`` of its moving tile is
    tissue (as :func:`pathotrain.registration.affine.tissue_pixels` sees it,
    against the moving slide's glass colour), e.g. where that section tore, folded
    away or lies off the slide. 0 keeps every pair."""
    if not isinstance(registration, Registration):
        registration = Registration.load(registration)
    root = Path(out_dir)
    source, target = root / "source", root / "target"
    source.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    stem = Path(registration.fixed).stem
    rows = []
    dropped = 0
    with (
        TileGenerator(registration.fixed, **generator_kwargs) as gen,
        open_slide(registration.moving) as moving,
    ):
        result = gen.result
        size = result.config.tile_size
        glass = glass_colour(moving)
        for (x, y), tile in gen:
            other = read_registered(moving, registration, (x, y), result.downsample, size)
            # Before ICC, so the tile and the glass colour share a colour space.
            fraction = float(tissue_pixels(other, glass)[1].mean())
            if fraction < min_target_tissue:
                dropped += 1
                continue
            if result.color.icc:
                other = moving.apply_icc(other)
            name = f"{stem}_{x}_{y}"
            Image.fromarray(tile).save(source / f"{name}.{fmt}")
            Image.fromarray(other).save(target / f"{name}.{fmt}")
            rows.append((name, x, y, round(fraction, 4)))
    with open(root / "tiles.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("name", "x", "y", "target_tissue"))
        writer.writerows(rows)
    logger.info(
        f"{stem}: wrote {len(rows)} tile pairs to {root}, dropped {dropped} with "
        f"< {min_target_tissue:.0%} tissue in the target"
    )
    return root
