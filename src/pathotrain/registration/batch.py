"""Register and tile every slide pair of two folders::

    python -m pathotrain.registration.batch he_slides/ ihc_slides/ out/ --tile-size 1024 --tile-mpp 1.0

Slides are paired by a key taken from the file stem with ``--key``, a regular
expression (default ``^[^_]+``, the text before the first ``_``: ``0_HE_val.tiff``
pairs with ``0_KI67_val.tiff``). Unpaired slides are skipped with a warning.

Writes::

    out/registrations/<H&E stem>.json   the registration (see Registration)
    out/registrations/<H&E stem>.jpg    its overlay, to check by eye
    out/source/, out/target/, out/tiles/ the tile pairs (see export_registered_tiles)
    out/registrations.csv               fixed, moving, ncc, status, tiles per pair

Pairs whose NCC is below ``--min-ncc`` (0.55, :func:`register`'s warning
threshold) are registered but not tiled, since a misaligned pair makes wrong
training pairs. Check their overlays; to tile one anyway, rerun with a lower
``--min-ncc``. Reruns reuse saved registrations and skip slides already tiled,
so an interrupted batch picks up where it stopped. Splits are up to you: run
once per split folder, or move slides between splits by patient beforehand.
"""

import argparse
import csv
import re
from pathlib import Path

import cv2
from loguru import logger

from patchology import TilingConfig
from pathotrain.dataloader.base import by_stem
from pathotrain.dataloader.registration import export_registered_tiles, tiles_csv
from pathotrain.registration.affine import Registration, overlay, register


def pair_slides(
    fixed_dir: str | Path, moving_dir: str | Path, key: str = r"^[^_]+"
) -> list[tuple[Path, Path]]:
    """``(fixed, moving)`` slides of two folders whose stems share the ``key`` match.

    Raises:
        ValueError: Two slides of one folder share a key, or no pair matches.
    """

    def keyed(d: str | Path) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for path in by_stem([d], None).values():
            match = re.search(key, path.stem)
            if match is None:
                logger.warning(f"{path.name} has no key matching {key!r}, skipped")
                continue
            if match.group(0) in out:
                raise ValueError(
                    f"{out[match.group(0)].name} and {path.name} share key {match.group(0)!r}"
                )
            out[match.group(0)] = path
        return out

    fixed, moving = keyed(fixed_dir), keyed(moving_dir)
    for k in sorted(fixed.keys() ^ moving.keys()):
        logger.warning(f"{(fixed | moving)[k].name} has no partner, skipped")
    pairs = [(fixed[k], moving[k]) for k in sorted(fixed.keys() & moving.keys())]
    if not pairs:
        raise ValueError(f"no slide in {fixed_dir} pairs with one in {moving_dir} by {key!r}")
    return pairs


def run(
    pairs: list[tuple[Path, Path]],
    out_dir: str | Path,
    min_ncc: float = 0.55,
    mpp: float = 16.0,
    **export_kwargs,
) -> list[dict]:
    """Register, check and tile each pair into ``out_dir``; return one summary row per
    pair (also written to ``out_dir/registrations.csv``). ``export_kwargs`` go to
    :func:`export_registered_tiles`."""
    out = Path(out_dir)
    registrations = out / "registrations"
    registrations.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (fixed, moving) in enumerate(pairs, 1):
        logger.info(f"pair {i}/{len(pairs)}: {fixed.name} + {moving.name}")
        saved = registrations / f"{fixed.stem}.json"
        if saved.exists() and Registration.load(saved).moving == str(moving):
            reg = Registration.load(saved)
        else:
            reg = register(fixed, moving, mpp, min_ncc=min_ncc)
            cv2.imwrite(str(saved.with_suffix(".jpg")), overlay(reg, mpp)[:, :, ::-1])
            reg.save(saved)  # after the overlay: a saved JSON means a finished pair
        row = {"fixed": str(fixed), "moving": str(moving), "ncc": round(reg.ncc, 4)}
        if reg.ncc < min_ncc:
            rows.append({**row, "status": "skipped: low ncc", "tiles": 0})
            continue
        listing = tiles_csv(out, reg)
        if not listing.exists():
            export_registered_tiles(reg, out, **export_kwargs)
        with open(listing) as f:
            rows.append({**row, "status": "tiled", "tiles": sum(1 for _ in f) - 1})
    with open(out / "registrations.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, ["fixed", "moving", "ncc", "status", "tiles"])
        writer.writeheader()
        writer.writerows(rows)
    skipped = [r for r in rows if r["status"] != "tiled"]
    logger.info(
        f"{len(rows) - len(skipped)} pairs tiled ({sum(r['tiles'] for r in rows)} tile pairs), "
        f"{len(skipped)} skipped for NCC < {min_ncc}; see {out / 'registrations.csv'}"
    )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("fixed_dir", help="folder of H&E slides")
    parser.add_argument("moving_dir", help="folder of slides to align to them, e.g. IHC")
    parser.add_argument("out_dir")
    parser.add_argument("--key", default=r"^[^_]+", help="regex on the stem that pairs slides")
    parser.add_argument("--tile-size", type=int, default=1024, help="tile side in pixels")
    parser.add_argument("--tile-mpp", type=float, default=1.0, help="tile microns per pixel")
    parser.add_argument(
        "--tissue-overlap", type=float, default=0.1, help="min H&E tissue fraction per tile"
    )
    parser.add_argument(
        "--min-target-tissue", type=float, default=0.1, help="min IHC tissue fraction per tile"
    )
    parser.add_argument("--min-ncc", type=float, default=0.55, help="tile pairs at or above this")
    parser.add_argument("--mpp", type=float, default=16.0, help="registration resolution")
    parser.add_argument("--fmt", default="png", help="tile image format")
    args = parser.parse_args(argv)
    tiling = TilingConfig(
        tile_size=args.tile_size, tile_mpp=args.tile_mpp, tissue_overlap=args.tissue_overlap
    )
    run(
        pair_slides(args.fixed_dir, args.moving_dir, args.key),
        args.out_dir,
        args.min_ncc,
        args.mpp,
        fmt=args.fmt,
        min_target_tissue=args.min_target_tissue,
        tiling=tiling,
    )


if __name__ == "__main__":
    main()
