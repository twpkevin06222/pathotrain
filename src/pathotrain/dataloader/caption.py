"""Image-caption pairs for contrastive image-text training, e.g.
:mod:`pathotrain.pathoclip`. One JSONL file per split::

    {"image": "images/3f2a....jpg", "caption": "Lung adenocarcinoma, acinar pattern.", "group": "fig12"}

``image`` is relative to ``root`` (default the JSONL's folder). ``group``
(optional, default the caption) names the pairs that share one caption, e.g. the
panels of one figure: retrieval counts any of them as a match. A sample is the
row with ``image`` as an RGB ``PIL.Image`` and ``group`` as an int index.

ARCH (Gamper and Rajpoot, CVPR 2021; CC BY-NC-SA 4.0, research use only;
``books_set.zip`` and ``pubmed_set.zip`` from
https://warwick.ac.uk/fac/cross_fac/tia/data/arch, unzipped into one folder)
converts to this layout with::

    python -m pathotrain.dataloader.caption arch ~/data/arch ~/data/arch
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from loguru import logger
from PIL import Image
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    """Rows of a caption JSONL file (see the module docstring).

    Raises:
        ValueError: An empty file, or a row without ``image`` or ``caption``.
    """

    def __init__(self, jsonl: str | Path, root: str | Path | None = None) -> None:
        jsonl = Path(jsonl).expanduser()
        base = Path(root).expanduser() if root is not None else jsonl.parent
        self.rows, groups = [], {}
        with open(jsonl) as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = {"image", "caption"} - row.keys()
                if missing:
                    raise ValueError(f"{jsonl}:{n} lacks {sorted(missing)}")
                group = groups.setdefault(row.get("group", row["caption"]), len(groups))
                self.rows.append({**row, "image": base / row["image"], "group": group})
        if not self.rows:
            raise ValueError(f"no rows in {jsonl}")
        logger.info(f"{len(self.rows)} images, {len(groups)} captions in {jsonl}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with Image.open(row["image"]) as im:
            return {**row, "image": im.convert("RGB")}


ARCH_SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}


def export_arch(arch_dir: str | Path, out_dir: str | Path, seed: int = 0) -> dict:
    """Write ``train.jsonl``, ``val.jsonl`` and ``test.jsonl`` under ``out_dir``
    from the unzipped ``books_set/`` and / or ``pubmed_set/`` in ``arch_dir``
    (ARCH has no official splits). Rows keep ``source`` (the subset); a figure's
    panels share a ``group`` and so a split, drawn from a hash of the group and
    ``seed`` in the :data:`ARCH_SPLITS` shares. Images stay where they are
    (``image`` is relative to ``out_dir``); captions whose image is missing are
    skipped with a warning (ARCH's counts differ from its paper's). Returns the
    rows and groups per split."""
    arch, out = Path(arch_dir), Path(out_dir)
    rows = []
    for subset in ("books_set", "pubmed_set"):
        folder = arch / subset
        if not (folder / "captions.json").is_file():
            continue
        images = {p.stem: p for p in (folder / "images").iterdir()}
        entries = json.loads((folder / "captions.json").read_text()).values()
        missing = [e["uuid"] for e in entries if e["uuid"] not in images]
        if missing:
            logger.warning(f"{subset}: {len(missing)} captions have no image, skipped")
        for e in entries:
            if e["uuid"] in images:
                # Books group a figure's panels by figure_id; a PubMed caption is its own group.
                group = f"{subset}:{e.get('figure_id', e['caption'])}"
                image = os.path.relpath(images[e["uuid"]], out)
                rows.append(
                    {"image": image, "caption": e["caption"], "group": group, "source": subset}
                )
    if not rows:
        raise ValueError(f"no books_set/ or pubmed_set/ with captions.json in {arch}")
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, rows_ in _split(rows, seed).items():
        with open(out / f"{split}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows_)
        stats[split] = {"images": len(rows_), "groups": len({r["group"] for r in rows_})}
    return stats


def _split(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    bounds, total = [], 0.0
    for split, share in ARCH_SPLITS.items():
        total += share
        bounds.append((total, split))
    bounds[-1] = (float("inf"), bounds[-1][1])  # the shares may sum to just under 1
    out = {split: [] for split in ARCH_SPLITS}
    for r in rows:
        digest = hashlib.sha1(f"{seed}:{r['group']}".encode()).digest()
        u = int.from_bytes(digest[:8], "big") / 2**64
        out[next(split for bound, split in bounds if u < bound)].append(r)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert a caption dataset to caption JSONL.")
    parser.add_argument("dataset", choices=["arch"])
    parser.add_argument("source", help="folder with the unzipped books_set/ and pubmed_set/")
    parser.add_argument("out_dir", help="folder for train.jsonl, val.jsonl, test.jsonl")
    parser.add_argument("--seed", type=int, default=0, help="seed of the split")
    args = parser.parse_args(argv)
    print(json.dumps(export_arch(args.source, args.out_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
