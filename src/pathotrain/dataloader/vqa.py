"""Image-question-answer rows for visual question answering, e.g. the VLM
fine-tuning of :mod:`pathotrain.pathovlm`. One JSONL file per split::

    {"image": "images/3f2a....jpg", "question": "is this a carcinoma?", "answer": "yes", "type": "closed"}

``image`` is relative to ``root`` (default the JSONL's folder). ``type`` is
``closed`` (yes / no) or ``open``; when absent it is ``closed`` for a yes / no
answer. A sample is a dict of the row with ``image`` as an RGB ``PIL.Image``
(what ``transform`` makes of it, with one): processors of VLMs take PIL images,
and prompt building belongs to the model, not the data.

PathVQA (`flaviagiammarino/path-vqa <https://huggingface.co/datasets/flaviagiammarino/path-vqa>`_,
MIT) converts to this layout with::

    python -m pathotrain.dataloader.vqa ~/data/pathvqa
"""

import argparse
import hashlib
import io
import json
from collections.abc import Callable, Iterable
from pathlib import Path

from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

TYPES = ("closed", "open")
PATHVQA = ("flaviagiammarino/path-vqa", "1685832883334b5bb5beaf4e4b333fdeecaa4ad9")


def answer_type(answer: str) -> str:
    """``closed`` for a yes / no answer, else ``open``."""
    return "closed" if answer.strip().lower() in ("yes", "no") else "open"


class VQADataset(Dataset):
    """Rows of a VQA JSONL file (see the module docstring).

    Raises:
        ValueError: An empty file, a row without ``image``, ``question`` or
            ``answer``, or a ``type`` other than closed / open.
    """

    def __init__(
        self, jsonl: str | Path, root: str | Path | None = None, transform: Callable | None = None
    ) -> None:
        base = Path(root) if root is not None else Path(jsonl).parent
        self.rows = []
        with open(jsonl) as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = {"image", "question", "answer"} - row.keys()
                if missing:
                    raise ValueError(f"{jsonl}:{n} lacks {sorted(missing)}")
                row.setdefault("type", answer_type(row["answer"]))
                if row["type"] not in TYPES:
                    raise ValueError(
                        f"{jsonl}:{n} type must be one of {TYPES}, got {row['type']!r}"
                    )
                self.rows.append({**row, "image": base / row["image"]})
        if not self.rows:
            raise ValueError(f"no rows in {jsonl}")
        self.transform = transform
        closed = sum(r["type"] == "closed" for r in self.rows)
        logger.info(f"{len(self.rows)} questions ({closed} closed) in {jsonl}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with Image.open(row["image"]) as im:
            image = im.convert("RGB")
        return {**row, "image": image if self.transform is None else self.transform(image)}


def export_vqa_parquet(splits: dict[str, Iterable[str | Path]], out_dir: str | Path) -> dict:
    """Write ``<split>.jsonl`` files and ``images/`` under ``out_dir`` from parquet
    files with ``image`` (``{"bytes": ...}``), ``question`` and ``answer`` columns
    (the Hugging Face ``image`` feature). Each image is stored once, named by the
    hash of its bytes, so repeats within and across splits share one file.

    Returns per split the question and image counts, and the images it shares
    with each other split (a warning is logged for any: leakage between splits).
    """
    import pyarrow.parquet as pq  # installed with mlflow

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    seen: dict[str, set[str]] = {}
    stats: dict = {}
    for split, files in splits.items():
        names, questions = set(), 0
        with open(out / f"{split}.jsonl", "w") as f:
            for file in files:
                for batch in pq.ParquetFile(file).iter_batches(batch_size=256):
                    for row in batch.to_pylist():
                        data = row["image"]["bytes"]
                        name = _save_image(data, out / "images")
                        names.add(name)
                        answer = row["answer"]
                        line = {
                            "image": f"images/{name}",
                            "question": row["question"],
                            "answer": answer,
                            "type": answer_type(answer),
                        }
                        f.write(json.dumps(line) + "\n")
                        questions += 1
        seen[split] = names
        stats[split] = {"questions": questions, "images": len(names)}
    for split, names in seen.items():
        shared = {other: len(names & seen[other]) for other in seen if other != split}
        stats[split]["shared_images"] = shared
        for other, n in shared.items():
            if n:
                logger.warning(f"{n} of {split}'s {len(names)} images are also in {other}")
    return stats


def _save_image(data: bytes, folder: Path) -> str:
    with Image.open(io.BytesIO(data)) as im:
        suffix = {"JPEG": ".jpg", "PNG": ".png"}.get(im.format, ".png")
    name = hashlib.sha1(data).hexdigest()[:20] + suffix
    path = folder / name
    if not path.exists():
        path.write_bytes(data)
    return name


def export_pathvqa(out_dir: str | Path) -> dict:
    """Download PathVQA (pinned revision) and write it as VQA JSONL under
    ``out_dir``: ``train.jsonl``, ``validation.jsonl``, ``test.jsonl``, ``images/``."""
    from huggingface_hub import snapshot_download

    repo, revision = PATHVQA
    local = Path(
        snapshot_download(
            repo, repo_type="dataset", revision=revision, allow_patterns=["data/*.parquet"]
        )
    )
    splits = {
        s: sorted((local / "data").glob(f"{s}-*.parquet")) for s in ("train", "validation", "test")
    }
    return export_vqa_parquet(splits, out_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert PathVQA to VQA JSONL + images.")
    parser.add_argument("out_dir", help="folder for <split>.jsonl and images/")
    args = parser.parse_args(argv)
    print(json.dumps(export_pathvqa(args.out_dir), indent=2))


if __name__ == "__main__":
    main()
