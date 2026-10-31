"""Labelled tiles for classification, e.g. the linear probe of
:mod:`pathotrain.histossl`. Two layouts::

    root/<class>/<tile>.png             ClassTileDataset("crc/train")
    labels.csv: path,label rows         ClassTileDataset(csv="crc/val.csv")

A sample is ``(image, label)``: what ``transform`` makes of the ``(3, H, W)``
uint8 ``tv_tensors.Image`` tile (a float32 tensor in ``[0, 1]`` without one), and
the class index into ``class_names``. Pass the train split's ``class_names`` to
the other splits so the indices agree even when a split lacks a class.
"""

import csv
from collections.abc import Callable, Sequence
from pathlib import Path

from loguru import logger
from torch.utils.data import Dataset

from pathotrain.dataloader.pathogen import IMAGE_SUFFIXES, read_image


class ClassTileDataset(Dataset):
    """Tiles with a class label, from class folders or a CSV.

    Args:
        root: Folder of one sub-folder per class (torchvision's ImageFolder
            layout); with ``csv``, the folder its paths are relative to (default
            the CSV's folder).
        csv: A file with ``path`` and ``label`` columns.
        class_names: The label order; default the sorted labels found.

    Raises:
        ValueError: Neither ``root`` nor ``csv``, no tiles, or a label outside
            ``class_names``.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        transform: Callable | None = None,
        *,
        csv: str | Path | None = None,
        class_names: Sequence[str] | None = None,
    ) -> None:
        if root is None and csv is None:
            raise ValueError("pass root (class folders) or csv")
        if csv is not None:
            base = Path(root) if root is not None else Path(csv).parent
            with open(csv, newline="") as f:
                items = [(base / r["path"], r["label"]) for r in _csv_rows(f)]
        else:
            folders = sorted(d for d in Path(root).iterdir() if d.is_dir())
            items = [
                (p, d.name)
                for d in folders
                for p in sorted(d.iterdir())
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            ]
        if not items:
            raise ValueError(f"no labelled tiles in {csv if csv is not None else root}")
        self.class_names = list(class_names or sorted({label for _, label in items}))
        index = {name: i for i, name in enumerate(self.class_names)}
        unknown = sorted({label for _, label in items} - index.keys())
        if unknown:
            raise ValueError(f"labels {unknown} are not in class_names {self.class_names}")
        self.samples = [(path, index[label]) for path, label in items]
        self.transform = transform
        logger.info(f"{len(self.samples)} tiles in {len(self.class_names)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, int]:
        path, label = self.samples[index]
        tile = read_image(path)
        return (tile.float() / 255 if self.transform is None else self.transform(tile)), label


def _csv_rows(f) -> list[dict[str, str]]:
    reader = csv.DictReader(f)
    if not {"path", "label"} <= set(reader.fieldnames or ()):
        raise ValueError(f"the CSV needs path and label columns, has {reader.fieldnames}")
    return list(reader)
