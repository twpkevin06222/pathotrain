"""Detection datasets built from directories of slides and GeoJSON annotations.

A slide is paired with the GeoJSON annotation of the same stem, as for
segmentation (see :func:`~pathotrain.dataloader.find_pairs`). Each annotated
object becomes the bounding box of its polygon.

Two ways to feed a PyTorch ``DataLoader``, as for segmentation:

* Lazy: :class:`DetectionDataset` tiles every slide in ``__init__`` and reads
  each ``(tile, boxes)`` from the slide on demand.
* Offline: :func:`export_detection_tiles` writes tiles and a COCO
  ``_annotations.coco.json`` to ``out_dir/<split>/`` (Roboflow's layout, which
  RF-DETR's own trainer also reads), and :class:`DetectionTileFolderDataset`
  reads them back.

A sample is ``(image, target)``: a ``(3, H, W)`` float32 tensor in ``[0, 1]``
and ``{"boxes": (N, 4) float32 [x1, y1, x2, y2] pixels, "labels": (N,) int64}``,
as :class:`~pathotrain.detection.DetectionTask` takes, or whatever
``transform(tile, target)`` returns, given the ``(H, W, 3)`` uint8 tile and the
COCO ``[x, y, w, h]`` target (use ``train_transform(boxes=True)``). Batch with
:func:`pathotrain.detection.collate`.

Example::

    kwargs = dict(
        class_map={"class01": 0, "class02": 1},
        roi_class="roi",
        min_box_visibility=0.6,
        min_objects=1,
        tiling=TilingConfig(tile_size=512, tile_mpp=0.5),
    )
    train = DetectionDataset(slides, annotations, transform=train_transform(boxes=True), **kwargs)
    loader = DataLoader(train, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate)

    export_detection_tiles(slides, annotations, "tiles", "train", **kwargs)
    train = DetectionTileFolderDataset("tiles/train", transform=train_transform(boxes=True))
"""

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

from patchology import DetectionTileGenerator
from pathotrain.dataloader.base import SlideDataset, find_pairs, to_image_tensor

COCO_FILE = "_annotations.coco.json"  # Roboflow's name, which RF-DETR looks for

Target = dict[str, np.ndarray]


def _check_class_map(class_map: Mapping[str, int]) -> None:
    labels = set(class_map.values())
    if not labels or labels != set(range(len(labels))):
        raise ValueError(f"class_map labels must be 0 .. num_classes - 1, got {sorted(labels)}")


def _sample(tile: np.ndarray, target: Target, transform: Callable | None) -> object:
    if transform is not None:
        return transform(tile, target)
    xywh = torch.from_numpy(target["boxes"]).float().reshape(-1, 4)
    boxes = torch.cat([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], dim=1)
    return to_image_tensor(tile), {
        "boxes": boxes,
        "labels": torch.from_numpy(target["labels"]).long(),
    }


class DetectionDataset(SlideDataset):
    """Lazy ``(image, target)`` tiles of every slide in ``slide_dirs`` with a GeoJSON annotation.

    Each slide is tiled once in ``__init__`` (tissue, ROI and box selection);
    tiles and boxes are read from the slide per item. Keyword arguments go to
    :class:`patchology.DetectionTileGenerator` (``min_box_visibility``,
    ``min_objects``, ``roi_class``, ``tiling``, ...). Samples are as described
    in the module docstring.

    Args:
        class_map: Annotation class name to label, with labels ``0 ..
            num_classes - 1`` (several names may share one), e.g.
            ``{"class01": 0, "class02": 1}``. Objects of other classes are left out.
    """

    generator_cls = DetectionTileGenerator

    def __init__(
        self,
        slide_dirs: Sequence[str | Path],
        annotation_dirs: Sequence[str | Path],
        class_map: Mapping[str, int],
        transform: Callable | None = None,
        **generator_kwargs,
    ) -> None:
        _check_class_map(class_map)
        super().__init__(
            slide_dirs, annotation_dirs, transform, class_map=class_map, **generator_kwargs
        )

    def __getitem__(self, index: int) -> object:
        gen, i = self.locate(index)
        _, tile, target = gen.get_tile(i)
        return _sample(tile, target, self.transform)


def export_detection_tiles(
    slide_dirs: Sequence[str | Path],
    annotation_dirs: Sequence[str | Path],
    out_dir: str | Path,
    split: str,
    class_map: Mapping[str, int],
    fmt: str = "png",
    **generator_kwargs,
) -> Path:
    """Write every annotated slide's tiles and their boxes under ``out_dir/split`` and return it.

    Layout: ``<slide>_<x>_<y>.<fmt>`` tiles and one COCO ``_annotations.coco.json``
    (boxes ``[x, y, w, h]`` in tile pixels, ``category_id = label + 1``, and each
    image's ``slide``, ``x`` and ``y`` in level-0 pixels). This is Roboflow's
    layout: name the splits ``train``, ``valid`` and ``test`` to train with
    RF-DETR's own trainer or other COCO tools as well. Keyword arguments are as
    for :class:`DetectionDataset`.
    """
    _check_class_map(class_map)
    root = Path(out_dir) / split
    root.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for slide, ground_truth in find_pairs(slide_dirs, annotation_dirs):
        with DetectionTileGenerator(
            slide, ground_truth, class_map=class_map, **generator_kwargs
        ) as gen:
            for (x, y), tile, target in gen:
                name = f"{slide.stem}_{x}_{y}.{fmt}"
                Image.fromarray(tile).save(root / name)
                image_id = len(images) + 1
                h, w = tile.shape[:2]
                image = {"id": image_id, "file_name": name, "width": w, "height": h}
                images.append({**image, "slide": slide.name, "x": x, "y": y})
                for box, label in zip(target["boxes"].tolist(), target["labels"].tolist()):
                    box = [round(v, 2) for v in box]
                    annotations.append(
                        {
                            "id": len(annotations) + 1,
                            "image_id": image_id,
                            "category_id": label + 1,
                            "bbox": box,
                            "area": round(box[2] * box[3], 2),
                            "iscrowd": 0,
                        }
                    )
        logger.info("{}: wrote {} tiles to {}", slide.name, len(gen), root)
    names: dict[int, list[str]] = {}
    for class_name, label in class_map.items():
        names.setdefault(label, []).append(class_name)
    categories = [
        {
            "id": label + 1,
            "name": " / ".join(names[label]),
            "supercategory": " / ".join(names[label]),
        }
        for label in sorted(names)
    ]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    (root / COCO_FILE).write_text(json.dumps(coco))
    return root


class DetectionTileFolderDataset(Dataset):
    """``(image, target)`` tiles from a COCO folder: ``root/_annotations.coco.json``
    with image paths relative to ``root``, as written by :func:`export_detection_tiles`.

    Any COCO detection folder in this layout works (e.g. a Roboflow export).
    Labels are the categories' positions in id order (``0 .. num_classes - 1``);
    the names are in :attr:`class_names`. ``iscrowd`` annotations are left out.
    Samples are as described in the module docstring.
    """

    def __init__(self, root: str | Path, transform: Callable | None = None) -> None:
        self.root = Path(root)
        self.transform = transform
        coco = json.loads((self.root / COCO_FILE).read_text())
        categories = sorted(coco["categories"], key=lambda c: c["id"])
        label_of = {c["id"]: i for i, c in enumerate(categories)}
        self.class_names = [c["name"] for c in categories]
        objects: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            if not a.get("iscrowd", 0):
                objects.setdefault(a["image_id"], []).append(a)
        self.files, self.targets = [], []
        for image in coco["images"]:
            anns = objects.get(image["id"], [])
            self.files.append(self.root / image["file_name"])
            self.targets.append(
                {
                    "boxes": np.array([a["bbox"] for a in anns], np.float32).reshape(-1, 4),
                    "labels": np.array([label_of[a["category_id"]] for a in anns], np.int64),
                }
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> object:
        tile = np.asarray(Image.open(self.files[index]).convert("RGB"))
        return _sample(tile, self.targets[index], self.transform)
