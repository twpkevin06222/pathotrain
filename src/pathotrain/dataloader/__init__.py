"""Datasets that feed whole-slide image tiles to PyTorch."""

from pathotrain.dataloader.base import find_pairs
from pathotrain.dataloader.detection import (
    DetectionDataset,
    DetectionTileFolderDataset,
    export_detection_tiles,
)
from pathotrain.dataloader.segmentation import (
    SegmentationDataset,
    TileFolderDataset,
    export_tiles,
)

__all__ = [
    "DetectionDataset",
    "DetectionTileFolderDataset",
    "SegmentationDataset",
    "TileFolderDataset",
    "export_detection_tiles",
    "export_tiles",
    "find_pairs",
]
