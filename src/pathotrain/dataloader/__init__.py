"""Datasets that feed whole-slide image tiles to PyTorch."""

from pathotrain.dataloader.base import find_pairs
from pathotrain.dataloader.detection import (
    DetectionDataset,
    DetectionTileFolderDataset,
    export_detection_tiles,
)
from pathotrain.dataloader.segmentation import (
    SegmentationDataset,
    SegmentationTileFolderDataset,
    export_segmentation_tiles,
)

__all__ = [
    "DetectionDataset",
    "DetectionTileFolderDataset",
    "SegmentationDataset",
    "SegmentationTileFolderDataset",
    "export_detection_tiles",
    "export_segmentation_tiles",
    "find_pairs",
]
