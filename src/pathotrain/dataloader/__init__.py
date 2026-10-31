"""Datasets that feed whole-slide image tiles to PyTorch."""

from pathotrain.dataloader.base import find_pairs
from pathotrain.dataloader.classification import ClassTileDataset
from pathotrain.dataloader.detection import (
    DetectionDataset,
    DetectionTileFolderDataset,
    export_detection_tiles,
)
from pathotrain.dataloader.pathogen import PairedTileDataset, paired_transform
from pathotrain.dataloader.registration import export_registered_tiles
from pathotrain.dataloader.segmentation import (
    SegmentationDataset,
    SegmentationTileFolderDataset,
    export_segmentation_tiles,
)
from pathotrain.dataloader.unlabelled import (
    MultiCropTransform,
    UnlabelledSlideDataset,
    UnlabelledTileFolderDataset,
    multicrop_collate,
)

__all__ = [
    "ClassTileDataset",
    "DetectionDataset",
    "DetectionTileFolderDataset",
    "MultiCropTransform",
    "PairedTileDataset",
    "SegmentationDataset",
    "SegmentationTileFolderDataset",
    "UnlabelledSlideDataset",
    "UnlabelledTileFolderDataset",
    "export_detection_tiles",
    "export_registered_tiles",
    "export_segmentation_tiles",
    "find_pairs",
    "multicrop_collate",
    "paired_transform",
]
