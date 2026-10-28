"""Object detection; needs the ``detection`` extra (``pip install 'pathotrain[detection]'``)."""

from pathotrain.detection.models import RFDETR_SIZES, Detector, build_detector
from pathotrain.detection.task import DetectionTask, collate

__all__ = ["RFDETR_SIZES", "DetectionTask", "Detector", "build_detector", "collate"]
