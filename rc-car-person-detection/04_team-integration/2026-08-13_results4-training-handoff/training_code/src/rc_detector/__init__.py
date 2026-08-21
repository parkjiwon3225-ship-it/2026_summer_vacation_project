"""Custom RC-car person detector package."""

from .dataset import PersonDetectionDataset, detection_collate
from .backbone import DSResidualBlock, LightweightBackbone, count_trainable_parameters
from .fpn import LightweightFPN
from .head import AnchorFreeHead
from .model import PersonDetector
from .assignment import AnchorFreeTargetAssigner
from .losses import DetectionLoss, decode_distances, generalized_iou_loss
from .inference import DetectionPostProcessor, pure_torch_nms
from .metrics import DetectionEvaluator, pairwise_iou

__all__ = [
    "DSResidualBlock",
    "AnchorFreeHead",
    "AnchorFreeTargetAssigner",
    "DetectionLoss",
    "DetectionPostProcessor",
    "DetectionEvaluator",
    "LightweightBackbone",
    "LightweightFPN",
    "PersonDetectionDataset",
    "PersonDetector",
    "count_trainable_parameters",
    "decode_distances",
    "detection_collate",
    "generalized_iou_loss",
    "pure_torch_nms",
    "pairwise_iou",
]
