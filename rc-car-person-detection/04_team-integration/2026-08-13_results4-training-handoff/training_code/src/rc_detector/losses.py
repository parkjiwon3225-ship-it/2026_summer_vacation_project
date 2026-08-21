from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from .assignment import AnchorFreeTargetAssigner


def decode_distances(points: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    """Decode point-relative l/t/r/b distances into xyxy boxes."""
    return torch.stack(
        (
            points[:, 0] - distances[:, 0],
            points[:, 1] - distances[:, 1],
            points[:, 0] + distances[:, 2],
            points[:, 1] + distances[:, 3],
        ),
        dim=1,
    )


def generalized_iou_loss(
    predicted_boxes: torch.Tensor, target_boxes: torch.Tensor
) -> torch.Tensor:
    predicted_width = (predicted_boxes[:, 2] - predicted_boxes[:, 0]).clamp_min(0)
    predicted_height = (predicted_boxes[:, 3] - predicted_boxes[:, 1]).clamp_min(0)
    target_width = (target_boxes[:, 2] - target_boxes[:, 0]).clamp_min(0)
    target_height = (target_boxes[:, 3] - target_boxes[:, 1]).clamp_min(0)
    predicted_area = predicted_width * predicted_height
    target_area = target_width * target_height

    intersection_left = torch.maximum(predicted_boxes[:, 0], target_boxes[:, 0])
    intersection_top = torch.maximum(predicted_boxes[:, 1], target_boxes[:, 1])
    intersection_right = torch.minimum(predicted_boxes[:, 2], target_boxes[:, 2])
    intersection_bottom = torch.minimum(predicted_boxes[:, 3], target_boxes[:, 3])
    intersection = (intersection_right - intersection_left).clamp_min(0) * (
        intersection_bottom - intersection_top
    ).clamp_min(0)
    union = predicted_area + target_area - intersection
    iou = intersection / union.clamp_min(1e-7)

    enclosing_left = torch.minimum(predicted_boxes[:, 0], target_boxes[:, 0])
    enclosing_top = torch.minimum(predicted_boxes[:, 1], target_boxes[:, 1])
    enclosing_right = torch.maximum(predicted_boxes[:, 2], target_boxes[:, 2])
    enclosing_bottom = torch.maximum(predicted_boxes[:, 3], target_boxes[:, 3])
    enclosing_area = (enclosing_right - enclosing_left).clamp_min(0) * (
        enclosing_bottom - enclosing_top
    ).clamp_min(0)
    generalized_iou = iou - (enclosing_area - union) / enclosing_area.clamp_min(1e-7)
    return 1.0 - generalized_iou


class DetectionLoss(nn.Module):
    def __init__(
        self,
        assigner: AnchorFreeTargetAssigner | None = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        box_weight: float = 2.0,
        quality_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.assigner = assigner or AnchorFreeTargetAssigner()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.box_weight = box_weight
        self.quality_weight = quality_weight

    def forward(
        self,
        predictions: dict[str, dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        first_prediction = next(iter(predictions.values()))["class_logits"]
        batch_size = first_prediction.shape[0]
        if len(targets) != batch_size:
            raise ValueError(f"Received {len(targets)} targets for batch size {batch_size}")
        feature_shapes = {
            level: tuple(output["class_logits"].shape[-2:])
            for level, output in predictions.items()
        }
        class_loss_sum = first_prediction.new_zeros(())
        quality_loss_sum = first_prediction.new_zeros(())
        box_loss_sum = first_prediction.new_zeros(())
        quality_weight_sum = first_prediction.new_zeros(())
        positive_count = 0

        for batch_index, target in enumerate(targets):
            boxes = target["boxes"].to(
                device=first_prediction.device, dtype=torch.float32
            )
            labels = target["labels"].to(device=first_prediction.device)
            assignments = self.assigner(feature_shapes, boxes, labels)
            for level, output in predictions.items():
                assigned = assignments[level]
                class_target = assigned["class_targets"]
                class_logits = output["class_logits"][batch_index]
                binary_cross_entropy = functional.binary_cross_entropy_with_logits(
                    class_logits, class_target, reduction="none"
                )
                probability = class_logits.sigmoid()
                probability_target = probability * (1.0 - class_target) + (
                    1.0 - probability
                ) * class_target
                alpha_target = self.focal_alpha * class_target + (
                    1.0 - self.focal_alpha
                ) * (1.0 - class_target)
                class_loss_sum = class_loss_sum + (
                    alpha_target
                    * probability_target.pow(self.focal_gamma)
                    * binary_cross_entropy
                ).sum()

                positive_mask = assigned["positive_mask"]
                level_positive_count = int(positive_mask.sum())
                positive_count += level_positive_count
                if not level_positive_count:
                    continue
                quality_target = assigned["quality_targets"][0][positive_mask]
                quality_logits = output["quality_logits"][batch_index, 0][positive_mask]
                quality_loss_sum = quality_loss_sum + functional.binary_cross_entropy_with_logits(
                    quality_logits, quality_target, reduction="sum"
                )

                points = assigned["points"][positive_mask]
                predicted_distances = output["distances"][batch_index].permute(1, 2, 0)[
                    positive_mask
                ]
                target_distances = assigned["distance_targets"].permute(1, 2, 0)[
                    positive_mask
                ]
                predicted_boxes = decode_distances(points, predicted_distances)
                target_boxes = decode_distances(points, target_distances)
                giou = generalized_iou_loss(predicted_boxes, target_boxes)
                box_loss_sum = box_loss_sum + (giou * quality_target).sum()
                quality_weight_sum = quality_weight_sum + quality_target.sum()

        normalizer = max(positive_count, 1)
        classification_loss = class_loss_sum / normalizer
        quality_loss = quality_loss_sum / normalizer
        box_loss = box_loss_sum / quality_weight_sum.clamp_min(1.0)
        total_loss = (
            classification_loss
            + self.quality_weight * quality_loss
            + self.box_weight * box_loss
        )
        return {
            "total": total_loss,
            "classification": classification_loss,
            "quality": quality_loss,
            "box": box_loss,
            "positive_count": first_prediction.new_tensor(float(positive_count)),
        }
