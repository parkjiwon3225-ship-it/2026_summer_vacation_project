from __future__ import annotations

import torch

from .assignment import AnchorFreeTargetAssigner
from .losses import decode_distances


def box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    left = torch.maximum(box[0], boxes[:, 0])
    top = torch.maximum(box[1], boxes[:, 1])
    right = torch.minimum(box[2], boxes[:, 2])
    bottom = torch.minimum(box[3], boxes[:, 3])
    intersection = (right - left).clamp_min(0) * (bottom - top).clamp_min(0)
    box_area = (box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)
    boxes_area = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp_min(0)
    return intersection / (box_area + boxes_area - intersection).clamp_min(1e-7)


def pure_torch_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=boxes.device)
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    order = scores.argsort(descending=True)
    kept: list[torch.Tensor] = []
    while order.numel():
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious = box_iou_one_to_many(boxes[current], boxes[remaining])
        order = remaining[ious <= iou_threshold]
    return torch.stack(kept)


class DetectionPostProcessor:
    def __init__(
        self,
        strides: dict[str, int] | None = None,
        score_threshold: float = 0.05,
        nms_iou_threshold: float = 0.5,
        pre_nms_topk: int = 1000,
        max_detections: int = 100,
    ) -> None:
        self.strides = strides or {"p2": 4, "p3": 8, "p4": 16, "p5": 32}
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.pre_nms_topk = pre_nms_topk
        self.max_detections = max_detections

    def __call__(
        self,
        predictions: dict[str, dict[str, torch.Tensor]],
        image_size: tuple[int, int] = (320, 240),
    ) -> list[dict[str, torch.Tensor]]:
        missing = set(self.strides) - set(predictions)
        if missing:
            raise KeyError(f"Missing prediction levels: {sorted(missing)}")
        output_width, output_height = image_size
        batch_size = next(iter(predictions.values()))["class_logits"].shape[0]
        batch_results: list[dict[str, torch.Tensor]] = []
        for batch_index in range(batch_size):
            image_boxes: list[torch.Tensor] = []
            image_scores: list[torch.Tensor] = []
            image_levels: list[torch.Tensor] = []
            for level_index, (level, stride) in enumerate(self.strides.items()):
                output = predictions[level]
                height, width = output["class_logits"].shape[-2:]
                points = AnchorFreeTargetAssigner.make_points(
                    height,
                    width,
                    stride,
                    output["distances"].device,
                    output["distances"].dtype,
                )
                class_scores = output["class_logits"][batch_index, 0].sigmoid().reshape(-1)
                quality_scores = output["quality_logits"][batch_index, 0].sigmoid().reshape(-1)
                scores = class_scores * quality_scores
                distances = output["distances"][batch_index].permute(1, 2, 0).reshape(-1, 4)
                keep = scores >= self.score_threshold
                if not keep.any():
                    continue
                scores = scores[keep]
                boxes = decode_distances(points[keep], distances[keep])
                levels = torch.full(
                    (len(scores),), level_index, dtype=torch.int64, device=scores.device
                )
                image_boxes.append(boxes)
                image_scores.append(scores)
                image_levels.append(levels)
            if not image_boxes:
                device = next(iter(predictions.values()))["class_logits"].device
                batch_results.append(
                    {
                        "boxes": torch.empty((0, 4), device=device),
                        "scores": torch.empty((0,), device=device),
                        "labels": torch.empty((0,), dtype=torch.int64, device=device),
                        "levels": torch.empty((0,), dtype=torch.int64, device=device),
                    }
                )
                continue
            boxes = torch.cat(image_boxes)
            scores = torch.cat(image_scores)
            levels = torch.cat(image_levels)
            if len(scores) > self.pre_nms_topk:
                scores, indices = scores.topk(self.pre_nms_topk)
                boxes = boxes[indices]
                levels = levels[indices]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, output_width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, output_height)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, scores, levels = boxes[valid], scores[valid], levels[valid]
            keep = pure_torch_nms(boxes, scores, self.nms_iou_threshold)
            keep = keep[: self.max_detections]
            batch_results.append(
                {
                    "boxes": boxes[keep],
                    "scores": scores[keep],
                    "labels": torch.zeros(len(keep), dtype=torch.int64, device=boxes.device),
                    "levels": levels[keep],
                }
            )
        return batch_results
