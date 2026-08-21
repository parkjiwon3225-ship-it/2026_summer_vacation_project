from __future__ import annotations

from dataclasses import dataclass

import torch


def pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((len(boxes1), len(boxes2)), dtype=torch.float32)
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection_wh = (bottom_right - top_left).clamp_min(0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp_min(0)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp_min(1e-7)


@dataclass
class MatchResult:
    scores: torch.Tensor
    true_positives: torch.Tensor
    matched_ious: torch.Tensor
    matched_gt_indices: torch.Tensor
    image_indices: torch.Tensor


def match_detections(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    iou_threshold: float,
) -> MatchResult:
    records: list[tuple[float, int, int]] = []
    prediction_boxes: list[torch.Tensor] = []
    target_boxes: list[torch.Tensor] = []
    for image_index, (prediction, target) in enumerate(zip(predictions, targets, strict=True)):
        boxes = prediction["boxes"].detach().cpu().float()
        scores = prediction["scores"].detach().cpu().float()
        prediction_boxes.append(boxes)
        target_boxes.append(target["boxes"].detach().cpu().float())
        records.extend(
            (float(score), image_index, detection_index)
            for detection_index, score in enumerate(scores)
        )
    records.sort(key=lambda item: item[0], reverse=True)
    used_gt = [torch.zeros(len(boxes), dtype=torch.bool) for boxes in target_boxes]
    scores_out: list[float] = []
    tp_out: list[bool] = []
    iou_out: list[float] = []
    gt_index_out: list[int] = []
    image_index_out: list[int] = []
    for score, image_index, detection_index in records:
        scores_out.append(score)
        image_index_out.append(image_index)
        gt_boxes = target_boxes[image_index]
        if not len(gt_boxes):
            tp_out.append(False)
            iou_out.append(0.0)
            gt_index_out.append(-1)
            continue
        ious = pairwise_iou(
            prediction_boxes[image_index][detection_index : detection_index + 1], gt_boxes
        )[0]
        ious[used_gt[image_index]] = -1
        best_iou, best_gt = ious.max(dim=0)
        matched = float(best_iou) >= iou_threshold
        tp_out.append(matched)
        iou_out.append(float(best_iou.clamp_min(0)))
        gt_index_out.append(int(best_gt) if matched else -1)
        if matched:
            used_gt[image_index][best_gt] = True
    return MatchResult(
        scores=torch.tensor(scores_out, dtype=torch.float32),
        true_positives=torch.tensor(tp_out, dtype=torch.bool),
        matched_ious=torch.tensor(iou_out, dtype=torch.float32),
        matched_gt_indices=torch.tensor(gt_index_out, dtype=torch.int64),
        image_indices=torch.tensor(image_index_out, dtype=torch.int64),
    )


def precision_recall_curve(
    matches: MatchResult, total_gt: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not len(matches.scores):
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty, empty
    true_positive = matches.true_positives.float().cumsum(0)
    false_positive = (~matches.true_positives).float().cumsum(0)
    precision = true_positive / (true_positive + false_positive).clamp_min(1e-7)
    recall = true_positive / max(total_gt, 1)
    return precision, recall, matches.scores


def interpolated_ap(precision: torch.Tensor, recall: torch.Tensor) -> float:
    if not len(precision):
        return 0.0
    recall_points = torch.linspace(0, 1, 101)
    interpolated: list[torch.Tensor] = []
    for recall_point in recall_points:
        valid = recall >= recall_point
        interpolated.append(precision[valid].max() if valid.any() else precision.new_zeros(()))
    return float(torch.stack(interpolated).mean())


class DetectionEvaluator:
    def __init__(self, operating_score_threshold: float = 0.25) -> None:
        self.operating_score_threshold = operating_score_threshold
        self.predictions: list[dict[str, torch.Tensor]] = []
        self.targets: list[dict[str, torch.Tensor]] = []

    def update(
        self,
        predictions: list[dict[str, torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
    ) -> None:
        if len(predictions) != len(targets):
            raise ValueError("Prediction and target batch lengths differ")
        for prediction, target in zip(predictions, targets, strict=True):
            self.predictions.append(
                {
                    "boxes": prediction["boxes"].detach().cpu(),
                    "scores": prediction["scores"].detach().cpu(),
                }
            )
            self.targets.append({"boxes": target["boxes"].detach().cpu()})

    def compute(self) -> dict[str, float | int | dict[str, float | int]]:
        total_gt = sum(len(target["boxes"]) for target in self.targets)
        iou_thresholds = [0.50 + 0.05 * index for index in range(10)]
        aps: dict[float, float] = {}
        matches_50: MatchResult | None = None
        for threshold in iou_thresholds:
            matches = match_detections(self.predictions, self.targets, threshold)
            precision, recall, _ = precision_recall_curve(matches, total_gt)
            aps[threshold] = interpolated_ap(precision, recall)
            if abs(threshold - 0.5) < 1e-6:
                matches_50 = matches
        assert matches_50 is not None
        operating = matches_50.scores >= self.operating_score_threshold
        tp = int(matches_50.true_positives[operating].sum())
        fp = int(operating.sum()) - tp
        fn = total_gt - tp
        precision_value = tp / max(tp + fp, 1)
        recall_value = tp / max(total_gt, 1)
        f1 = 2 * precision_value * recall_value / max(precision_value + recall_value, 1e-7)
        detection_accuracy = tp / max(tp + fp + fn, 1)

        curve_precision, curve_recall, curve_scores = precision_recall_curve(matches_50, total_gt)
        if len(curve_precision):
            curve_f1 = 2 * curve_precision * curve_recall / (
                curve_precision + curve_recall
            ).clamp_min(1e-7)
            best_index = int(curve_f1.argmax())
            best_f1 = float(curve_f1[best_index])
            best_threshold = float(curve_scores[best_index])
        else:
            best_f1, best_threshold = 0.0, 1.0

        tp_mask = operating & matches_50.true_positives
        mean_tp_iou = (
            float(matches_50.matched_ious[tp_mask].mean()) if tp_mask.any() else 0.0
        )
        detections_per_image = [
            int((prediction["scores"] >= self.operating_score_threshold).sum())
            for prediction in self.predictions
        ]
        no_detection_rate = sum(count == 0 for count in detections_per_image) / max(
            len(detections_per_image), 1
        )
        matched_images = set(matches_50.image_indices[tp_mask].tolist())
        images_with_gt = {
            image_index for image_index, target in enumerate(self.targets) if len(target["boxes"])
        }
        zero_recall_rate = len(images_with_gt - matched_images) / max(len(images_with_gt), 1)

        size_bins = {
            "tiny_lt16": (0.0, 16.0),
            "small_16_32": (16.0, 32.0),
            "medium_32_96": (32.0, 96.0),
            "large_ge96": (96.0, float("inf")),
        }
        matched_pairs = {
            (int(image_index), int(gt_index))
            for image_index, gt_index in zip(
                matches_50.image_indices[tp_mask],
                matches_50.matched_gt_indices[tp_mask],
                strict=True,
            )
        }
        size_metrics: dict[str, dict[str, float | int]] = {}
        for name, (lower, upper) in size_bins.items():
            count = 0
            matched_count = 0
            for image_index, target in enumerate(self.targets):
                heights = target["boxes"][:, 3] - target["boxes"][:, 1]
                for gt_index, height in enumerate(heights.tolist()):
                    if lower <= height < upper:
                        count += 1
                        matched_count += (image_index, gt_index) in matched_pairs
            size_metrics[name] = {
                "gt": count,
                "matched": matched_count,
                "recall": matched_count / max(count, 1),
            }

        return {
            "ap50": aps[0.5],
            "ap75": aps[0.75],
            "map50_95": sum(aps.values()) / len(aps),
            "precision": precision_value,
            "recall": recall_value,
            "f1": f1,
            "best_f1": best_f1,
            "best_f1_threshold": best_threshold,
            "detection_accuracy": detection_accuracy,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_tp_iou": mean_tp_iou,
            "average_detections_per_image": sum(detections_per_image)
            / max(len(detections_per_image), 1),
            "no_detection_image_rate": no_detection_rate,
            "zero_recall_image_rate": zero_recall_rate,
            "score_threshold": self.operating_score_threshold,
            "size_recall": size_metrics,
        }
