from __future__ import annotations

import math
from collections import OrderedDict

import torch


class AnchorFreeTargetAssigner:
    """FCOS-style point assignment with center sampling and level ranges."""

    def __init__(
        self,
        strides: dict[str, int] | None = None,
        regression_ranges: dict[str, tuple[float, float]] | None = None,
        center_sampling_radius: float = 1.5,
    ) -> None:
        self.strides = strides or {"p2": 4, "p3": 8, "p4": 16, "p5": 32}
        self.regression_ranges = regression_ranges or {
            "p2": (0.0, 32.0),
            "p3": (32.0, 64.0),
            "p4": (64.0, 128.0),
            "p5": (128.0, math.inf),
        }
        if set(self.strides) != set(self.regression_ranges):
            raise ValueError("strides and regression_ranges must have identical levels")
        if center_sampling_radius <= 0:
            raise ValueError("center_sampling_radius must be positive")
        self.center_sampling_radius = center_sampling_radius

    @staticmethod
    def make_points(
        height: int,
        width: int,
        stride: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * stride
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)

    def assign_level(
        self,
        level: str,
        feature_shape: tuple[int, int],
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if level not in self.strides:
            raise KeyError(f"Unknown feature level: {level}")
        height, width = feature_shape
        stride = self.strides[level]
        points = self.make_points(height, width, stride, boxes.device, boxes.dtype)
        location_count = len(points)
        class_targets = torch.zeros(location_count, dtype=torch.float32, device=boxes.device)
        quality_targets = torch.zeros_like(class_targets)
        distance_targets = torch.zeros(
            (location_count, 4), dtype=boxes.dtype, device=boxes.device
        )
        matched_gt_indices = torch.full(
            (location_count,), -1, dtype=torch.int64, device=boxes.device
        )
        if boxes.numel() == 0:
            return self._reshape_targets(
                height,
                width,
                points,
                class_targets,
                quality_targets,
                distance_targets,
                matched_gt_indices,
            )
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"boxes must have shape [N, 4], found {tuple(boxes.shape)}")
        if labels.shape != (len(boxes),) or not torch.all(labels == 0):
            raise ValueError("labels must contain one class-0 entry per box")

        point_x = points[:, 0:1]
        point_y = points[:, 1:2]
        left = point_x - boxes[None, :, 0]
        top = point_y - boxes[None, :, 1]
        right = boxes[None, :, 2] - point_x
        bottom = boxes[None, :, 3] - point_y
        distances = torch.stack((left, top, right, bottom), dim=2)
        inside_box = distances.min(dim=2).values > 0

        center_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        center_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
        radius = self.center_sampling_radius * stride
        center_left = torch.maximum(center_x - radius, boxes[:, 0])
        center_top = torch.maximum(center_y - radius, boxes[:, 1])
        center_right = torch.minimum(center_x + radius, boxes[:, 2])
        center_bottom = torch.minimum(center_y + radius, boxes[:, 3])
        inside_center = (
            (point_x > center_left[None, :])
            & (point_x < center_right[None, :])
            & (point_y > center_top[None, :])
            & (point_y < center_bottom[None, :])
        )

        lower, upper = self.regression_ranges[level]
        max_distance = distances.max(dim=2).values
        in_level_range = (max_distance >= lower) & (max_distance < upper)
        candidates = inside_box & inside_center & in_level_range

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        candidate_areas = areas[None, :].expand(location_count, -1).clone()
        candidate_areas[~candidates] = math.inf
        minimum_area, matched = candidate_areas.min(dim=1)
        positive = torch.isfinite(minimum_area)
        positive_indices = torch.where(positive)[0]
        matched_positive = matched[positive]
        if len(positive_indices):
            selected_distances = distances[positive_indices, matched_positive]
            distance_targets[positive] = selected_distances
            matched_gt_indices[positive] = matched_positive
            class_targets[positive] = 1.0
            left_right = selected_distances[:, [0, 2]]
            top_bottom = selected_distances[:, [1, 3]]
            quality_targets[positive] = torch.sqrt(
                (left_right.min(dim=1).values / left_right.max(dim=1).values.clamp_min(1e-7))
                * (top_bottom.min(dim=1).values / top_bottom.max(dim=1).values.clamp_min(1e-7))
            )
        return self._reshape_targets(
            height,
            width,
            points,
            class_targets,
            quality_targets,
            distance_targets,
            matched_gt_indices,
        )

    @staticmethod
    def _reshape_targets(
        height: int,
        width: int,
        points: torch.Tensor,
        class_targets: torch.Tensor,
        quality_targets: torch.Tensor,
        distance_targets: torch.Tensor,
        matched_gt_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "points": points.reshape(height, width, 2),
            "class_targets": class_targets.reshape(1, height, width),
            "quality_targets": quality_targets.reshape(1, height, width),
            "distance_targets": distance_targets.transpose(0, 1).reshape(4, height, width),
            "positive_mask": class_targets.bool().reshape(height, width),
            "matched_gt_indices": matched_gt_indices.reshape(height, width),
        }

    def __call__(
        self,
        feature_shapes: dict[str, tuple[int, int]],
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> OrderedDict[str, dict[str, torch.Tensor]]:
        missing = set(self.strides) - set(feature_shapes)
        if missing:
            raise KeyError(f"Missing feature shapes: {sorted(missing)}")
        return OrderedDict(
            (
                level,
                self.assign_level(level, feature_shapes[level], boxes, labels),
            )
            for level in self.strides
        )
