from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn.functional as functional
from torch import nn

from .fpn import DepthwiseRefinement


class HeadTower(nn.Sequential):
    def __init__(self, channels: int, depth: int = 2) -> None:
        if depth < 1:
            raise ValueError("head tower depth must be positive")
        super().__init__(*(DepthwiseRefinement(channels) for _ in range(depth)))


class AnchorFreeHead(nn.Module):
    """Shared anchor-free head for single-class person detection."""

    level_names = ("p2", "p3", "p4", "p5")

    def __init__(
        self,
        channels: int = 64,
        num_classes: int = 1,
        strides: tuple[int, int, int, int] = (4, 8, 16, 32),
        tower_depth: int = 2,
        prior_probability: float = 0.01,
    ) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("V1 head is intentionally restricted to the person class")
        if len(strides) != len(self.level_names):
            raise ValueError("One stride is required for every pyramid level")
        self.num_classes = num_classes
        self.strides = dict(zip(self.level_names, strides, strict=True))
        self.classification_tower = HeadTower(channels, tower_depth)
        self.regression_tower = HeadTower(channels, tower_depth)
        self.classification_output = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.quality_output = nn.Conv2d(channels, 1, kernel_size=1)
        self.regression_output = nn.Conv2d(channels, 4, kernel_size=1)
        self.regression_scales = nn.ParameterDict(
            {level: nn.Parameter(torch.zeros(())) for level in self.level_names}
        )
        self._initialize_weights(prior_probability)

    def _initialize_weights(self, prior_probability: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        prior_bias = -math.log((1.0 - prior_probability) / prior_probability)
        nn.init.normal_(self.classification_output.weight, std=0.01)
        nn.init.normal_(self.quality_output.weight, std=0.01)
        nn.init.normal_(self.regression_output.weight, std=0.01)
        nn.init.constant_(self.classification_output.bias, prior_bias)
        nn.init.constant_(self.quality_output.bias, prior_bias)
        nn.init.zeros_(self.regression_output.bias)

    def forward(
        self, features: dict[str, torch.Tensor]
    ) -> OrderedDict[str, dict[str, torch.Tensor]]:
        missing = set(self.level_names) - set(features)
        if missing:
            raise KeyError(f"Missing pyramid features: {sorted(missing)}")
        predictions: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        for level in self.level_names:
            feature = features[level]
            classification_feature = self.classification_tower(feature)
            regression_feature = self.regression_tower(feature)
            class_logits = self.classification_output(classification_feature)
            quality_logits = self.quality_output(regression_feature)
            raw_distances = self.regression_output(regression_feature)
            scale = self.regression_scales[level].exp()
            distances = functional.softplus(raw_distances * scale) * self.strides[level]
            predictions[level] = {
                "class_logits": class_logits,
                "quality_logits": quality_logits,
                "distances": distances,
            }
        return predictions
