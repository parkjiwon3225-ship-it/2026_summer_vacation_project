from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as functional
from torch import nn

from .backbone import ConvBNAct


class DepthwiseRefinement(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            ConvBNAct(
                channels,
                channels,
                kernel_size=3,
                groups=channels,
            ),
            ConvBNAct(channels, channels, kernel_size=1),
        )


class LightweightFPN(nn.Module):
    """Top-down feature pyramid with exact-size interpolation."""

    def __init__(
        self,
        in_channels: dict[str, int] | None = None,
        out_channels: int = 64,
    ) -> None:
        super().__init__()
        in_channels = in_channels or {"c2": 24, "c3": 48, "c4": 96, "c5": 160}
        if set(in_channels) != {"c2", "c3", "c4", "c5"}:
            raise ValueError("in_channels must contain exactly c2, c3, c4, and c5")
        self.out_channels = out_channels
        self.lateral2 = ConvBNAct(
            in_channels["c2"], out_channels, kernel_size=1, activation=False
        )
        self.lateral3 = ConvBNAct(
            in_channels["c3"], out_channels, kernel_size=1, activation=False
        )
        self.lateral4 = ConvBNAct(
            in_channels["c4"], out_channels, kernel_size=1, activation=False
        )
        self.lateral5 = ConvBNAct(
            in_channels["c5"], out_channels, kernel_size=1, activation=False
        )
        self.refine2 = DepthwiseRefinement(out_channels)
        self.refine3 = DepthwiseRefinement(out_channels)
        self.refine4 = DepthwiseRefinement(out_channels)
        self.refine5 = DepthwiseRefinement(out_channels)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, features: dict[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        missing = {"c2", "c3", "c4", "c5"} - set(features)
        if missing:
            raise KeyError(f"Missing backbone features: {sorted(missing)}")
        lateral5 = self.lateral5(features["c5"])
        lateral4 = self.lateral4(features["c4"])
        lateral3 = self.lateral3(features["c3"])
        lateral2 = self.lateral2(features["c2"])
        merged4 = lateral4 + functional.interpolate(
            lateral5, size=lateral4.shape[-2:], mode="nearest"
        )
        merged3 = lateral3 + functional.interpolate(
            merged4, size=lateral3.shape[-2:], mode="nearest"
        )
        merged2 = lateral2 + functional.interpolate(
            merged3, size=lateral2.shape[-2:], mode="nearest"
        )
        p5 = self.refine5(lateral5)
        p4 = self.refine4(merged4)
        p3 = self.refine3(merged3)
        p2 = self.refine2(merged2)
        return OrderedDict(p2=p2, p3=p3, p4=p4, p5=p5)
