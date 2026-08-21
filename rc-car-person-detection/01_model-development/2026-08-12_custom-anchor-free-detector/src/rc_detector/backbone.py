from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class DSResidualBlock(nn.Module):
    """Pointwise expansion, depthwise spatial conv, and pointwise projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: float = 2.0,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        hidden_channels = max(out_channels, int(round(in_channels * expansion)))
        self.use_residual = stride == 1 and in_channels == out_channels
        self.expand = ConvBNAct(in_channels, hidden_channels, kernel_size=1)
        self.depthwise = ConvBNAct(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=stride,
            groups=hidden_channels,
        )
        self.project = ConvBNAct(
            hidden_channels,
            out_channels,
            kernel_size=1,
            activation=False,
        )
        self.output_activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.project(self.depthwise(self.expand(inputs)))
        if self.use_residual:
            outputs = outputs + inputs
        return self.output_activation(outputs)


def make_stage(
    in_channels: int,
    out_channels: int,
    repeats: int,
    expansion: float,
) -> nn.Sequential:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    blocks: list[nn.Module] = [
        DSResidualBlock(
            in_channels,
            out_channels,
            stride=2,
            expansion=expansion,
        )
    ]
    blocks.extend(
        DSResidualBlock(
            out_channels,
            out_channels,
            stride=1,
            expansion=expansion,
        )
        for _ in range(repeats - 1)
    )
    return nn.Sequential(*blocks)


class LightweightBackbone(nn.Module):
    """Custom DSConv residual backbone for 320x240 person detection."""

    output_channels = OrderedDict(c2=24, c3=48, c4=96, c5=160)
    output_strides = OrderedDict(c2=4, c3=8, c4=16, c5=32)

    def __init__(self, expansion: float = 2.0) -> None:
        super().__init__()
        self.stem = ConvBNAct(3, 16, kernel_size=3, stride=2)
        self.stage1 = make_stage(16, 24, repeats=2, expansion=expansion)
        self.stage2 = make_stage(24, 48, repeats=3, expansion=expansion)
        self.stage3 = make_stage(48, 96, repeats=3, expansion=expansion)
        self.stage4 = make_stage(96, 160, repeats=2, expansion=expansion)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        outputs = self.stem(inputs)
        c2 = self.stage1(outputs)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return OrderedDict(c2=c2, c3=c3, c4=c4, c5=c5)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
