from __future__ import annotations

import torch
from torch import nn

from .backbone import LightweightBackbone
from .fpn import LightweightFPN
from .head import AnchorFreeHead


class PersonDetector(nn.Module):
    """Backbone, FPN, and anchor-free prediction head."""

    def __init__(
        self,
        fpn_channels: int = 64,
        backbone_expansion: float = 2.0,
    ) -> None:
        super().__init__()
        self.backbone = LightweightBackbone(expansion=backbone_expansion)
        self.fpn = LightweightFPN(
            dict(self.backbone.output_channels), out_channels=fpn_channels
        )
        self.head = AnchorFreeHead(channels=fpn_channels)

    def forward(
        self, inputs: torch.Tensor
    ) -> dict[str, dict[str, torch.Tensor]]:
        return self.head(self.fpn(self.backbone(inputs)))
