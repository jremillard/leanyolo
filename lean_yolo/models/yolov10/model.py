from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn

from .backbone import YOLOv10Backbone
from .neck import YOLOv10Neck
from .head import DecoupledHead


@dataclass(frozen=True)
class YOLOv10Spec:
    depth_mult: float
    width_mult: float
    max_channels: int


class YOLOv10(nn.Module):
    """Minimal, readable YOLOv10-style model.

    Notes:
    - This implementation focuses on clarity. It mirrors the common YOLO
      pattern (Backbone -> PAN-FPN Neck -> Decoupled Head) and returns raw
      per-level predictions [P3, P4, P5].
    - Output format per scale: [B, (4 + num_classes), H, W]. Box branch is
      [tx, ty, tw, th] in raw units; postprocessing (sigmoid/exp/grid) is not
      applied here. That is handled by inference utilities to be added later.
    """

    def __init__(self, num_classes: int = 80, in_channels: int = 3, spec: YOLOv10Spec | None = None):
        super().__init__()
        if spec is None:
            spec = YOLOv10Spec(depth_mult=1.0, width_mult=1.0, max_channels=1024)

        self.backbone = YOLOv10Backbone(
            in_channels=in_channels,
            width_mult=spec.width_mult,
            depth_mult=spec.depth_mult,
            max_channels=spec.max_channels,
        )
        c3, c4, c5 = self.backbone.out_c
        self.neck = YOLOv10Neck(
            width_mult=spec.width_mult,
            depth_mult=spec.depth_mult,
            max_channels=spec.max_channels,
            c3=c3,
            c4=c4,
            c5=c5,
        )
        p3, p4, p5 = self.neck.out_c
        self.head = DecoupledHead((p3, p4, p5), num_classes=num_classes, width_mult=spec.width_mult, max_channels=spec.max_channels)

        # Initialize bias for better startup (optional minor improvement)
        self._init_head_bias()

    def _init_head_bias(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels in (4,) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c3, c4, c5)
        out = self.head((p3, p4, p5))
        return out
