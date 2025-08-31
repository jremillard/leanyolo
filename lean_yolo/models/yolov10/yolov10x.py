from __future__ import annotations

from types import SimpleNamespace
from typing import List

import torch
import torch.nn as nn

from .backbone import YOLOv10Backbone
from .neck import YOLOv10Neck
from .head_v10 import V10Detect


class YOLOv10x(nn.Module):
    CH = {0: 80, 1: 160, 2: 160, 3: 320, 4: 320, 5: 640, 6: 640, 7: 640, 8: 640, 9: 640, 10: 640}
    HCH = {13: 640, 16: 320, 19: 640, 22: 640}
    REPS = {2: 3, 4: 6, 6: 6, 8: 3, 13: 3, 16: 3, 19: 3, 22: 3}
    TYPES = {"c6": "C2fCIB", "c8": "C2fCIB", "p5_p4": "C2fCIB", "p3_p4": "C2fCIB", "p4_p5": "C2fCIB"}
    LK = {"c8": False, "p5_p4": False, "p4_p5": False}

    def __init__(self, num_classes: int = 80, in_channels: int = 3):
        super().__init__()
        cfg = SimpleNamespace(CH=self.CH, HCH=self.HCH, reps=self.REPS, types=self.TYPES, lk=self.LK)

        self.backbone = YOLOv10Backbone(
            in_channels=in_channels,
            width_mult=1.25,
            depth_mult=1.00,
            max_channels=512,
            variant="x",
            cfg=cfg,
        )
        c3, c4, c5 = self.backbone.out_c
        self.neck = YOLOv10Neck(
            width_mult=1.25,
            depth_mult=1.00,
            max_channels=512,
            c3=c3,
            c4=c4,
            c5=c5,
            variant="x",
            cfg=cfg,
        )
        p3, p4, p5 = self.neck.out_c
        self.head = V10Detect(nc=num_classes, ch=(p3, p4, p5), reg_max=16)
        self._init_head_bias()

    def _init_head_bias(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels in (4,) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c3, c4, c5)
        return self.head((p3, p4, p5))

