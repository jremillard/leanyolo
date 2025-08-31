from __future__ import annotations

from types import SimpleNamespace
from typing import List

import torch
import torch.nn as nn

from .backbone import YOLOv10Backbone
from .neck import YOLOv10Neck
from .head_v10 import V10Detect


class YOLOv10n(nn.Module):
    CH = {0: 16, 1: 32, 2: 32, 3: 64, 4: 64, 5: 128, 6: 128, 7: 256, 8: 256, 9: 256, 10: 256}
    HCH = {13: 128, 16: 64, 19: 128, 22: 256}
    REPS = {2: 1, 4: 2, 6: 2, 8: 1, 13: 1, 16: 1, 19: 1, 22: 1}
    TYPES = {"c6": "C2f", "c8": "C2f", "p5_p4": "C2f", "p3_p4": "C2f", "p4_p5": "C2fCIB"}
    LK = {"c8": False, "p5_p4": False, "p4_p5": True}

    def __init__(self, num_classes: int = 80, in_channels: int = 3):
        super().__init__()
        cfg = SimpleNamespace(CH=self.CH, HCH=self.HCH, reps=self.REPS, types=self.TYPES, lk=self.LK)

        self.backbone = YOLOv10Backbone(
            in_channels=in_channels,
            width_mult=0.33,
            depth_mult=0.33,
            max_channels=1024,
            variant="n",
            cfg=cfg,
        )
        c3, c4, c5 = self.backbone.out_c
        self.neck = YOLOv10Neck(
            width_mult=0.33,
            depth_mult=0.33,
            max_channels=1024,
            c3=c3,
            c4=c4,
            c5=c5,
            variant="n",
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

