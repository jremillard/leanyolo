from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .backbone import YOLOv10Backbone
from .neck import YOLOv10Neck
from .head_v10 import V10Detect


class YOLOv10s(nn.Module):
    CH = {0: 32, 1: 64, 2: 64, 3: 128, 4: 128, 5: 256, 6: 256, 7: 512, 8: 512, 9: 512, 10: 512}
    HCH = {13: 256, 16: 128, 19: 256, 22: 512}
    REPS = {2: 1, 4: 2, 6: 2, 8: 1, 13: 1, 16: 1, 19: 1, 22: 1}
    TYPES = {"c6": "C2f", "c8": "C2fCIB", "p5_p4": "C2f", "p3_p4": "C2f", "p4_p5": "C2fCIB"}

    def __init__(self, *, num_classes: int, in_channels: int):
        super().__init__()
        self.backbone = YOLOv10Backbone(
            in_channels=in_channels,
            CH=self.CH,
            reps=self.REPS,
            types=self.TYPES,
            use_lk_c8=True,
        )
        c3, c4, c5 = self.backbone.out_c
        self.neck = YOLOv10Neck(
            c3=c3,
            c4=c4,
            c5=c5,
            HCH=self.HCH,
            reps=self.REPS,
            types=self.TYPES,
            use_lk_p5_p4=False,
            use_lk_p4_p5=True,
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
