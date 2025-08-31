from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .backbone import YOLOv10Backbone
from .neck import YOLOv10Neck
from .head_v10 import V10Detect


class YOLOv10b(nn.Module):
    CH = {0: 64, 1: 128, 2: 128, 3: 256, 4: 256, 5: 512, 6: 512, 7: 512, 8: 512, 9: 512, 10: 512}
    HCH = {13: 512, 16: 256, 19: 512, 22: 512}
    REPS = {2: 2, 4: 4, 6: 4, 8: 2, 13: 2, 16: 2, 19: 2, 22: 2}
    TYPES = {"c6": "C2f", "c8": "C2fCIB", "p5_p4": "C2fCIB", "p3_p4": "C2fCIB", "p4_p5": "C2fCIB"}
    LK = {"c8": False, "p5_p4": False, "p4_p5": False}

    def __init__(self, *, num_classes: int, in_channels: int):
        super().__init__()
        self.backbone = YOLOv10Backbone(
            in_channels=in_channels,
            CH=self.CH,
            reps=self.REPS,
            types=self.TYPES,
            use_lk_c8=self.LK.get("c8", False),
        )
        c3, c4, c5 = self.backbone.out_c
        self.neck = YOLOv10Neck(
            c3=c3,
            c4=c4,
            c5=c5,
            HCH=self.HCH,
            reps=self.REPS,
            types=self.TYPES,
            use_lk_p5_p4=self.LK.get("p5_p4", False),
            use_lk_p4_p5=self.LK.get("p4_p5", False),
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
