from __future__ import annotations

"""YOLOv10-l model definition.

The large (l) variant increases capacity for higher accuracy on larger devices.
Structure is the same as other variants, with bigger channels and more repeats.
"""

from typing import List

import torch
import torch.nn as nn

from .backbone import YOLOv10Backbone
from .neck import YOLOv10Neck
from .head_v10 import V10Detect


class YOLOv10l(nn.Module):
    CH = {0: 64, 1: 128, 2: 128, 3: 256, 4: 256, 5: 512, 6: 512, 7: 512, 8: 512, 9: 512, 10: 512}
    HCH = {13: 512, 16: 256, 19: 512, 22: 512}
    REPS = {2: 3, 4: 6, 6: 6, 8: 3, 13: 3, 16: 3, 19: 3, 22: 3}
    TYPES = {"c6": "C2f", "c8": "C2fCIB", "p5_p4": "C2fCIB", "p3_p4": "C2fCIB", "p4_p5": "C2fCIB"}

    def __init__(self, *, num_classes: int, in_channels: int):
        super().__init__()
        self.backbone = YOLOv10Backbone(
            in_channels=in_channels,
            CH=self.CH,
            reps=self.REPS,
            types=self.TYPES,
            use_lk_c8=False,
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
            use_lk_p4_p5=False,
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
