from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .layers import Conv, C2f, C2fCIB, SPPF, PSA, SCDown, make_divisible


class YOLOv10Backbone(nn.Module):
    def __init__(self, in_channels: int = 3, width_mult: float = 1.0, depth_mult: float = 1.0, max_channels: int = 1024):
        super().__init__()

        def c(ch: int) -> int:
            return min(max_channels, make_divisible(int(ch * width_mult)))

        def d(n: int) -> int:
            return max(int(round(n * depth_mult)), 1) if n > 0 else 0

        # Following YOLOv10s topology (scaled by width/depth multipliers)
        self.cv0 = Conv(in_channels, c(32), 3, 2)
        self.cv1 = Conv(c(32), c(64), 3, 2)
        self.c2 = C2f(c(64), c(64), n=d(1))
        self.cv3 = Conv(c(64), c(128), 3, 2)
        self.c4 = C2f(c(128), c(128), n=d(2))
        self.sc5 = SCDown(c(128), c(256), 3, 2)
        self.c6 = C2f(c(256), c(256), n=d(2))
        self.sc7 = SCDown(c(256), c(512), 3, 2)
        self.c8 = C2fCIB(c(512), c(512), n=d(1))
        self.sppf9 = SPPF(c(512), c(512))
        self.psa10 = PSA(c(512), c(512))

        self.out_c = (c(128), c(256), c(512))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.cv0(x)
        x = self.cv1(x)
        x = self.c2(x)
        x = self.cv3(x)
        c3 = self.c4(x)
        x = self.sc5(c3)
        c4 = self.c6(x)
        x = self.sc7(c4)
        x = self.c8(x)
        x = self.sppf9(x)
        c5 = self.psa10(x)
        return c3, c4, c5
