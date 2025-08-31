from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .layers import Conv, C2f, C2fCIB, SPPF, PSA, SCDown, make_divisible


class YOLOv10Backbone(nn.Module):
    def __init__(self, in_channels: int = 3, width_mult: float = 1.0, depth_mult: float = 1.0, max_channels: int = 1024, variant: str = "s"):
        super().__init__()

        def c(ch: int) -> int:
            return min(max_channels, make_divisible(int(ch * width_mult)))

        def d(n: int) -> int:
            return max(int(round(n * depth_mult)), 1) if n > 0 else 0

        # Following YOLOv10s topology (scaled by width/depth multipliers)
        self.cv0 = Conv(in_channels, c(64), 3, 2)
        self.cv1 = Conv(c(64), c(128), 3, 2)
        self.c2 = C2f(c(128), c(128), n=d(3))
        self.cv3 = Conv(c(128), c(256), 3, 2)
        self.c4 = C2f(c(256), c(256), n=d(6))
        self.sc5 = SCDown(c(256), c(512), 3, 2)
        if variant == "x":
            self.c6 = C2fCIB(c(512), c(512), n=d(6))
        else:
            self.c6 = C2f(c(512), c(512), n=d(6))
        self.sc7 = SCDown(c(512), c(1024), 3, 2)
        # lk usage varies by YAML; enable for s/x; n uses plain C2f here
        if variant == "n":
            from .layers import C2f as _C2f
            self.c8 = _C2f(c(1024), c(1024), n=d(1))
        else:
            self.c8 = C2fCIB(c(1024), c(1024), n=d(3), lk=(variant in {"s", "x"}))
        self.sppf9 = SPPF(c(1024), c(1024))
        self.psa10 = PSA(c(1024), c(1024))

        self.out_c = (c(256), c(512), c(1024))

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
