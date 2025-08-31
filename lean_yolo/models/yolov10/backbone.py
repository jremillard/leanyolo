from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .layers import Conv, C2f, C2fCIB, SPPF, PSA, SCDown


class YOLOv10Backbone(nn.Module):
    def __init__(self, in_channels: int = 3, width_mult: float = 1.0, depth_mult: float = 1.0, max_channels: int = 1024, variant: str = "s"):
        super().__init__()

        def d(n: int) -> int:
            return max(int(round(n * depth_mult)), 1) if n > 0 else 0
        # Exact per-variant channels (from official YAMLs)
        CH = {
            "n": {0: 16, 1: 32, 2: 32, 3: 64, 4: 64, 5: 128, 6: 128, 7: 256, 8: 256, 9: 256, 10: 256},
            "s": {0: 32, 1: 64, 2: 64, 3: 128, 4: 128, 5: 256, 6: 256, 7: 512, 8: 512, 9: 512, 10: 512},
            "m": {0: 48, 1: 96, 2: 96, 3: 192, 4: 192, 5: 384, 6: 384, 7: 576, 8: 576, 9: 576, 10: 576},
            "b": {0: 64, 1: 128, 2: 128, 3: 256, 4: 256, 5: 512, 6: 512, 7: 512, 8: 512, 9: 512, 10: 512},
            "l": {0: 64, 1: 128, 2: 128, 3: 256, 4: 256, 5: 512, 6: 512, 7: 512, 8: 512, 9: 512, 10: 512},
            "x": {0: 80, 1: 160, 2: 160, 3: 320, 4: 320, 5: 640, 6: 640, 7: 640, 8: 640, 9: 640, 10: 640},
        }[variant]

        self.cv0 = Conv(in_channels, CH[0], 3, 2)
        self.cv1 = Conv(CH[0], CH[1], 3, 2)
        rep2 = {"n": 1, "s": 1, "m": 2, "b": 2, "l": 3, "x": 3}[variant]
        self.c2 = C2f(CH[1], CH[2], n=rep2)
        self.cv3 = Conv(CH[2], CH[3], 3, 2)
        rep4 = {"n": 2, "s": 2, "m": 4, "b": 4, "l": 6, "x": 6}[variant]
        self.c4 = C2f(CH[3], CH[4], n=rep4)
        self.sc5 = SCDown(CH[4], CH[5], 3, 2)
        # stage 6
        rep6 = {"n": 2, "s": 2, "m": 4, "b": 4, "l": 6, "x": 6}[variant]
        if variant == "x":
            self.c6 = C2fCIB(CH[5], CH[6], n=rep6)
        else:
            self.c6 = C2f(CH[5], CH[6], n=rep6)
        self.sc7 = SCDown(CH[6], CH[7], 3, 2)
        # stage 8
        rep8 = {"n": 1, "s": 1, "m": 2, "b": 2, "l": 1, "x": 3}[variant]
        if variant == "n":
            self.c8 = C2f(CH[7], CH[8], n=rep8)
        else:
            self.c8 = C2fCIB(CH[7], CH[8], n=rep8, lk=(variant in {"s", "x"}))
        self.sppf9 = SPPF(CH[8], CH[9])
        self.psa10 = PSA(CH[9], CH[10])

        self.out_c = (CH[3], CH[5], CH[7])

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
