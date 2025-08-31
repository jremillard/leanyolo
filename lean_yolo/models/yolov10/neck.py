from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .layers import Conv, C2f, C2fCIB, SCDown, UpSample, make_divisible


class YOLOv10Neck(nn.Module):
    def __init__(self, width_mult: float = 1.0, depth_mult: float = 1.0, max_channels: int = 1024, c3: int = 128, c4: int = 256, c5: int = 512, variant: str = "s"):
        super().__init__()

        def c(ch: int) -> int:
            return min(max_channels, make_divisible(int(ch * width_mult)))

        def d(n: int) -> int:
            return max(int(round(n * depth_mult)), 1) if n > 0 else 0

        # Neck as per YOLOv10 (base channels; width_mult applied inside c())
        self.upsample = UpSample()
        # P5 -> P4
        self.p5_p4_c2f = C2f(c5 + c4, c(512), n=d(3))
        # P4 -> P3
        self.p4_p3_c2f = C2f(c(512) + c3, c(256), n=d(3))
        # P3 -> P4
        self.p3_down = Conv(c(256), c(256), 3, 2)
        if variant in {"m", "b", "l", "x"}:
            self.p3_p4_c2f = C2fCIB(c(256) + c(512), c(512), n=d(3))
        else:
            self.p3_p4_c2f = C2f(c(256) + c(512), c(512), n=d(3))
        # P4 -> P5
        self.p4_down = SCDown(c(512), c(512), 3, 2)
        self.p4_p5_c2f = C2fCIB(c(512) + c5, c(1024), n=d(3), lk=(variant in {"n", "s", "x"}))

        self.out_c = (c(256), c(512), c(1024))

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Top-down
        up4 = self.upsample(c5)
        p4 = self.p5_p4_c2f(torch.cat([up4, c4], dim=1))

        up3 = self.upsample(p4)
        p3 = self.p4_p3_c2f(torch.cat([up3, c3], dim=1))

        # Bottom-up
        down3 = self.p3_down(p3)
        p4 = self.p3_p4_c2f(torch.cat([down3, p4], dim=1))

        down4 = self.p4_down(p4)
        p5 = self.p4_p5_c2f(torch.cat([down4, c5], dim=1))

        return p3, p4, p5
