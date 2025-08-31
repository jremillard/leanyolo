from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .layers import Conv, C2f, C2fCIB, SCDown, UpSample, make_divisible


class YOLOv10Neck(nn.Module):
    def __init__(self, width_mult: float = 1.0, depth_mult: float = 1.0, max_channels: int = 1024, c3: int = 128, c4: int = 256, c5: int = 512, variant: str = "s"):
        super().__init__()

        def d(n: int) -> int:
            return max(int(round(n * depth_mult)), 1) if n > 0 else 0

        # Exact per-variant head channels
        HCH = {
            "n": {13: 128, 16: 64, 19: 128, 22: 256},
            "s": {13: 256, 16: 128, 19: 256, 22: 512},
            "m": {13: 384, 16: 192, 19: 384, 22: 576},
            "b": {13: 512, 16: 256, 19: 512, 22: 512},
            "l": {13: 512, 16: 256, 19: 512, 22: 512},
            "x": {13: 640, 16: 320, 19: 640, 22: 640},
        }[variant]

        self.upsample = UpSample()
        # P5 -> P4
        rep13 = {"n": 1, "s": 1, "m": 2, "b": 2, "l": 3, "x": 3}[variant]
        if variant in {"b", "l", "x"}:
            self.p5_p4_c2f = C2fCIB(c5 + c4, HCH[13], n=rep13, lk=(variant in {"x"}))
        else:
            self.p5_p4_c2f = C2f(c5 + c4, HCH[13], n=rep13)
        # P4 -> P3
        rep16 = {"n": 1, "s": 1, "m": 2, "b": 2, "l": 3, "x": 3}[variant]
        self.p4_p3_c2f = C2f(HCH[13] + c3, HCH[16], n=rep16)
        # P3 -> P4
        self.p3_down = Conv(HCH[16], HCH[16], 3, 2)
        rep19 = {"n": 1, "s": 1, "m": 2, "b": 2, "l": 3, "x": 3}[variant]
        if variant in {"m", "b", "l", "x"}:
            self.p3_p4_c2f = C2fCIB(HCH[16] + HCH[13], HCH[19], n=rep19)
        else:
            self.p3_p4_c2f = C2f(HCH[16] + HCH[13], HCH[19], n=rep19)
        # P4 -> P5
        self.p4_down = SCDown(HCH[19], HCH[19], 3, 2)
        rep22 = {"n": 1, "s": 1, "m": 2, "b": 2, "l": 3, "x": 3}[variant]
        self.p4_p5_c2f = C2fCIB(HCH[19] + c5, HCH[22], n=rep22, lk=(variant in {"n", "s", "x"}))

        self.out_c = (HCH[16], HCH[19], HCH[22])

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
