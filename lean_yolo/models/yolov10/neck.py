from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .layers import Conv, C2f, UpSample, make_divisible


class YOLOv10Neck(nn.Module):
    def __init__(self, width_mult: float = 1.0, depth_mult: float = 1.0, max_channels: int = 1024, c3: int = 256, c4: int = 512, c5: int = 1024):
        super().__init__()

        def c(ch: int) -> int:
            return min(max_channels, make_divisible(int(ch * width_mult)))

        def d(n: int) -> int:
            return max(int(round(n * depth_mult)), 1) if n > 0 else 0

        # Lateral reduce
        self.cv5 = Conv(c5, c(512), 1, 1)
        self.cv4 = Conv(c4, c(256), 1, 1)

        # Top-down pathway
        self.upsample = UpSample()
        self.p5_to_p4 = C2f(c(512) + c(256), c(256), n=d(3))
        self.p4_to_p3 = C2f(c(256) + c3, c(128), n=d(3))

        # Bottom-up pathway
        self.p3_to_p4 = C2f(c(128) + c(128), c(256), n=d(3))
        self.p4_to_p5 = C2f(c(256) + c(256), c(512), n=d(3))

        # Downsample convs for PAN
        self.down_p3 = Conv(c(128), c(128), 3, 2)
        self.down_p4 = Conv(c(256), c(256), 3, 2)

        self.out_c = (c(128), c(256), c(512))

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p5 = self.cv5(c5)
        p4 = self.cv4(c4)

        up4 = self.upsample(p5)
        p4 = self.p5_to_p4(torch.cat([up4, p4], dim=1))

        up3 = self.upsample(p4)
        p3 = self.p4_to_p3(torch.cat([up3, c3], dim=1))

        down3 = self.down_p3(p3)
        p4 = self.p3_to_p4(torch.cat([down3, p4], dim=1))

        down4 = self.down_p4(p4)
        p5 = self.p4_to_p5(torch.cat([down4, p5], dim=1))

        return p3, p4, p5

