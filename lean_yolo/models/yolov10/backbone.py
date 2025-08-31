from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .layers import Conv, C2f, SPPF, make_divisible


class YOLOv10Backbone(nn.Module):
    def __init__(self, in_channels: int = 3, width_mult: float = 1.0, depth_mult: float = 1.0, max_channels: int = 1024):
        super().__init__()

        def c(ch: int) -> int:
            return min(max_channels, make_divisible(int(ch * width_mult)))

        def d(n: int) -> int:
            # round depth, keep >=1 when n>0
            return max(int(round(n * depth_mult)), 1) if n > 0 else 0

        # Stem
        self.stem = Conv(in_channels, c(64), 3, 2)

        # Stages
        self.stage2 = nn.Sequential(
            Conv(c(64), c(128), 3, 2),
            C2f(c(128), c(128), n=d(3)),
        )
        self.stage3 = nn.Sequential(
            Conv(c(128), c(256), 3, 2),
            C2f(c(256), c(256), n=d(6)),
        )
        self.stage4 = nn.Sequential(
            Conv(c(256), c(512), 3, 2),
            C2f(c(512), c(512), n=d(6)),
        )
        self.stage5 = nn.Sequential(
            Conv(c(512), c(1024), 3, 2),
            C2f(c(1024), c(1024), n=d(3)),
            SPPF(c(1024), c(1024)),
        )

        self.out_c = (c(256), c(512), c(1024))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x2 = self.stage2(x)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        # return C3, C4, C5 pyramid
        return x3, x4, x5

