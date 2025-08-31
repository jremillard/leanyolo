from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def make_divisible(v: int, divisor: int = 8) -> int:
    return int(math.ceil(v / divisor) * divisor)


def _autopad(k: int, p: int | None = None) -> int:
    # same padding by default
    return k // 2 if p is None else p


class Conv(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, _autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c_in: int, c_out: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_hidden = int(c_out * e)
        self.cv1 = Conv(c_in, c_hidden, 1, 1)
        self.cv2 = Conv(c_hidden, c_out, 3, 1, g=g)
        self.add = shortcut and c_in == c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """Lightweight C2f block (split, bottlenecks, concat) used in YOLOv8/10.

    This is a simplified variant that balances readability and performance.
    """

    def __init__(self, c_in: int, c_out: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c = int(c_out * e)
        self.cv1 = Conv(c_in, 2 * c, 1, 1)
        self.cv2 = Conv((2 + n) * c, c_out, 1)
        self.m = nn.ModuleList([Bottleneck(c, c, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y1, y2 = y.chunk(2, 1)
        ys = [y1, y2]
        for m in self.m:
            y2 = m(y2)
            ys.append(y2)
        return self.cv2(torch.cat(ys, 1))


class SPPF(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 5):
        super().__init__()
        c_hidden = c_in // 2
        self.cv1 = Conv(c_in, c_hidden, 1, 1)
        self.cv2 = Conv(c_hidden * 4, c_out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class UpSample(nn.Module):
    def __init__(self, scale_factor: float = 2.0):
        super().__init__()
        self.scale = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")

