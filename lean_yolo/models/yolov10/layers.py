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
    def __init__(self, *, c_in: int, c_out: int, k: int, s: int, p: int | None, g: int, act: bool):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, _autopad(k, p), groups=g, bias=False)
        # Match Ultralytics BN defaults for YOLOv8/10
        self.bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, *, c_in: int, c_out: int, shortcut: bool, g: int, e: float):
        super().__init__()
        c_hidden = int(c_out * e)
        self.cv1 = Conv(c_in=c_in, c_out=c_hidden, k=3, s=1, p=None, g=1, act=True)
        self.cv2 = Conv(c_in=c_hidden, c_out=c_out, k=3, s=1, p=None, g=g, act=True)
        self.add = shortcut and c_in == c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """Lightweight C2f block (split, bottlenecks, concat) used in YOLOv8/10.

    This is a simplified variant that balances readability and performance.
    """

    def __init__(self, *, c_in: int, c_out: int, n: int, shortcut: bool, g: int, e: float):
        super().__init__()
        c = int(c_out * e)
        self.cv1 = Conv(c_in=c_in, c_out=2 * c, k=1, s=1, p=None, g=1, act=True)
        self.cv2 = Conv(c_in=(2 + n) * c, c_out=c_out, k=1, s=1, p=None, g=1, act=True)
        self.m = nn.ModuleList([Bottleneck(c_in=c, c_out=c, shortcut=shortcut, g=g, e=1.0) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y1, y2 = y.chunk(2, 1)
        ys = [y1, y2]
        for m in self.m:
            y2 = m(y2)
            ys.append(y2)
        return self.cv2(torch.cat(ys, 1))


class SPPF(nn.Module):
    def __init__(self, *, c_in: int, c_out: int, k: int):
        super().__init__()
        c_hidden = c_in // 2
        self.cv1 = Conv(c_in=c_in, c_out=c_hidden, k=1, s=1, p=None, g=1, act=True)
        self.cv2 = Conv(c_in=c_hidden * 4, c_out=c_out, k=1, s=1, p=None, g=1, act=True)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class UpSample(nn.Module):
    def __init__(self, *, scale_factor: float):
        super().__init__()
        self.scale = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")


class CIB(nn.Module):
    def __init__(self, *, c_in: int, c_out: int, shortcut: bool, e: float, lk: bool):
        super().__init__()
        c_hidden = int(c_out * e)
        # Depthwise 3x3
        class RepVGGDW(nn.Module):
            def __init__(self, ch: int):
                super().__init__()
                # Match Ultralytics RepVGGDW: depthwise 7x7 and 3x3 branches
                self.conv = Conv(c_in=ch, c_out=ch, k=7, s=1, p=3, g=ch, act=False)
                self.conv1 = Conv(c_in=ch, c_out=ch, k=3, s=1, p=1, g=ch, act=False)
                self.act = nn.SiLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.act(self.conv(x) + self.conv1(x))

        mid = 2 * c_hidden
        self.cv1 = nn.Sequential(
            Conv(c_in=c_in, c_out=c_in, k=3, s=1, p=None, g=c_in, act=True),
            Conv(c_in=c_in, c_out=mid, k=1, s=1, p=None, g=1, act=True),
            (RepVGGDW(mid) if lk else Conv(c_in=mid, c_out=mid, k=3, s=1, p=None, g=mid, act=True)),
            Conv(c_in=mid, c_out=c_out, k=1, s=1, p=None, g=1, act=True),
            Conv(c_in=c_out, c_out=c_out, k=3, s=1, p=None, g=c_out, act=True),
        )
        self.add = shortcut and c_in == c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        return x + y if self.add else y


class C2fCIB(nn.Module):
    def __init__(self, *, c_in: int, c_out: int, n: int, shortcut: bool, lk: bool, e: float):
        super().__init__()
        c = int(c_out * e)
        self.cv1 = Conv(c_in=c_in, c_out=2 * c, k=1, s=1, p=None, g=1, act=True)
        self.cv2 = Conv(c_in=(2 + n) * c, c_out=c_out, k=1, s=1, p=None, g=1, act=True)
        self.m = nn.ModuleList([CIB(c_in=c, c_out=c, shortcut=shortcut, e=1.0, lk=lk) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y1, y2 = y.chunk(2, 1)
        ys = [y1, y2]
        for m in self.m:
            y2 = m(y2)
            ys.append(y2)
        return self.cv2(torch.cat(ys, 1))


class Attention(nn.Module):
    def __init__(self, *, dim: int, num_heads: int, attn_ratio: float):
        super().__init__()
        self.num_heads = max(1, num_heads)
        self.head_dim = dim // self.num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * self.num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(c_in=dim, c_out=h, k=1, s=1, p=None, g=1, act=False)
        self.proj = Conv(c_in=dim, c_out=dim, k=1, s=1, p=None, g=1, act=False)
        self.pe = Conv(c_in=dim, c_out=dim, k=3, s=1, p=None, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x_attn = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        x_proj = self.proj(x_attn)
        return x_proj


class PSA(nn.Module):
    def __init__(self, *, c_in: int, c_out: int, e: float):
        super().__init__()
        assert c_in == c_out
        self.c = int(c_in * e)
        self.cv1 = Conv(c_in=c_in, c_out=2 * self.c, k=1, s=1, p=None, g=1, act=True)
        self.cv2 = Conv(c_in=2 * self.c, c_out=c_in, k=1, s=1, p=None, g=1, act=True)
        self.attn = Attention(dim=self.c, attn_ratio=0.5, num_heads=max(1, self.c // 64))
        self.ffn = nn.Sequential(
            Conv(c_in=self.c, c_out=self.c * 2, k=1, s=1, p=None, g=1, act=True),
            Conv(c_in=self.c * 2, c_out=self.c, k=1, s=1, p=None, g=1, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), dim=1))


class SCDown(nn.Module):
    def __init__(self, *, c_in: int, c_out: int, k: int, s: int):
        super().__init__()
        self.cv1 = Conv(c_in=c_in, c_out=c_out, k=1, s=1, p=None, g=1, act=True)
        self.cv2 = Conv(c_in=c_out, c_out=c_out, k=k, s=s, p=None, g=c_out, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))
