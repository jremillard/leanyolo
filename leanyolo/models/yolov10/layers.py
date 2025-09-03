from __future__ import annotations

"""Core YOLOv10 building blocks used by the backbone and neck.

This file implements the small, composable modules that make up the backbone and
neck in the lean YOLOv10 models. The designs follow the intent of the YOLOv10
paper ("YOLOv10: Real-Time End-to-End Object Detection"), which emphasizes:

- Efficiency-driven blocks: compact inverted blocks (CIB) and spatial–channel
  decoupled downsampling (SCDown) to reduce FLOPs without hurting accuracy.
- Accuracy-driven context: a partial self-attention (PSA) module to inject
  lightweight global context in later stages where features are more semantic.
- Standard multi-scale fusion: keep proven components like C2f and SPPF for
  rich feature reuse and multi-scale context.

If you’ve seen basic CNNs, you can think of these as familiar conv blocks with
residual connections, some efficient depthwise tricks, and a small attention
layer where it yields the most benefit.

Legend
- B, C, H, W: batch, channels, height, width
- DWConv: depthwise conv (groups=channels)
- PWConv: pointwise (1x1) conv
- BN+SiLU: BatchNorm + SiLU activation

Reference mapping to the paper
- SCDown here corresponds to the "spatial–channel decoupled downsampling"
  optimization described in the paper’s efficiency design.
- CIB and C2fCIB implement the compact inverted block and its use inside a C2f
  (split-transform-merge) scaffold.
- PSA implements the partial self-attention block the paper places at deeper
  stages to add global context at low cost.
"""

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
    """Conv → BN → SiLU convenience block.

    Input/Output
    - x: (B, c_in, H, W)
    - y: (B, c_out, H/stride, W/stride)

    Args
    - c_in: input channels
    - c_out: output channels
    - k: kernel size (1 or 3 are common)
    - s: stride (1 keeps size, 2 downsamples)
    - p: padding (None gives "same" for odd k)
    - g: groups (g=c_out means depthwise)
    - act: apply SiLU if True, else Identity

    ASCII
        [C_in,H,W]
           │ Conv(k,s,g)
           ▼
        [C_out,H',W'] → BN → SiLU
    """
    def __init__(self, *, c_in: int, c_out: int, k: int, s: int, p: int | None, g: int, act: bool):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, _autopad(k, p), groups=g, bias=False)
        # Match Ultralytics BN defaults for YOLOv8/10
        self.bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Residual bottleneck: Conv(3x3) → Conv(3x3) with optional skip.

    Purpose: stabilize training and increase capacity with little cost.

    ASCII
        x ──┐
            ├─ Conv3×3 → Conv3×3 ──┐
        ────┘            (add if shapes match) ─▶ y
    """
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
    """C2f: split → transform (stack) → concat → fuse.

    Rationale: splitting channels allows a cheap path (skip) and an expressive
    path (n small bottlenecks). Concatenation reuses intermediate features.

    ASCII
        x → Conv1×1 → [y1 | y2]  (split channels)
                       │
                       ├─ Bottleneck → Bottleneck → … (n×) → y2'
        concat([y1, y2, y2', …]) → Conv1×1 → out
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
    """Spatial Pyramid Pooling – Fast (multi-scale context).

    Idea: repeatedly apply a k×k max-pool (stride 1) to gather context at
    increasing receptive fields, then concatenate with the original and fuse.

    ASCII
        x → Conv1×1 → x0
               │  MaxPool(k)
               ├─ x1
               │  MaxPool(k)
               ├─ x2
               │  MaxPool(k)
               └─ x3
        concat([x0,x1,x2,x3]) → Conv1×1 → out
    """
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
    """Nearest-neighbor upsampling for top-down neck fusion.

    Simple and fast; doubles resolution when ``scale_factor=2``.
    """
    def __init__(self, *, scale_factor: float):
        super().__init__()
        self.scale = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")


class CIB(nn.Module):
    """Compact Inverted Block (paper’s efficiency-driven block).

    Design: use cheap depthwise spatial mixing and 1×1 pointwise channel mixing.
    If ``lk=True`` we add a long-kernel depthwise branch (7×7 + 3×3) in the
    style of RepVGGDW for larger receptive fields.

    ASCII (simplified)
        x → DWConv3×3 → PWConv1×1 → [DWConv7×7 + DWConv3×3] → PWConv1×1 → DWConv3×3 → y
        (optional skip-add if shapes match)
    """
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
    """C2f variant that uses CIB blocks inside (accuracy/efficiency balance).

    Same split-transform-merge idea as :class:`C2f`, but each inner block is a
    :class:`CIB`, optionally with long-kernel depthwise branches.
    """
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
    """Lightweight multi-head self-attention on spatial tokens.

    Treat the H×W grid as a sequence of tokens per head. Compute
    q·k^T/√d → softmax → v·attn, add a shallow positional branch, then project.

    Shapes
    - x: (B, C, H, W) → n=H·W tokens per head
    - q,k: (B, heads, d_k, n), v: (B, heads, d_v, n)

    Notes: Kept intentionally small (few heads, 1×1 conv projections) to fit
    real-time budgets.
    """
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
    """Partial Self-Attention (paper’s accuracy-driven context module).

    Split channels in half: leave one half as a local path, apply
    lightweight attention + MLP to the other half, then fuse and project.
    This injects global context at low cost (only half the channels attend).

    ASCII
        x → Conv1×1 → [a | b]
                 b ← b + Attention(b)
                 b ← b + MLP(b)
        concat([a,b]) → Conv1×1 → out
    """
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
    """Spatial–Channel decoupled downsampling (paper’s efficiency idea).

    Replace a single costly 3×3 stride-2 standard conv with:
    - 1×1 PWConv to set channels, then
    - 3×3 DWConv with stride s for spatial reduction.

    This reduces parameters/FLOPs while retaining information.

    ASCII
        x → PWConv1×1 (C_in→C_out)
            → DWConv3×3 (stride=s) → out
    """
    def __init__(self, *, c_in: int, c_out: int, k: int, s: int):
        super().__init__()
        self.cv1 = Conv(c_in=c_in, c_out=c_out, k=1, s=1, p=None, g=1, act=True)
        self.cv2 = Conv(c_in=c_out, c_out=c_out, k=k, s=s, p=None, g=c_out, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))
