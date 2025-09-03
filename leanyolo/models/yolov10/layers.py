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

    Goal
    - Provide a common conv+norm+activation building block used throughout
      the backbone and neck.

    Why it works
    - BatchNorm stabilizes optimization; SiLU is a smooth nonlinearity that
      works well in modern CNNs; grouped conv enables efficient depthwise ops.

    What it does
    - Applies a 2D conv with optional groups, then BN, then SiLU (or Identity).

    Args
    - c_in: input channels
    - c_out: output channels
    - k: kernel size (e.g., 1 or 3)
    - s: stride (1 keeps size, 2 downsamples)
    - p: padding (None → same padding for odd k)
    - g: groups (g=c_out yields depthwise conv)
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
    """Residual bottleneck: Conv(3×3) → Conv(3×3) with optional skip.

    Goal
    - Increase representational capacity while keeping gradients flowing.

    Why it works
    - Residual connections ease optimization; two small convolutions are
      efficient and expressive for spatial mixing.

    What it does
    - Reduces channels (if e<1), processes with Conv blocks, and optionally
      adds the input back if shapes match.

    Args
    - c_in: input channels
    - c_out: output channels
    - shortcut: enable residual add when c_in==c_out
    - g: groups for second conv (1 commonly)
    - e: expansion ratio for hidden channels (0<e≤1)

    ASCII
        x ──┐
            ├─ Conv3×3 → Conv3×3 ──┐
        ────┘      (add if shapes match) ─▶ y
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

    Goal
    - Capture richer features at low cost by splitting channels and reusing
      intermediate representations.

    Why it works
    - Provides a shortcut path and a deeper path; concatenation broadens the
      feature basis without expensive wide convolutions.

    What it does
    - Splits channels, runs a stack of Bottlenecks on one part, concatenates
      all intermediates, then fuses with a 1×1 conv.

    Args
    - c_in: input channels
    - c_out: output channels
    - n: number of inner Bottlenecks
    - shortcut: residual add inside Bottlenecks
    - g: groups for Bottlenecks
    - e: expansion ratio for internal channels

    ASCII
        x → Conv1×1 → [y1 | y2]
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
    """Spatial Pyramid Pooling – Fast (multi‑scale context).

    Goal
    - Enrich features with multi‑scale context without changing spatial size.

    Why it works
    - Reusing the same k×k max‑pool repeatedly approximates larger receptive
      fields and gathers context at different scales cheaply.

    What it does
    - Applies a 1×1 conv, then three successive max‑pools to create pyramids,
      concatenates them, and fuses with a 1×1 conv.

    Args
    - c_in: input channels
    - c_out: output channels
    - k: max‑pool kernel size (odd), stride fixed to 1

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
    """Nearest‑neighbor upsampling for top‑down neck fusion.

    Goal
    - Match spatial resolutions across scales for feature fusion.

    Why it works
    - Nearest‑neighbor is simple and fast, sufficient before 1×1 fusion convs.

    What it does
    - Uses interpolate with mode="nearest" by a given scale factor.

    Args
    - scale_factor: upsample factor (e.g., 2.0)
    """
    def __init__(self, *, scale_factor: float):
        super().__init__()
        self.scale = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")


class CIB(nn.Module):
    """Compact Inverted Block (paper’s efficiency‑driven block).

    Goal
    - Reduce compute by preferring depthwise spatial mixing and inexpensive
      pointwise channel mixing; optionally expand receptive field with a long‑
      kernel branch.

    Why it works
    - Depthwise separable convs drastically cut FLOPs/params; long‑kernel DW
      adds context affordably when enabled.

    What it does
    - Applies DWConv→PWConv, then either a RepVGGDW (7×7+3×3 DW) or DWConv,
      followed by PWConv and a final DWConv; residual add when shapes match.

    Args
    - c_in: input channels
    - c_out: output channels
    - shortcut: enable residual add when c_in==c_out
    - e: expansion ratio to internal channels
    - lk: enable long‑kernel RepVGGDW branch if True

    ASCII (simplified)
        x → DWConv3×3 → PWConv1×1 → [DWConv7×7 + DWConv3×3] or DWConv3×3
            → PWConv1×1 → DWConv3×3 → (+x) → y
    """
    def __init__(self, *, c_in: int, c_out: int, shortcut: bool, e: float, lk: bool):
        super().__init__()
        c_hidden = int(c_out * e)
        # Depthwise 3x3
        class RepVGGDW(nn.Module):
            """Depthwise RepVGG‑style branch (7×7 + 3×3) with SiLU.

            Goal: enlarge receptive field cheaply with depthwise long kernels.
            """
            def __init__(self, ch: int):
                super().__init__()
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
    """C2f with CIB inner blocks (accuracy/efficiency balance).

    Goal
    - Combine split‑transform‑merge with compact inverted blocks to trade a bit
      of capacity for efficiency where analysis finds redundancy.

    Why it works
    - YOLOv10 uses rank‑guided allocation to place compact blocks where stages
      are redundant; CIBs reduce cost with minimal accuracy loss.

    What it does
    - Same topology as C2f, but inner modules are CIB (optionally long‑kernel).

    Args
    - c_in, c_out, n, shortcut, e: as in C2f
    - lk: enable long‑kernel depthwise branches in CIB
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
    """Lightweight multi‑head self‑attention on spatial tokens.

    Goal
    - Inject global context into deep features at modest cost.

    Why it works
    - Treats H×W locations as tokens; attention reweights features based on
      content similarity while a small positional branch adds locality.

    What it does
    - Projects to q/k/v with 1×1 conv, computes attention per head, aggregates
      values, adds a depthwise 3×3 positional term, then projects back.

    Args
    - dim: channel dimension
    - num_heads: number of attention heads
    - attn_ratio: key dimension ratio relative to head dim (0<r≤1)
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
    """Partial Self‑Attention (accuracy‑driven context module).

    Goal
    - Add global context to deep features at low cost by attending only a
      portion of channels.

    Why it works
    - Half the channels pass unchanged (local path); the other half get
    **attention + MLP**. Fusing them injects context without full attention cost.

    What it does
    - Split channels with 1×1 conv, apply Attention + MLP to the second half,
      concatenate halves, and fuse.

    Args
    - c_in: input channels (must equal c_out)
    - c_out: output channels
    - e: expansion ratio for internal channels

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

    Goal
    - Lower downsampling cost vs. a single stride‑2 standard conv while keeping
      information flow healthy.

    Why it works
    - 1×1 pointwise conv handles channel change; depthwise stride‑s 3×3 handles
      spatial reduction with far fewer parameters/FLOPs than a full conv.

    What it does
    - Applies PWConv1×1 (C_in→C_out), then DWConv3×3 with stride s.

    Args
    - c_in: input channels
    - c_out: output channels
    - k: depthwise kernel (typically 3)
    - s: stride (e.g., 2)

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
