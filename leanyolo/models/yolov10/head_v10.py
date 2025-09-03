from __future__ import annotations

"""YOLOv10 Detection Head (training‑style outputs)

Goal
- Predict class probabilities and high‑quality box regressions from P3/P4/P5
  features using a decoupled head suitable for end‑to‑end training.

Why it works
- Decoupled branches (cls/reg) reduce competition and are a proven design in
  modern YOLOs (e.g., YOLOv5/6/8). YOLOv10 follows this pattern and pairs it
  with distributional box regression (DFL) to improve localization quality.
  The paper also trains with consistent dual assignments (one‑to‑many and
  one‑to‑one) to enable NMS‑free inference, which this head supports by
  exposing both branches.

What it does
- For each scale, apply a small regression stack to output 4×reg_max logits and
  a classification stack to output nc logits. Optionally compute the expected
  distances via DFL. In training, return both one‑to‑many and one‑to‑one raw
  outputs; in eval, return the one‑to‑many branch.
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .layers import Conv


class DFL(nn.Module):
    """Distribution Focal Loss projection (logits → expected distances).

    Goal
    - Convert per‑bin logits for left/top/right/bottom distances into expected
      continuous offsets for better localization stability.

    Why it works
    - Modeling box sides as categorical distributions over discrete bins (DFL)
      captures uncertainty and sub‑pixel precision better than a single scalar
      regression target; taking the expectation yields smooth, accurate offsets.

    What it does
    - Initializes a fixed 1×1 conv with weights [0..reg_max‑1] that computes
      the expectation over softmaxed bin logits. Operates independently per
      side and per location.
    """
    def __init__(self, *, c1: int):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False)
        x = torch.arange(c1, dtype=torch.float)
        with torch.no_grad():
            self.conv.weight.copy_(x.view(1, c1, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class V10Detect(nn.Module):
    """Decoupled YOLOv10 detection head.

    Goal
    - Produce raw per‑scale tensors for classification and DFL‑style box
      regression, suitable for dual‑assignment training.

    Why it works
    - Separate light branches for cls/reg improve convergence and accuracy; DFL
      improves box quality. Dual branches (one‑to‑many and one‑to‑one) align
      with YOLOv10’s NMS‑free training paradigm.

    What it does
    - Builds per‑scale conv stacks for regression (→ 4×reg_max) and
      classification (→ nc). Exposes cloned branches for one‑to‑one training.
      The forward returns dicts in training and lists in eval for downstream
      loss/decoding.

    Args
    - nc: number of classes
    - ch: channels of (P3,P4,P5)
    - reg_max: DFL bin count per side
    """
    def __init__(self, *, nc: int, ch: Sequence[int], reg_max: int):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.stride = torch.zeros(self.nl)

        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(self.nc, 100))

        # Regression branch per level
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(c_in=x, c_out=c2, k=3, s=1, p=None, g=1, act=True),
                Conv(c_in=c2, c_out=c2, k=3, s=1, p=None, g=1, act=True),
                nn.Conv2d(c2, 4 * reg_max, 1),
            )
            for x in ch
        )
        # Classification branch per level
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(
                    Conv(c_in=x, c_out=x, k=3, s=1, p=None, g=x, act=True),
                    Conv(c_in=x, c_out=c3, k=1, s=1, p=None, g=1, act=True),
                ),
                nn.Sequential(
                    Conv(c_in=c3, c_out=c3, k=3, s=1, p=None, g=c3, act=True),
                    Conv(c_in=c3, c_out=c3, k=1, s=1, p=None, g=1, act=True),
                ),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )

        # One-to-one branches (copied)
        import copy as _copy
        self.one2one_cv2 = _copy.deepcopy(self.cv2)
        self.one2one_cv3 = _copy.deepcopy(self.cv3)

        self.dfl = DFL(c1=self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward_feat(self, x: Sequence[torch.Tensor], cv2: nn.ModuleList, cv3: nn.ModuleList) -> List[torch.Tensor]:
        y: List[torch.Tensor] = []
        for i in range(self.nl):
            y.append(torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1))
        return y

    def forward(self, x: Sequence[torch.Tensor]):
        """Return raw outputs.

        - Training: return dict with 'one2many' and 'one2one' branches (lists of 3 tensors each).
        - Eval: return one2many branch only (list of 3 tensors).
        """
        if self.training:
            return {
                "one2many": self.forward_feat(x, self.cv2, self.cv3),
                "one2one": self.forward_feat(x, self.one2one_cv2, self.one2one_cv3),
            }
        return self.forward_feat(x, self.cv2, self.cv3)
