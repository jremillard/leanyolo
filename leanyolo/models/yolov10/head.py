from __future__ import annotations

"""Alternative simple decoupled head (anchor‑free, minimal example).

Context
- YOLOv10’s main head (see ``head_v10.py``) uses DFL (distribution focal loss)
  with per‑bin logits for box regression and other design details.
- This file provides a compact, decoupled head that predicts 4 direct box
  offsets and class logits per location — useful for demonstrations and unit
  tests, not intended to exactly match the official head.

Inputs
- A tuple of feature maps (P3, P4, P5) from the neck, typically with strides
  (8, 16, 32). Shapes: (B, Ci, Hi, Wi)

Outputs
- A list of three tensors [Y3, Y4, Y5], where Yi has shape (B, 4+nc, Hi, Wi)
  with the first 4 channels as box offsets and the remaining nc channels as
  class logits (anchor‑free, per‑cell predictions).

Decoupled design (ASCII per scale)
    x → Stem(1×1) → ┌─ Conv3×3 → Conv1×1 → cls (nc)
                     └─ Conv3×3 → Conv1×1 → box (4)
    concat([box, cls]) → out (4+nc channels)

For background on decoupled heads and anchor‑free detection, see the README.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .layers import Conv, make_divisible


class DecoupledHead(nn.Module):
    """Simple anchor‑free, decoupled detection head.

    Produces per‑scale outputs of shape (B, 4+nc, H, W). The 4 channels
    correspond to box offsets; the remaining channels are class logits.
    """

    def __init__(self, *, ch: Tuple[int, int, int], num_classes: int, width_mult: float, max_channels: int):
        super().__init__()

        def c(v: int) -> int:
            return min(max_channels, make_divisible(int(v * width_mult)))

        self.nc = num_classes
        self.cls_layers = nn.ModuleList()
        self.box_layers = nn.ModuleList()
        self.stems = nn.ModuleList()

        for c_in in ch:
            hidden = c(max(64, c_in // 2))
            self.stems.append(Conv(c_in=c_in, c_out=hidden, k=1, s=1, p=None, g=1, act=True))

            # classification branch
            self.cls_layers.append(
                nn.Sequential(
                    Conv(c_in=hidden, c_out=hidden, k=3, s=1, p=None, g=1, act=True),
                    Conv(c_in=hidden, c_out=num_classes, k=1, s=1, p=None, g=1, act=False),
                )
            )
            # box regression branch
            self.box_layers.append(
                nn.Sequential(
                    Conv(c_in=hidden, c_out=hidden, k=3, s=1, p=None, g=1, act=True),
                    Conv(c_in=hidden, c_out=4, k=1, s=1, p=None, g=1, act=False),
                )
            )

    def forward(self, feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> List[torch.Tensor]:
        """Apply the head to (P3, P4, P5) and concatenate box/class outputs.

        Input: tuple of three tensors (B, Ci, Hi, Wi)
        Output: list of three tensors (B, 4+nc, Hi, Wi)
        """
        outs: List[torch.Tensor] = []
        for x, stem, cls, box in zip(feats, self.stems, self.cls_layers, self.box_layers):
            x = stem(x)
            c = cls(x)
            b = box(x)
            outs.append(torch.cat([b, c], dim=1))
        return outs
