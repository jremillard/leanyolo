from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .layers import Conv, make_divisible


class DecoupledHead(nn.Module):
    """Simple anchor-free, decoupled detection head.

    Produces per-scale outputs of shape [B, (num_classes + 4), H, W].
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
        outs: List[torch.Tensor] = []
        for x, stem, cls, box in zip(feats, self.stems, self.cls_layers, self.box_layers):
            x = stem(x)
            c = cls(x)
            b = box(x)
            outs.append(torch.cat([b, c], dim=1))
        return outs
