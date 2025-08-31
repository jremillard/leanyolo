from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .layers import Conv, make_divisible


class DecoupledHead(nn.Module):
    """Simple anchor-free, decoupled detection head.

    Produces per-scale outputs of shape [B, (num_classes + 4), H, W].
    """

    def __init__(self, ch: Tuple[int, int, int], num_classes: int, width_mult: float = 1.0, max_channels: int = 1024):
        super().__init__()

        def c(v: int) -> int:
            return min(max_channels, make_divisible(int(v * width_mult)))

        self.nc = num_classes
        self.cls_layers = nn.ModuleList()
        self.box_layers = nn.ModuleList()
        self.stems = nn.ModuleList()

        for c_in in ch:
            hidden = c(max(64, c_in // 2))
            self.stems.append(Conv(c_in, hidden, 1, 1))

            # classification branch
            self.cls_layers.append(
                nn.Sequential(
                    Conv(hidden, hidden, 3, 1),
                    Conv(hidden, num_classes, 1, 1, act=False),
                )
            )
            # box regression branch
            self.box_layers.append(
                nn.Sequential(
                    Conv(hidden, hidden, 3, 1),
                    Conv(hidden, 4, 1, 1, act=False),
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

