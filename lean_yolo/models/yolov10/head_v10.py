from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .layers import Conv


class DFL(nn.Module):
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

    def forward(self, x: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        # Return training-style outputs per feature level
        return self.forward_feat(x, self.cv2, self.cv3)
