from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .layers import Conv, C2f, C2fCIB, SPPF, PSA, SCDown


class YOLOv10Backbone(nn.Module):
    def __init__(self, *, in_channels: int, cfg):
        super().__init__()

        CH = cfg.CH if cfg is not None else {}
        reps = cfg.reps if cfg is not None else {}
        types = cfg.types if cfg is not None else {}
        lk = cfg.lk if cfg is not None else {}

        self.cv0 = Conv(c_in=in_channels, c_out=CH[0], k=3, s=2, p=None, g=1, act=True)
        self.cv1 = Conv(c_in=CH[0], c_out=CH[1], k=3, s=2, p=None, g=1, act=True)
        self.c2 = C2f(c_in=CH[1], c_out=CH[2], n=reps.get(2, 1), shortcut=True, g=1, e=0.5)
        self.cv3 = Conv(c_in=CH[2], c_out=CH[3], k=3, s=2, p=None, g=1, act=True)
        self.c4 = C2f(c_in=CH[3], c_out=CH[4], n=reps.get(4, 1), shortcut=True, g=1, e=0.5)
        self.sc5 = SCDown(c_in=CH[4], c_out=CH[5], k=3, s=2)
        if types.get("c6", "C2f") == "C2fCIB":
            self.c6 = C2fCIB(c_in=CH[5], c_out=CH[6], n=reps.get(6, 1), shortcut=True, lk=False, e=0.5)
        else:
            self.c6 = C2f(c_in=CH[5], c_out=CH[6], n=reps.get(6, 1), shortcut=True, g=1, e=0.5)
        self.sc7 = SCDown(c_in=CH[6], c_out=CH[7], k=3, s=2)
        if types.get("c8", "C2f") == "C2fCIB":
            self.c8 = C2fCIB(c_in=CH[7], c_out=CH[8], n=reps.get(8, 1), shortcut=True, lk=lk.get("c8", False), e=0.5)
        else:
            self.c8 = C2f(c_in=CH[7], c_out=CH[8], n=reps.get(8, 1), shortcut=True, g=1, e=0.5)
        self.sppf9 = SPPF(c_in=CH[8], c_out=CH[9], k=5)
        self.psa10 = PSA(c_in=CH[9], c_out=CH[10], e=0.5)

        self.out_c = (CH[3], CH[5], CH[7])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.cv0(x)
        x = self.cv1(x)
        x = self.c2(x)
        x = self.cv3(x)
        c3 = self.c4(x)
        x = self.sc5(c3)
        c4 = self.c6(x)
        x = self.sc7(c4)
        x = self.c8(x)
        x = self.sppf9(x)
        c5 = self.psa10(x)
        return c3, c4, c5
