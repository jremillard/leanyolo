from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .layers import Conv, C2f, C2fCIB, SPPF, PSA, SCDown


class YOLOv10Backbone(nn.Module):
    def __init__(self, in_channels: int = 3, width_mult: float = 1.0, depth_mult: float = 1.0, max_channels: int = 1024, variant: str = "s", cfg=None):
        super().__init__()

        CH = cfg.CH if cfg is not None else {}
        reps = cfg.reps if cfg is not None else {}
        types = cfg.types if cfg is not None else {}
        lk = cfg.lk if cfg is not None else {}

        self.cv0 = Conv(in_channels, CH[0], 3, 2)
        self.cv1 = Conv(CH[0], CH[1], 3, 2)
        self.c2 = C2f(CH[1], CH[2], n=reps.get(2, 1), shortcut=True)
        self.cv3 = Conv(CH[2], CH[3], 3, 2)
        self.c4 = C2f(CH[3], CH[4], n=reps.get(4, 1), shortcut=True)
        self.sc5 = SCDown(CH[4], CH[5], 3, 2)
        if types.get("c6", "C2f") == "C2fCIB":
            self.c6 = C2fCIB(CH[5], CH[6], n=reps.get(6, 1), shortcut=True)
        else:
            self.c6 = C2f(CH[5], CH[6], n=reps.get(6, 1), shortcut=True)
        self.sc7 = SCDown(CH[6], CH[7], 3, 2)
        if types.get("c8", "C2f") == "C2fCIB":
            self.c8 = C2fCIB(CH[7], CH[8], n=reps.get(8, 1), shortcut=True, lk=lk.get("c8", False))
        else:
            self.c8 = C2f(CH[7], CH[8], n=reps.get(8, 1), shortcut=True)
        self.sppf9 = SPPF(CH[8], CH[9])
        self.psa10 = PSA(CH[9], CH[10])

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
