from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .layers import Conv, C2f, C2fCIB, SCDown, UpSample, make_divisible


class YOLOv10Neck(nn.Module):
    def __init__(self, *, c3: int, c4: int, c5: int, cfg):
        super().__init__()

        HCH = cfg.HCH if cfg is not None else {}
        reps = cfg.reps if cfg is not None else {}
        types = cfg.types if cfg is not None else {}
        lk = cfg.lk if cfg is not None else {}

        self.upsample = UpSample(scale_factor=2.0)
        # P5 -> P4
        if types.get("p5_p4", "C2f") == "C2fCIB":
            self.p5_p4_c2f = C2fCIB(c_in=c5 + c4, c_out=HCH[13], n=reps.get(13, 1), shortcut=True, lk=lk.get("p5_p4", False), e=0.5)
        else:
            self.p5_p4_c2f = C2f(c_in=c5 + c4, c_out=HCH[13], n=reps.get(13, 1), shortcut=False, g=1, e=0.5)
        # P4 -> P3
        self.p4_p3_c2f = C2f(c_in=HCH[13] + c3, c_out=HCH[16], n=reps.get(16, 1), shortcut=False, g=1, e=0.5)
        # P3 -> P4
        self.p3_down = Conv(c_in=HCH[16], c_out=HCH[16], k=3, s=2, p=None, g=1, act=True)
        if types.get("p3_p4", "C2f") == "C2fCIB":
            self.p3_p4_c2f = C2fCIB(c_in=HCH[16] + HCH[13], c_out=HCH[19], n=reps.get(19, 1), shortcut=True, lk=False, e=0.5)
        else:
            self.p3_p4_c2f = C2f(c_in=HCH[16] + HCH[13], c_out=HCH[19], n=reps.get(19, 1), shortcut=False, g=1, e=0.5)
        # P4 -> P5
        self.p4_down = SCDown(c_in=HCH[19], c_out=HCH[19], k=3, s=2)
        self.p4_p5_c2f = C2fCIB(c_in=HCH[19] + c5, c_out=HCH[22], n=reps.get(22, 1), shortcut=True, lk=lk.get("p4_p5", False), e=0.5)

        self.out_c = (HCH[16], HCH[19], HCH[22])

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Top-down
        up4 = self.upsample(c5)
        p4 = self.p5_p4_c2f(torch.cat([up4, c4], dim=1))

        up3 = self.upsample(p4)
        p3 = self.p4_p3_c2f(torch.cat([up3, c3], dim=1))

        # Bottom-up
        down3 = self.p3_down(p3)
        p4 = self.p3_p4_c2f(torch.cat([down3, p4], dim=1))

        down4 = self.p4_down(p4)
        p5 = self.p4_p5_c2f(torch.cat([down4, c5], dim=1))

        return p3, p4, p5
