from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .layers import Conv, C2f, C2fCIB, SCDown, UpSample, make_divisible


class YOLOv10Neck(nn.Module):
    def __init__(self, width_mult: float = 1.0, depth_mult: float = 1.0, max_channels: int = 1024, c3: int = 128, c4: int = 256, c5: int = 512, cfg=None):
        super().__init__()

        HCH = cfg.HCH if cfg is not None else {}
        reps = cfg.reps if cfg is not None else {}
        types = cfg.types if cfg is not None else {}
        lk = cfg.lk if cfg is not None else {}

        self.upsample = UpSample()
        # P5 -> P4
        if types.get("p5_p4", "C2f") == "C2fCIB":
            self.p5_p4_c2f = C2fCIB(c5 + c4, HCH[13], n=reps.get(13, 1), shortcut=True, lk=lk.get("p5_p4", False))
        else:
            self.p5_p4_c2f = C2f(c5 + c4, HCH[13], n=reps.get(13, 1))
        # P4 -> P3
        self.p4_p3_c2f = C2f(HCH[13] + c3, HCH[16], n=reps.get(16, 1))
        # P3 -> P4
        self.p3_down = Conv(HCH[16], HCH[16], 3, 2)
        if types.get("p3_p4", "C2f") == "C2fCIB":
            self.p3_p4_c2f = C2fCIB(HCH[16] + HCH[13], HCH[19], n=reps.get(19, 1), shortcut=True)
        else:
            self.p3_p4_c2f = C2f(HCH[16] + HCH[13], HCH[19], n=reps.get(19, 1))
        # P4 -> P5
        self.p4_down = SCDown(HCH[19], HCH[19], 3, 2)
        self.p4_p5_c2f = C2fCIB(HCH[19] + c5, HCH[22], n=reps.get(22, 1), shortcut=True, lk=lk.get("p4_p5", False))

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
