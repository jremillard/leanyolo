from __future__ import annotations

"""YOLOv10 Backbone

This module builds a hierarchical feature extractor from an input image. The
backbone gradually downsamples the spatial resolution while increasing channels,
producing three feature maps at different scales (commonly called C3, C4, C5).

If you have seen ResNet or similar CNNs, this is the analogous part that turns
pixels into meaningful feature tensors for downstream detection heads.

See the README for links to the YOLOv10 paper and background references.
"""

from typing import Tuple, Dict

import torch
import torch.nn as nn

from .layers import Conv, C2f, C2fCIB, SPPF, PSA, SCDown


class YOLOv10Backbone(nn.Module):
    """Backbone producing multi-scale features (C3, C4, C5).

    Args:
        in_channels: Input image channels, usually 3 for RGB.
        CH: Channel dictionary for each stage (indices align with paper/code).
        reps: How many times to repeat certain blocks at each stage.
        types: Which block variant to use at specific points (e.g., C2f vs C2fCIB).
        use_lk_c8: Whether to enable the long-kernel depthwise path in the c8 block.
    """
    def __init__(
        self,
        *,
        in_channels: int,
        CH: Dict[int, int],
        reps: Dict[int, int],
        types: Dict[str, str],
        use_lk_c8: bool,
    ):
        super().__init__()


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
            self.c8 = C2fCIB(c_in=CH[7], c_out=CH[8], n=reps.get(8, 1), shortcut=True, lk=use_lk_c8, e=0.5)
        else:
            self.c8 = C2f(c_in=CH[7], c_out=CH[8], n=reps.get(8, 1), shortcut=True, g=1, e=0.5)
        self.sppf9 = SPPF(c_in=CH[8], c_out=CH[9], k=5)
        self.psa10 = PSA(c_in=CH[9], c_out=CH[10], e=0.5)

        self.out_c = (CH[3], CH[5], CH[7])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute backbone features.

        Returns:
            Tuple of tensors (C3, C4, C5) at progressively lower resolutions.
        """
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
