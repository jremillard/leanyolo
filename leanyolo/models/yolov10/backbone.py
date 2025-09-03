from __future__ import annotations

"""YOLOv10 Backbone (hierarchical feature extractor)

Purpose
- Turn an input image into a pyramid of semantically richer features at lower
  spatial resolutions. Outputs three maps (C3, C4, C5) commonly used by the
  neck and head.

Input/Output (typical)
- x: (B, 3, H, W)
- C3: (B, C3, H/8,  W/8)
- C4: (B, C4, H/16, W/16)
- C5: (B, C5, H/32, W/32)

Design highlights tied to the YOLOv10 paper
- Use SCDown (spatial–channel decoupled downsampling) instead of a single
  stride-2 standard conv for better efficiency at downsampling points.
- Allow C2fCIB (compact inverted block) in deeper stages to reduce compute with
  minimal accuracy impact, optionally enabling a long‑kernel depthwise branch.
- Add SPPF (multi-scale context) and a lightweight PSA (partial self-attention)
  at the end to inject global context efficiently.

Stage flow (ASCII)
    x → Conv s=2 → Conv s=2 → C2f → Conv s=2 → C2f →  C3 (stride 8)
                       │              │
                       └─ SCDown s=2 → C2f/C2fCIB →  C4 (stride 16)
                                       │
                                       └─ SCDown s=2 → C2f/C2fCIB → SPPF → PSA → C5 (stride 32)

See the README and the paper for background and rationale.
"""

from typing import Tuple, Dict

import torch
import torch.nn as nn

from .layers import Conv, C2f, C2fCIB, SPPF, PSA, SCDown


class YOLOv10Backbone(nn.Module):
    """Backbone producing multi-scale features (C3, C4, C5).

    Parameters
    - in_channels: input image channels (3 for RGB)
    - CH: dict mapping stage indices to channel widths; indices 0..10 correspond
      to the consecutive nodes in this graph (conv/c2f/scdown/sppf/psa)
    - reps: repeat counts for blocks at those stages (e.g., number of C2f repeats)
    - types: choose per-stage block type, e.g. {"c6": "C2fCIB", "c8": "C2f"}
    - use_lk_c8: if True and c8 uses C2fCIB, enable long‑kernel depthwise branch

    Returns
    - tuple (C3, C4, C5) with strides (8, 16, 32)
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
        """Compute (C3, C4, C5) pyramid.

        - C3: output of the first mid‑level C2f stage (stride 8)
        - C4: output of the next stage after SCDown + C2f/C2fCIB (stride 16)
        - C5: output after final SCDown, C2f/C2fCIB, SPPF, PSA (stride 32)
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
