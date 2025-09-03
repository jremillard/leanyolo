from __future__ import annotations

"""YOLOv10 Neck (PAN-FPN style multi-scale fusion)

Purpose
- Fuse multi-scale backbone features using a top-down path and a bottom-up path
  so each output scale benefits from both local detail and global context.

Inputs (typical strides)
- ``c3``: (B, C3, H/8,  W/8)
- ``c4``: (B, C4, H/16, W/16)
- ``c5``: (B, C5, H/32, W/32)

Outputs
- ``p3``: (B, P3C, H/8,  W/8)
- ``p4``: (B, P4C, H/16, W/16)
- ``p5``: (B, P5C, H/32, W/32)

Design notes (aligned with the YOLOv10 paper)
- Top-down uses nearest-neighbor upsample to reduce cost.
- Bottom-up uses an efficient downsample. We employ ``SCDown`` (spatial–channel
  decoupled downsampling) on the second leg to reduce FLOPs while preserving
  information.
- Merge blocks can be the classic ``C2f`` or the efficiency-oriented ``C2fCIB``
  (which wraps compact inverted blocks, CIB). Some merges allow an optional
  long-kernel depthwise branch for slightly larger receptive fields.

ASCII overview
    Top-down (coarse → fine)
        c5 ──Upsample──▶ concat(c4) ─▶ C2f/C2fCIB ─▶ p4
        p4 ──Upsample──▶ concat(c3) ─▶ C2f        ─▶ p3

    Bottom-up (fine → coarse)
        p3 ──Conv s=2──▶ concat(p4) ─▶ C2f/C2fCIB ─▶ p4
        p4 ──SCDown s=2▶ concat(c5) ─▶ C2fCIB     ─▶ p5

See the README and the paper for background and rationale behind the block
choices.
"""

from typing import Tuple, Dict

import torch
import torch.nn as nn

from .layers import Conv, C2f, C2fCIB, SCDown, UpSample


class YOLOv10Neck(nn.Module):
    """Fuse backbone features into P3, P4, P5 for the detection head.

    Parameters
    - c3, c4, c5: channel counts of backbone outputs ``c3``, ``c4``, ``c5``.
    - HCH: mapping of internal node ids → channel width for the fused tensors.
      This mirrors widths used in the original graphs; keys 13/16/19/22
      correspond to the four fusion nodes shown in the ASCII overview.
    - reps: number of inner block repetitions at those fusion nodes.
    - types: choose block type per merge, e.g. ``{"p5_p4": "C2fCIB", "p3_p4": "C2f"}``.
    - use_lk_p5_p4: if ``True`` and using ``C2fCIB`` on P5→P4, enable long-kernel
      depthwise branch (slightly larger receptive field).
    - use_lk_p4_p5: same, for the final P4→P5 fusion.

    Returns
    - tuple ``(p3, p4, p5)`` with channels ``(HCH[16], HCH[19], HCH[22])`` and
      strides matching ``(c3, c4, c5)``.
    """
    def __init__(
        self,
        *,
        c3: int,
        c4: int,
        c5: int,
        HCH: Dict[int, int],
        reps: Dict[int, int],
        types: Dict[str, str],
        use_lk_p5_p4: bool,
        use_lk_p4_p5: bool,
    ):
        super().__init__()


        self.upsample = UpSample(scale_factor=2.0)
        # P5 -> P4
        if types.get("p5_p4", "C2f") == "C2fCIB":
            self.p5_p4_c2f = C2fCIB(c_in=c5 + c4, c_out=HCH[13], n=reps.get(13, 1), shortcut=True, lk=use_lk_p5_p4, e=0.5)
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
        self.p4_p5_c2f = C2fCIB(c_in=HCH[19] + c5, c_out=HCH[22], n=reps.get(22, 1), shortcut=True, lk=use_lk_p4_p5, e=0.5)

        self.out_c = (HCH[16], HCH[19], HCH[22])

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top‑down and bottom‑up fusion.

        Inputs
        - c3: (B, c3, H/8,  W/8)
        - c4: (B, c4, H/16, W/16)
        - c5: (B, c5, H/32, W/32)

        Outputs
        - p3: (B, HCH[16], H/8,  W/8)
        - p4: (B, HCH[19], H/16, W/16)
        - p5: (B, HCH[22], H/32, W/32)
        """
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
