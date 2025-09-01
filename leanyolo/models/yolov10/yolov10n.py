from __future__ import annotations

"""YOLOv10-n model definition.

The nano (n) variant trades accuracy for speed by using smaller channel sizes
and fewer block repetitions. Structure is identical to other variants.

Input format:
- Tensor layout: CHW, shape (N, C, H, W)
- Color order: RGB (not BGR)
- Dtype/range: float32 in [0, 1] (scale by 1/255)
- Tip: If loading images with OpenCV (BGR), convert to RGB first

Output format:
- Returns a list of 3 tensors [P3, P4, P5]
- Each tensor has shape (N, 4*reg_max + num_classes, H, W)
- Channels 0..(4*reg_max-1): DFL logits for [l, t, r, b] distances
- Channels (4*reg_max)..: class logits (unnormalized)
- Post-processing: use leanyolo.utils.postprocess.decode_predictions and NMS

Note:
- These YOLOv10x classes (family) are raw model modules. For inference, wrap
  preprocessing (RGB, letterbox, normalization) and postprocessing (decode,
  NMS, unletterbox). See leanyolo.engine.infer for a reference pipeline.
"""

from typing import List

import torch
import torch.nn as nn

from .backbone import YOLOv10Backbone
from .neck import YOLOv10Neck
from .head_v10 import V10Detect


class YOLOv10n(nn.Module):
    CH = {0: 16, 1: 32, 2: 32, 3: 64, 4: 64, 5: 128, 6: 128, 7: 256, 8: 256, 9: 256, 10: 256}
    HCH = {13: 128, 16: 64, 19: 128, 22: 256}
    REPS = {2: 1, 4: 2, 6: 2, 8: 1, 13: 1, 16: 1, 19: 1, 22: 1}
    TYPES = {"c6": "C2f", "c8": "C2f", "p5_p4": "C2f", "p3_p4": "C2f", "p4_p5": "C2fCIB"}

    def __init__(self, *, num_classes: int, in_channels: int):
        super().__init__()
        self.backbone = YOLOv10Backbone(
            in_channels=in_channels,
            CH=self.CH,
            reps=self.REPS,
            types=self.TYPES,
            use_lk_c8=False,
        )
        c3, c4, c5 = self.backbone.out_c
        self.neck = YOLOv10Neck(
            c3=c3,
            c4=c4,
            c5=c5,
            HCH=self.HCH,
            reps=self.REPS,
            types=self.TYPES,
            use_lk_p5_p4=False,
            use_lk_p4_p5=True,
        )
        p3, p4, p5 = self.neck.out_c
        self.head = V10Detect(nc=num_classes, ch=(p3, p4, p5), reg_max=16)
        self._init_head_bias()

    def _init_head_bias(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels in (4,) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c3, c4, c5)
        return self.head((p3, p4, p5))
