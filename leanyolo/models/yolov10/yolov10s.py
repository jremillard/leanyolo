from __future__ import annotations

"""YOLOv10-s model definition.

This is the small (s) variant of YOLOv10. Variants differ mainly in channel
counts and block repetitions; the code below wires up the backbone, neck, and
head with the appropriate sizes for this model.

If you are new to detection models, read the README’s high-level overview and
paper references. The forward pass is: image → backbone (C3/C4/C5) → neck
(P3/P4/P5) → head (per-scale outputs).

Input format:
- Tensor layout: CHW, shape (N, C, H, W)
- Color order: RGB (not BGR)
- Dtype/range: float32 in [0, 1] (scale by 1/255)
- Tip: If loading images with OpenCV (BGR), convert to RGB first

- Image size: Models are fully convolutional and accept arbitrary H×W. 640 is
  a convenient default (divisible by 32 for strides 8/16/32 → 80/40/20 grids).

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

from typing import List, Sequence

import torch
import torch.nn as nn

from .backbone import YOLOv10Backbone
from .neck import YOLOv10Neck
from .head_v10 import V10Detect


class YOLOv10s(nn.Module):
    CH = {0: 32, 1: 64, 2: 64, 3: 128, 4: 128, 5: 256, 6: 256, 7: 512, 8: 512, 9: 512, 10: 512}
    HCH = {13: 256, 16: 128, 19: 256, 22: 512}
    REPS = {2: 1, 4: 2, 6: 2, 8: 1, 13: 1, 16: 1, 19: 1, 22: 1}
    TYPES = {"c6": "C2f", "c8": "C2fCIB", "p5_p4": "C2f", "p3_p4": "C2f", "p4_p5": "C2fCIB"}

    def __init__(self, *, class_names: Sequence[str], in_channels: int, input_norm_subtract: Sequence[float], input_norm_divide: Sequence[float]):
        super().__init__()
        self.class_names = list(class_names)
        # Input normalization buffers
        import torch
        sub = torch.tensor(list(input_norm_subtract), dtype=torch.float32).view(1, in_channels, 1, 1)
        div = torch.tensor(list(input_norm_divide), dtype=torch.float32).view(1, in_channels, 1, 1)
        self.register_buffer("input_subtract", sub)
        self.register_buffer("input_divide", div)
        self._skip_subtract = bool((sub == 0).all().item())
        self._skip_divide = bool((div == 1).all().item())
        self.backbone = YOLOv10Backbone(
            in_channels=in_channels,
            CH=self.CH,
            reps=self.REPS,
            types=self.TYPES,
            use_lk_c8=True,
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
        self.head = V10Detect(nc=len(self.class_names), ch=(p3, p4, p5), reg_max=16)
        self._init_head_bias()

    def _init_head_bias(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels in (4,) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Apply optional per-channel normalization: (x - mean) / divide
        if not self._skip_subtract or not self._skip_divide:
            x = x.float()
        if not self._skip_subtract:
            x = x - self.input_subtract
        if not self._skip_divide:
            x = x / self.input_divide
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c3, c4, c5)
        return self.head((p3, p4, p5))
