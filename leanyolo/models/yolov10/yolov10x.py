from __future__ import annotations

"""YOLOv10‑x model (extra‑large) definition.

Goal
- Maximize accuracy for the most capable hardware in the family.

Why it works
- Scaling up while preserving YOLOv10’s efficient blocks and decoupled DFL head
  yields strong accuracy at higher compute footprints.

What it does
- Wires backbone→neck→head with extra‑large widths/repeats; includes
  normalization and decode helper for inference.

Input format:
- Tensor layout: CHW, shape (N, C, H, W)
- Color order: RGB (not BGR)
- Dtype/range: float32 in [0, 1] (scale by 1/255)
- Tip: If loading images with OpenCV (BGR), convert to RGB first
- Normalization: controlled by `input_norm_subtract` and `input_norm_divide` per
  channel: x' = (x - subtract) / divide. The raw model expects inputs in [0,1]
  (linear, i.e., 1/255). Recommended defaults for pretrained COCO are
  subtract=[0,0,0], divide=[255,255,255]. Use [0,0,0] and [1,1,1] to skip.

- Image size: Models are fully convolutional and accept arbitrary H×W. 640 is
  a convenient default (divisible by 32 for strides 8/16/32 → 80/40/20 grids).

Output format:
- Training mode: returns a list of 3 tensors [P3, P4, P5], each
  (N, 4*reg_max + num_classes, H, W). Channels 0..(4*reg_max-1) are DFL logits
  for [l,t,r,b], remaining are class logits.
- Eval mode: returns decoded detections per image as List[List[Tensor]], where
  each inner tensor is [N,6] = [x1,y1,x2,y2,score,cls] in pixels of the input
  (letterboxed) size. Thresholds via attributes: post_conf_thresh (0.25),
  post_iou_thresh (0.45), post_max_det (300).

Note:
- These YOLOv10x classes (family) are raw model modules. For inference, wrap
  preprocessing (RGB, letterbox, normalization) and postprocessing (decode,
  NMS, unletterbox). See tools/infer.py for a reference pipeline.
"""

from typing import List, Sequence

import torch
import torch.nn as nn

from .backbone import YOLOv10Backbone
from .neck import YOLOv10Neck
from .head_v10 import V10Detect
from .postprocess import decode_v10_predictions


class YOLOv10x(nn.Module):
    CH = {0: 80, 1: 160, 2: 160, 3: 320, 4: 320, 5: 640, 6: 640, 7: 640, 8: 640, 9: 640, 10: 640}
    HCH = {13: 640, 16: 320, 19: 640, 22: 640}
    REPS = {2: 3, 4: 6, 6: 6, 8: 3, 13: 3, 16: 3, 19: 3, 22: 3}
    TYPES = {"c6": "C2fCIB", "c8": "C2fCIB", "p5_p4": "C2fCIB", "p3_p4": "C2fCIB", "p4_p5": "C2fCIB"}

    def __init__(self, *, class_names: Sequence[str], in_channels: int, input_norm_subtract: Sequence[float], input_norm_divide: Sequence[float]):
        super().__init__()
        self.class_names = list(class_names)
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
            use_lk_p4_p5=False,
        )
        p3, p4, p5 = self.neck.out_c
        self.head = V10Detect(nc=len(self.class_names), ch=(p3, p4, p5), reg_max=16)
        self._init_head_bias()

    def _init_head_bias(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels in (4,) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        if not self._skip_subtract or not self._skip_divide:
            x = x.float()
        if not self._skip_subtract:
            x = x - self.input_subtract
        if not self._skip_divide:
            x = x / self.input_divide
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c3, c4, c5)
        return self.head((p3, p4, p5))

    def decode_forward(self, raw: List[torch.Tensor]):
        """Decode raw head outputs into final detections per image.

        Returns: List[List[Tensor]] with one entry per image; each inner tensor
        has shape [N, 6] where the columns are:
        - x1, y1: top-left corner in pixels (input letterboxed size)
        - x2, y2: bottom-right corner in pixels
        - score: confidence score (max class probability)
        - cls: class index (float; cast to int as needed)
        """
        conf = getattr(self, "post_conf_thresh", 0.25)
        iou = getattr(self, "post_iou_thresh", 0.45)
        mdet = getattr(self, "post_max_det", 300)
        strides = (8, 16, 32)
        _, _, h3, w3 = raw[0].shape
        ih, iw = h3 * strides[0], w3 * strides[0]
        return decode_v10_predictions(
            raw,
            num_classes=len(self.class_names),
            strides=strides,
            conf_thresh=conf,
            iou_thresh=iou,
            max_det=mdet,
            img_size=(ih, iw),
        )
