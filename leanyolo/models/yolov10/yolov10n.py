from __future__ import annotations

"""YOLOv10‑n model (nano) definition.

Goal
- Provide the fastest, smallest YOLOv10 variant for constrained devices while
  retaining the core architectural ideas.

Why it works
- Width/depth scaling reduces channels and repeats while keeping efficient
  blocks (SCDown, C2f/C2fCIB) and the decoupled head pattern pioneered across
  YOLOs (from v3’s multi‑scale, v5/6/8 decoupled heads, to v10’s DFL + dual
  assignments). This preserves good accuracy‑per‑FLOP.

What it does
- Wires the common backbone→neck→head with nano channel widths and repeats, and
  exposes simple normalization and a decode helper for inference.

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

Note
- These YOLOv10* classes are raw model modules. For inference, wrap
  preprocessing (RGB, letterbox, normalization) and postprocessing (decode,
  NMS, unletterbox). See tools/infer.py for a reference pipeline.
"""

from typing import List, Sequence

import torch
import torch.nn as nn

from .backbone import YOLOv10Backbone
from .neck import YOLOv10Neck
from .head import V10Detect
from .postprocess import decode_v10_official_topk as decode_v10_predictions


class YOLOv10n(nn.Module):
    CH = {0: 16, 1: 32, 2: 32, 3: 64, 4: 64, 5: 128, 6: 128, 7: 256, 8: 256, 9: 256, 10: 256}
    HCH = {13: 128, 16: 64, 19: 128, 22: 256}
    REPS = {2: 1, 4: 2, 6: 2, 8: 1, 13: 1, 16: 1, 19: 1, 22: 1}
    TYPES = {"c6": "C2f", "c8": "C2f", "p5_p4": "C2f", "p3_p4": "C2f", "p4_p5": "C2fCIB"}

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
            use_lk_p4_p5=True,
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
        if self.training:
            return self.head((p3, p4, p5))
        one2many = self.head((p3, p4, p5))
        one2one = self.head.forward_feat((p3, p4, p5), self.head.one2one_cv2, self.head.one2one_cv3)
        self._eval_branches = {"one2many": one2many, "one2one": one2one}
        return one2many

    def decode_forward(self, raw: List[torch.Tensor] | dict):
        """Decode raw head outputs into final detections per image.

        Returns: List[List[Tensor]] with one entry per image; each inner tensor
        has shape [N, 6] where the columns are:
        - x1, y1: top-left corner in pixels (input letterboxed size)
        - x2, y2: bottom-right corner in pixels
        - score: confidence score (max class probability)
        - cls: class index (float; cast to int as needed)
        """
        strides = (8, 16, 32)
        if isinstance(raw, dict):
            seq = raw.get("one2one", raw.get("one2many"))
        else:
            seq = getattr(self, "_eval_branches", {}).get("one2one", raw)
        return decode_v10_predictions(
            seq,
            num_classes=len(self.class_names),
            strides=strides,
        )
