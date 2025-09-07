from __future__ import annotations

"""ONNX export utilities for YOLOv10 models.

This module provides a small, export‑friendly wrapper that:
- Runs the model forward (including built‑in input normalization)
- Decodes YOLOv10 DFL outputs to pixel‑space boxes and class scores
- Applies a confidence threshold and top‑k selection to produce fixed‑shape
  detections per image: [B, N, 6] with columns [x1,y1,x2,y2,score,cls]
- Returns an additional [B] tensor with the valid detection counts

Notes
- Uses the official top‑k style decode (NMS‑free), which is stable and export‑
  friendly. This matches the default Python path in this repo.
- If class‑wise NMS is desired in ONNX, a future extension can swap the top‑k
  selection with an ONNX NonMaxSuppression‑based routine.
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
try:
    from torchvision.ops import nms as _tv_nms  # type: ignore
except Exception:  # pragma: no cover - optional; only used when decode='nms'
    _tv_nms = None  # type: ignore

from .head import V10Detect
from ...utils.tal import make_anchors, dist2bbox


class YOLOv10ONNXExport(nn.Module):
    """Wrap a YOLOv10 model to emit final detections for ONNX export.

    Args:
        model: An instantiated YOLOv10 variant (e.g., YOLOv10s).
        imgsz: Input spatial size (assumed square H=W).
        max_dets: Maximum detections per image (fixed N in ONNX output).
        conf: Confidence threshold to filter (applied on class probs).
        strides: Feature map strides; defaults to (8,16,32).
    Returns:
        forward(images) -> (detections, num_dets)
            detections: [B, N, 6] = [x1,y1,x2,y2,score,cls]
            num_dets: [B] int64 valid count per image
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        imgsz: int = 640,
        max_dets: int = 300,
        conf: float = 0.25,
        strides: Sequence[int] = (8, 16, 32),
        nms: bool = False,
        iou: float = 0.45,
        pre_topk: int = 1000,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.imgsz = int(imgsz)
        self.max_dets = int(max(1, max_dets))
        self.conf = float(conf)
        self.strides = tuple(int(s) for s in strides)
        self.nms = bool(nms)
        self.iou = float(iou)
        self.pre_topk = int(max(1, pre_topk))

        # Cache class count and reg_max from the head
        if not hasattr(self.model, "head") or not isinstance(self.model.head, V10Detect):
            raise ValueError("Provided model does not appear to be a YOLOv10 model with V10Detect head.")
        self.num_classes = int(self.model.head.nc)
        self.reg_max = int(self.model.head.reg_max)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning fixed‑shape detections and valid counts.

        The forward is export‑friendly: it uses tensor ops only and avoids
        Python‑side postprocessing.
        """
        device = images.device
        dtype = images.dtype

        # Run model to get raw per‑scale outputs [B, (4*reg_max + C), H, W]
        preds: List[torch.Tensor] = self.model(images)  # type: ignore[assignment]
        if not isinstance(preds, (list, tuple)):
            raise RuntimeError("Unexpected model output; expected list of tensors per scale.")
        assert len(preds) == len(self.strides), "preds/strides mismatch"

        b = preds[0].shape[0]
        # Build anchors over all levels
        anc_points, stride_tensor = make_anchors(preds, self.strides)
        # Shapes to match dist2bbox expectations: anchors -> [1, 2, A], strides -> [1, A]
        anc_points = anc_points.transpose(0, 1).unsqueeze(0)
        stride_tensor = stride_tensor.transpose(0, 1)

        # Prepare DFL bin indices
        idx = torch.arange(self.reg_max, device=device, dtype=dtype).view(1, 1, self.reg_max, 1)

        dists: List[torch.Tensor] = []  # [B, 4, A]
        clss: List[torch.Tensor] = []   # [B, C, A]
        for p in preds:
            _, c, h, w = p.shape
            a = h * w
            p = p.view(b, c, a)
            box = p[:, : 4 * self.reg_max]
            cls = p[:, 4 * self.reg_max :]
            probs = box.view(b, 4, self.reg_max, a).softmax(2)
            dist = (probs * idx).sum(2)
            dists.append(dist)
            clss.append(cls)

        # Concatenate all levels along anchors dimension
        dist_all = torch.cat(dists, dim=2)
        cls_all = torch.cat(clss, dim=2).sigmoid()  # [B, C, A_total]

        # Decode distances to xyxy in pixels; dist2bbox expects anchors [B,2,A]
        dbox = dist2bbox(dist_all, anc_points, xywh=False, dim=1) * stride_tensor  # [B, 4, A]
        boxes = dbox.permute(0, 2, 1)  # [B, A, 4]
        scores = cls_all.permute(0, 2, 1)  # [B, A, C]

        conf_t = torch.tensor(self.conf, device=device, dtype=dtype)
        H = W = self.imgsz

        if not self.nms:
            # NMS-free top-k per anchor by best class score (official eval style)
            best_scores, best_cls = scores.max(dim=2)  # [B, A]
            masked_scores = torch.where(best_scores >= conf_t, best_scores, torch.full_like(best_scores, -1.0))
            k = min(self.max_dets, masked_scores.shape[1])
            top_vals, top_idx = torch.topk(masked_scores, k, dim=1)
            batch_idx = torch.arange(b, device=device).view(b, 1)
            sel_boxes = boxes[batch_idx, top_idx]                 # [B, k, 4]
            sel_scores = best_scores[batch_idx, top_idx]          # [B, k]
            sel_cls = best_cls[batch_idx, top_idx].to(dtype)      # [B, k]
            # Clamp to image bounds
            sel_boxes[..., 0].clamp_(0, W)
            sel_boxes[..., 2].clamp_(0, W)
            sel_boxes[..., 1].clamp_(0, H)
            sel_boxes[..., 3].clamp_(0, H)
            sel_scores = torch.clamp(sel_scores, min=0.0)
            detections = torch.cat((sel_boxes, sel_scores.unsqueeze(-1), sel_cls.unsqueeze(-1)), dim=-1)
            num_dets = (sel_scores >= conf_t).sum(dim=1).to(torch.int64)
            return detections, num_dets
        else:
            # Class-wise NMS via class+image offset trick and a single NMS pass
            # 1) Pre-select top candidates across all (anchor,class) pairs
            A = boxes.shape[1]
            C = scores.shape[2]
            flat_scores = scores.reshape(b, A * C)
            k_pre = min(self.pre_topk, flat_scores.shape[1])
            pre_vals, pre_idx = torch.topk(flat_scores, k_pre, dim=1)
            anc_idx = pre_idx // C                      # [B, k_pre]
            cls_idx = (pre_idx % C).to(torch.int64)     # [B, k_pre]
            batch_idx = torch.arange(b, device=device).view(b, 1)
            cand_boxes = boxes[batch_idx, anc_idx]      # [B, k_pre, 4]
            cand_scores = pre_vals                      # [B, k_pre]

            # 2) Build large offsets per (image, class) group to emulate class-wise NMS in one call
            group_offset = float(max(H, W) * 10.0)
            img_ids = torch.arange(b, device=device).view(b, 1).to(dtype)
            group_id = img_ids * float(C) + cls_idx.to(dtype)     # [B, k_pre]
            off = (group_id * group_offset).unsqueeze(-1)         # [B, k_pre, 1]
            cand_boxes_off = cand_boxes + torch.cat([off, off, off, off], dim=-1)

            # 3) Single NMS pass on flattened candidates
            if _tv_nms is None:
                raise RuntimeError("torchvision.ops.nms not available; install torchvision to use decode='nms'")
            boxes_all = cand_boxes_off.reshape(b * k_pre, 4)
            scores_all = cand_scores.reshape(b * k_pre)
            keep = _tv_nms(boxes_all, scores_all, float(self.iou))           # [M]

            # 4) Map kept indices back to per-image positions and take per-image top-k by score
            grid = torch.full((b * k_pre,), -1.0, dtype=dtype, device=device)
            grid.scatter_(0, keep, scores_all[keep])
            grid = grid.view(b, k_pre)
            vals, pos = torch.topk(grid, k=min(self.max_dets, k_pre), dim=1)

            # 5) Gather final boxes/scores/classes from pre-selected candidate arrays
            final_boxes = cand_boxes[batch_idx, pos]    # [B, k, 4]
            final_scores = cand_scores[batch_idx, pos]  # [B, k]
            final_cls = cls_idx[batch_idx, pos].to(dtype)  # [B, k]

            # Clip to image bounds
            final_boxes[..., 0].clamp_(0, W)
            final_boxes[..., 2].clamp_(0, W)
            final_boxes[..., 1].clamp_(0, H)
            final_boxes[..., 3].clamp_(0, H)

            # Zero out entries with score < conf
            valid = vals >= conf_t
            final_boxes = torch.where(valid.unsqueeze(-1), final_boxes, torch.zeros_like(final_boxes))
            final_scores = torch.where(valid, final_scores, torch.zeros_like(final_scores))
            final_cls = torch.where(valid, final_cls, torch.zeros_like(final_cls))
            num_dets = valid.sum(dim=1).to(torch.int64)

            detections = torch.cat((final_boxes, final_scores.unsqueeze(-1), final_cls.unsqueeze(-1)), dim=-1)
            return detections, num_dets


def build_export_wrapper(
    model: nn.Module,
    *,
    imgsz: int = 640,
    max_dets: int = 300,
    conf: float = 0.25,
    decode: str = "topk",
    iou: float = 0.45,
    pre_topk: int = 1000,
) -> YOLOv10ONNXExport:
    """Helper to wrap a YOLOv10 model for ONNX export with defaults."""
    use_nms = (decode.lower() == "nms")
    return YOLOv10ONNXExport(
        model,
        imgsz=int(imgsz),
        max_dets=int(max_dets),
        conf=float(conf),
        nms=use_nms,
        iou=float(iou),
        pre_topk=int(pre_topk),
    )


@torch.no_grad()
def export_onnx(
    model: nn.Module,
    path: str,
    *,
    dummy_batch: int = 1,
    imgsz: int = 640,
    opset: int = 19,
    half: bool = False,
    max_dets: int = 300,
    conf: float = 0.25,
    decode: str = "topk",
    iou: float = 0.45,
    pre_topk: int = 1000,
) -> str:
    """Export a YOLOv10 model to ONNX with dynamic batch axis.

    Returns the output path on success.
    """
    model = model.eval()
    wrapper = build_export_wrapper(
        model,
        imgsz=imgsz,
        max_dets=max_dets,
        conf=conf,
        decode=decode,
        iou=iou,
        pre_topk=pre_topk,
    )
    dtype = torch.float16 if half else torch.float32
    dummy = torch.zeros((int(dummy_batch), 3, int(imgsz), int(imgsz)), dtype=dtype)

    # Names and dynamic axes for ONNX
    input_names = ["images"]
    output_names = ["detections", "num_dets"]
    dynamic_axes = {
        "images": {0: "batch"},
        "detections": {0: "batch"},
        "num_dets": {0: "batch"},
    }

    torch.onnx.export(
        wrapper,
        dummy,
        path,
        export_params=True,
        opset_version=int(opset),
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    return path
