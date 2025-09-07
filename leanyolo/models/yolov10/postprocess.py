from __future__ import annotations

"""YOLOv10 post‑processing utilities.

Goal
- Turn raw head outputs into final, per‑image detections (boxes, scores, labels).

Why it works
- Anchor‑free heads predict either direct offsets (cx,cy,bw,bh) or DFL
  distributions for distances (l,t,r,b). Converting these to pixel‑space boxes
  and applying thresholding and NMS yields a clean set of detections consistent
  with the training targets used across YOLO families.

What it does
- For each feature map and stride: decode boxes (direct offsets or DFL
  distances), compute per‑class scores, filter by confidence, merge all scales
  and run NMS, cap to ``max_det`` per image, and return [x1,y1,x2,y2,score,cls].

Return format per image
- A tensor of shape (N, 6): [x1, y1, x2, y2, score, cls]
  - x1, y1: top‑left corner in pixels
  - x2, y2: bottom‑right corner in pixels
  - score: confidence (max class prob)
  - cls: class index (float for convenience; cast to int as needed)

Parameters
- num_classes: number of classes
- strides: receptive‑field strides per map (default (8,16,32))
- conf_thresh: filter boxes below this score (sigmoid applied) before NMS
- iou_thresh: IoU threshold for NMS
- max_det: maximum detections per image after NMS
- img_size: optional (H, W) to clamp boxes to input bounds

Note
- Supports both layouts: [B, 4+nc, H, W] (direct offsets) and
  [B, 4*reg_max+nc, H, W] (DFL distances + classes).
"""

from typing import List, Sequence, Tuple

import torch

from ...utils.box_ops import box_xywh_to_xyxy, nms
from ...utils.tal import make_anchors, dist2bbox


@torch.no_grad()
def decode_v10_predictions(
    preds: Sequence[torch.Tensor],
    *,
    num_classes: int,
    strides: Sequence[int] = (8, 16, 32),
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_det: int = 300,
    img_size: Tuple[int, int] | None = None,
) -> List[List[torch.Tensor]]:
    assert len(preds) == len(strides), "preds and strides length mismatch"
    bsz = preds[0].shape[0]
    device = preds[0].device
    out: List[List[torch.Tensor]] = [[] for _ in range(bsz)]

    all_boxes = []
    all_scores = []
    all_labels = []
    all_batch = []

    for p, s in zip(preds, strides):
        b, c, h, w = p.shape
        if c == 4 + num_classes:
            p = p.permute(0, 2, 3, 1).contiguous()
            p = p.view(b, -1, c)
            bbox = p[..., :4]
            cls = p[..., 4:]

            gy, gx = torch.meshgrid(
                torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
            )
            gx = gx.reshape(1, -1)
            gy = gy.reshape(1, -1)

            cx = (bbox[..., 0].sigmoid() + gx) * s
            cy = (bbox[..., 1].sigmoid() + gy) * s
            bw = bbox[..., 2].exp() * s
            bh = bbox[..., 3].exp() * s
            boxes_xyxy = box_xywh_to_xyxy(torch.stack((cx, cy, bw, bh), dim=-1))
            if img_size is not None:
                H, W = img_size
                boxes_xyxy[..., 0].clamp_(0, W)
                boxes_xyxy[..., 2].clamp_(0, W)
                boxes_xyxy[..., 1].clamp_(0, H)
                boxes_xyxy[..., 3].clamp_(0, H)

            scores, labels = cls.sigmoid().max(dim=-1)
            mask = scores > conf_thresh
            for i in range(b):
                mi = mask[i]
                if mi.any():
                    all_boxes.append(boxes_xyxy[i][mi])
                    all_scores.append(scores[i][mi])
                    all_labels.append(labels[i][mi])
                    all_batch.append(torch.full((mi.sum().item(),), i, dtype=torch.int64, device=device))
        else:
            # v10-style DFL outputs: c = 4*reg_max + nc
            reg_max = (c - num_classes) // 4
            assert reg_max * 4 + num_classes == c, "invalid DFL channels"
            p = p.view(b, c, h * w)
            box_dist = p[:, : 4 * reg_max]
            cls = p[:, 4 * reg_max :]

            gy, gx = torch.meshgrid(
                torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
            )
            cx = (gx.reshape(-1).float() + 0.5) * s
            cy = (gy.reshape(-1).float() + 0.5) * s

            idx = torch.arange(reg_max, device=device, dtype=torch.float).view(1, 1, reg_max, 1)
            probs = box_dist.view(b, 4, reg_max, -1).softmax(2)
            dist = (probs * idx).sum(2)  # [B,4,A]
            l = dist[:, 0] * s
            t = dist[:, 1] * s
            r = dist[:, 2] * s
            btm = dist[:, 3] * s

            x1 = cx[None, :] - l
            y1 = cy[None, :] - t
            x2 = cx[None, :] + r
            y2 = cy[None, :] + btm
            boxes_xyxy = torch.stack((x1, y1, x2, y2), dim=-1)  # [B,A,4]

            scores, labels = cls.sigmoid().max(dim=1)  # [B, A]
            mask = scores > conf_thresh
            for i in range(b):
                mi = mask[i]
                if mi.any():
                    all_boxes.append(boxes_xyxy[i][mi])
                    all_scores.append(scores[i][mi])
                    all_labels.append(labels[i][mi])
                    all_batch.append(torch.full((mi.sum().item(),), i, dtype=torch.int64, device=device))

    if not all_boxes:
        return [[torch.empty((0, 6), device=device)] for _ in range(bsz)]

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    batch = torch.cat(all_batch, dim=0)

    results: List[List[torch.Tensor]] = [[] for _ in range(bsz)]
    for i in range(bsz):
        sel = batch == i
        bi = boxes[sel]
        si = scores[sel]
        ci = labels[sel].float()
        if bi.numel() == 0:
            results[i] = [torch.empty((0, 6), device=device)]
            continue
        keep = nms(bi, si, iou_thresh)
        keep = keep[:max_det]
        di = torch.cat((bi[keep], si[keep, None], ci[keep, None]), dim=1)
        results[i] = [di]

    return results


@torch.no_grad()
def decode_v10_official_topk(
    preds: Sequence[torch.Tensor],
    *,
    num_classes: int,
    strides: Sequence[int] = (8, 16, 32),
    max_det: int = 300,
    # Back-compat kwargs (ignored for top-k decode)
    conf_thresh: float | None = None,
    iou_thresh: float | None = None,
    img_size: Tuple[int, int] | None = None,
) -> List[List[torch.Tensor]]:
    """Replicate official YOLOv10 eval decoding (no NMS, top‑k by class scores).

    This mirrors ultralytics v10Detect._inference + postprocess:
    - Compute DFL expectation per side over reg_max bins.
    - Create anchor points over all feature maps with given strides.
    - Decode to pixel‑space xyxy boxes, concatenate with sigmoid class scores.
    - Select top‑k anchors by max class score, then flatten per‑class scores
      and select top‑k elements, producing [x,y,w,h,score,cls] per anchor.

    Args:
        preds: list of [B, 4*reg_max+num_classes, H, W]
        num_classes: number of classes (nc)
        strides: receptive field strides per level
        max_det: maximum elements to keep after top‑k

    Returns:
        List (len B) of single‑element lists with tensors of shape
        [min(max_det, A), 6] formatted as [x1, y1, x2, y2, score, cls].
        Nested list structure matches the NMS decode API for compatibility.
    """
    assert len(preds) == len(strides)
    b = preds[0].shape[0]
    device = preds[0].device

    c_total = preds[0].shape[1]
    reg_max = (c_total - num_classes) // 4
    assert 4 * reg_max + num_classes == c_total, "Invalid channel layout for v10 head"

    # Anchors across all levels
    anc_points, stride_tensor = make_anchors(preds, strides)
    # Shapes to match official ops: anchors -> [1, 2, A], strides -> [1, A]
    anc_points = anc_points.transpose(0, 1).unsqueeze(0)
    stride_tensor = stride_tensor.transpose(0, 1)

    # Precompute DFL bin indices
    idx = torch.arange(reg_max, device=device, dtype=preds[0].dtype).view(1, 1, reg_max, 1)

    dists: List[torch.Tensor] = []  # [B, 4, A]
    clss: List[torch.Tensor] = []   # [B, C, A]
    for p in preds:
        _, c, h, w = p.shape
        a = h * w
        p = p.view(b, c, a)
        box = p[:, : 4 * reg_max]
        cls = p[:, 4 * reg_max :]
        probs = box.view(b, 4, reg_max, a).softmax(2)
        dist = (probs * idx).sum(2)
        dists.append(dist)
        clss.append(cls)

    dist_all = torch.cat(dists, dim=2)
    cls_all = torch.cat(clss, dim=2).sigmoid()

    # Decode to xyxy in pixels; dist2bbox expects anchors shape [B,2,A]
    dbox = dist2bbox(dist_all, anc_points, xywh=False, dim=1) * stride_tensor

    # Concatenate into [B, A, 4+nc] and perform two-stage top‑k without NMS
    preds_cat = torch.cat((dbox, cls_all), dim=1).permute(0, 2, 1)
    B, A, C = preds_cat.shape
    nc = C - 4
    k = min(max_det, A)

    boxes = preds_cat[..., :4]     # [B, A, 4]
    scores = preds_cat[..., 4:]    # [B, A, nc]

    # Stage 1: select top‑k anchors by their best class score
    max_per_anchor, _ = scores.max(dim=-1)                          # [B, A]
    top_anchor_vals, top_anchor_idx = torch.topk(max_per_anchor, k, dim=1)

    # Gather class score vectors for those anchors using advanced indexing
    batch_idx = torch.arange(B, device=device).view(B, 1)
    sel_scores = scores[batch_idx, top_anchor_idx]                   # [B, k, nc]

    # Stage 2: from the selected anchors, take the global top‑k (anchor,class) pairs
    flat_scores = sel_scores.reshape(B, -1)                          # [B, k*nc]
    flat_vals, flat_idx = torch.topk(flat_scores, k, dim=1)
    rel_anchor = (flat_idx // nc)                                    # [B, k]
    cls_idx = (flat_idx % nc).to(torch.int64)                        # [B, k]
    final_anchor_idx = top_anchor_idx.gather(1, rel_anchor)          # [B, k]

    # Gather final boxes and assemble output tensor [B, k, 6]
    final_boxes = boxes[batch_idx, final_anchor_idx]
    final = torch.cat([final_boxes, flat_vals.unsqueeze(-1), cls_idx.float().unsqueeze(-1)], dim=-1)
    return [[final[i]] for i in range(B)]
