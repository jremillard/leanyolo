from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from .box_ops import box_xywh_to_xyxy, nms


@torch.no_grad()
def decode_predictions(
    preds: Sequence[torch.Tensor],
    *,
    num_classes: int,
    strides: Sequence[int] = (8, 16, 32),
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_det: int = 300,
    img_size: Tuple[int, int] | None = None,
) -> List[List[torch.Tensor]]:
    """Decode anchor-free predictions into final detections per image.

    Args:
        preds: list of 3 tensors, each [B, 4+nc, H, W]
        num_classes: number of classes
        strides: strides associated with each tensor
        conf_thresh: score threshold
        iou_thresh: NMS IoU threshold
        max_det: max detections per image
        img_size: optional network input size (h, w). If provided, boxes are
                  clamped to this range.
    Returns:
        List of length B, each is [N, 6] tensor: (x1,y1,x2,y2,score,cls)
    """
    assert len(preds) == len(strides), "preds and strides length mismatch"
    bsz = preds[0].shape[0]
    device = preds[0].device
    out: List[List[torch.Tensor]] = [[] for _ in range(bsz)]

    all_boxes = []
    all_scores = []
    all_labels = []
    all_batch = []

    for p, s in zip(preds, strides):
        # [B, C, H, W] -> [B, H, W, C]
        b, c, h, w = p.shape
        assert c == 4 + num_classes, "unexpected channel count"
        p = p.permute(0, 2, 3, 1).contiguous()
        # Flatten spatial
        p = p.view(b, -1, c)  # [B, HW, 4+nc]

        # Split bbox and class logits
        bbox = p[..., :4]
        cls = p[..., 4:]

        # Build grid
        gy, gx = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
        )
        gx = gx.reshape(1, -1)
        gy = gy.reshape(1, -1)

        # Decode center + size
        # Center offsets in [0,1] via sigmoid, plus grid; size via exp
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

        # Class scores
        scores, labels = cls.sigmoid().max(dim=-1)  # [B, HW]

        # Apply threshold
        mask = scores > conf_thresh
        for i in range(b):
            mi = mask[i]
            if mi.any():
                all_boxes.append(boxes_xyxy[i][mi])
                all_scores.append(scores[i][mi])
                all_labels.append(labels[i][mi])
                all_batch.append(torch.full((mi.sum().item(),), i, dtype=torch.int64, device=device))

    if not all_boxes:
        return [ [torch.empty((0, 6), device=device)] for _ in range(bsz) ]

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

