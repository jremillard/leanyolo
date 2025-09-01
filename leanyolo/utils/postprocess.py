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
            # reshape to [B, (4*reg_max+nc), H*W]
            p = p.view(b, c, h * w)
            box_dist = p[:, : 4 * reg_max]
            cls = p[:, 4 * reg_max :]

            # grid centers in pixels
            gy, gx = torch.meshgrid(
                torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
            )
            cx = (gx.reshape(-1).float() + 0.5) * s
            cy = (gy.reshape(-1).float() + 0.5) * s

            # DFL to distances [B,4,A]
            # Build fixed conv weights on the fly
            idx = torch.arange(reg_max, device=device, dtype=torch.float).view(1, 1, reg_max, 1)
            probs = box_dist.view(b, 4, reg_max, -1).softmax(2)
            dist = (probs * idx).sum(2)  # [B,4,A]
            # Distances are in bins; scale by stride s to pixels
            l = dist[:, 0] * s
            t = dist[:, 1] * s
            r = dist[:, 2] * s
            btm = dist[:, 3] * s

            # xyxy per level
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
