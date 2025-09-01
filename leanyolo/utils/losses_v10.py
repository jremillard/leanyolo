from __future__ import annotations

"""YOLOv10-style training losses (compact approximation).

This module implements a practical loss suitable for transfer learning:
- Positive assignment at the target's center cell per stride level (8/16/32).
- Regression uses Distribution Focal Loss (DFL) over distance bins plus an IoU term.
- Classification uses BCE-with-logits at the positive location (one-vs-rest).

Notes:
- This is a compact reference; it is not a full reproduction of the official
  training pipeline, but aligns more closely than the naive baseline.
"""

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .box_ops import box_iou


def _exp_from_dfl(logits: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Compute expectation over DFL bins.

    Args:
        logits: [B, 4*reg_max]
        reg_max: number of bins per side
    Returns:
        [B,4] expected distances in bin units
    """
    b = logits.shape[0]
    probs = logits.view(b, 4, reg_max).softmax(dim=2)
    idx = torch.arange(reg_max, device=logits.device, dtype=logits.dtype).view(1, 1, reg_max)
    return (probs * idx).sum(dim=2)


def _dfl_loss(logits: torch.Tensor, target: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Distribution Focal Loss for a single location.

    Args:
        logits: [4*reg_max], raw logits for 4 sides concatenated
        target: [4], target distances in bin units (float)
    Returns:
        scalar DFL loss (sum over 4 sides)
    """
    x = logits.view(4, reg_max)
    t = target.clamp(0, reg_max - 1 - 1e-3)
    l = t.floor()
    u = l + 1
    wl = (u - t).detach()
    wu = (t - l).detach()
    l = l.long()
    u = u.long()
    ce = F.cross_entropy  # type: ignore[assignment]
    loss = ce(x, l, reduction="none") * wl + ce(x, u, reduction="none") * wu
    return loss.sum()


def _level_for_box(w: float, h: float) -> int:
    m = max(w, h)
    if m < 64:
        return 0
    if m < 128:
        return 1
    return 2


def detection_loss_v10(
    raw: Sequence[torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    *,
    num_classes: int,
    reg_max: int = 16,
    strides: Tuple[int, int, int] = (8, 16, 32),
    lambda_dfl: float = 1.5,
    lambda_iou: float = 1.0,
    lambda_cls: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute a YOLOv10-style loss with DFL + IoU + BCE.

    Args:
        raw: list of 3 tensors [P3,P4,P5], each [B, 4*reg_max + nc, H, W]
        targets: list of dicts with 'boxes' (Nx4 xyxy) and 'labels' (N)
        num_classes: number of classes
    Returns:
        dict with keys: total, cls, reg
    """
    b = raw[0].shape[0]
    device = raw[0].device
    cls_loss = torch.zeros((), device=device)
    reg_loss = torch.zeros((), device=device)

    for bi in range(b):
        boxes = targets[bi]["boxes"].to(device)
        labels = targets[bi]["labels"].to(device)
        for j in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[j]
            w = (x2 - x1).item()
            h = (y2 - y1).item()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            lvl = _level_for_box(w, h)
            stride = strides[lvl]
            p = raw[lvl]
            _, c, hmap, wmap = p.shape
            gx = int(torch.clamp(cx / stride, 0, wmap - 1))
            gy = int(torch.clamp(cy / stride, 0, hmap - 1))
            # Slice predictions at this location
            logits = p[bi, :, gy, gx]
            reg_logits = logits[: 4 * reg_max]
            cls_logits = logits[4 * reg_max : 4 * reg_max + num_classes]

            # Regression: DFL against distances (in bins), plus IoU on decoded box
            l = (cx - x1) / stride
            t = (cy - y1) / stride
            r = (x2 - cx) / stride
            btm = (y2 - cy) / stride
            tgt_bins = torch.stack((l, t, r, btm), dim=0)
            dfl = _dfl_loss(reg_logits, tgt_bins, reg_max)
            # IoU on decoded expected distances
            pred_bins = _exp_from_dfl(reg_logits.view(1, -1), reg_max)[0]
            px1 = cx - pred_bins[0] * stride
            py1 = cy - pred_bins[1] * stride
            px2 = cx + pred_bins[2] * stride
            py2 = cy + pred_bins[3] * stride
            iou = box_iou(
                torch.tensor([[px1, py1, px2, py2]], device=device),
                torch.tensor([[x1, y1, x2, y2]], device=device),
            )[0, 0]
            iou_loss = 1.0 - iou.clamp(0, 1)
            reg_loss = reg_loss + (lambda_dfl * dfl + lambda_iou * iou_loss)

            # Classification one-vs-rest at this cell
            target = torch.zeros_like(cls_logits)
            target[labels[j].item()] = 1.0
            cls_loss = cls_loss + lambda_cls * F.binary_cross_entropy_with_logits(cls_logits, target, reduction="sum")

    # Normalize by number of targets
    nt = sum(t["boxes"].shape[0] for t in targets)
    nt = max(nt, 1)
    cls_loss = cls_loss / nt
    reg_loss = reg_loss / nt
    total = cls_loss + reg_loss
    return {"total": total, "cls": cls_loss, "reg": reg_loss}

