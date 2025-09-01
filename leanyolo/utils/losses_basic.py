from __future__ import annotations

"""Basic YOLOv10 training losses with naive assignment.

This is a simple reference implementation intended for small transfer-learning
tasks. It does not aim to match the official training recipe.
"""

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


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


def _level_for_box(w: float, h: float) -> int:
    m = max(w, h)
    # Naive thresholds for 8/16/32 strides
    if m < 64:
        return 0
    if m < 128:
        return 1
    return 2


def detection_loss_naive(
    raw: Sequence[torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    *,
    num_classes: int,
    reg_max: int = 16,
    strides: Tuple[int, int, int] = (8, 16, 32),
) -> Dict[str, torch.Tensor]:
    """Compute a basic detection loss with center-cell assignment.

    Args:
        raw: list of 3 tensors [P3,P4,P5], each [B, 4*reg_max + nc, H, W]
        targets: length-B list of dicts with 'boxes' (Nx4 xyxy) and 'labels' (N)
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
            # Regression target in bins (float)
            l = (cx - x1) / stride
            t = (cy - y1) / stride
            r = (x2 - cx) / stride
            btm = (y2 - cy) / stride
            pred_bins = _exp_from_dfl(reg_logits.view(1, -1), reg_max)[0]
            tgt_bins = torch.stack((l, t, r, btm), dim=0)
            reg_loss = reg_loss + F.smooth_l1_loss(pred_bins, tgt_bins, reduction="sum")

            # Classification one-vs-rest at this cell
            target = torch.zeros_like(cls_logits)
            target[labels[j].item()] = 1.0
            cls_loss = cls_loss + F.binary_cross_entropy_with_logits(cls_logits, target, reduction="sum")

    # Normalize by number of targets to avoid scale issues
    nt = sum(t["boxes"].shape[0] for t in targets)
    nt = max(nt, 1)
    cls_loss = cls_loss / nt
    reg_loss = reg_loss / nt
    total = cls_loss + reg_loss
    return {"total": total, "cls": cls_loss, "reg": reg_loss}

