"""Common bounding box operations.

This module collects small utilities used across the project for converting
between ``xyxy`` and ``xywh`` formats, computing IoU/area, performing greedy
non‑maximum suppression, and adjusting box coordinates after letterboxing.
All functions expect and return ``torch.Tensor`` objects with shape
``[N, 4]`` unless otherwise noted.
"""

from __future__ import annotations

from typing import Tuple

import torch


def box_xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)


def box_xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    cx = x1 + w / 2
    cy = y1 + h / 2
    return torch.stack((cx, cy, w, h), dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0))


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    # boxes are Nx4 and Mx4 in xyxy
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-9)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """Greedy NMS implemented in PyTorch.

    Args:
        boxes: [N, 4] in xyxy
        scores: [N]
        iou_thresh: IoU threshold for suppression
    Returns:
        keep indices tensor [K]
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)

    order = scores.argsort(descending=True)
    keep = []
    boxes = boxes[order]
    scores = scores[order]

    while boxes.size(0) > 0:
        keep.append(order[0])
        if boxes.size(0) == 1:
            break
        ious = box_iou(boxes[:1], boxes[1:]).squeeze(0)
        mask = ious <= iou_thresh
        # advance
        boxes = boxes[1:][mask]
        scores = scores[1:][mask]
        order = order[1:][mask]

    return torch.stack(keep)


def scale_coords(from_shape: Tuple[int, int], boxes: torch.Tensor, to_shape: Tuple[int, int]) -> torch.Tensor:
    """Scale xyxy boxes from from_shape (h,w) to to_shape (h,w)."""
    fh, fw = from_shape
    th, tw = to_shape
    gain_w = tw / max(fw, 1)
    gain_h = th / max(fh, 1)
    # boxes in xyxy
    x1, y1, x2, y2 = boxes.unbind(-1)
    x1 = x1 * gain_w
    x2 = x2 * gain_w
    y1 = y1 * gain_h
    y2 = y2 * gain_h
    return torch.stack((x1, y1, x2, y2), dim=-1)


def unletterbox_coords(
    boxes: torch.Tensor,
    gain: Tuple[float, float],
    pad: Tuple[int, int],
    to_shape: Tuple[int, int],
) -> torch.Tensor:
    """Invert letterbox: boxes are xyxy in resized-padded space.

    Args:
        boxes: [N,4]
        gain: (gw, gh) gains returned by letterbox
        pad: (px, py) padding (left, top)
        to_shape: target original shape (h, w)
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    px, py = pad
    gw, gh = gain
    # remove pad
    x1 = (x1 - px) / gw
    x2 = (x2 - px) / gw
    y1 = (y1 - py) / gh
    y2 = (y2 - py) / gh
    # clip to image
    H, W = to_shape
    x1 = x1.clamp(0, W)
    x2 = x2.clamp(0, W)
    y1 = y1.clamp(0, H)
    y2 = y2.clamp(0, H)
    return torch.stack((x1, y1, x2, y2), dim=-1)
