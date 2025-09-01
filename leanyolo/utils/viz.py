from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
import torch


def draw_detections(
    img: np.ndarray,
    dets: torch.Tensor,
    class_names: List[str] | None = None,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw Nx6 dets (x1,y1,x2,y2,score,cls) on BGR image."""
    out = img.copy()
    if dets is None or dets.numel() == 0:
        return out
    dets = dets.cpu().numpy()
    for x1, y1, x2, y2, s, c in dets:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_id = int(c)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_id}:{s:.2f}"
        if class_names and 0 <= cls_id < len(class_names):
            label = f"{class_names[cls_id]}:{s:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - h - 4), (x1 + w + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out

