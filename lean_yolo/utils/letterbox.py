from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def letterbox(
    img: np.ndarray,
    new_shape: int | Tuple[int, int] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    """Resize and pad image to meet stride-multiple constraints.

    Returns (image, (gain_w, gain_h), (pad_w, pad_h)) where pad is half on each side.
    """
    shape = img.shape[:2]  # current shape [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:  # make sure padding is a multiple of stride
        dw %= stride
        dh %= stride

    if scale_fill:  # stretch
        new_unpad = (new_shape[1], new_shape[0])
        dw, dh = 0, 0
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]
    else:
        r = (r, r)

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top = int(round(dh / 2.0))
    bottom = int(round(dh - top))
    left = int(round(dw / 2.0))
    right = int(round(dw - left))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (r if isinstance(r, tuple) else (r, r)), (left, top)

