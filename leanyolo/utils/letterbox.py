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
    """Aspect-preserving resize with constant padding.

    Scales an image to fit inside a target shape while keeping aspect ratio,
    then pads the borders with a solid color. When ``auto=True``, the final
    output size is the smallest multiple of ``stride`` that can contain the
    resized image (useful for models with stride constraints). When
    ``scale_fill=True``, the image is stretched to exactly match the target
    size (no padding), potentially changing aspect ratio.

    Args:
        img: Input RGB image of shape (H, W, 3), dtype uint8 or float.
        new_shape: Target size as int (square) or tuple (H, W).
        color: Border color used for padding in (R, G, B).
        auto: If True, reduce padding so output dims are stride-multiples.
        scale_fill: If True, stretch to target size without preserving aspect.
        scaleup: If False, never scale above 1.0 (no upsampling).
        stride: Stride granularity used when ``auto=True``.

    Returns:
        (img_out, (gain_w, gain_h), (pad_w, pad_h)) where ``pad`` represents
        the amount added on the left and top respectively. ``gain`` maps
        original pixel coordinates to the resized (pre-pad) image space.
    """
    orig_h, orig_w = img.shape[:2]

    # Normalize target shape to (H, W)
    if isinstance(new_shape, int):
        tgt_h, tgt_w = new_shape, new_shape
    else:
        tgt_h, tgt_w = int(new_shape[0]), int(new_shape[1])

    # Compute resize scales and new spatial size
    if scale_fill:
        # Axis-wise stretch to exactly match target size
        gain_w = tgt_w / max(orig_w, 1)
        gain_h = tgt_h / max(orig_h, 1)
        new_w, new_h = tgt_w, tgt_h
        pad_w, pad_h = 0.0, 0.0
    else:
        # Uniform scale (preserve aspect)
        r = min(tgt_w / max(orig_w, 1), tgt_h / max(orig_h, 1))
        if not scaleup:
            r = min(r, 1.0)

        new_w = int(round(orig_w * r))
        new_h = int(round(orig_h * r))
        gain_w = r
        gain_h = r

        # Padding required to reach target size
        pad_w = float(tgt_w - new_w)
        pad_h = float(tgt_h - new_h)

        # Optionally reduce padding so output is a multiple of stride
        if auto and stride > 1:
            pad_w = pad_w % stride
            pad_h = pad_h % stride

    # Resize if needed (OpenCV expects (W, H))
    if (orig_w, orig_h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Split padding equally left/right and top/bottom
    left = int(round(pad_w / 2.0))
    right = int(round(pad_w - left))
    top = int(round(pad_h / 2.0))
    bottom = int(round(pad_h - top))

    if any(v != 0 for v in (top, bottom, left, right)):
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=color
        )

    return img, (float(gain_w), float(gain_h)), (left, top)
