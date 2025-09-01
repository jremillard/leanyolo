from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

# Re-export for backward compatibility with existing tests/utilities.
from leanyolo.models.yolov10.postprocess import decode_v10_predictions as decode_predictions  # noqa: F401
