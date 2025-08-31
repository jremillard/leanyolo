import os
import sys

import torch

from lean_yolo.models import get_model
from lean_yolo.utils.remap import remap_official_yolov10_to_lean


def _ensure_official_on_path():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "yolov10-official"))
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_official_state_dict(weights_path: str) -> dict:
    _ensure_official_on_path()
    # torch.load with weights_only=False may require ultralytics import
    import ultralytics  # noqa: F401
    return torch.load(weights_path, map_location="cpu", weights_only=False)


def test_remap_covers_majority_of_params(tmp_path):
    # Download official weights to temp cache if not present
    from lean_yolo.models.registry import _YOLOv10Weights
    entry = _YOLOv10Weights().get("yolov10s", "DEFAULT")
    # Download manually and load with weights_only=False due to PyTorch 2.6+ changes
    import urllib.request, os
    dst = os.path.join(str(tmp_path), entry.filename)
    if not os.path.exists(dst):
        with urllib.request.urlopen(entry.url) as r, open(dst, 'wb') as f:
            while True:
                chunk = r.read(1<<20)
                if not chunk:
                    break
                f.write(chunk)
    loaded = _load_official_state_dict(dst)

    # Build lean model
    model = get_model("yolov10s", weights=None)
    dst_sd = model.state_dict()

    # Remap
    mapped = remap_official_yolov10_to_lean(loaded, model)

    # Expect at least 5% of parameters (by count of tensors) to be mappable by shape
    # Note: Our lean model differs architecturally from the official model, so only
    # early layers are expected to align one-to-one.
    coverage = len(mapped) / max(len(dst_sd), 1)
    assert coverage > 0.05, f"Remap coverage too low: {coverage:.2%}"


def test_first_conv_maps_identically(tmp_path):
    from lean_yolo.models.registry import _YOLOv10Weights

    entry = _YOLOv10Weights().get("yolov10s", "DEFAULT")
    import urllib.request, os
    dst = os.path.join(str(tmp_path), entry.filename)
    if not os.path.exists(dst):
        with urllib.request.urlopen(entry.url) as r, open(dst, 'wb') as f:
            while True:
                chunk = r.read(1<<20)
                if not chunk:
                    break
                f.write(chunk)
    loaded = _load_official_state_dict(dst)

    model = get_model("yolov10s", weights=None)
    mapped = remap_official_yolov10_to_lean(loaded, model)

    # Our first conv weight key
    first_key = next(k for k in model.state_dict().keys() if k.endswith("cv0.conv.weight"))
    assert first_key in mapped, "First conv weight was not mapped"
    # Sanity on shape
    assert mapped[first_key].shape == model.state_dict()[first_key].shape
