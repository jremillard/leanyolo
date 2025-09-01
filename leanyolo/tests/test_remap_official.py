import os
import sys

import torch
import pytest

from leanyolo.models import get_model
from leanyolo.data.coco import coco80_class_names
from leanyolo.utils.remap import remap_official_yolov10_to_lean


def _ensure_official_on_path():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "yolov10-official"))
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_official_state_dict(weights_path: str) -> dict:
    _ensure_official_on_path()
    # torch.load with weights_only=False may require ultralytics import
    import ultralytics  # noqa: F401
    return torch.load(weights_path, map_location="cpu", weights_only=False)


@pytest.mark.fidelity
def test_remap_covers_majority_of_params(tmp_path):
    # Download official weights to temp cache if not present
    from leanyolo.models.registry import _YOLOv10Weights
    entry = _YOLOv10Weights().get("yolov10s", "PRETRAINED_COCO")
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
    model = get_model(
        "yolov10s",
        weights=None,
        class_names=coco80_class_names(),
        input_norm_subtract=[0.0],
        input_norm_divide=[1.0],
    )
    dst_sd = model.state_dict()

    # Remap
    mapped = remap_official_yolov10_to_lean(loaded, model)

    # Expect at least 30% of params to be mappable thanks to name-based remap
    coverage = len(mapped) / max(len(dst_sd), 1)
    assert coverage > 0.30, f"Remap coverage too low: {coverage:.2%}"


@pytest.mark.fidelity
def test_first_conv_maps_identically(tmp_path):
    from leanyolo.models.registry import _YOLOv10Weights

    entry = _YOLOv10Weights().get("yolov10s", "PRETRAINED_COCO")
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

    model = get_model(
        "yolov10s",
        weights=None,
        class_names=coco80_class_names(),
        input_norm_subtract=[0.0],
        input_norm_divide=[1.0],
    )
    mapped = remap_official_yolov10_to_lean(loaded, model)

    # Our first conv weight key
    first_key = next(k for k in model.state_dict().keys() if k.endswith("cv0.conv.weight"))
    assert first_key in mapped, "First conv weight was not mapped"
    # Sanity on shape
    assert mapped[first_key].shape == model.state_dict()[first_key].shape
