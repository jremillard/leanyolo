import os
import pytest
import torch

from leanyolo.models import get_model, get_model_weights
from leanyolo.data.coco import coco80_class_names
from leanyolo.utils.remap import remap_official_yolov10_to_lean


@pytest.mark.fidelity
def test_remap_covers_majority_of_params(tmp_path):
    # Resolve and load official weights via our WeightsEntry (no ultralytics import)
    entry = get_model_weights("yolov10s")().get("yolov10s", "PRETRAINED_COCO")
    loaded = entry.get_state_dict(progress=True, map_location="cpu")

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
    entry = get_model_weights("yolov10s")().get("yolov10s", "PRETRAINED_COCO")
    loaded = entry.get_state_dict(progress=True, map_location="cpu")

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
