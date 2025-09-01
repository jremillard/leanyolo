import pytest

from leanyolo.models import get_model
from leanyolo.data.coco import coco80_class_names


def test_get_model_rejects_unknown_weights_key():
    # Anything other than PRETRAINED_COCO or None should raise
    with pytest.raises(ValueError):
        _ = get_model(
            "yolov10s",
            weights="DEFAULT",
            class_names=coco80_class_names(),
            input_norm_subtract=[0.0],
            input_norm_divide=[1.0],
        )
    with pytest.raises(ValueError):
        _ = get_model(
            "yolov10s",
            weights="something_else",
            class_names=coco80_class_names(),
            input_norm_subtract=[0.0],
            input_norm_divide=[1.0],
        )


def test_get_model_allows_none_weights():
    # None is valid and should not raise
    m = get_model(
        "yolov10s",
        weights=None,
        class_names=coco80_class_names(),
        input_norm_subtract=[0.0],
        input_norm_divide=[1.0],
    )
    assert hasattr(m, "class_names") and len(m.class_names) == 80


def test_norm_broadcast_and_validation():
    # One-element vectors should broadcast to 3 channels
    m = get_model(
        "yolov10n",
        weights=None,
        class_names=coco80_class_names(),
        input_norm_subtract=[10.0],
        input_norm_divide=[2.0],
    )
    import torch
    assert m.input_subtract.shape == (1, 3, 1, 1)
    assert torch.allclose(m.input_subtract, torch.tensor([10.0]).view(1, 1, 1, 1).repeat(1, 3, 1, 1))
    assert m.input_divide.shape == (1, 3, 1, 1)
    # Bad lengths should raise
    with pytest.raises(ValueError):
        _ = get_model(
            "yolov10n",
            weights=None,
            class_names=coco80_class_names(),
            input_norm_subtract=[0.0, 0.0],
            input_norm_divide=[1.0],
        )
