import pytest

from leanyolo.models import get_model
from leanyolo.data.coco import coco80_class_names


def test_get_model_rejects_unknown_weights_key():
    # Anything other than PRETRAINED_COCO or None should raise
    with pytest.raises(ValueError):
        _ = get_model("yolov10s", weights="DEFAULT", class_names=coco80_class_names())
    with pytest.raises(ValueError):
        _ = get_model("yolov10s", weights="something_else", class_names=coco80_class_names())


def test_get_model_allows_none_weights():
    # None is valid and should not raise
    m = get_model("yolov10s", weights=None, class_names=coco80_class_names())
    assert hasattr(m, "class_names") and len(m.class_names) == 80

