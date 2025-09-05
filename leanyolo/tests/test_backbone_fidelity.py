import torch
import pytest

from leanyolo.tests.fidelity.common import ref_path, load_tensor


@pytest.mark.fidelity
def test_backbone_feature_shapes_match_references():
    # Use saved official references instead of importing the official repo
    # Compare our backbone outputs against reference shapes for yolov10s
    from leanyolo.models import get_model
    from leanyolo.data.coco import coco80_class_names

    x = torch.zeros(1, 3, 320, 320)
    m = get_model(
        'yolov10s',
        weights=None,
        class_names=coco80_class_names(),
        input_norm_subtract=[0.0],
        input_norm_divide=[1.0],
    ).eval()

    with torch.no_grad():
        c3, c4, c5 = m.backbone(x)

    r_c3 = load_tensor(ref_path('yolov10s', 'backbone_c3'))
    r_c4 = load_tensor(ref_path('yolov10s', 'backbone_c4'))
    r_c5 = load_tensor(ref_path('yolov10s', 'backbone_c5'))

    assert tuple(c3.shape) == tuple(r_c3.shape)
    assert tuple(c4.shape) == tuple(r_c4.shape)
    assert tuple(c5.shape) == tuple(r_c5.shape)
