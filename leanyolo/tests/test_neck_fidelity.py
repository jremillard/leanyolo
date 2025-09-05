import torch
import pytest

from leanyolo.tests.fidelity.common import ref_path, load_tensor


@pytest.mark.fidelity
def test_neck_feature_shapes_match_references():
    # Compare our neck outputs against saved reference shapes for yolov10s
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
        p3, p4, p5 = m.neck(c3, c4, c5)

    r_p3 = load_tensor(ref_path('yolov10s', 'neck_p3'))
    r_p4 = load_tensor(ref_path('yolov10s', 'neck_p4'))
    r_p5 = load_tensor(ref_path('yolov10s', 'neck_p5'))

    assert tuple(p3.shape) == tuple(r_p3.shape)
    assert tuple(p4.shape) == tuple(r_p4.shape)
    assert tuple(p5.shape) == tuple(r_p5.shape)
