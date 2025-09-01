import torch
import pytest


@pytest.mark.fidelity
def test_neck_feature_shapes_match_official():
    import sys
    sys.path.insert(0, 'yolov10-official')
    from ultralytics.nn.tasks import YOLOv10DetectionModel

    off = YOLOv10DetectionModel('ultralytics/cfg/models/v10/yolov10s.yaml')
    off.eval()

    feats = {}
    idxs = [16, 19, 22]
    hooks = [off.model[i].register_forward_hook(lambda m, inp, out, i=i: feats.__setitem__(i, out)) for i in idxs]

    x = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        _ = off(x)

    for h in hooks:
        h.remove()

    from leanyolo.models import get_model
    m = get_model('yolov10s', weights=None).eval()
    with torch.no_grad():
        c3, c4, c5 = m.backbone(x)
        p3, p4, p5 = m.neck(c3, c4, c5)

    assert p3.shape == feats[16].shape
    assert p4.shape == feats[19].shape
    assert p5.shape == feats[22].shape
