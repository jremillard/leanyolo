import torch
import pytest


@pytest.mark.fidelity
def test_backbone_feature_shapes_match_official():
    # Import official model
    import sys, os
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    candidates = [
        os.path.join(repo_root, "references", "yolov10", "THU-MIG.yoloe"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            sys.path.insert(0, p)
            break
    from ultralytics.nn.tasks import YOLOv10DetectionModel

    off = YOLOv10DetectionModel('ultralytics/cfg/models/v10/yolov10s.yaml')
    off.eval()

    # Capture intermediate outputs at indices 4, 6, 10
    feats = {}
    idxs = [4, 6, 10]
    hooks = []
    for i in idxs:
        hooks.append(off.model[i].register_forward_hook(lambda m, inp, out, i=i: feats.__setitem__(i, out)))

    x = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        _ = off(x)

    for h in hooks:
        h.remove()

    # Our backbone
    from leanyolo.models import get_model
    from leanyolo.data.coco import coco80_class_names
    m = get_model(
        'yolov10s',
        weights=None,
        class_names=coco80_class_names(),
        input_norm_subtract=[0.0],
        input_norm_divide=[1.0],
    )
    m.eval()

    with torch.no_grad():
        c3, c4, c5 = m.backbone(x)

    # Compare shapes (B,C,H,W)
    assert c3.shape == feats[4].shape
    assert c4.shape == feats[6].shape
    assert c5.shape == feats[10].shape
