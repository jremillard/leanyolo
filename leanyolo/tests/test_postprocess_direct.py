import torch

from leanyolo.models.yolov10.postprocess import decode_v10_predictions


def _make_direct_preds(b: int, nc: int, h: int, w: int, s: int):
    # Layout [B, 4+nc, H, W]
    c = 4 + nc
    p = torch.zeros((b, c, h, w), dtype=torch.float32)
    # Put two confident boxes at adjacent cells to trigger NMS
    # First at (0,0)
    p[0, 0, 0, 0] = 0.0   # dx -> sigmoid 0.5
    p[0, 1, 0, 0] = 0.0   # dy
    p[0, 2, 0, 0] = torch.log(torch.tensor(4.0))  # log bw -> exp 4.0
    p[0, 3, 0, 0] = torch.log(torch.tensor(4.0))  # log bh
    p[0, 4, 0, 0] = 10.0  # class 0 logit high

    # Second at (0,1) same class, similar size to overlap heavily
    if w > 1:
        p[0, 0, 0, 1] = 0.0
        p[0, 1, 0, 1] = 0.0
        p[0, 2, 0, 1] = torch.log(torch.tensor(4.0))
        p[0, 3, 0, 1] = torch.log(torch.tensor(4.0))
        p[0, 4, 0, 1] = 10.0
    return p


def test_decode_direct_offsets_basic_nms_and_clamp():
    b, nc = 1, 1
    strides = (8, 16, 32)
    # Three levels as expected by the API; only first has positives
    preds = [
        _make_direct_preds(b, nc, 2, 2, strides[0]),
        torch.zeros((b, 4 + nc, 1, 1), dtype=torch.float32),
        torch.zeros((b, 4 + nc, 1, 1), dtype=torch.float32),
    ]

    out = decode_v10_predictions(
        preds,
        num_classes=nc,
        strides=strides,
        conf_thresh=0.9,
        iou_thresh=0.5,  # standard; overlap should suppress now
        max_det=10,
        img_size=(16, 16),  # clamps boxes to image bounds
    )

    assert len(out) == b and len(out[0]) == 1
    det = out[0][0]
    # Expect a single detection after NMS
    assert det.shape[1] == 6 and det.shape[0] == 1
    # Score is high (sigmoid(10)) and class is 0.0
    assert det[0, 4] > 0.99 and det[0, 5] == 0.0


def test_decode_direct_offsets_no_detections_returns_empty_tensor():
    b, nc = 1, 2
    strides = (8, 16, 32)
    preds = [
        torch.zeros((b, 4 + nc, 1, 1), dtype=torch.float32),
        torch.zeros((b, 4 + nc, 1, 1), dtype=torch.float32),
        torch.zeros((b, 4 + nc, 1, 1), dtype=torch.float32),
    ]
    out = decode_v10_predictions(
        preds,
        num_classes=nc,
        strides=strides,
        conf_thresh=0.99,
        iou_thresh=0.5,
        max_det=10,
    )
    assert out[0][0].numel() == 0
