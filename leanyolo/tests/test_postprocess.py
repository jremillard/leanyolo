import torch

from leanyolo.models.yolov10.postprocess import decode_v10_official_topk as decode_predictions


def test_decode_predictions_basic():
    # Single-image, single-level signal producing a confident class
    B, C, H, W, NC = 1, 4 + 3, 2, 2, 3
    p3 = torch.zeros(B, C, H, W)
    # Make cell (0,0) produce a high class-0 score
    p3[0, 4 + 0, 0, 0] = 10.0  # large positive -> sigmoid ~ 1
    # bbox tx,ty near 0 -> center near grid corner
    p3[0, 0, 0, 0] = 0.0
    p3[0, 1, 0, 0] = 0.0
    # tw,th = 0 -> exp(0)=1 -> size ~ stride
    p3[0, 2, 0, 0] = 0.0
    p3[0, 3, 0, 0] = 0.0

    preds = [p3, torch.zeros_like(p3), torch.zeros_like(p3)]
    dets = decode_predictions(preds, num_classes=NC, strides=(8, 16, 32), conf_thresh=0.5, iou_thresh=0.5, img_size=(64, 64))
    d = dets[0][0]
    assert d.shape[1] == 6
    assert d.size(0) >= 1
    # Top-left cell should decode to within image bounds
    assert (d[:, :4] >= 0).all() and (d[:, :4] <= 64).all()
