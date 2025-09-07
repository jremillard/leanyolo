import torch

from leanyolo.models.yolov10.postprocess import (
    decode_v10_official_topk,
    decode_v10_predictions,
)


def _make_dfl_logits_for_anchor(b, c, h, w, reg_max, nc, sel_anchor, ltrb_bins, cls_idx, cls_logit=8.0, bin_logit=20.0):
    # Produce logits shaped [B, 4*reg_max+nc, H, W] with one anchor having clear signal
    x = torch.zeros(b, 4 * reg_max + nc, h, w)
    a = sel_anchor  # (gy, gx)
    by, bx = a
    for side, tb in enumerate(ltrb_bins):
        ch = side * reg_max + int(tb)
        x[0, ch, by, bx] = bin_logit
    x[0, 4 * reg_max + cls_idx, by, bx] = cls_logit
    return x


def test_decode_topk_dfl_known_anchor_geometry():
    # Single-level, controlled DFL bins -> exact geometry check
    B, H, W, s = 1, 3, 4, 8
    nc, reg_max = 5, 8
    sel = (1, 2)
    # choose bins that produce distances exactly equal to those bins
    ltrb_bins = (2, 3, 4, 5)
    cls_idx = 2
    p3 = _make_dfl_logits_for_anchor(B, 4 * reg_max + nc, H, W, reg_max, nc, sel, ltrb_bins, cls_idx)
    dets = decode_v10_official_topk([p3], num_classes=nc, strides=(s,), max_det=10)
    out = dets[0][0]
    assert out.shape[1] == 6 and out.size(0) >= 1
    # Expected box from anchor center (gx+0.5, gy+0.5)*s and ltrb*s distances
    gy, gx = sel
    cx = (gx + 0.5) * s
    cy = (gy + 0.5) * s
    # Compute expectation using the same softmax over bins to avoid tiny numerical drift
    reg_max = 8
    idx = torch.arange(reg_max).view(1, 1, reg_max, 1).float()
    # construct the local logits again for the 4 sides
    logits = torch.zeros(1, 4, reg_max, 1)
    for i, tb in enumerate(ltrb_bins):
        logits[0, i, int(tb), 0] = 20.0
    probs = logits.softmax(2)
    dist = (probs * idx).sum(2).view(4)
    l, t, r, btm = (dist * s).tolist()
    exp = torch.tensor([cx - l, cy - t, cx + r, cy + btm], dtype=out.dtype)
    # Find row matching our class index
    # Top-k can include multiple classes, so select class match
    rows = (out[:, 5].long() == cls_idx).nonzero(as_tuple=False).view(-1)
    assert rows.numel() >= 1
    box = out[rows[0], :4]
    assert torch.allclose(box, exp, rtol=0, atol=1e-5)


def test_decode_nms_filters_and_clamps():
    # Direct-offset path: 4+nc channels, overlapping boxes -> NMS keeps 1; clamp within bounds
    B, H, W, s = 1, 2, 2, 8
    nc = 3
    C = 4 + nc
    p = torch.zeros(B, C, H, W)
    p[:, 4:, :, :] = -10.0  # suppress all classes by default
    # Two anchors: (0,0) high score, (0,1) slightly lower but overlapping heavily
    p[0, 4 + 0, 0, 0] = 8.0
    p[0, 4 + 0, 0, 1] = 6.0
    # tx,ty ~ 0 centers at grid indices; tw,th large -> big boxes; clamp will apply
    p[0, 0, 0, 0] = 0.0
    p[0, 1, 0, 0] = 0.0
    p[0, 2, 0, 0] = 3.0  # exp ~ 20x stride
    p[0, 3, 0, 0] = 3.0
    p[0, 0, 0, 1] = 0.0
    p[0, 1, 0, 1] = 0.0
    p[0, 2, 0, 1] = 3.0
    p[0, 3, 0, 1] = 3.0

    pneg = torch.zeros_like(p)
    pneg[:, 4:, :, :] = -10.0  # suppress by conf threshold
    dets = decode_v10_predictions([p, pneg, pneg], num_classes=nc, strides=(s, s, s), conf_thresh=0.25, iou_thresh=0.5, max_det=10, img_size=(64, 64))
    out = dets[0][0]
    assert out.size(0) == 1  # NMS removed overlapping one
    # Clamped to image
    x1, y1, x2, y2 = out[0, :4]
    assert 0.0 <= x1 <= 64.0 and 0.0 <= y1 <= 64.0 and 0.0 <= x2 <= 64.0 and 0.0 <= y2 <= 64.0


def test_decode_nms_conf_thresholding_and_batch():
    # Batch=2, ensure per-image filtering on conf
    B, H, W, s = 2, 1, 1, 16
    nc = 2
    C = 4 + nc
    p = torch.zeros(B, C, H, W)
    p[0, 4 + 1, 0, 0] = 2.0  # keep
    p[1, 4 + 1, 0, 0] = -5.0  # below threshold
    dets = decode_v10_predictions([p, torch.zeros_like(p), torch.zeros_like(p)], num_classes=nc, strides=(s, s, s), conf_thresh=0.5, iou_thresh=0.5, max_det=5, img_size=(64, 64))
    out0 = dets[0][0]
    out1 = dets[1][0]
    assert out0.size(0) >= 1
    assert out1.size(0) == 0


def test_decode_topk_respects_max_det():
    B, H, W, s = 1, 2, 2, 8
    nc, reg_max = 4, 8
    # Fill every anchor with same class high; expect limited by max_det
    p3 = torch.zeros(B, 4 * reg_max + nc, H, W)
    p3[:, 4 * reg_max + 0] = 5.0
    dets = decode_v10_official_topk([p3], num_classes=nc, strides=(s,), max_det=3)
    assert dets[0][0].size(0) == 3
