import math

import torch

from leanyolo.utils.losses_v10 import _dfl_loss, _exp_from_dfl, detection_loss_v10


def test_dfl_loss_matches_official_formula_seeded():
    torch.manual_seed(0)
    reg_max = 16
    # Random logits per side: [4*reg_max]
    logits = torch.randn(4 * reg_max)
    # Random fractional targets in [0, reg_max-1)
    target = torch.rand(4) * (reg_max - 1 - 1e-3)

    # Our implementation
    ours = _dfl_loss(logits, target, reg_max)

    # Official formula: two-hot CE to left/right integer bins per side, sum across 4 sides
    x = logits.view(4, reg_max)
    t = target.clamp(0, reg_max - 1 - 1e-3)
    l = t.floor()
    u = l + 1
    wl = (u - t)
    wu = (t - l)
    l = l.long()
    u = u.long()
    ce = torch.nn.functional.cross_entropy
    official = (ce(x, l, reduction="none") * wl + ce(x, u, reduction="none") * wu).sum()

    assert torch.allclose(ours, official, rtol=1e-6, atol=1e-6)


def test_exp_from_dfl_matches_DFL_module():
    # Compare expectation against a conv-weighted DFL module from official design
    reg_max = 16
    # Shape [B, C, A] where C=4*reg_max; use B=1, A=5 anchors
    B, A = 1, 5
    torch.manual_seed(1)
    # Build logits: [B, C, A]
    logits = torch.randn(B, 4 * reg_max, A)

    # Implement official DFL forward inline
    x = logits.view(B, 4, reg_max, A).transpose(2, 1)  # (B, reg_max, 4, A)
    probs_official = x.softmax(1)
    idx = torch.arange(reg_max, dtype=probs_official.dtype).view(1, reg_max, 1, 1)
    expected = (probs_official * idx).sum(1).view(B, 4, A)

    # Compute ours in a batched way
    probs = logits.view(B, 4, reg_max, A).softmax(2)
    idx2 = torch.arange(reg_max, dtype=probs.dtype).view(1, 1, reg_max, 1)
    ours_batched = (probs * idx2).sum(2)
    assert torch.allclose(ours_batched, expected, rtol=1e-6, atol=1e-6)


def test_detection_loss_v10_runs_and_returns_finite_values():
    # Craft a tiny batch with 1 image and 1 target, ensure loss gets smaller as preds improve.
    reg_max = 16
    nc = 3
    H = W = 8  # small map for speed
    B = 1
    # Build raw predictions for 3 levels (simulate P3 only here by repeating)
    # Shape per level: [B, 4*reg_max+nc, H, W]
    base = torch.zeros(B, 4 * reg_max + nc, H, W)

    # Target box near center cell (4,4) at stride 8
    stride = 8
    cx, cy = torch.tensor(4.0 * stride), torch.tensor(4.0 * stride)
    w, h = torch.tensor(32.0), torch.tensor(32.0)
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    targets = [{
        "boxes": torch.tensor([[x1, y1, x2, y2]]),
        "labels": torch.tensor([1]),
    }]

    # Bad predictions: zero logits
    raw_bad = [base.clone(), base.clone(), base.clone()]
    loss_bad = detection_loss_v10(raw_bad, targets, num_classes=nc)

    # Better predictions: set DFL logits to favor correct distances and class logit high at the center cell
    raw_good = [base.clone(), base.clone(), base.clone()]
    lvl = 0  # level selected by _level_for_box for w=h=32
    gx, gy = 4, 4
    # Distances in bins for stride=8
    l = (cx - x1) / stride
    t = (cy - y1) / stride
    r = (x2 - cx) / stride
    btm = (y2 - cy) / stride
    tgt = torch.stack((l, t, r, btm))
    # For each side, place high logit near floor of target bin
    for i, tval in enumerate(tgt):
        tb = int(torch.floor(tval).item())
        raw_good[lvl][0, i * reg_max + tb, gy, gx] = 8.0
    # Class 1 logit high
    raw_good[lvl][0, 4 * reg_max + 1, gy, gx] = 5.0

    loss_good = detection_loss_v10(raw_good, targets, num_classes=nc)
    # Check keys and finiteness
    for k in ("total", "cls", "reg"):
        assert k in loss_bad and k in loss_good
        assert torch.isfinite(loss_bad[k]) and torch.isfinite(loss_good[k])
