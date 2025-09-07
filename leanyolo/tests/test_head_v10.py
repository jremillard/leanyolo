import torch

from leanyolo.models.yolov10.head import V10Detect, DFL as HeadDFL


def _fake_feats(b=2):
    # Create 3 FPN levels with increasing channels and decreasing spatial size
    P3 = torch.randn(b, 64, 16, 16)
    P4 = torch.randn(b, 128, 8, 8)
    P5 = torch.randn(b, 256, 4, 4)
    return [P3, P4, P5]


def test_v10detect_training_and_eval_outputs():
    feats = _fake_feats(b=1)
    nc, reg_max = 7, 12
    head = V10Detect(nc=nc, ch=[x.shape[1] for x in feats], reg_max=reg_max)

    # Training path: dict with two branches
    head.train()
    out = head(feats)
    assert isinstance(out, dict) and set(out.keys()) == {"one2many", "one2one"}
    for k in ("one2many", "one2one"):
        assert isinstance(out[k], list) and len(out[k]) == 3
    # Shapes per level
    for i, xi in enumerate(feats):
        H, W = xi.shape[-2:]
        for k in ("one2many", "one2one"):
            y = out[k][i]
            assert y.shape == (1, 4 * reg_max + nc, H, W)

    # Eval path: list of tensors
    head.eval()
    ev = head(feats)
    assert isinstance(ev, list) and len(ev) == 3
    for i, yi in enumerate(ev):
        H, W = feats[i].shape[-2:]
        assert yi.shape == (1, 4 * reg_max + nc, H, W)


def test_head_dfl_matches_manual_expectation():
    # Validate HeadDFL projection is consistent with softmax-weighted bin indices
    b, A, reg_max = 2, 5, 10
    logits = torch.randn(b, 4 * reg_max, A)
    dfl = HeadDFL(c1=reg_max)
    got = dfl(logits)
    # Manual expectation
    probs = logits.view(b, 4, reg_max, A).softmax(dim=2)
    idx = torch.arange(reg_max, dtype=probs.dtype).view(1, 1, reg_max, 1)
    want = (probs * idx).sum(dim=2)
    assert torch.allclose(got, want, rtol=1e-6, atol=1e-6)

