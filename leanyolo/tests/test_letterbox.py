import numpy as np
import pytest

from leanyolo.utils.letterbox import letterbox


def _expected_dims_and_pad(h, w, new_shape, auto=False, scale_fill=False, scaleup=True, stride=32):
    if isinstance(new_shape, int):
        tgt_h, tgt_w = new_shape, new_shape
    else:
        tgt_h, tgt_w = int(new_shape[0]), int(new_shape[1])

    if scale_fill:
        gain_w = tgt_w / max(w, 1)
        gain_h = tgt_h / max(h, 1)
        new_w, new_h = tgt_w, tgt_h
        pad_w = 0.0
        pad_h = 0.0
    else:
        r = min(tgt_w / max(w, 1), tgt_h / max(h, 1))
        if not scaleup:
            r = min(r, 1.0)
        gain_w = r
        gain_h = r
        new_w = int(round(w * r))
        new_h = int(round(h * r))
        pad_w = float(tgt_w - new_w)
        pad_h = float(tgt_h - new_h)

        if auto and stride > 1:
            pad_w = pad_w % stride
            pad_h = pad_h % stride

    left = int(round(pad_w / 2.0))
    top = int(round(pad_h / 2.0))
    out_w = new_w + int(round(pad_w))
    out_h = new_h + int(round(pad_h))
    return (out_h, out_w), (gain_w, gain_h), (left, top)


def test_letterbox_basic_square_pad():
    # Input HxW = 200x100, square target 320 keeps aspect, pads equally on width
    img = np.zeros((200, 100, 3), dtype=np.uint8)
    out, gain, pad = letterbox(img, new_shape=320, auto=False, scale_fill=False, scaleup=True, stride=32)

    # Expected
    exp_shape, exp_gain, exp_pad = _expected_dims_and_pad(200, 100, 320, auto=False, scale_fill=False, scaleup=True, stride=32)

    assert out.shape[:2] == exp_shape, f"Unexpected output shape: {out.shape[:2]} vs {exp_shape}"
    assert pytest.approx(gain[0], rel=0, abs=1e-6) == exp_gain[0]
    assert pytest.approx(gain[1], rel=0, abs=1e-6) == exp_gain[1]
    assert pad == exp_pad


def test_letterbox_scaleup_false_no_upsample():
    # Input smaller than target; with scaleup=False we don't upscale, only pad
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    out, gain, pad = letterbox(img, new_shape=320, auto=False, scale_fill=False, scaleup=False, stride=32)

    exp_shape, exp_gain, exp_pad = _expected_dims_and_pad(50, 50, 320, auto=False, scale_fill=False, scaleup=False, stride=32)

    assert out.shape[:2] == exp_shape
    assert gain == pytest.approx(exp_gain, rel=0, abs=1e-6)
    assert pad == exp_pad


def test_letterbox_scale_fill_stretches_exactly():
    # scale_fill stretches to exact size, no padding, possibly non-uniform gain
    img = np.zeros((200, 100, 3), dtype=np.uint8)
    out, gain, pad = letterbox(img, new_shape=320, auto=False, scale_fill=True, scaleup=True, stride=32)

    exp_shape, exp_gain, exp_pad = _expected_dims_and_pad(200, 100, 320, auto=False, scale_fill=True, scaleup=True, stride=32)

    assert out.shape[:2] == exp_shape
    assert gain == pytest.approx(exp_gain, rel=0, abs=1e-6)
    assert pad == exp_pad


def test_letterbox_auto_stride_reduces_padding():
    # When auto=True and stride>1, padding is reduced via modulo, potentially shrinking output dims
    img = np.zeros((300, 200, 3), dtype=np.uint8)
    out, gain, pad = letterbox(img, new_shape=320, auto=True, scale_fill=False, scaleup=True, stride=32)

    exp_shape, exp_gain, exp_pad = _expected_dims_and_pad(300, 200, 320, auto=True, scale_fill=False, scaleup=True, stride=32)

    assert out.shape[:2] == exp_shape
    # Ensure stride multiple when target is a stride multiple
    assert exp_shape[0] % 32 == 0
    assert exp_shape[1] % 32 == 0
    assert gain == pytest.approx(exp_gain, rel=0, abs=1e-6)
    assert pad == exp_pad


@pytest.mark.parametrize("stride", [8, 16, 32, 64])
def test_letterbox_auto_true_stride_variants(stride: int):
    # Ensure output dims are multiples of the given stride and <= target dims
    h, w = 301, 219  # pick values to avoid accidental perfect alignment
    img = np.zeros((h, w, 3), dtype=np.uint8)
    out, gain, pad = letterbox(img, new_shape=320, auto=True, scale_fill=False, scaleup=True, stride=stride)

    exp_shape, exp_gain, exp_pad = _expected_dims_and_pad(h, w, 320, auto=True, scale_fill=False, scaleup=True, stride=stride)

    assert out.shape[:2] == exp_shape
    assert exp_shape[0] % stride == 0 and exp_shape[1] % stride == 0
    assert exp_shape[0] <= 320 and exp_shape[1] <= 320
    assert gain == pytest.approx(exp_gain, rel=0, abs=1e-6)
    assert pad == exp_pad


@pytest.mark.parametrize("stride", [1, 8, 32])
def test_letterbox_auto_false_ignores_stride(stride: int):
    # With auto=False, output exactly matches target dims regardless of stride
    img = np.zeros((123, 77, 3), dtype=np.uint8)
    out, gain, pad = letterbox(img, new_shape=320, auto=False, scale_fill=False, scaleup=True, stride=stride)

    exp_shape, exp_gain, exp_pad = _expected_dims_and_pad(123, 77, 320, auto=False, scale_fill=False, scaleup=True, stride=stride)
    assert out.shape[:2] == (320, 320)  # explicit
    assert out.shape[:2] == exp_shape
    assert gain == pytest.approx(exp_gain, rel=0, abs=1e-6)
    assert pad == exp_pad
