import math
import torch

from leanyolo.models.yolov10.layers import (
    Conv,
    Bottleneck,
    C2f,
    SPPF,
    UpSample,
    CIB,
    C2fCIB,
    Attention,
    PSA,
    SCDown,
)


def _rand(ch, h=16, w=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, ch, h, w, generator=g)


def test_conv_basic_and_depthwise_groups():
    x = _rand(8)
    # standard conv
    m = Conv(c_in=8, c_out=16, k=3, s=1, p=None, g=1, act=True)
    y = m(x)
    assert y.shape == (2, 16, 16, 16)
    # depthwise: groups=c_out only if c_in==c_out and groups==c_out
    m_dw = Conv(c_in=8, c_out=8, k=3, s=1, p=None, g=8, act=True)
    ydw = m_dw(x)
    assert ydw.shape == (2, 8, 16, 16)
    assert isinstance(m_dw.conv, torch.nn.Conv2d) and m_dw.conv.groups == 8


def test_bottleneck_residual_shapes():
    x = _rand(16)
    m_add = Bottleneck(c_in=16, c_out=16, shortcut=True, g=1, e=0.5)
    y = m_add(x)
    assert y.shape == x.shape
    # when channels differ, no residual add
    m_nadd = Bottleneck(c_in=16, c_out=24, shortcut=True, g=1, e=0.5)
    y2 = m_nadd(x)
    assert y2.shape == (2, 24, 16, 16)


def test_c2f_channel_fusion_counts():
    x = _rand(32)
    m = C2f(c_in=32, c_out=64, n=3, shortcut=True, g=1, e=0.5)
    y = m(x)
    assert y.shape == (2, 64, 16, 16)


def test_sppf_pyramid_context():
    x = _rand(24)
    m = SPPF(c_in=24, c_out=48, k=5)
    y = m(x)
    assert y.shape == (2, 48, 16, 16)


def test_upsample_nearest():
    x = _rand(12, 10, 7)
    m = UpSample(scale_factor=2.0)
    y = m(x)
    assert y.shape == (2, 12, 20, 14)


def test_cib_paths_and_residual():
    x = _rand(32)
    # No long-kernel path
    m = CIB(c_in=32, c_out=32, shortcut=True, e=0.5, lk=False)
    y = m(x)
    assert y.shape == x.shape
    # Long-kernel variant
    mlk = CIB(c_in=32, c_out=48, shortcut=False, e=0.5, lk=True)
    y2 = mlk(x)
    assert y2.shape == (2, 48, 16, 16)


def test_c2fcib_stacks_cib():
    x = _rand(24)
    m = C2fCIB(c_in=24, c_out=40, n=2, shortcut=True, lk=True, e=0.5)
    y = m(x)
    assert y.shape == (2, 40, 16, 16)


def test_attention_shapes_heads_scale():
    dim = 64
    x = _rand(dim)
    m = Attention(dim=dim, num_heads=4, attn_ratio=0.5)
    y = m(x)
    assert y.shape == x.shape
    # num_heads must divide dim per implementation
    assert m.num_heads == 4 and m.head_dim * m.num_heads == dim


def test_psa_splits_and_fuses():
    c = 48
    x = _rand(c)
    m = PSA(c_in=c, c_out=c, e=0.5)
    y = m(x)
    assert y.shape == x.shape


def test_scdown_stride_and_channels():
    x = _rand(16, 15, 17)  # odd spatial dims
    m = SCDown(c_in=16, c_out=24, k=3, s=2)
    y = m(x)
    # For k=3, p=1, s=2: out = floor((in + 2p - (k-1) - 1)/s + 1)
    h_exp = math.floor(((15 + 2*1 - (3 - 1) - 1) / 2) + 1)
    w_exp = math.floor(((17 + 2*1 - (3 - 1) - 1) / 2) + 1)
    assert y.shape == (2, 24, h_exp, w_exp)
    # ensure depthwise groups on cv2
    assert isinstance(m.cv2.conv, torch.nn.Conv2d) and m.cv2.conv.groups == 24
