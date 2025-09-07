import torch

from leanyolo.utils.box_ops import (
    box_area,
    box_iou,
    box_xywh_to_xyxy,
    box_xyxy_to_xywh,
    nms,
    scale_coords,
    unletterbox_coords,
)


def test_xy_conversions_and_area_edge_cases():
    xywh = torch.tensor([[10.0, 10.0, 4.0, 6.0]])
    xyxy = box_xywh_to_xyxy(xywh)
    back = box_xyxy_to_xywh(xyxy)
    assert torch.allclose(xywh, back)

    # Negative width/height clamp to zero area
    bad = torch.tensor([[5.0, 5.0, 3.0, 2.0]])  # x2<x1,y2<y1
    assert box_area(bad).item() == 0.0


def test_iou_and_nms_behaviors():
    # Two identical boxes -> IoU 1; one disjoint -> IoU 0
    a = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    b = torch.tensor([[0.0, 0.0, 2.0, 2.0], [10.0, 10.0, 12.0, 12.0]])
    iou = box_iou(a, b)
    assert torch.allclose(iou, torch.tensor([[1.0, 0.0]]), atol=1e-6)

    # Empty NMS input
    keep = nms(torch.zeros((0, 4)), torch.zeros((0,)), 0.5)
    assert keep.numel() == 0

    # NMS suppresses overlapping lower-score boxes
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],  # high score
            [1.0, 1.0, 9.0, 9.0],    # overlaps heavily
            [20.0, 20.0, 21.0, 21.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.1])
    keep = nms(boxes, scores, 0.5)
    # Expect to keep the first and the third
    assert set(keep.tolist()) == {0, 2}


def test_scale_and_unletterbox_coords():
    boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0]])
    # Scale from (h=100,w=200) to (h=50,w=100) -> halves coords
    scaled = scale_coords((100, 200), boxes, (50, 100))
    assert torch.allclose(scaled, boxes / 2)

    # Unletterbox inverse: remove pad and divide by gain, then clip to HxW
    gain = (2.0, 2.0)
    pad = (5, 3)
    letterboxed = torch.tensor([[15.0, 13.0, 35.0, 33.0]])  # after pad+gain
    unlb = unletterbox_coords(letterboxed, gain, pad, (20, 20))
    # ((x - 5)/2, (y - 3)/2) clipped to (20,20)
    expect = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
    assert torch.allclose(unlb, expect)

