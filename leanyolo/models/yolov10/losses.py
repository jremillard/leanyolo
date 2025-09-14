from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from leanyolo.utils.tal import make_anchors, dist2bbox, bbox2dist, TaskAlignedAssigner, _bbox_iou_ciou


def _exp_from_dfl(logits: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Return per-side expectation from DFL logits.

    Accepts a tensor shaped [N, 4*reg_max] or [4*reg_max] and returns
    the expected distances per side [N, 4] (or [4] if input is 1D).
    """
    x = logits
    squeeze_out = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze_out = True
    n = x.shape[0]
    probs = x.view(n, 4, reg_max).softmax(dim=2)
    bins = torch.arange(reg_max, device=x.device, dtype=x.dtype).view(1, 1, reg_max)
    # Avoid torch.einsum to prevent optional backend import issues.
    # Equivalent to summing over the last dimension after elementwise multiply.
    expect = (probs * bins.expand_as(probs)).sum(dim=2)
    return expect.squeeze(0) if squeeze_out else expect


def _dfl_loss(logits: torch.Tensor, target: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Distribution Focal Loss (sum over items and sides).

    Supports vector input ([4*reg_max]) or batch input ([N, 4*reg_max]).
    Targets are fractional bin locations in [0, reg_max-1].
    """
    x = logits
    t = target
    if x.dim() == 1:
        x = x.unsqueeze(0)
        t = t.unsqueeze(0)
    n = x.shape[0]
    x = x.view(n, 4, reg_max)
    t = t.view(n, 4).clamp(0, reg_max - 1 - 1e-3)

    l = t.floor()  # lower bin
    u = l + 1      # upper bin
    wl = (u - t).detach()
    wu = (t - l).detach()
    l = l.long()
    u = u.long()

    logp = F.log_softmax(x, dim=2)
    # Gather negative log-likelihood at lower/upper bins
    nll_l = -logp.gather(2, l.unsqueeze(-1)).squeeze(-1)
    nll_u = -logp.gather(2, u.unsqueeze(-1)).squeeze(-1)
    loss = (nll_l * wl + nll_u * wu).sum()
    return loss


def _flatten_feats_to_preds(
    feats: Sequence[torch.Tensor], num_classes: int, reg_max: int
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Flatten multi-level feature maps to per-anchor predictions.

    Returns:
        pred_distri: [B, A, 4*reg_max]
        pred_scores: [B, A, num_classes]
        feats_out: original feats list (identity) for downstream anchor gen
    """
    bsz = feats[0].shape[0]
    flat = [x.flatten(2) for x in feats]  # [B, C, HW]
    y = torch.cat(flat, dim=2)
    box_logits, cls_logits = y.split((reg_max * 4, num_classes), dim=1)
    return (
        box_logits.permute(0, 2, 1).contiguous(),
        cls_logits.permute(0, 2, 1).contiguous(),
        list(feats),
    )


def _build_targets_from_list(
    targets: List[Dict[str, torch.Tensor]], max_boxes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a Python list of targets into padded tensors.

    Each target dict has keys 'boxes' [Ni,4] (xyxy) and 'labels' [Ni].
    Outputs are BxNmax tensors with zero-padding and a boolean mask.
    """
    bsz = len(targets)
    dev = targets[0]["boxes"].device if bsz else torch.device("cpu")
    gt_labels = torch.zeros((bsz, max_boxes, 1), dtype=torch.long, device=dev)
    gt_bboxes = torch.zeros((bsz, max_boxes, 4), dtype=torch.float32, device=dev)
    mask_gt = torch.zeros((bsz, max_boxes, 1), dtype=torch.bool, device=dev)
    for i, t in enumerate(targets):
        n = min(int(t["boxes"].shape[0]), max_boxes)
        if n:
            gt_bboxes[i, :n].copy_(t["boxes"][:n])
            gt_labels[i, :n, 0].copy_(t["labels"][:n])
            mask_gt[i, :n, 0] = True
    return gt_labels, gt_bboxes, mask_gt


def _v8_detection_loss(
    feats: Sequence[torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    *,
    num_classes: int,
    reg_max: int = 16,
    strides: Tuple[int, int, int] = (8, 16, 32),
    tal_topk: int = 10,
    lambda_cls: float = 1.0,
    lambda_iou: float = 1.0,
    lambda_dfl: float = 1.5,
) -> Dict[str, torch.Tensor]:
    device = feats[0].device
    bsz = feats[0].shape[0]

    # 1) Flatten raw predictions into per-anchor tensors
    pred_distri, pred_scores, feats_cat = _flatten_feats_to_preds(feats, num_classes, reg_max)

    # 2) Build anchor centers and per-anchor stride
    anchor_xy, stride_tensor = make_anchors(feats_cat, strides)

    # 3) Convert DFL logits -> expected distances -> decode to xyxy (feature coords)
    ba, a, _ = pred_distri.shape
    exp_ltrb = _exp_from_dfl(pred_distri.view(-1, 4 * reg_max), reg_max).view(ba, a, 4)
    pred_bboxes = dist2bbox(exp_ltrb, anchor_xy[None, ...], xywh=False)

    # 4) Prepare padded GT tensors [B, Nmax, ...]
    max_boxes = max((int(t["boxes"].shape[0]) for t in targets), default=0)
    gt_labels, gt_bboxes, mask_gt = _build_targets_from_list(targets, max_boxes)

    # 5) Task-aligned assignment in pixel space (scale predictions and anchors by stride)
    assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=num_classes, alpha=0.5, beta=6.0)
    tgt_labels, tgt_bboxes, tgt_scores, fg_mask, _ = assigner(
        pred_scores,
        pred_bboxes * stride_tensor[None, ...],
        anchor_xy * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )

    # 6) Classification loss (BCE with logits), normalized by sum of assigned scores
    denom = max(tgt_scores.sum().item(), 1.0)
    bce = torch.nn.BCEWithLogitsLoss(reduction="sum")
    cls_loss = bce(pred_scores, tgt_scores.to(pred_scores.dtype)) / denom

    # 7) Regression losses (IoU + DFL) on positives only
    reg_loss = torch.zeros((), device=device)
    if fg_mask.any():
        # Move targets back to feature space (divide by stride)
        tgt_bboxes = tgt_bboxes / stride_tensor[None, ...]
        for b in range(bsz):
            pos = fg_mask[b]
            if not pos.any():
                continue
            # IoU (CIoU) between predicted and target boxes for positives
            ious = _bbox_iou_ciou(pred_bboxes[b][pos], tgt_bboxes[b][pos]).diag()
            iou_term = (1.0 - ious).sum() / denom
            # DFL term
            t_ltrb = bbox2dist(anchor_xy[pos], tgt_bboxes[b][pos], reg_max - 1)
            pd_logits = pred_distri[b][pos].view(-1, reg_max * 4)
            dfl_term = _dfl_loss(pd_logits, t_ltrb.view(-1, 4), reg_max) / denom
            reg_loss = reg_loss + (lambda_iou * iou_term + lambda_dfl * dfl_term)

    total = lambda_cls * cls_loss + reg_loss
    return {"total": total, "cls": cls_loss, "reg": reg_loss}


def detection_loss_v10(
    raw: Sequence[torch.Tensor] | Dict[str, Sequence[torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    *,
    num_classes: int,
    reg_max: int = 16,
    strides: Tuple[int, int, int] = (8, 16, 32),
) -> Dict[str, torch.Tensor]:
    """YOLOv10 parity loss wrapper.

    If raw is a dict with 'one2many'/'one2one', compute both (topk=10 and topk=1) and sum.
    Otherwise, fall back to one2many only (topk=10) on the given list of 3 tensors.
    """
    if isinstance(raw, dict):
        l_many = _v8_detection_loss(raw["one2many"], targets, num_classes=num_classes, reg_max=reg_max, strides=strides, tal_topk=10)
        l_one = _v8_detection_loss(raw["one2one"], targets, num_classes=num_classes, reg_max=reg_max, strides=strides, tal_topk=1)
        return {"total": l_many["total"] + l_one["total"], "cls": l_many["cls"] + l_one["cls"], "reg": l_many["reg"] + l_one["reg"]}
    else:
        return _v8_detection_loss(raw, targets, num_classes=num_classes, reg_max=reg_max, strides=strides, tal_topk=10)
