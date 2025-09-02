from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .tal import make_anchors, dist2bbox, bbox2dist, TaskAlignedAssigner, _bbox_iou_ciou


def _exp_from_dfl(logits: torch.Tensor, reg_max: int) -> torch.Tensor:
    b = logits.shape[0]
    probs = logits.view(b, 4, reg_max).softmax(dim=2)
    idx = torch.arange(reg_max, device=logits.device, dtype=logits.dtype).view(1, 1, reg_max)
    return (probs * idx).sum(dim=2)


def _dfl_loss(logits: torch.Tensor, target: torch.Tensor, reg_max: int) -> torch.Tensor:
    """DFL loss supporting vector or batch inputs.

    Args:
        logits: [..., 4*reg_max]
        target: [..., 4]
    Returns:
        scalar DFL loss (sum over items and sides)
    """
    if logits.dim() == 1:
        x = logits.view(4, reg_max)
        t = target.clamp(0, reg_max - 1 - 1e-3)
        l = t.floor()
        u = l + 1
        wl = (u - t).detach()
        wu = (t - l).detach()
        l = l.long()
        u = u.long()
        ce = F.cross_entropy
        loss = ce(x, l, reduction="none") * wl + ce(x, u, reduction="none") * wu
        return loss.sum()
    else:
        N = logits.shape[0]
        x = logits.view(N, 4, reg_max)
        t = target.clamp(0, reg_max - 1 - 1e-3).view(N, 4)
        l = t.floor()
        u = l + 1
        wl = (u - t).detach()
        wu = (t - l).detach()
        l = l.long()
        u = u.long()
        ce = F.cross_entropy
        # compute per-side losses then sum over sides and batch
        loss_l = ce(x.view(-1, reg_max), l.view(-1), reduction="none").view(N, 4)
        loss_u = ce(x.view(-1, reg_max), u.view(-1), reduction="none").view(N, 4)
        loss = loss_l * wl + loss_u * wu
        return loss.sum()


def _flatten_feats_to_preds(feats: Sequence[torch.Tensor], num_classes: int, reg_max: int) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    B = feats[0].shape[0]
    cat = []
    for xi in feats:
        _, C, H, W = xi.shape
        cat.append(xi.view(B, C, H * W))
    y = torch.cat(cat, dim=2)
    pd, ps = y.split((reg_max * 4, num_classes), dim=1)
    return pd.permute(0, 2, 1).contiguous(), ps.permute(0, 2, 1).contiguous(), list(feats)


def _build_targets_from_list(targets: List[Dict[str, torch.Tensor]], max_boxes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = len(targets)
    device = targets[0]["boxes"].device
    gt_labels = torch.zeros((B, max_boxes, 1), dtype=torch.long, device=device)
    gt_bboxes = torch.zeros((B, max_boxes, 4), dtype=torch.float32, device=device)
    mask_gt = torch.zeros((B, max_boxes, 1), dtype=torch.bool, device=device)
    for b, t in enumerate(targets):
        n = min(t["boxes"].shape[0], max_boxes)
        if n > 0:
            gt_bboxes[b, :n] = t["boxes"][:n]
            gt_labels[b, :n, 0] = t["labels"][:n]
            mask_gt[b, :n, 0] = True
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
    B = feats[0].shape[0]
    # Flatten predictions
    pred_distri, pred_scores, feats_cat = _flatten_feats_to_preds(feats, num_classes, reg_max)
    # Anchors
    anchor_points, stride_tensor = make_anchors(feats_cat, strides)
    # Decode to xyxy in feature space
    # Convert distributions to expected distances per side
    B, A, C = pred_distri.shape
    probs = pred_distri.view(B, A, 4, reg_max).softmax(3)
    proj = torch.arange(reg_max, dtype=probs.dtype, device=probs.device)
    exp_ltrb = torch.matmul(probs, proj)  # [B, A, 4]
    pred_bboxes = dist2bbox(exp_ltrb, anchor_points[None, ...], xywh=False)
    # Build padded targets
    max_boxes = max((t["boxes"].shape[0] for t in targets), default=0)
    gt_labels, gt_bboxes, mask_gt = _build_targets_from_list(targets, max_boxes)
    # Assignment
    assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=num_classes, alpha=0.5, beta=6.0)
    target_labels, target_bboxes, target_scores, fg_mask, _ = assigner(
        pred_scores, pred_bboxes * stride_tensor[None, ...], anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
    )
    # Normalize target scores sum
    target_scores_sum = max(target_scores.sum().item(), 1.0)
    # Classification loss (BCE with logits)
    bce = torch.nn.BCEWithLogitsLoss(reduction="sum")
    cls_loss = bce(pred_scores, target_scores.to(pred_scores.dtype)) / target_scores_sum
    # Bbox + DFL for positives
    reg_loss = torch.zeros((), device=device)
    if fg_mask.any():
        target_bboxes = target_bboxes / stride_tensor[None, ...]
        for b in range(B):
            pos = fg_mask[b]
            if pos.any():
                iou = _bbox_iou_ciou(pred_bboxes[b][pos], target_bboxes[b][pos]).diag()
                iou_loss = (1.0 - iou).sum() / target_scores_sum
                t_ltrb = bbox2dist(anchor_points[pos], target_bboxes[b][pos], reg_max - 1)
                pd = pred_distri[b][pos].view(-1, reg_max * 4)
                dfl = _dfl_loss(pd, t_ltrb.view(-1, 4), reg_max) / target_scores_sum
                reg_loss = reg_loss + (lambda_iou * iou_loss + lambda_dfl * dfl)
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
