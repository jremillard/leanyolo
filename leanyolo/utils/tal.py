from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_anchors(feats: Sequence[torch.Tensor], strides: Sequence[int], grid_cell_offset: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate anchor points and stride tensor from per-level feature maps.

    Args:
        feats: list of tensors [B, C, H, W]
        strides: strides per level (e.g., (8,16,32))
        grid_cell_offset: fractional offset for centers (default 0.5)
    Returns:
        anchor_points: [sum(H*W), 2] with (x,y) in feature-cell coordinates
        stride_tensor: [sum(H*W), 1] with stride per anchor
    """
    anchor_points: List[torch.Tensor] = []
    stride_tensor: List[torch.Tensor] = []
    assert len(feats) == len(strides)
    dtype = feats[0].dtype
    device = feats[0].device
    for i, s in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if hasattr(torch, "vmap") else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), float(s), dtype=dtype, device=device))
    return torch.cat(anchor_points, 0), torch.cat(stride_tensor, 0)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xywh: bool = False, dim: int = -1) -> torch.Tensor:
    """Transform distance(l,t,r,b) to box(xyxy or xywh)."""
    assert distance.shape[dim] == 4
    lt, rb = distance.split([2, 2], dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points: torch.Tensor, bbox_xyxy: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Transform bbox(xyxy) to dist(l,t,r,b) clamped to [0, reg_max)."""
    x1y1, x2y2 = bbox_xyxy.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)


def _bbox_iou_ciou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute Complete IoU (CIoU) between boxes1 [N,4] and boxes2 [M,4] (xyxy)."""
    # IoU
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2 - inter + 1e-9
    iou = inter / union
    # Enclosing box
    cw = (torch.max(boxes1[:, None, 2], boxes2[:, 2]) - torch.min(boxes1[:, None, 0], boxes2[:, 0])).clamp(min=0)
    ch = (torch.max(boxes1[:, None, 3], boxes2[:, 3]) - torch.min(boxes1[:, None, 1], boxes2[:, 1])).clamp(min=0)
    c2 = cw.pow(2) + ch.pow(2) + 1e-9
    # Center distance
    b1cx = (boxes1[:, 0] + boxes1[:, 2]) / 2
    b1cy = (boxes1[:, 1] + boxes1[:, 3]) / 2
    b2cx = (boxes2[:, 0] + boxes2[:, 2]) / 2
    b2cy = (boxes2[:, 1] + boxes2[:, 3]) / 2
    rho2 = (b1cx[:, None] - b2cx).pow(2) + (b1cy[:, None] - b2cy).pow(2)
    # Aspect ratio term
    w1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=1e-9)
    h1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=1e-9)
    w2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=1e-9)
    h2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=1e-9)
    v = (4 / (3.141592653589793 ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1[:, None] / h1[:, None])).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-9)
    ciou = iou - (rho2 / c2) - alpha * v
    return ciou.clamp(0, 1)


class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk: int = 10, num_classes: int = 80, alpha: float = 0.5, beta: float = 6.0, eps: float = 1e-9):
        super().__init__()
        self.topk = int(topk)
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,  # [B, A, C]
        pd_bboxes: torch.Tensor,  # [B, A, 4] in xyxy (feature space)
        anc_points: torch.Tensor,  # [A, 2]
        gt_labels: torch.Tensor,  # [B, N, 1]
        gt_bboxes: torch.Tensor,  # [B, N, 4]
        mask_gt: torch.Tensor,  # [B, N, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n_max, _ = gt_labels.shape
        if n_max == 0:
            device = pd_scores.device
            A = pd_scores.shape[1]
            return (
                torch.full((bsz, A), self.num_classes, device=device, dtype=torch.int64),
                torch.zeros((bsz, A, 4), device=device, dtype=pd_bboxes.dtype),
                torch.zeros((bsz, A, self.num_classes), device=device, dtype=pd_scores.dtype),
                torch.zeros((bsz, A), device=device, dtype=torch.bool),
                torch.zeros((bsz, A), device=device, dtype=torch.int64),
            )

        # mask of anchors inside gts
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes).to(torch.bool) & mask_gt.bool()
        # alignment metric
        overlaps = self.iou_calculation(gt_bboxes, pd_bboxes)  # [B,N,A]
        b_idx = torch.arange(bsz, device=pd_scores.device)[:, None, None]
        gt_ind = gt_labels.long().squeeze(-1).clamp(min=0)
        cls_scores = torch.gather(pd_scores.sigmoid().transpose(1, 2), 1, gt_ind.unsqueeze(-1).expand(-1, -1, pd_scores.shape[1]))  # [B,N,A]
        align_metric = (cls_scores.clamp(0, 1) ** self.alpha) * (overlaps.clamp(0, 1) ** self.beta)
        align_metric = align_metric * mask_in_gts.float()

        # topk candidates per gt
        topk_metrics, topk_idxs = torch.topk(align_metric, k=min(self.topk, align_metric.shape[-1]), dim=-1)
        # create mask
        mask_topk = topk_metrics.max(-1, keepdim=True)[0] > self.eps
        topk_idxs = topk_idxs.masked_fill(~mask_topk, 0)
        count_tensor = torch.zeros_like(align_metric, dtype=torch.int8)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8)
        for k in range(topk_idxs.shape[-1]):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        mask_pos = (count_tensor > 0) & mask_in_gts

        # resolve multiple gts per anchor by highest overlap
        fg_mask = mask_pos.sum(dim=1)  # [B,A]
        if fg_mask.max() > 1:
            max_overlaps_idx = overlaps.argmax(1)  # [B,A]
            is_max = torch.zeros_like(mask_pos, dtype=mask_pos.dtype)
            is_max.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where((fg_mask.unsqueeze(1) > 1), is_max, mask_pos)
            fg_mask = mask_pos.sum(1)
        target_gt_idx = mask_pos.float().argmax(1)  # [B,A]

        # Build targets
        batch_ind = torch.arange(bsz, device=pd_scores.device)[:, None]
        gather_idx = target_gt_idx + batch_ind * n_max
        target_labels = gt_labels.long().flatten()[gather_idx]
        target_bboxes = gt_bboxes.view(-1, 4)[gather_idx]
        target_scores = torch.zeros((bsz, pd_scores.shape[1], self.num_classes), device=pd_scores.device, dtype=torch.int64)
        target_labels = target_labels.clamp(0)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        target_scores = target_scores * fg_mask.bool().unsqueeze(-1)
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def iou_calculation(self, gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
        # gt_bboxes: [B,N,4], pd_bboxes: [B,A,4]
        B, N, _ = gt_bboxes.shape
        A = pd_bboxes.shape[1]
        out = torch.zeros((B, N, A), device=gt_bboxes.device, dtype=gt_bboxes.dtype)
        for b in range(B):
            out[b] = _bbox_iou_ciou(gt_bboxes[b], pd_bboxes[b])
        return out

    @staticmethod
    def select_candidates_in_gts(xy_centers: torch.Tensor, gt_bboxes: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        # xy_centers: [A,2], gt_bboxes: [B,N,4]
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return deltas.amin(3).gt_(eps)
