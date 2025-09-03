#!/usr/bin/env python3
"""Debug micro-overfit on a single image with live viz and loss curves.

Usage (Aquarium example):
  ./.venv/bin/python tools/debug_overfit.py \
    --images data/aquarium/images/train \
    --ann data/aquarium/train.json \
    --model yolov10s \
    --weights PRETRAINED_COCO \
    --imgsz 320 \
    --steps 200 \
    --device cuda \
    --save-dir runs/debug_overfit/aquarium

This script:
- Picks one train image with at least one annotation
- Runs a tight training loop (micro-overfit) for N steps
- Every K steps saves an annotated visualization and logs loss values
- Plots loss curves to save_dir/loss_curve.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
import matplotlib.pyplot as plt

from leanyolo.models import get_model
from leanyolo.data.coco_simple import CocoDetection
from leanyolo.utils.losses_v10 import detection_loss_v10, _v8_detection_loss, _flatten_feats_to_preds
from leanyolo.utils.tal import make_anchors, dist2bbox, TaskAlignedAssigner, _bbox_iou_ciou
from leanyolo.utils.letterbox import letterbox
from leanyolo.utils.box_ops import unletterbox_coords
from leanyolo.utils.viz import draw_detections


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Micro-overfit YOLOv10 on a single image with live viz")
    p.add_argument("--images", required=True, help="Train images directory (COCO)")
    p.add_argument("--ann", required=True, help="Train annotations JSON (COCO)")
    p.add_argument("--model", default="yolov10s", help="Model variant")
    p.add_argument("--weights", default="PRETRAINED_COCO", help="Weights key or path (PRETRAINED_COCO or ckpt)")
    p.add_argument("--imgsz", type=int, default=320, help="Letterbox size")
    p.add_argument("--device", default="cuda", help="cpu or cuda")
    p.add_argument("--steps", type=int, default=200, help="Number of training steps")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--log-interval", type=int, default=20, help="Viz/log interval (steps)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for debug viz")
    p.add_argument("--save-dir", default="runs/debug_overfit/exp", help="Output directory")
    p.add_argument("--one2one-only", action="store_true", help="Train only on one2one branch (topk=1)")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--head-reset", action="store_true")
    p.add_argument("--pick-stem", default=None, help="Optional image stem to pick (e.g., IMG_2464_jpeg_jpg)")
    p.add_argument("--min-classes", type=int, default=1, help="Pick an image with at least this many unique GT classes")
    return p.parse_args()


def load_class_names(ann_json: str) -> List[str]:
    with open(ann_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = sorted(data.get("categories", []), key=lambda c: c.get("id", 0))
    return [c.get("name", str(i)) for i, c in enumerate(cats)]


def pick_one_index(ds: CocoDetection, *, pick_stem: str | None, min_classes: int) -> int:
    if pick_stem:
        for i in range(len(ds)):
            img_id = ds.ids[i]
            fn = ds.images[img_id]["file_name"]
            if Path(fn).stem == pick_stem:
                return i
    for i in range(len(ds)):
        _, tgt = ds[i]
        if tgt["boxes"].shape[0] == 0:
            continue
        uniq = len(torch.unique(tgt["labels"]))
        if uniq >= max(1, min_classes):
            return i
    return 0


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    save_dir = Path(args.save_dir)
    (save_dir / "viz").mkdir(parents=True, exist_ok=True)

    # Dataset (no augment for determinism)
    ds = CocoDetection(args.images, args.ann, imgsz=args.imgsz, augment=False)
    idx = pick_one_index(ds, pick_stem=args.pick_stem, min_classes=args.min_classes)
    print(f"[debug] Using sample index: {idx} (min_classes={args.min_classes}, pick_stem={args.pick_stem})")

    # Class names and model
    class_names = load_class_names(args.ann)
    model = get_model(
        args.model,
        weights=(None if args.weights.lower() == "none" else args.weights),
        class_names=class_names,
    ).to(device).train()
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        for p in model.neck.parameters():
            p.requires_grad_(False)
    if args.head_reset:
        for m in model.head.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # Loss curves
    steps, totals, clss, regs, det_counts, max_scores = [], [], [], [], [], []

    # Prepare original image path for viz
    # Rebuild filename from dataset internals
    img_id = ds.ids[idx]
    file_name = ds.images[img_id]["file_name"]
    img_path = Path(args.images) / file_name

    # Training loop
    for step in range(1, args.steps + 1):
        x, tgt = ds[idx]
        x = x.unsqueeze(0).to(device)
        targets = [{"boxes": tgt["boxes"].to(device), "labels": tgt["labels"].to(device)}]

        raw = model(x)
        if isinstance(raw, dict) and args.one2one_only:
            # Train only one2one branch with topk=1
            loss_dict = _v8_detection_loss(
                raw["one2one"], targets, num_classes=len(class_names), reg_max=16, strides=(8, 16, 32), tal_topk=1
            )
        else:
            loss_dict = detection_loss_v10(raw, targets, num_classes=len(class_names))
        loss = loss_dict["total"]
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Log
        steps.append(step)
        totals.append(float(loss_dict["total"].item()))
        clss.append(float(loss_dict["cls"].item()))
        regs.append(float(loss_dict["reg"].item()))

        if step % max(1, args.log_interval) == 0 or step == 1 or step == args.steps:
            # Decode on original image for visualization
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            lb_img, gain, pad = letterbox(rgb, new_shape=args.imgsz)
            xx = torch.from_numpy(lb_img).permute(2, 0, 1).float().unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                model.post_conf_thresh = args.conf
                raw_eval = model(xx)
                dets = model.decode_forward(raw_eval)[0][0]
            model.train()
            # Assigner diagnostics (one2many/one2one)
            try:
                def branch_stats(feats: List[torch.Tensor]) -> Tuple[int, float, float]:
                    # Flatten preds
                    reg_max = 16
                    nc = len(class_names)
                    B = feats[0].shape[0]
                    pd, ps, feats_cat = _flatten_feats_to_preds(feats, nc, reg_max)
                    # Decode expected distances
                    probs = pd.view(B, -1, 4, reg_max).softmax(3)
                    proj = torch.arange(reg_max, dtype=probs.dtype, device=probs.device)
                    exp_ltrb = torch.matmul(probs, proj)  # [B,A,4]
                    anchors, stride_t = make_anchors(feats_cat, (8, 16, 32))
                    pb = dist2bbox(exp_ltrb, anchors[None, ...], xywh=False)
                    # Build GT pack
                    gt_b = targets[0]["boxes"][None, ...]
                    gt_l = targets[0]["labels"][None, :, None]
                    mask = torch.ones_like(gt_l, dtype=torch.bool)
                    assign = TaskAlignedAssigner(topk=10, num_classes=nc, alpha=0.5, beta=6.0)
                    tl, tb, ts, fg, _ = assign(ps, pb * stride_t[None, ...], anchors * stride_t, gt_l, gt_b, mask)
                    pos = fg[0]
                    num_pos = int(pos.sum().item())
                    mean_iou = 0.0
                    mean_cls = 0.0
                    if num_pos > 0:
                        iou = _bbox_iou_ciou(pb[0][pos] * stride_t[pos], (tb[0][pos]))
                        mean_iou = float(iou.diag().mean().item()) if iou.numel() else 0.0
                        # Gather predicted class prob for assigned label
                        labels_pos = tl[0][pos]
                        cls_logits = ps[0][pos]
                        cls_prob = cls_logits.sigmoid().gather(1, labels_pos.unsqueeze(-1)).mean()
                        mean_cls = float(cls_prob.item())
                    return num_pos, mean_iou, mean_cls

                if isinstance(raw, dict):
                    npos_m, miou_m, mcls_m = branch_stats(raw["one2many"])  # type: ignore[arg-type]
                    npos_o, miou_o, mcls_o = branch_stats(raw["one2one"])  # type: ignore[arg-type]
                    print(
                        f"[assign] one2many: pos={npos_m}, miou={miou_m:.3f}, mcls={mcls_m:.3f} | "
                        f"one2one: pos={npos_o}, miou={miou_o:.3f}, mcls={mcls_o:.3f}"
                    )
                else:
                    npos, miou, mcls = branch_stats(raw)  # type: ignore[arg-type]
                    print(f"[assign] pos={npos}, miou={miou:.3f}, mcls={mcls:.3f}")
            except Exception as e:
                print("[assign] diagnostics error:", e)
            if dets.numel() > 0:
                dets[:, :4] = unletterbox_coords(dets[:, :4], gain=gain, pad=pad, to_shape=rgb.shape[:2])
                det_counts.append(int(dets.shape[0]))
                max_scores.append(float(dets[:, 4].max().item()))
            else:
                det_counts.append(0)
                max_scores.append(0.0)
            vis = draw_detections(bgr, dets, class_names=class_names)
            out_path = save_dir / "viz" / f"step_{step:04d}.jpg"
            cv2.imwrite(str(out_path), vis)
            print(
                f"[step {step}/{args.steps}] total={totals[-1]:.4f} cls={clss[-1]:.4f} reg={regs[-1]:.4f} "
                f"dets={det_counts[-1]} max_score={max_scores[-1]:.3f}"
            )

    # Save checkpoint and quick reload sanity
    ckpt = {
        "leanyolo_version": "0.1",
        "model_name": args.model,
        "class_names": class_names,
        "input_norm_subtract": [0.0, 0.0, 0.0],
        "input_norm_divide": [255.0, 255.0, 255.0],
        "state_dict": model.state_dict(),
    }
    ckpt_path = save_dir / "ckpt.pt"
    torch.save(ckpt, str(ckpt_path))
    print(f"[debug] Saved ckpt: {ckpt_path}")

    # Plot curves
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(steps, totals, label="total")
        plt.plot(steps, clss, label="cls")
        plt.plot(steps, regs, label="reg")
        if det_counts:
            ax2 = plt.twinx()
            ax2.plot(steps[: len(det_counts)], det_counts, "g--", label="dets")
            ax2.set_ylabel("#dets")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Micro-overfit debug")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(save_dir / "loss_curve.png")
        print(f"[debug] Saved plot: {save_dir/'loss_curve.png'}")
    except Exception as e:
        print("[warn] Could not save plot:", e)


if __name__ == "__main__":
    main()
