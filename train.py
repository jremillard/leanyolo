#!/usr/bin/env python3
from __future__ import annotations

"""
Minimal training script (baseline) and example usage.

This file is intentionally compact and explicit so you can copy/paste and
adapt it to your needs. It shows how to:
- Build a YOLOv10 model via `get_model`
- Create COCO-style datasets and a DataLoader
- Train with a simple loss and evaluate mAP each epoch
- Save per-epoch checkpoints you can later load
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from leanyolo.models import get_model
from leanyolo.data.coco_simple import CocoDetection, coco_collate
from leanyolo.utils.losses_v10 import detection_loss_v10


def load_class_names_from_coco(ann_path: str) -> List[str]:
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = sorted(data.get("categories", []), key=lambda c: c["id"])
    return [c["name"] for c in cats]


def evaluate_coco(
    model: torch.nn.Module,
    images_dir: str,
    ann_json: str,
    *,
    imgsz: int,
    device: str,
    conf: float = 0.25,
    iou: float = 0.65,
    progress: bool = False,
    log_every: int = 20,
) -> Dict[str, float]:
    """Compute COCO-style mAP for a trained model.

    Returns a dict with keys mAP50-95, mAP50, mAP75. Kept small on purpose so
    you can transplant it into your own training scripts.
    """
    import cv2
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from leanyolo.utils.letterbox import letterbox
    from leanyolo.utils.box_ops import unletterbox_coords

    model.eval()
    model.post_conf_thresh = conf
    model.post_iou_thresh = iou
    device_t = torch.device(device)
    model.to(device_t)

    coco = COCO(str(ann_json))
    # Map prediction class index -> dataset category id by sorted id order
    cats = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c["id"])  # type: ignore
    cat_ids = [c["id"] for c in cats]

    # List image paths by ids in annotations
    imgs_info = coco.loadImgs(coco.getImgIds())  # type: ignore
    img_paths = [Path(images_dir) / img["file_name"] for img in imgs_info]
    fname_to_id = {img["file_name"]: int(img["id"]) for img in imgs_info}

    results = []
    total = len(img_paths)
    if progress:
        print(f"[eval] Running inference on {total} images (conf={conf}, iou={iou})...")
    for idx, p in enumerate(img_paths, 1):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_shape = img.shape[:2]
        lb_img, gain, pad = letterbox(img, new_shape=imgsz)
        x = torch.from_numpy(lb_img).to(device_t).permute(2, 0, 1).float().unsqueeze(0)
        with torch.no_grad():
            raw = model(x)
            dets = model.decode_forward(raw)[0][0]
        if dets.numel() == 0:
            continue
        dets[:, :4] = unletterbox_coords(dets[:, :4], gain=gain, pad=pad, to_shape=orig_shape)
        image_id = int(fname_to_id.get(Path(p).name, -1))
        if image_id == -1:
            stem = Path(p).stem
            for fn, iid in fname_to_id.items():
                if Path(fn).stem == stem:
                    image_id = int(iid)
                    break
            if image_id == -1:
                continue
        for x1, y1, x2, y2, score, cls in dets.cpu().numpy():
            w, h = x2 - x1, y2 - y1
            cls = int(cls)
            cat_id = cat_ids[cls] if cls < len(cat_ids) else cat_ids[-1]
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(cat_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                }
            )
        if progress and (idx % max(1, log_every) == 0 or idx == total):
            print(f"[eval] {idx}/{total} images processed; results so far: {len(results)}")

    if not results:
        return {"mAP50-95": 0.0, "mAP50": 0.0, "mAP75": 0.0}

    coco_dt = coco.loadRes(results)
    coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {
        "mAP50-95": float(coco_eval.stats[0]),
        "mAP50": float(coco_eval.stats[1]),
        "mAP75": float(coco_eval.stats[2]),
    }


def parse_args() -> argparse.Namespace:
    """CLI for basic training. See README for dataset layout."""
    p = argparse.ArgumentParser(description="Train YOLOv10 on COCO-format dataset (basic trainer)")
    # Data
    p.add_argument("--train-images", required=True, help="Path to training images directory")
    p.add_argument("--train-ann", required=True, help="Path to training annotations JSON (COCO)")
    p.add_argument("--val-images", default=None, help="Optional: path to validation images directory (skip eval if unset)")
    p.add_argument("--val-ann", default=None, help="Optional: path to validation annotations JSON (skip eval if unset)")
    p.add_argument("--imgsz", type=int, default=640, help="Letterbox size")
    # Model
    p.add_argument("--model", default="yolov10n", choices=[
        "yolov10n","yolov10s","yolov10m","yolov10b","yolov10l","yolov10x"
    ])
    p.add_argument("--weights", default="PRETRAINED_COCO", help="PRETRAINED_COCO, None, or checkpoint filename")
    # Train
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2, help="Dataloader workers (set 0 if hangs)")
    p.add_argument("--no-tqdm", action="store_true", help="Disable tqdm bars; print every --log-interval steps")
    p.add_argument("--log-interval", type=int, default=20, help="Step interval for logging when --no-tqdm")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--head-reset", action="store_true")
    p.add_argument("--save-dir", default="runs/train/exp")
    # Eval
    p.add_argument("--eval-conf", type=float, default=0.25, help="Eval confidence threshold (higher = faster)")
    p.add_argument("--eval-iou", type=float, default=0.65, help="Eval IoU threshold for NMS")
    p.add_argument("--eval-progress", action="store_true", help="Print progress during evaluation")
    return p.parse_args()


def main() -> None:
    """Train a YOLOv10 model on a COCO-style dataset (baseline)."""
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Dataset & loaders
    train_ds = CocoDetection(args.train_images, args.train_ann, imgsz=args.imgsz, augment=True)
    has_val = bool(args.val_images) and bool(args.val_ann)
    val_ds = CocoDetection(args.val_images, args.val_ann, imgsz=args.imgsz, augment=False) if has_val else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=coco_collate,
    )

    # Classes
    class_names = load_class_names_from_coco(args.train_ann)

    # Model (defaults normalize by /255 inside)
    weights_arg: str | None
    if args.weights.lower() == "none":
        weights_arg = None
    else:
        weights_arg = args.weights
    print(f"[train] Device: {device}, model={args.model}, imgsz={args.imgsz}")
    print(f"[train] Weights: {('None' if (args.weights.lower()=='none') else args.weights)}")
    print(f"[train] Classes (nc={len(class_names)}): {class_names}")
    val_count = len(val_ds) if val_ds is not None else 0
    print(f"[train] Train images: {len(train_ds)}, Val images: {val_count}, Batches/Epoch: {len(train_loader)}")

    model = get_model(
        args.model,
        weights=weights_arg,
        class_names=class_names,
    ).to(device)

    # Transfer learning options
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

    # Optimizer & scheduler
    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = CosineAnnealingLR(optim, T_max=args.epochs)

    # Train loop
    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        running = {"total": 0.0, "cls": 0.0, "reg": 0.0}
        nb = 0
        if args.no_tqdm:
            for i, (x, targets) in enumerate(train_loader, 1):
                x = x.to(device, non_blocking=True)
                targets_dev = [
                    {"boxes": t["boxes"].to(device, non_blocking=True), "labels": t["labels"].to(device, non_blocking=True)}
                    for t in targets
                ]
                raw = model(x)
                loss_dict = detection_loss_v10(raw, targets_dev, num_classes=len(class_names))
                loss = loss_dict["total"]
                optim.zero_grad()
                loss.backward()
                optim.step()
                nb += 1
                for k in running.keys():
                    running[k] += float(loss_dict[k].item())
                if i % max(1, args.log_interval) == 0:
                    avg = {k: running[k] / nb for k in running.keys()}
                    print(f"[epoch {epoch+1}/{args.epochs}] step {i}/{len(train_loader)} total={avg['total']:.4f} cls={avg['cls']:.4f} reg={avg['reg']:.4f}", flush=True)
        else:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)
            for x, targets in pbar:
                x = x.to(device, non_blocking=True)
                targets_dev = [
                    {"boxes": t["boxes"].to(device, non_blocking=True), "labels": t["labels"].to(device, non_blocking=True)}
                    for t in targets
                ]
                raw = model(x)
                loss_dict = detection_loss_v10(raw, targets_dev, num_classes=len(class_names))
                loss = loss_dict["total"]
                optim.zero_grad()
                loss.backward()
                optim.step()
                nb += 1
                for k in running.keys():
                    running[k] += float(loss_dict[k].item())
                pbar.set_postfix({k: f"{running[k]/nb:.4f}" for k in running.keys()})
        sched.step()

        # Full mAP evaluation on val (if provided)
        if has_val:
            model.eval()
            with torch.no_grad():
                stats = evaluate_coco(
                    model,
                    args.val_images,
                    args.val_ann,
                    imgsz=args.imgsz,
                    device=args.device,
                    conf=args.eval_conf,
                    iou=args.eval_iou,
                    progress=args.eval_progress,
                    log_every=20,
                )
            print({k: round(v, 5) for k, v in stats.items()})
        else:
            print("[eval] Skipping validation (no --val-images/--val-ann provided)")

        # Save checkpoint each epoch
        ckpt = {
            "leanyolo_version": "0.1",
            "model_name": args.model,
            "class_names": class_names,
            "input_norm_subtract": [0.0, 0.0, 0.0],
            "input_norm_divide": [255.0, 255.0, 255.0],
            "state_dict": model.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"epoch{epoch+1:03d}.pt"))

    # Save final
    torch.save(ckpt, os.path.join(args.save_dir, "ckpt.pt"))


if __name__ == "__main__":
    main()
