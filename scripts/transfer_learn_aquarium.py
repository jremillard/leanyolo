#!/usr/bin/env python3
from __future__ import annotations

"""
Transfer learning example: fine‑tune YOLOv10 on the Aquarium dataset.

This file was previously named `transfer_aquarium.py`. It has been renamed
to emphasize its role as a generic transfer‑learning example you can copy for
other datasets.
"""

# NOTE: this file mirrors scripts/transfer_aquarium.py content at rename time.
# If you enhance one, mirror changes in the other history if needed.

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Ensure repository root is on sys.path when running as a script from scripts/
sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from leanyolo.models import get_model
from leanyolo.data.coco_simple import CocoDetection, coco_collate
from leanyolo.utils.losses_v10 import detection_loss_v10
from leanyolo.utils.viz import draw_detections
from leanyolo.utils.box_ops import unletterbox_coords
from leanyolo.utils.letterbox import letterbox


def seed_everything(seed: int = 42) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def setup_logger(save_dir: Path, filename: str = "train.log") -> logging.Logger:
    logger = logging.getLogger("transfer")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    save_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(save_dir / filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def load_class_names_from_coco(ann_path: str | Path) -> List[str]:
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = sorted(data.get("categories", []), key=lambda c: c.get("id", 0))
    return [c.get("name", str(i)) for i, c in enumerate(cats)]


@torch.no_grad()
def evaluate_coco(
    model: torch.nn.Module,
    images_dir: str | Path,
    ann_json: str | Path,
    *,
    imgsz: int,
    device: str,
    conf: float = 0.25,
    iou: float = 0.65,
    max_images: int | None = None,
    progress: bool = False,
    log_every: int = 20,
) -> Dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    model.eval()
    model.post_conf_thresh = conf
    model.post_iou_thresh = iou
    device_t = torch.device(device)
    model.to(device_t)
    coco = COCO(str(ann_json))
    cats = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c["id"])  # type: ignore
    cat_ids = [c["id"] for c in cats]
    imgs_info = coco.loadImgs(coco.getImgIds())  # type: ignore
    img_paths = [Path(images_dir) / img["file_name"] for img in imgs_info]
    if max_images is not None:
        img_paths = img_paths[: max(0, int(max_images))]
    fname_to_id = {img["file_name"]: int(img["id"]) for img in imgs_info}
    results = []
    total = len(img_paths)
    for idx, p in enumerate(img_paths, 1):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_shape = img.shape[:2]
        lb_img, gain, pad = letterbox(img, new_shape=imgsz)
        x = torch.from_numpy(lb_img).to(device_t).permute(2, 0, 1).float().unsqueeze(0)
        raw = model(x)
        dets = model.decode_forward(raw)[0][0]
        if dets.numel() == 0:
            if progress and (idx % max(1, log_every) == 0 or idx == total):
                print(f"[eval] {idx}/{total} images processed; results so far: {len(results)}")
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
    return {"mAP50-95": float(coco_eval.stats[0]), "mAP50": float(coco_eval.stats[1]), "mAP75": float(coco_eval.stats[2])}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transfer learn YOLOv10 on Aquarium (COCO‑style)")
    p.add_argument("--root", default="data/aquarium", help="Aquarium dataset root containing images/ and *.json")
    p.add_argument("--train-images", default=None)
    p.add_argument("--train-ann", default=None)
    p.add_argument("--val-images", default=None)
    p.add_argument("--val-ann", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--model", default="yolov10s", choices=["yolov10n","yolov10s","yolov10m","yolov10b","yolov10l","yolov10x"])
    p.add_argument("--weights", default="PRETRAINED_COCO")
    p.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true")
    p.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    p.add_argument("--head-reset", dest="head_reset", action="store_true")
    p.add_argument("--no-head-reset", dest="head_reset", action="store_false")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-conf", type=float, default=0.25)
    p.add_argument("--eval-iou", type=float, default=0.65)
    p.add_argument("--viz-interval", type=int, default=200)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-train-size", type=int, default=64)
    p.add_argument("--debug-val-size", type=int, default=64)
    p.add_argument("--debug-eval-every", type=int, default=1)
    p.add_argument("--save-dir", default="runs/transfer/aquarium")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-tqdm", action="store_true")
    p.set_defaults(freeze_backbone=True, head_reset=True)
    return p.parse_args()


def resolve_dataset_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path | None, Path | None]:
    root = Path(args.root)
    train_images = Path(args.train_images) if args.train_images else root / "images" / "train"
    train_ann = Path(args.train_ann) if args.train_ann else root / "train.json"
    val_images = Path(args.val_images) if args.val_images else root / "images" / "val"
    val_ann = Path(args.val_ann) if args.val_ann else root / "val.json"
    if not train_images.exists() or not train_ann.exists():
        raise FileNotFoundError(
            f"Could not find training data. Expected images at '{train_images}' and annotations at '{train_ann}'.\n"
            "Run scripts/prepare_acquirium.py or provide --train-images/--train-ann explicitly."
        )
    if not val_images.exists() or not val_ann.exists():
        val_images = None  # type: ignore
        val_ann = None     # type: ignore
    return train_images, train_ann, val_images, val_ann


def build_dataloaders(
    train_images: Path,
    train_ann: Path,
    val_images: Path | None,
    val_ann: Path | None,
    *,
    imgsz: int,
    batch_size: int,
    workers: int,
    debug: bool,
    debug_train_size: int,
    debug_val_size: int,
) -> Tuple[DataLoader, CocoDetection, DataLoader | None, CocoDetection | None]:
    train_ds = CocoDetection(train_images, train_ann, imgsz=imgsz, augment=True)
    if debug:
        n = min(len(train_ds), max(1, int(debug_train_size)))
        train_ds = Subset(train_ds, list(range(n)))  # type: ignore
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=coco_collate)
    if val_images is None or val_ann is None:
        return train_loader, train_ds, None, None
    val_ds = CocoDetection(val_images, val_ann, imgsz=imgsz, augment=False)
    if debug:
        n = min(len(val_ds), max(1, int(debug_val_size)))
        val_ds = Subset(val_ds, list(range(n)))  # type: ignore
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=coco_collate)
    return train_loader, train_ds, val_loader, val_ds


def maybe_freeze_and_reset(model: torch.nn.Module, *, freeze_backbone: bool, head_reset: bool) -> None:
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        for p in model.neck.parameters():
            p.requires_grad_(False)
    if head_reset:
        for m in model.head.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


def save_train_viz(*, save_dir: Path, step: int, imgs: torch.Tensor, dets_batched: List[torch.Tensor], class_names: List[str]) -> Path | None:
    viz_dir = save_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    if imgs.ndim != 4 or imgs.shape[0] == 0:
        return None
    x0 = imgs[0].detach().cpu().float()
    img_hwc = x0.permute(1, 2, 0).numpy()
    bgr = cv2.cvtColor(img_hwc.astype("uint8"), cv2.COLOR_RGB2BGR)
    dets = dets_batched[0] if dets_batched else torch.zeros((0, 6))
    vis = draw_detections(bgr, dets, class_names=class_names)
    out_path = viz_dir / f"step_{step:06d}.jpg"
    ok = cv2.imwrite(str(out_path), vis)
    return out_path if ok else None


def main() -> None:
    args = parse_args()
    start_wall = time.time()
    start_iso = datetime.now().isoformat(timespec="seconds")
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_images, train_ann, val_images, val_ann = resolve_dataset_paths(args)

    save_dir = Path(args.save_dir)
    logger = setup_logger(save_dir)
    logger.info(f"RUN START | {start_iso}")
    logger.info("CLI: " + " ".join(map(str, sys.argv)))
    try:
        logger.info("ARGS: " + json.dumps(vars(args), sort_keys=True))
    except Exception:
        logger.info("ARGS: " + repr(args))

    train_loader, train_ds, val_loader, val_ds = build_dataloaders(
        train_images, train_ann, val_images, val_ann,
        imgsz=args.imgsz, batch_size=args.batch_size, workers=args.workers,
        debug=args.debug, debug_train_size=args.debug_train_size, debug_val_size=args.debug_val_size,
    )

    class_names = load_class_names_from_coco(train_ann)
    weights_arg = None if (isinstance(args.weights, str) and args.weights.lower() in {"none", "null", ""}) else args.weights
    logger.info(f"Device={device} | model={args.model} | imgsz={args.imgsz} | weights={weights_arg or 'None'} | nc={len(class_names)}")
    try:
        n_train = len(train_ds)
    except Exception:
        n_train = -1
    try:
        n_val = len(val_ds) if val_ds is not None else 0
    except Exception:
        n_val = -1
    logger.info(f"DATA: train_images={n_train} | val_images={n_val} | batches/epoch={len(train_loader)}")

    model = get_model(
        args.model,
        weights=weights_arg,
        class_names=class_names,
        input_norm_subtract=[0.0, 0.0, 0.0],
        input_norm_divide=[255.0, 255.0, 255.0],
    ).to(device)

    maybe_freeze_and_reset(model, freeze_backbone=args.freeze_backbone, head_reset=args.head_reset)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"TRANSFER: freeze_backbone={args.freeze_backbone} | head_reset={args.head_reset} | params={num_params:,} | trainable={num_trainable:,}")

    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    steps_seen = 0
    for epoch in range(args.epochs):
        model.train()
        running = {"total": 0.0, "cls": 0.0, "reg": 0.0}
        nb = 0
        epoch_t0 = time.perf_counter()
        iterator = enumerate(train_loader if args.no_tqdm else tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True), 1)
        for i, (x, targets) in iterator:
            x = x.to(device, non_blocking=True)
            targets_dev = [{"boxes": t["boxes"].to(device, non_blocking=True), "labels": t["labels"].to(device, non_blocking=True)} for t in targets]
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                raw = model(x)
                loss_dict = detection_loss_v10(raw, targets_dev, num_classes=len(class_names))
                loss = loss_dict["total"]
            optim.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()
            nb += 1
            steps_seen += 1
            for k in running.keys():
                running[k] += float(loss_dict[k].item())
            if args.no_tqdm or (not args.no_tqdm and i % 10 == 0):
                avg = {k: running[k] / max(1, nb) for k in running.keys()}
                logger.info(f"[epoch {epoch+1}/{args.epochs}] step {i}/{len(train_loader)} total={avg['total']:.4f} cls={avg['cls']:.4f} reg={avg['reg']:.4f}")
            if args.viz_interval and (steps_seen % max(1, args.viz_interval) == 0 or (args.debug and steps_seen <= 5)):
                model.eval()
                with torch.no_grad():
                    model.post_conf_thresh = args.eval_conf
                    model.post_iou_thresh = args.eval_iou
                    raw_eval = model(x[:1])
                    dets_list = model.decode_forward(raw_eval)[0]
                model.train()
                saved = save_train_viz(save_dir=save_dir, step=steps_seen, imgs=x[:1].detach().cpu(), dets_batched=dets_list, class_names=class_names)
                if saved is not None:
                    logger.info(f"[viz] saved: {saved}")
        sched.step()
        avg_epoch = {k: running[k] / max(1, nb) for k in running.keys()}
        lr0 = optim.param_groups[0].get("lr", args.lr)
        epoch_dt = time.perf_counter() - epoch_t0
        logger.info(f"EPOCH {epoch+1}/{args.epochs} | loss_total={avg_epoch['total']:.4f} loss_cls={avg_epoch['cls']:.4f} loss_reg={avg_epoch['reg']:.4f} | lr={lr0:.6f} | steps={nb} | dur={epoch_dt:.2f}s")
        do_eval = val_images is not None and val_ann is not None
        if do_eval and (not args.debug or ((epoch + 1) % max(1, args.debug_eval_every) == 0)):
            try:
                stats = evaluate_coco(model, images_dir=val_images or "", ann_json=val_ann or "", imgsz=args.imgsz, device=args.device, conf=args.eval_conf, iou=args.eval_iou, max_images=(args.debug_val_size if args.debug else None), progress=False)
                logger.info("[val] " + ", ".join(f"{k}={v:.5f}" for k, v in stats.items()))
            except Exception as e:
                logger.info(f"[val] evaluation failed: {e}")
        else:
            logger.info("[val] Skipping validation (no data or deferred in debug mode)")
        try:
            ckpt = {"leanyolo_version": "0.1", "model_name": args.model, "class_names": class_names, "input_norm_subtract": [0.0, 0.0, 0.0], "input_norm_divide": [255.0, 255.0, 255.0], "state_dict": model.state_dict(), "epoch": epoch + 1}
            torch.save(ckpt, os.path.join(save_dir, f"epoch{epoch+1:03d}.pt"))
        except Exception as e:
            logger.info(f"[ckpt] save failed: {e}")

    try:
        torch.save({"leanyolo_version": "0.1", "model_name": args.model, "class_names": class_names, "input_norm_subtract": [0.0, 0.0, 0.0], "input_norm_divide": [255.0, 255.0, 255.0], "state_dict": model.state_dict(), "epoch": args.epochs}, os.path.join(save_dir, "ckpt.pt"))
    except Exception as e:
        logger.info(f"[ckpt] final save failed: {e}")
    end_iso = datetime.now().isoformat(timespec="seconds")
    total_sec = time.time() - start_wall
    logger.info(f"RUN END | {end_iso} | duration={total_sec:.2f}s | save_dir={save_dir}")


if __name__ == "__main__":
    main()

