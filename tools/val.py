#!/usr/bin/env python3
"""
COCO validation script and example pipeline.

This file doubles as copy‑and‑paste friendly sample code for using leanyolo:
- Loads a YOLOv10 model via `get_model` (no YAML)
- Preprocesses with letterbox, runs forward + decode, and evaluates with COCOeval
- Can save JSON detections and visualization images

Design goal: be easy to read and modify for your own datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import sys
from pathlib import Path as _Path

# Ensure repo root on path for 'leanyolo' imports when run from any CWD
_repo_root = _Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from leanyolo.data.coco import ensure_coco_val, load_coco_categories, list_images, coco80_class_names
from leanyolo.models import get_model
from leanyolo.utils.box_ops import unletterbox_coords
from leanyolo.utils.letterbox import letterbox
from leanyolo.utils.val_log import (
    COLUMNS as VAL_COLUMNS,
    append_row as append_val_row,
    collect_env_info,
    now_iso,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Lean YOLOv10 COCO validation")
    ap.add_argument("--data-root", default="data/coco", help="COCO root directory (or ignored if --images/--ann provided)")
    ap.add_argument("--download", action="store_true", help="Download COCO val2017 if missing")
    ap.add_argument("--model", default="yolov10s", help="Model name")
    ap.add_argument("--weights", default="PRETRAINED_COCO", help="Weights key or None for random init")
    ap.add_argument("--images", default=None, help="Optional: explicit images directory (COCO)")
    ap.add_argument("--ann", default=None, help="Optional: explicit annotations JSON (COCO)")
    ap.add_argument("--imgsz", type=int, default=640, help="Image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.65, help="IoU threshold")
    ap.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    ap.add_argument("--max-images", type=int, default=None, help="Validate on first N images")
    ap.add_argument("--save-json", default=None, help="Optional: path to save detections JSON")
    ap.add_argument("--save-viz-dir", default=None, help="Optional: directory to save annotated images")
    ap.add_argument(
        "--viz-name",
        choices=["file", "id", "prefix"],
        default="file",
        help="How to name saved viz images: original file name, COCO image id, or prefix before '.rf.'",
    )
    # Logging and runtime flags
    ap.add_argument("--log-csv", default="runs/val/val_runs.csv", help="CSV file to append validation results")
    ap.add_argument("--run-id", default=None, help="Optional run id; defaults to ISO timestamp")
    ap.add_argument(
        "--runtime",
        default="torch",
        choices=["torch", "onnxrt", "tensorrt", "torchscript"],
        help="Inference runtime (for logging)",
    )
    ap.add_argument("--precision", default="fp32", help="Precision label for logging (fp32/fp16/int8)")
    ap.add_argument("--notes", default="", help="Freeform notes for this run")
    # Removed: --latency-iters (FPS sampling now uses a fixed internal iteration count)
    ap.add_argument("--warmup-iters", type=int, default=5, help="Warmup iterations before FPS measurement")
    # Optional: skip second model build for perf sampling (useful for custom class counts)
    ap.add_argument("--skip-perf", action="store_true", help="Skip throughput measurement to avoid rebuilding model")
    return ap.parse_args()


@torch.no_grad()
def validate_coco(
    *,
    model_name: str = "yolov10s",
    weights: str | None = "PRETRAINED_COCO",
    data_root: str = "data/coco",
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.65,
    device: str = "cpu",
    max_images: int | None = None,
    save_json: str | None = None,
    images_dir: str | None = None,
    ann_json: str | None = None,
    save_viz_dir: str | None = None,
    viz_name: str = "file",
) -> Dict[str, float]:
    """Run COCO mAP on a folder + annotations.

    Arguments mirror the CLI. Returns a dict with mAP metrics. The function is
    intentionally small and explicit so you can copy it into your own projects
    and adapt as needed.
    """
    device_t = torch.device(device)
    images_dir_p, ann_json_p, img_paths = _resolve_dataset_paths(
        data_root=data_root, images_dir=images_dir, ann_json=ann_json, max_images=max_images
    )

    coco = COCO(str(ann_json_p))
    cat_ids = load_coco_categories(ann_json_p)

    # Determine class names: use COCO-80 by default, or derive from provided ann_json
    if ann_json_p is not None and Path(ann_json_p).exists():
        with open(ann_json_p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cats = sorted(data.get('categories', []), key=lambda c: c.get('id', 0))
        cn = [c.get('name', str(i)) for i, c in enumerate(cats)]
    else:
        cn = coco80_class_names()
    model = get_model(
        model_name,
        weights=weights,
        class_names=cn,
        input_norm_subtract=[0.0, 0.0, 0.0],
        input_norm_divide=[255.0, 255.0, 255.0],
    )
    model.to(device_t).eval()
    model.post_conf_thresh = conf
    model.post_iou_thresh = iou

    # Build mapping filename -> image_id for robust id assignment
    imgs_info = coco.loadImgs(coco.getImgIds())  # type: ignore
    fname_to_id = {img["file_name"]: int(img["id"]) for img in imgs_info}

    results = []
    # Prepare viz dir
    if save_viz_dir:
        Path(save_viz_dir).mkdir(parents=True, exist_ok=True)

    for p in img_paths:
        img = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        orig_shape = img.shape[:2]
        lb_img, gain, pad = letterbox(img, new_shape=imgsz)
        x = torch.from_numpy(lb_img).to(device_t).permute(2, 0, 1).float().unsqueeze(0)

        raw = model(x)
        dets = model.decode_forward(raw)[0][0]
        # Scale boxes back if present
        if dets.numel() > 0:
            dets[:, :4] = unletterbox_coords(dets[:, :4], gain=gain, pad=pad, to_shape=orig_shape)
        # Optional visualization save
        if save_viz_dir:
            from leanyolo.utils.viz import draw_detections
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            vis = draw_detections(bgr, dets, class_names=cn)
            # Choose output filename format
            if viz_name == "id":
                out_name = f"{image_id}.jpg"
            elif viz_name == "prefix":
                fname = Path(p).name
                out_name = (fname.split(".rf.")[0] + ".jpg") if ".rf." in fname else fname
            else:
                out_name = Path(p).name
            out_path = Path(save_viz_dir) / out_name
            ok = cv2.imwrite(str(out_path), vis)
            if not ok:
                print(f"[viz] Failed to write: {out_path}")
            else:
                print(f"[viz] Saved: {out_path} ({int(dets.shape[0])} dets)")
        # Convert to COCO json
        # COCO expects [x, y, w, h] with category_id being dataset category IDs
        image_id = int(fname_to_id.get(Path(p).name, -1))
        if image_id == -1:
            # Fallback: try stem lookup without extension
            stem = Path(p).stem
            # Try to find first match by stem
            for fn, iid in fname_to_id.items():
                if Path(fn).stem == stem:
                    image_id = int(iid)
                    break
            if image_id == -1:
                continue
        if dets.numel() > 0:
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

    if not results:
        return {"mAP50-95": 0.0}

    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(save_json).write_text(json.dumps(results))

    coco_dt = coco.loadRes(results)
    coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract mAP .5:.95
    stats = {
        "mAP50-95": float(coco_eval.stats[0]),
        "mAP50": float(coco_eval.stats[1]),
        "mAP75": float(coco_eval.stats[2]),
    }
    return stats


def _resolve_dataset_paths(
    *, data_root: str, images_dir: str | None, ann_json: str | None, max_images: int | None
) -> Tuple[Path, Path, List[Path]]:
    root = Path(data_root)
    if images_dir is None or ann_json is None:
        subset_ann = root / "annotations.json"
        subset_imgs = root / "images"
        if subset_ann.exists() and subset_imgs.exists():
            images_dir_p, ann_json_p = subset_imgs, subset_ann
        else:
            images_dir_p, ann_json_p = ensure_coco_val(str(root), download=False)
    else:
        images_dir_p, ann_json_p = Path(images_dir), Path(ann_json)

    img_paths = list_images(images_dir_p)
    if max_images is not None:
        img_paths = img_paths[:max_images]
    return images_dir_p, ann_json_p, img_paths


@torch.no_grad()
def _measure_latency(
    model: torch.nn.Module,
    sample_image: Path,
    *,
    device: str,
    imgsz: int,
    iters: int,
    warmup: int,
) -> Dict[str, float]:
    import time
    import cv2

    if iters <= 0:
        return {"throughput_fps": 0.0}

    device_t = torch.device(device)
    img = cv2.cvtColor(cv2.imread(str(sample_image), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    lb_img, _, _ = letterbox(img, new_shape=imgsz)
    x = torch.from_numpy(lb_img).to(device_t).permute(2, 0, 1).float().unsqueeze(0)

    # Warmup
    for _ in range(max(0, int(warmup))):
        _ = model.decode_forward(model(x))
    if device_t.type == "cuda":
        torch.cuda.synchronize()

    iters = max(1, int(iters))
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model.decode_forward(model(x))
        if device_t.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.perf_counter()
    total_s = max(1e-9, t1 - t0)
    fps = float(iters / total_s)
    return {"throughput_fps": fps}

def main():
    args = parse_args()
    if args.download:
        ensure_coco_val(args.data_root, download=True)
    # Resolve dataset paths for logging/latency regardless of CLI combination
    images_dir_p, ann_json_p, img_paths = _resolve_dataset_paths(
        data_root=args.data_root, images_dir=args.images, ann_json=args.ann, max_images=args.max_images
    )

    stats = validate_coco(
        model_name=args.model,
        weights=None if args.weights in {"", "none", "None", "NONE"} else args.weights,
        data_root=args.data_root,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_images=args.max_images,
        save_json=args.save_json,
        images_dir=str(images_dir_p),
        ann_json=str(ann_json_p),
        save_viz_dir=args.save_viz_dir,
        viz_name=args.viz_name,
    )
    print({k: round(v, 5) for k, v in stats.items()})

    # Optional latency measurement (single image)
    perf = {"throughput_fps": 0.0}
    if img_paths and not getattr(args, "skip_perf", False):
        # Build model again to reuse already-loaded weights with same settings
        cn = coco80_class_names()
        model = get_model(
            args.model,
            weights=None if args.weights in {"", "none", "None", "NONE"} else args.weights,
            class_names=cn,
            input_norm_subtract=[0.0, 0.0, 0.0],
            input_norm_divide=[255.0, 255.0, 255.0],
        ).to(args.device)
        model.eval()
        # Fixed sampling iterations for FPS (keeps CLI simple)
        _iters = 30
        perf.update(
            _measure_latency(
                model,
                img_paths[0],
                device=args.device,
                imgsz=args.imgsz,
                iters=_iters,
                warmup=args.warmup_iters,
            )
        )

    # Append CSV log
    env = collect_env_info(device=args.device)
    row: Dict[str, object] = {
        "timestamp": now_iso(),
        "run_id": args.run_id or now_iso(),
        "commit": env.get("commit", ""),
        "host": env.get("host", ""),
        "runtime": args.runtime,
        "precision": args.precision,
        "device": env.get("device", ""),
        "device_name": env.get("device_name", ""),
        "model": args.model,
        "weights": args.weights,
        "dataset": "coco",
        "images_dir": str(images_dir_p),
        "ann_json": str(ann_json_p),
        "split": "val",
        "n_images": len(img_paths),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_images": args.max_images or "",
        "map_50_95": stats.get("mAP50-95", 0.0),
        "map_50": stats.get("mAP50", 0.0),
        "map_75": stats.get("mAP75", 0.0),
        "fps": perf.get("throughput_fps", 0.0),
        "export_path": "",
        "detections_json": args.save_json or "",
        "viz_dir": args.save_viz_dir or "",
        "notes": args.notes,
    }
    append_val_row(Path(args.log_csv), row, columns=VAL_COLUMNS)


if __name__ == "__main__":
    main()
