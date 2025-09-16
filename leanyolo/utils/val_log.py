from __future__ import annotations

import csv
import platform
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


# Canonical CSV column order. Add new columns only at the end.
COLUMNS: List[str] = [
    "timestamp",
    "run_id",
    "commit",
    "host",
    "runtime",  # torch | onnxrt | tensorrt | torchscript
    "precision",  # fp32 | fp16 | int8
    "device",  # cpu | cuda
    "device_name",
    "model",
    "weights",
    "dataset",
    "images_dir",
    "ann_json",
    "split",
    "n_images",
    "imgsz",
    "conf",
    "iou",
    "max_images",
    "map_50_95",
    "map_50",
    "map_75",
    "fps",
    "export_path",
    "detections_json",
    "viz_dir",
    "notes",
]


def _git_commit() -> str:
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        return sha
    except Exception:
        return ""


def _get_device_name(device: str) -> str:
    device = (device or "").lower()
    if device.startswith("cuda"):
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except Exception:
            return "cuda"
    return platform.processor() or platform.machine() or "cpu"


def collect_env_info(*, device: str) -> Dict[str, str]:
    return {
        "commit": _git_commit(),
        "host": socket.gethostname(),
        "device": device,
        "device_name": _get_device_name(device),
    }


def ensure_csv(path: Path, *, columns: Iterable[str] = COLUMNS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(cols)
        return
    # If exists, verify header; migrate if different
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                raise ValueError("empty CSV")
            if header == cols:
                return
            # Migrate: remap existing rows by header->value into new column order
            rows = []
            for r in reader:
                # pad/truncate row to header length
                if len(r) < len(header):
                    r = r + [""] * (len(header) - len(r))
                elif len(r) > len(header):
                    r = r[: len(header)]
                rows.append(dict(zip(header, r)))
        # Rewrite file with new header/columns
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for rowmap in rows:
                w.writerow([rowmap.get(c, "") for c in cols])
    except Exception:
        # On any failure, write a fresh header (non-destructive append behavior will follow)
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(cols)


def append_row(path: Path, values: Mapping[str, object], *, columns: Iterable[str] = COLUMNS) -> None:
    ensure_csv(path, columns=columns)
    row = [values.get(col, "") for col in columns]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def now_iso() -> str:
    """Return current UTC time in ISO 8601 format without microseconds.

    Uses timezone-aware :func:`datetime.now` with :data:`datetime.UTC` to avoid
    the deprecated :func:`datetime.utcnow`. The trailing ``"+00:00"`` is
    replaced with ``"Z"`` for brevity.
    """
    return (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
