from __future__ import annotations

import os
import time
from typing import Dict

from .common import REPORTS_DIR, write_json, ensure_dirs


def record_report(model_name: str, results: Dict) -> str:
    ensure_dirs()
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(REPORTS_DIR, f"{model_name}-{ts}.json")
    write_json(path, results)
    return path

