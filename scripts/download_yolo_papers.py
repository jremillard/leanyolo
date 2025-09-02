#!/usr/bin/env python3
"""Download YOLO research papers in TeX format with PDF fallback.

This script fetches reference papers for several YOLO versions. Each paper is
stored under ``references/<name>/`` where ``<name>`` is the YOLO version. The
script first attempts to download the original TeX source from arXiv; if the
TeX source is unavailable it falls back to downloading the PDF.

Example:
    python scripts/download_yolo_papers.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import requests

# Mapping of YOLO version/name -> URLs for tex and pdf.
# tex may be ``None`` if no source tarball is provided.
PAPERS: Dict[str, Dict[str, Optional[str]]] = {
    "yolov1": {
        "tex": "https://arxiv.org/e-print/1506.02640",
        "pdf": "https://arxiv.org/pdf/1506.02640.pdf",
    },
    "yolov2": {
        "tex": "https://arxiv.org/e-print/1612.08242",
        "pdf": "https://arxiv.org/pdf/1612.08242.pdf",
    },
    "yolov3": {
        "tex": "https://arxiv.org/e-print/1804.02767",
        "pdf": "https://arxiv.org/pdf/1804.02767.pdf",
    },
    "yolov4": {
        "tex": "https://arxiv.org/e-print/2004.10934",
        "pdf": "https://arxiv.org/pdf/2004.10934.pdf",
    },
    "yolov5": {
        "tex": None,
        "pdf": "https://www.ultralytics.com/assets/YOLOv5_Paper.pdf",
    },
    "pp-yolo": {
        "tex": "https://arxiv.org/e-print/2007.12099",
        "pdf": "https://arxiv.org/pdf/2007.12099.pdf",
    },
    "yolov6": {
        "tex": "https://arxiv.org/e-print/2209.02976",
        "pdf": "https://arxiv.org/pdf/2209.02976.pdf",
    },
    "yolov7": {
        "tex": "https://arxiv.org/e-print/2207.02696",
        "pdf": "https://arxiv.org/pdf/2207.02696.pdf",
    },
    "damo-yolo": {
        "tex": "https://arxiv.org/e-print/2211.15444",
        "pdf": "https://arxiv.org/pdf/2211.15444.pdf",
    },
    "yolov8": {
        "tex": None,
        "pdf": "https://www.ultralytics.com/assets/YOLOv8_Paper.pdf",
    },
    "yolo-nas": {
        "tex": None,
        "pdf": None,
    },
    "yolo-world": {
        "tex": "https://arxiv.org/e-print/2401.17270",
        "pdf": "https://arxiv.org/pdf/2401.17270.pdf",
    },
    "yolov9": {
        "tex": "https://arxiv.org/e-print/2402.13616",
        "pdf": "https://arxiv.org/pdf/2402.13616.pdf",
    },
    "yolov10": {
        "tex": "https://arxiv.org/e-print/2405.14458",
        "pdf": "https://arxiv.org/pdf/2405.14458.pdf",
    },
    "yolov11": {
        "tex": None,
        "pdf": "https://www.ultralytics.com/assets/YOLOv11_Paper.pdf",
    },
    "yolov12": {
        "tex": "https://arxiv.org/e-print/2502.12524",
        "pdf": "https://arxiv.org/pdf/2502.12524.pdf",
    },
}


def download_file(url: str, dest: Path) -> bool:
    """Download ``url`` to ``dest``.

    Returns ``True`` on success, ``False`` otherwise.
    """

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            dest.write_bytes(resp.content)
            return True
    except requests.RequestException:
        pass
    return False


def download_papers(out_dir: Path) -> None:
    """Download all YOLO papers into ``out_dir``."""

    for name, urls in PAPERS.items():
        paper_dir = out_dir / name
        paper_dir.mkdir(parents=True, exist_ok=True)
        downloaded = False
        tex_url = urls.get("tex")
        if tex_url:
            tex_path = paper_dir / "paper.tar"
            downloaded = download_file(tex_url, tex_path)
        if not downloaded:
            pdf_url = urls.get("pdf")
            if pdf_url:
                pdf_path = paper_dir / "paper.pdf"
                downloaded = download_file(pdf_url, pdf_path)
        if downloaded:
            print(f"[{name}] download complete")
        else:
            print(f"[{name}] failed to download")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YOLO papers")
    default_dir = Path(__file__).resolve().parents[1] / "references"
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_dir,
        help="Directory to store papers (default: references/)",
    )
    args = parser.parse_args()
    download_papers(args.out_dir)


if __name__ == "__main__":
    main()
