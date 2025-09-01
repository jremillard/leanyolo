from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Tolerance:
    """Numerical tolerance for comparisons.

    rtol/atol follow torch.allclose semantics.
    max_abs is an optional hard cap for maximum absolute error.
    """

    rtol: float = 1e-4
    atol: float = 1e-4
    max_abs: float | None = None


@dataclass(frozen=True)
class Rubric:
    """Criteria per component for a given weight variant."""

    # Structural fidelity: shapes must match exactly
    require_shape_match: bool = True

    # Numerical tolerances per component
    backbone_tol: Tolerance = Tolerance(rtol=1e-4, atol=1e-4, max_abs=5e-4)
    neck_tol: Tolerance = Tolerance(rtol=1e-4, atol=1e-4, max_abs=5e-4)
    head_tol: Tolerance = Tolerance(rtol=1e-4, atol=1e-4, max_abs=5e-4)
    end2end_tol: Tolerance = Tolerance(rtol=1e-4, atol=1e-4, max_abs=5e-4)

    # Report formatting options
    report_version: str = "1.0"


# Default rubric per model size. Kept identical unless a size needs special handling.
RUBRICS: Dict[str, Rubric] = {
    "yolov10n": Rubric(),
    "yolov10s": Rubric(),
    "yolov10m": Rubric(),
    "yolov10b": Rubric(),
    "yolov10l": Rubric(),
    "yolov10x": Rubric(),
}


def get_rubric(model_name: str) -> Rubric:
    if model_name not in RUBRICS:
        raise KeyError(f"No rubric defined for '{model_name}'.")
    return RUBRICS[model_name]


def tolerance_summary(model_name: str) -> Dict[str, Tuple[float, float, float | None]]:
    r = get_rubric(model_name)
    return {
        "backbone": (r.backbone_tol.rtol, r.backbone_tol.atol, r.backbone_tol.max_abs),
        "neck": (r.neck_tol.rtol, r.neck_tol.atol, r.neck_tol.max_abs),
        "head": (r.head_tol.rtol, r.head_tol.atol, r.head_tol.max_abs),
        "end2end": (r.end2end_tol.rtol, r.end2end_tol.atol, r.end2end_tol.max_abs),
    }
