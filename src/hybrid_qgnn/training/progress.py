"""Structured progress snapshots for long-running experiments (UI / jobs)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional

StepStatus = Literal["pending", "running", "done", "error"]


def build_step_list(
    prepare: StepStatus,
    lightgcn: StepStatus,
    hybrid_warmup: StepStatus,
    hybrid_train: StepStatus,
    analysis: StepStatus,
    epochs_lg: int,
    epochs_hyb: int,
) -> List[Dict[str, str]]:
    return [
        {"id": "prepare", "label": "Data & graph", "status": prepare},
        {"id": "lightgcn", "label": f"LightGCN ({epochs_lg} ep)", "status": lightgcn},
        {"id": "hybrid_warmup", "label": "Hybrid QGNN warmup", "status": hybrid_warmup},
        {"id": "hybrid_train", "label": f"Hybrid QGNN ({epochs_hyb} ep)", "status": hybrid_train},
        {"id": "analysis", "label": "Metrics & tables", "status": analysis},
    ]


def progress_payload(
    completed_units: int,
    total_units: int,
    steps: List[Dict[str, str]],
    phase: str,
    detail: Optional[str] = None,
) -> Dict[str, Any]:
    pct = 100.0 if total_units <= 0 else min(100.0, (completed_units / total_units) * 100.0)
    out: Dict[str, Any] = {
        "progress_pct": round(pct, 2),
        "steps": steps,
        "phase": phase,
    }
    if detail is not None:
        out["detail"] = detail
    return out


def make_emit(
    epochs_lg: int,
    epochs_hyb: int,
    on_progress: Optional[Callable[[Dict[str, Any]], None]],
) -> Callable[..., None]:
    """Returns emit(completed, phase, detail=..., **status overrides for 5 steps)."""
    total = 1 + epochs_lg + 1 + epochs_hyb + 1

    def emit(
        completed: int,
        phase: str,
        detail: Optional[str] = None,
        *,
        prepare: StepStatus = "pending",
        lightgcn: StepStatus = "pending",
        hybrid_warmup: StepStatus = "pending",
        hybrid_train: StepStatus = "pending",
        analysis: StepStatus = "pending",
    ) -> None:
        if not on_progress:
            return
        steps = build_step_list(prepare, lightgcn, hybrid_warmup, hybrid_train, analysis, epochs_lg, epochs_hyb)
        on_progress(progress_payload(completed, total, steps, phase, detail))

    return emit
