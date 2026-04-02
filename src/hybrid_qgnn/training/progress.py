"""Structured progress snapshots for long-running experiments (UI / jobs)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional

StepStatus = Literal["pending", "running", "done", "error"]


def build_multibaseline_steps(
    prepare: StepStatus,
    baseline_ids: List[str],
    baseline_statuses: List[StepStatus],
    hybrid_warmup: StepStatus,
    hybrid_train: StepStatus,
    analysis: StepStatus,
    epochs_lg: int,
    epochs_hyb: int,
) -> List[Dict[str, str]]:
    from hybrid_qgnn.models.graph_encoders import BASELINE_DISPLAY_NAMES

    if len(baseline_statuses) != len(baseline_ids):
        raise ValueError("baseline_statuses must align with baseline_ids")
    out: List[Dict[str, str]] = [{"id": "prepare", "label": "Data & graph", "status": prepare}]
    for bid, st in zip(baseline_ids, baseline_statuses):
        dn = BASELINE_DISPLAY_NAMES.get(bid, bid)
        out.append({"id": f"baseline_{bid}", "label": f"{dn} ({epochs_lg} ep)", "status": st})
    out.extend(
        [
            {"id": "hybrid_warmup", "label": "Hybrid QGNN warmup", "status": hybrid_warmup},
            {"id": "hybrid_train", "label": f"Hybrid QGNN ({epochs_hyb} ep)", "status": hybrid_train},
            {"id": "analysis", "label": "Metrics & tables", "status": analysis},
        ]
    )
    return out


def progress_payload(
    completed_units: float,
    total_units: float,
    steps: List[Dict[str, str]],
    phase: str,
    detail: Optional[str] = None,
    activity: Optional[Dict[str, Any]] = None,
    event: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pct = 100.0 if total_units <= 0 else min(100.0, (completed_units / total_units) * 100.0)
    out: Dict[str, Any] = {
        "progress_pct": round(pct, 2),
        "steps": steps,
        "phase": phase,
    }
    if detail is not None:
        out["detail"] = detail
    if activity is not None:
        out["activity"] = activity
    if event is not None:
        out["event"] = event
    return out


def should_emit_batch(batch_idx_1based: int, total_batches: int, max_reports: int = 96) -> bool:
    """Throttle batch-level updates so the UI stays responsive on large loaders."""
    if total_batches <= 0:
        return False
    if total_batches <= max_reports:
        return True
    step = max(1, total_batches // max_reports)
    return batch_idx_1based % step == 0 or batch_idx_1based >= total_batches


def segment_train_val_position(
    segment_base: float,
    train_idx_1based: int,
    train_total: int,
    val_idx_1based: int,
    val_total: int,
    in_train: bool,
    train_weight: float = 0.88,
) -> float:
    """Map train/val batch progress into [segment_base, segment_base + 1)."""
    tw, vw = train_weight, 1.0 - train_weight
    if in_train:
        if train_total <= 0:
            return segment_base
        return segment_base + tw * min(1.0, train_idx_1based / train_total)
    if val_total <= 0:
        return segment_base + tw
    return segment_base + tw + vw * min(1.0, val_idx_1based / val_total)


class PipelineReporter:
    """
    Pipeline position in [0, M] with M = 1 + n_baselines * epochs_lg + 1 + epochs_hyb + 1.
    Baseline step list is built from ``baseline_ids`` (e.g. lightgcn, sgl, …).
    """

    def __init__(
        self,
        baseline_ids: List[str],
        epochs_lg: int,
        epochs_hyb: int,
        on_progress: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        self.baseline_ids = list(baseline_ids)
        self.epochs_lg = epochs_lg
        self.epochs_hyb = epochs_hyb
        self.M = float(1 + len(self.baseline_ids) * epochs_lg + 1 + epochs_hyb + 1)
        self.on_progress = on_progress
        self._bs: List[StepStatus] = ["pending"] * len(self.baseline_ids)

    def _iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def set_baseline_active(self, index: int) -> None:
        for j in range(len(self._bs)):
            if j < index:
                self._bs[j] = "done"
            elif j == index:
                self._bs[j] = "running"
            else:
                self._bs[j] = "pending"

    def mark_baseline_done(self, index: int) -> None:
        if 0 <= index < len(self._bs):
            self._bs[index] = "done"

    def all_baselines_finished(self) -> None:
        self._bs = ["done"] * len(self.baseline_ids)

    def push(
        self,
        position: float,
        phase: str,
        detail: Optional[str] = None,
        *,
        prepare: StepStatus = "pending",
        hybrid_warmup: StepStatus = "pending",
        hybrid_train: StepStatus = "pending",
        analysis: StepStatus = "pending",
        activity: Optional[Dict[str, Any]] = None,
        event_message: Optional[str] = None,
        event_kind: str = "info",
    ) -> None:
        if not self.on_progress:
            return
        steps = build_multibaseline_steps(
            prepare,
            self.baseline_ids,
            list(self._bs),
            hybrid_warmup,
            hybrid_train,
            analysis,
            self.epochs_lg,
            self.epochs_hyb,
        )
        ev: Optional[Dict[str, Any]] = None
        if event_message:
            ev = {"ts": self._iso(), "kind": event_kind, "message": event_message}
        payload = progress_payload(position, self.M, steps, phase, detail, activity, ev)
        self.on_progress(payload)

    def push_fine(
        self,
        position: float,
        phase: str,
        detail: Optional[str],
        activity: Dict[str, Any],
        event_message: Optional[str] = None,
    ) -> None:
        """Update percentage + activity without rebuilding steps (smaller payloads)."""
        if not self.on_progress:
            return
        out: Dict[str, Any] = {
            "progress_pct": round(min(100.0, (position / self.M) * 100.0), 2),
            "phase": phase,
            "detail": detail,
            "activity": activity,
        }
        if event_message:
            out["event"] = {"ts": self._iso(), "kind": "info", "message": event_message}
        self.on_progress(out)

    def base_prepare(self) -> float:
        return 0.0

    def base_baseline_epoch(self, baseline_index: int, ep_1based: int) -> float:
        return float(1 + baseline_index * self.epochs_lg + (ep_1based - 1))

    def base_hybrid_warmup(self) -> float:
        return float(1 + len(self.baseline_ids) * self.epochs_lg)

    def base_hybrid_epoch(self, ep_1based: int) -> float:
        return float(1 + len(self.baseline_ids) * self.epochs_lg + 1 + (ep_1based - 1))

    def base_analysis(self) -> float:
        return float(1 + len(self.baseline_ids) * self.epochs_lg + 1 + self.epochs_hyb)
