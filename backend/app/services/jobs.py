from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from hybrid_qgnn.exceptions import ExperimentCancelled

from app.schemas.experiment import JobActivity, JobEvent, JobPublic, JobStatus, JobStep, JobStepStatus

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qgnn_train")
_lock = threading.Lock()
_jobs: Dict[str, JobPublic] = {}
_cancel_events: Dict[str, threading.Event] = {}
_MAX_EVENTS = 200


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def get_job(job_id: str) -> Optional[JobPublic]:
    with _lock:
        return _jobs.get(job_id)


def list_jobs() -> list[JobPublic]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)


def clear_all_job_records() -> int:
    """Drop every job from memory. Only call when no queued/running jobs exist."""
    with _lock:
        n = len(_jobs)
        _jobs.clear()
        _cancel_events.clear()
        return n


def cancel_job(job_id: str) -> bool:
    with _lock:
        ev = _cancel_events.get(job_id)
    if ev is None:
        return False
    ev.set()
    return True


def _parse_steps(raw: list) -> list[JobStep]:
    out: list[JobStep] = []
    for s in raw:
        if isinstance(s, JobStep):
            out.append(s)
        else:
            out.append(
                JobStep(
                    id=str(s["id"]),
                    label=str(s["label"]),
                    status=JobStepStatus(str(s["status"])),
                )
            )
    return out


def submit_training(
    fn: Callable[[str, Callable[[Dict[str, Any]], None], threading.Event], Any],
) -> JobPublic:
    job_id = str(uuid.uuid4())
    now = _utcnow()
    cancel_ev = threading.Event()
    job = JobPublic(
        id=job_id,
        status=JobStatus.queued,
        phase="queued",
        progress_pct=0.0,
        steps=[],
        created_at=now,
        updated_at=now,
    )
    with _lock:
        _jobs[job_id] = job
        _cancel_events[job_id] = cancel_ev

    def _progress(update: Dict[str, Any]) -> None:
        with _lock:
            j = _jobs.get(job_id)
            if not j:
                return
            j.status = JobStatus.running
            if "progress_pct" in update:
                j.progress_pct = float(update["progress_pct"])
            if "steps" in update and update["steps"] is not None:
                j.steps = _parse_steps(list(update["steps"]))
            if "phase" in update and update["phase"] is not None:
                j.phase = str(update["phase"])
            if "detail" in update:
                j.detail = update["detail"] if update["detail"] is not None else None
            if "activity" in update:
                act = update["activity"]
                if act is None:
                    j.activity = None
                elif isinstance(act, dict):
                    j.activity = JobActivity(**{k: v for k, v in act.items() if k in JobActivity.model_fields})
                else:
                    j.activity = act
            if "event" in update and update["event"]:
                ev_raw = update["event"]
                if isinstance(ev_raw, dict):
                    ev = JobEvent(
                        ts=str(ev_raw.get("ts", "")),
                        kind=str(ev_raw.get("kind", "info")),
                        message=str(ev_raw.get("message", "")),
                    )
                    j.events = [*j.events, ev]
                    if len(j.events) > _MAX_EVENTS:
                        j.events = j.events[-_MAX_EVENTS:]
            sd_raw = update.get("save_dir")
            if isinstance(sd_raw, str) and sd_raw.strip():
                j.save_dir = sd_raw.strip().replace("\\", "/")
            j.updated_at = _utcnow()

    def _run():
        with _lock:
            j = _jobs[job_id]
            j.status = JobStatus.running
            j.phase = "starting"
            j.updated_at = _utcnow()
        try:
            result = fn(job_id, _progress, cancel_ev)
            with _lock:
                j = _jobs[job_id]
                if cancel_ev.is_set():
                    j.status = JobStatus.cancelled
                    j.phase = "cancelled"
                    j.detail = "Stopped by user"
                    j.result = None
                    j.updated_at = _utcnow()
                else:
                    j.status = JobStatus.completed
                    j.phase = "done"
                    j.progress_pct = 100.0
                    j.detail = None
                    j.result = result
                    j.steps = [s.model_copy(update={"status": JobStepStatus.done}) for s in j.steps]
                    j.updated_at = _utcnow()
        except ExperimentCancelled:
            with _lock:
                j = _jobs[job_id]
                j.status = JobStatus.cancelled
                j.phase = "cancelled"
                j.detail = "Stopped by user"
                j.error = None
                j.updated_at = _utcnow()
        except Exception as e:
            with _lock:
                j = _jobs[job_id]
                j.status = JobStatus.failed
                j.phase = "failed"
                j.error = str(e)
                j.updated_at = _utcnow()
                j.steps = [
                    s.model_copy(
                        update={
                            "status": JobStepStatus.error
                            if s.status == JobStepStatus.running
                            else s.status
                        }
                    )
                    for s in j.steps
                ]
        finally:
            with _lock:
                _cancel_events.pop(job_id, None)

    _executor.submit(_run)
    return job
