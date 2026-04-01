from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from app.schemas.experiment import JobPublic, JobStatus, JobStep, JobStepStatus

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qgnn_train")
_lock = threading.Lock()
_jobs: Dict[str, JobPublic] = {}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def get_job(job_id: str) -> Optional[JobPublic]:
    with _lock:
        return _jobs.get(job_id)


def list_jobs() -> list[JobPublic]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)


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


def submit_training(fn: Callable[[str, Callable[[Dict[str, Any]], None]], None]) -> JobPublic:
    job_id = str(uuid.uuid4())
    now = _utcnow()
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
            j.updated_at = _utcnow()

    def _run():
        with _lock:
            j = _jobs[job_id]
            j.status = JobStatus.running
            j.phase = "starting"
            j.updated_at = _utcnow()
        try:
            result = fn(job_id, _progress)
            with _lock:
                j = _jobs[job_id]
                j.status = JobStatus.completed
                j.phase = "done"
                j.progress_pct = 100.0
                j.detail = None
                j.result = result
                j.steps = [
                    s.model_copy(update={"status": JobStepStatus.done})
                    for s in j.steps
                ]
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

    _executor.submit(_run)
    return job
