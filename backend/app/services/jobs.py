from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from app.schemas.experiment import JobPublic, JobStatus

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


def submit_training(fn: Callable[[str, Callable[..., None]], None]) -> JobPublic:
    job_id = str(uuid.uuid4())
    now = _utcnow()
    job = JobPublic(
        id=job_id,
        status=JobStatus.queued,
        phase="queued",
        created_at=now,
        updated_at=now,
    )
    with _lock:
        _jobs[job_id] = job

    def _progress(phase: str, detail: Optional[str]):
        with _lock:
            j = _jobs.get(job_id)
            if j:
                j.phase = phase
                j.detail = detail
                j.status = JobStatus.running
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
                j.result = result
                j.updated_at = _utcnow()
        except Exception as e:
            with _lock:
                j = _jobs[job_id]
                j.status = JobStatus.failed
                j.phase = "failed"
                j.error = str(e)
                j.updated_at = _utcnow()

    _executor.submit(_run)
    return job
