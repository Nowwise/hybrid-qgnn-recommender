from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class ExperimentStartBody(BaseModel):
    """Base preset plus optional overrides (omit a field to keep preset default)."""

    preset: Optional[str] = Field(
        default=None,
        description='Base configuration: "quick" or "full". Ignored if quick_demo is true.',
    )
    quick_demo: bool = Field(
        default=False,
        description="If true, same as preset=quick (small fast run).",
    )
    save_dir: Optional[str] = Field(default=None, description="Relative to project root, e.g. runs/my_exp")
    data_dir: Optional[str] = None
    max_users: Optional[int] = None
    max_pos_per_user: Optional[int] = None
    neg_per_pos: Optional[int] = None
    val_ratio: Optional[float] = None
    epochs_lg: Optional[int] = None
    epochs_hyb: Optional[int] = None
    batch_size: Optional[int] = None
    micro_bs: Optional[int] = None
    d: Optional[int] = None
    K: Optional[int] = None
    q: Optional[int] = None
    L: Optional[int] = None
    lr: Optional[float] = None
    wd: Optional[float] = None
    eval_every: Optional[int] = None
    hybrid_lr_mult: Optional[float] = None
    backend: Optional[str] = None
    p_quantum_start: Optional[float] = None
    p_quantum_end: Optional[float] = None
    seed: Optional[int] = None


class JobStepStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    error = "error"


class JobStep(BaseModel):
    id: str
    label: str
    status: JobStepStatus


class JobPublic(BaseModel):
    id: str
    status: JobStatus
    phase: str = ""
    detail: Optional[str] = None
    progress_pct: float = 0.0
    steps: List[JobStep] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class RunSummary(BaseModel):
    run_id: str
    path: str
    has_metrics: bool
    has_summary: bool
    best_lightgcn_auc: Optional[float] = None
    best_hybrid_auc: Optional[float] = None


class DatasetStatus(BaseModel):
    data_dir: str
    exists: bool
    train_txt: bool
    test_txt: bool


