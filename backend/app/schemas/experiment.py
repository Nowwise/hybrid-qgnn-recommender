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
    """Optional overrides; omitted fields use defaults from preset or full config."""

    quick_demo: bool = Field(
        default=False,
        description="If true, run a small configuration suitable for interactive testing.",
    )
    save_dir: Optional[str] = Field(default=None, description="Relative to project root, e.g. runs/my_exp")
    data_dir: Optional[str] = None
    max_users: Optional[int] = None
    max_pos_per_user: Optional[int] = None
    epochs_lg: Optional[int] = None
    epochs_hyb: Optional[int] = None
    batch_size: Optional[int] = None
    d: Optional[int] = None
    K: Optional[int] = None
    q: Optional[int] = None
    L: Optional[int] = None
    lr: Optional[float] = None
    seed: Optional[int] = None


class JobPublic(BaseModel):
    id: str
    status: JobStatus
    phase: str = ""
    detail: Optional[str] = None
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


