from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class ExperimentStartBody(BaseModel):
    """Base preset plus optional overrides (omit a field to keep preset default)."""

    preset: Optional[str] = Field(
        default=None,
        description='Base: "lightweight" | "notebook" | "custom" (aliases: quick, full). Ignored if quick_demo is true.',
    )
    quick_demo: bool = Field(
        default=False,
        description="If true, same as lightweight preset (small fast run).",
    )
    save_dir: Optional[str] = Field(default=None, description="Relative to project root, e.g. runs/my_exp")
    experiment_name: Optional[str] = Field(
        default=None,
        description="Friendly run label. If save_dir is omitted or empty, output goes to runs/<slug>_<UTC timestamp>.",
        max_length=200,
    )
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
    lightgcn_lr: Optional[float] = None
    hybrid_lr: Optional[float] = None
    wd: Optional[float] = None
    eval_every: Optional[int] = None
    hybrid_lr_mult: Optional[float] = None
    backend: Optional[str] = None
    p_quantum_start: Optional[float] = None
    p_quantum_end: Optional[float] = None
    seed: Optional[int] = None
    device: Optional[str] = Field(
        default=None,
        description='Training device: omit for env QGNN_DEVICE or auto. '
        'Values: "auto", "cpu", "cuda", "cuda:0", …',
    )
    compute_mode: Optional[Literal["auto", "cpu", "gpu"]] = Field(
        default=None,
        description='Applied last: "auto" keeps merged preset/device/backend; '
        '"cpu" forces CPU PyTorch + lightning.qubit; '
        '"gpu" uses CUDA + lightning.gpu (falls back in training if unavailable).',
    )
    eval_ranking: Optional[bool] = None
    ranking_max_users: Optional[int] = None
    ranking_negatives: Optional[int] = None
    ranking_ks: Optional[List[int]] = Field(
        default=None,
        description="K values for Recall/NDCG/HitRatio (e.g. [10, 20, 50]); need ranking_negatives >= max(K).",
    )

    @field_validator("experiment_name", mode="before")
    @classmethod
    def _strip_experiment_name(cls, v: Any) -> Optional[str]:
        if v is None or v == "":
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator("save_dir", mode="before")
    @classmethod
    def _empty_save_dir_to_none(cls, v: Any) -> Optional[str]:
        if v is None or v == "":
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator("ranking_ks", mode="before")
    @classmethod
    def _coerce_ranking_ks(cls, v: Any) -> Optional[List[int]]:
        """Accept JSON list or comma-separated string (some clients send form-style strings)."""
        if v is None or v == "":
            return None
        if isinstance(v, list):
            return [int(x) for x in v]
        if isinstance(v, str):
            parts = [p for p in re.split(r"[,;\s]+", v.strip()) if p]
            if not parts:
                return None
            return [int(p) for p in parts]
        raise ValueError("ranking_ks must be a list of integers or a comma-separated string")

    eval_test_ranking: Optional[bool] = None
    eval_hybrid_ablation: Optional[bool] = None
    log_phase_timings: Optional[bool] = None


class JobStepStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    error = "error"


class JobStep(BaseModel):
    id: str
    label: str
    status: JobStepStatus


class JobActivity(BaseModel):
    """Current fine-grained training activity (batch / epoch level)."""

    model: Optional[str] = None
    split: Optional[str] = None
    phase: Optional[str] = None
    epoch: Optional[int] = None
    batch: Optional[int] = None
    total_batches: Optional[int] = None
    loss: Optional[float] = None
    p_quantum: Optional[float] = None


class JobEvent(BaseModel):
    ts: str
    kind: str = "info"
    message: str


class JobPublic(BaseModel):
    id: str
    status: JobStatus
    phase: str = ""
    detail: Optional[str] = None
    progress_pct: float = 0.0
    steps: List[JobStep] = Field(default_factory=list)
    activity: Optional[JobActivity] = None
    events: List[JobEvent] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class RunSummary(BaseModel):
    run_id: str
    path: str
    has_metrics: bool
    has_summary: bool
    experiment_name: Optional[str] = None
    best_lightgcn_auc: Optional[float] = None
    best_hybrid_auc: Optional[float] = None


class DatasetStatus(BaseModel):
    data_dir: str
    exists: bool
    train_txt: bool
    test_txt: bool


class DatasetsOverview(BaseModel):
    """Known benchmark folders under project root (LightGCN-style train.txt / test.txt)."""

    datasets: List[DatasetStatus]


