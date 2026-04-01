from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.core.settings import Settings, get_settings
from app.schemas.experiment import ExperimentStartBody, JobPublic, RunSummary
from app.services.jobs import get_job, list_jobs, submit_training
from app.services.runs import list_runs, read_run_config

router = APIRouter(prefix="/experiments", tags=["experiments"])


def _base_from_body(body: ExperimentStartBody):
    from hybrid_qgnn.config import ExperimentConfig

    if body.quick_demo or (body.preset or "").lower() == "quick":
        return ExperimentConfig.quick_demo()
    return ExperimentConfig()


def _merge_experiment_config(body: ExperimentStartBody, settings: Settings):
    cfg = _base_from_body(body)

    if body.save_dir is not None:
        cfg.save_dir = body.save_dir
    if body.data_dir is not None:
        cfg.data_dir = body.data_dir
    if body.max_users is not None:
        cfg.max_users = body.max_users
    if body.max_pos_per_user is not None:
        cfg.max_pos_per_user = body.max_pos_per_user
    if body.neg_per_pos is not None:
        cfg.neg_per_pos = body.neg_per_pos
    if body.val_ratio is not None:
        cfg.val_ratio = body.val_ratio
    if body.epochs_lg is not None:
        cfg.epochs_lg = body.epochs_lg
    if body.epochs_hyb is not None:
        cfg.epochs_hyb = body.epochs_hyb
    if body.batch_size is not None:
        cfg.batch_size = body.batch_size
    if body.micro_bs is not None:
        cfg.micro_bs = body.micro_bs
    if body.d is not None:
        cfg.d = body.d
    if body.K is not None:
        cfg.K = body.K
    if body.q is not None:
        cfg.q = body.q
    if body.L is not None:
        cfg.L = body.L
    if body.lr is not None:
        cfg.lr = body.lr
    if body.wd is not None:
        cfg.wd = body.wd
    if body.eval_every is not None:
        cfg.eval_every = body.eval_every
    if body.hybrid_lr_mult is not None:
        cfg.hybrid_lr_mult = body.hybrid_lr_mult
    if body.backend is not None:
        cfg.backend = body.backend
    if body.p_quantum_start is not None:
        cfg.p_quantum_start = body.p_quantum_start
    if body.p_quantum_end is not None:
        cfg.p_quantum_end = body.p_quantum_end
    if body.seed is not None:
        cfg.seed = body.seed

    is_quick = body.quick_demo or (body.preset or "").lower() == "quick"
    if is_quick and body.save_dir is None:
        cfg.save_dir = f"./runs/quick_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    return cfg


@router.get("/presets")
def experiment_presets():
    from hybrid_qgnn.config import ExperimentConfig

    return {
        "quick": ExperimentConfig.quick_demo().to_dict(),
        "full": ExperimentConfig().to_dict(),
    }


@router.post("/runs", response_model=JobPublic)
def start_run(body: ExperimentStartBody, settings: Settings = Depends(get_settings)):
    cfg = _merge_experiment_config(body, settings)
    data_root = settings.project_root / cfg.data_dir
    train_f = data_root / "train.txt"
    test_f = data_root / "test.txt"
    if not train_f.is_file() or not test_f.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Dataset not found. Expected {cfg.data_dir}/train.txt and test.txt under project root.",
        )

    def run(_job_id: str, on_progress: Callable[[Dict[str, Any]], None]) -> dict[str, Any]:
        from hybrid_qgnn.training import run_experiment

        return run_experiment(
            cfg,
            project_root=settings.project_root,
            on_progress=on_progress,
            show_progress=False,
        )

    return submit_training(run)


@router.get("/jobs", response_model=List[JobPublic])
def jobs():
    return list_jobs()


@router.get("/jobs/{job_id}", response_model=JobPublic)
def job(job_id: str):
    j = get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    return j


@router.get("/history", response_model=List[RunSummary])
def history(settings: Settings = Depends(get_settings)):
    runs_dir = settings.project_root / "runs"
    return list_runs(runs_dir)


@router.get("/history/{run_id}/config")
def run_config(run_id: str, settings: Settings = Depends(get_settings)):
    run_dir = settings.project_root / "runs" / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    cfg = read_run_config(run_dir)
    if cfg is None:
        raise HTTPException(status_code=404, detail="run_config.json missing")
    return cfg


@router.get("/history/{run_id}/metrics")
def run_metrics(run_id: str, settings: Settings = Depends(get_settings)):
    run_dir = settings.project_root / "runs" / run_id
    csv_path = run_dir / "metrics.csv"
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail="metrics.csv not found")
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


@router.get("/history/{run_id}/comparative")
def run_comparative(run_id: str, settings: Settings = Depends(get_settings)):
    run_dir = settings.project_root / "runs" / run_id
    p = run_dir / "val_best_comparative.csv"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="val_best_comparative.csv not found")
    df = pd.read_csv(p)
    return df.to_dict(orient="records")


@router.get("/history/{run_id}/download/metrics.csv")
def download_metrics(run_id: str, settings: Settings = Depends(get_settings)):
    run_dir = settings.project_root / "runs" / run_id
    csv_path = run_dir / "metrics.csv"
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail="metrics.csv not found")
    return FileResponse(csv_path, filename=f"{run_id}_metrics.csv", media_type="text/csv")
