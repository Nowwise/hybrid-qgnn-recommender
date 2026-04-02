from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import re
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.core.settings import Settings, get_settings
from app.schemas.experiment import ExperimentStartBody, JobPublic, JobStatus, RunSummary
from app.services.jobs import cancel_job, get_job, list_jobs, submit_training
from app.services.runs import list_runs, read_run_config

router = APIRouter(prefix="/experiments", tags=["experiments"])


def _slugify_run_dir_segment(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s[:120] if s else "run"


def _base_from_body(body: ExperimentStartBody):
    from hybrid_qgnn.config import ExperimentConfig

    if body.quick_demo:
        return ExperimentConfig.lightweight()
    p = (body.preset or "").lower()
    if p in ("quick", "lightweight", "light"):
        return ExperimentConfig.lightweight()
    if p in ("notebook", "balanced", "balanced_cpu"):
        return ExperimentConfig.notebook_balanced()
    return ExperimentConfig()


def _merge_experiment_config(body: ExperimentStartBody, settings: Settings):
    cfg = _base_from_body(body)

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
    if body.lightgcn_lr is not None:
        cfg.lightgcn_lr = body.lightgcn_lr
    if body.hybrid_lr is not None:
        cfg.hybrid_lr = body.hybrid_lr
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
    if body.device is not None:
        cfg.device = body.device
    if body.eval_ranking is not None:
        cfg.eval_ranking = body.eval_ranking
    if body.ranking_max_users is not None:
        cfg.ranking_max_users = body.ranking_max_users
    if body.ranking_negatives is not None:
        cfg.ranking_negatives = body.ranking_negatives
    if body.ranking_ks is not None:
        cfg.ranking_ks = list(body.ranking_ks)
    if body.eval_test_ranking is not None:
        cfg.eval_test_ranking = body.eval_test_ranking
    if body.eval_hybrid_ablation is not None:
        cfg.eval_hybrid_ablation = body.eval_hybrid_ablation
    if body.log_phase_timings is not None:
        cfg.log_phase_timings = body.log_phase_timings

    # Last: whole-run compute profile (overrides preset device/backend for this job)
    if body.compute_mode is not None:
        cm = body.compute_mode.strip().lower()
        if cm == "cpu":
            cfg.device = "cpu"
            cfg.backend = "lightning.qubit"
        elif cm == "gpu":
            cfg.device = "cuda"
            cfg.backend = "lightning.gpu"
        elif cm == "auto":
            pass
        cfg.compute_mode = cm

    exp_name = (body.experiment_name or "").strip()
    if exp_name:
        cfg.experiment_name = exp_name

    p = (body.preset or "").lower()
    is_light = body.quick_demo or p in ("quick", "lightweight", "light")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if body.save_dir is not None:
        cfg.save_dir = body.save_dir
    elif exp_name:
        slug = _slugify_run_dir_segment(exp_name)
        cfg.save_dir = f"./runs/{slug}_{ts}"
    elif is_light:
        cfg.save_dir = f"./runs/quick_{ts}"

    return cfg


@router.get("/device")
def experiment_device():
    """Report PyTorch/CUDA visibility and selectable device ids (for UI or debugging)."""
    from hybrid_qgnn.device import compute_device_summary

    return compute_device_summary()


@router.get("/presets")
def experiment_presets():
    from hybrid_qgnn.config import ExperimentConfig

    lw = ExperimentConfig.lightweight().to_dict()
    nb = ExperimentConfig.notebook_balanced().to_dict()
    cu = ExperimentConfig().to_dict()
    return {
        "lightweight": lw,
        "notebook": nb,
        "custom": cu,
        "quick": lw,
        "full": cu,
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
            detail=f"Dataset not found. Expected project_root/{cfg.data_dir}/train.txt and test.txt.",
        )

    def run(
        _job_id: str,
        on_progress: Callable[[Dict[str, Any]], None],
        cancel_ev,
    ) -> dict[str, Any]:
        from hybrid_qgnn.training import run_experiment

        return run_experiment(
            cfg,
            project_root=settings.project_root,
            on_progress=on_progress,
            show_progress=False,
            cancel_event=cancel_ev,
        )

    return submit_training(run)


@router.post("/jobs/{job_id}/cancel", response_model=JobPublic)
def cancel_experiment_job(job_id: str):
    j = get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    if j.status not in (JobStatus.queued, JobStatus.running):
        raise HTTPException(status_code=400, detail="Job is not queued or running")
    cancel_job(job_id)
    out = get_job(job_id)
    return out if out else j


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
    """Wide table: val@best metrics + val/test ranking + Δ row. Built from metrics.csv if needed."""
    from hybrid_qgnn.analysis.comparative import build_full_model_comparative

    run_dir = settings.project_root / "runs" / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")

    def _rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
        dfo = df.astype(object).where(pd.notnull(df), None)
        return dfo.to_dict(orient="records")

    full_p = run_dir / "full_model_comparative.csv"
    if full_p.is_file():
        return _rows(pd.read_csv(full_p))

    metrics_p = run_dir / "metrics.csv"
    if not metrics_p.is_file():
        raise HTTPException(status_code=404, detail="metrics.csv not found")

    raw = pd.read_csv(metrics_p)
    merged = build_full_model_comparative(raw)
    if merged is None or merged.empty:
        raise HTTPException(
            status_code=404,
            detail="No comparable metrics in metrics.csv (need val classification and/or ranking rows).",
        )
    return _rows(merged)


@router.get("/history/{run_id}/download/metrics.csv")
def download_metrics(run_id: str, settings: Settings = Depends(get_settings)):
    run_dir = settings.project_root / "runs" / run_id
    csv_path = run_dir / "metrics.csv"
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail="metrics.csv not found")
    return FileResponse(csv_path, filename=f"{run_id}_metrics.csv", media_type="text/csv")
