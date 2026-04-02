from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import re
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response

from app.core.settings import Settings, get_settings
from app.schemas.experiment import (
    ExperimentStartBody,
    JobPublic,
    JobStatus,
    RunSummary,
    ScorePairsBody,
    ScorePairsResponse,
)
from app.services.jobs import cancel_job, clear_all_job_records, get_job, list_jobs, submit_training
from app.services.runs import (
    clear_runs_directory,
    delete_run_directory,
    job_output_overlaps_run_dir,
    list_runs,
    read_run_config,
    resolved_run_dir,
)

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
    if p == "large":
        return ExperimentConfig.large()
    if p in ("extra_large", "xlarge", "xl", "extra-large"):
        return ExperimentConfig.extra_large()
    return ExperimentConfig()


def _batch_progress_wrap(
    on_progress: Optional[Callable[[Dict[str, Any]], None]],
    run_idx: int,
    n_runs: int,
) -> Optional[Callable[[Dict[str, Any]], None]]:
    if on_progress is None or n_runs <= 1:
        return on_progress
    lo = run_idx / n_runs
    span = 1.0 / n_runs

    def _wrapped(update: Dict[str, Any]) -> None:
        u = dict(update)
        if "progress_pct" in u:
            inner_frac = float(u["progress_pct"]) / 100.0
            u["progress_pct"] = round((lo + inner_frac * span) * 100.0, 2)
        ev = u.get("event")
        if isinstance(ev, dict):
            msg = ev.get("message")
            if isinstance(msg, str) and msg:
                u["event"] = {**ev, "message": f"[{run_idx + 1}/{n_runs}] {msg}"}
        on_progress(u)

    return _wrapped


def _expand_run_variants(cfg, body: ExperimentStartBody) -> List[Any]:
    if body.seeds:
        seed_vals = list(dict.fromkeys(int(s) for s in body.seeds))
    else:
        seed_vals = [cfg.seed]
    q_vals = list(body.sweep_q) if body.sweep_q else [cfg.q]
    L_vals = list(body.sweep_L) if body.sweep_L else [cfg.L]
    ent_vals = [bool(x) for x in body.sweep_entangle] if body.sweep_entangle is not None else [bool(cfg.quantum_entangle)]

    n_total = len(seed_vals) * len(q_vals) * len(L_vals) * len(ent_vals)
    base_save = Path(cfg.save_dir)
    out: List[Any] = []
    idx = 0
    for s in seed_vals:
        for q in q_vals:
            for L in L_vals:
                for ent in ent_vals:
                    c = replace(
                        cfg,
                        seed=int(s),
                        q=int(q),
                        L=int(L),
                        quantum_entangle=bool(ent),
                    )
                    if n_total > 1:
                        c.save_dir = str(
                            base_save / f"run_{idx:03d}_seed{s}_q{q}_L{L}_e{int(ent)}"
                        )
                    out.append(c)
                    idx += 1
    return out


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
    if body.live_plots is not None:
        cfg.live_plots = body.live_plots

    if body.train_baseline_lightgcn is not None:
        cfg.train_baseline_lightgcn = bool(body.train_baseline_lightgcn)
    if body.train_baseline_ultragcn is not None:
        cfg.train_baseline_ultragcn = bool(body.train_baseline_ultragcn)
    if body.train_baseline_sgl is not None:
        cfg.train_baseline_sgl = bool(body.train_baseline_sgl)
    if body.train_baseline_ncl is not None:
        cfg.train_baseline_ncl = bool(body.train_baseline_ncl)
    if body.train_baseline_xsimgcl is not None:
        cfg.train_baseline_xsimgcl = bool(body.train_baseline_xsimgcl)
    if body.hybrid_backbone is not None:
        cfg.hybrid_backbone = str(body.hybrid_backbone).strip().lower()

    if body.training_loss is not None:
        cfg.training_loss = body.training_loss
    if body.early_stopping is not None:
        cfg.early_stopping = body.early_stopping
    if body.early_stopping_patience is not None:
        cfg.early_stopping_patience = body.early_stopping_patience
    if body.early_stopping_min_delta is not None:
        cfg.early_stopping_min_delta = body.early_stopping_min_delta
    if body.early_stopping_monitor is not None:
        cfg.early_stopping_monitor = body.early_stopping_monitor
    if body.quantum_entangle is not None:
        cfg.quantum_entangle = body.quantum_entangle

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
    lg = ExperimentConfig.large().to_dict()
    xl = ExperimentConfig.extra_large().to_dict()
    cu = ExperimentConfig().to_dict()
    return {
        "lightweight": lw,
        "notebook": nb,
        "large": lg,
        "extra_large": xl,
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

    variants = _expand_run_variants(cfg, body)
    batch_parent = Path(variants[0].save_dir).parent if len(variants) > 1 else None

    def run(
        _job_id: str,
        on_progress: Callable[[Dict[str, Any]], None],
        cancel_ev,
    ) -> dict[str, Any]:
        from hybrid_qgnn.training import run_experiment

        results: List[Dict[str, Any]] = []
        for i, sub_cfg in enumerate(variants):
            if cancel_ev.is_set():
                break
            wp = _batch_progress_wrap(on_progress, i, len(variants))
            results.append(
                run_experiment(
                    sub_cfg,
                    project_root=settings.project_root,
                    on_progress=wp,
                    show_progress=False,
                    cancel_event=cancel_ev,
                )
            )
        if batch_parent is not None and len(results) > 0:
            batch_parent.mkdir(parents=True, exist_ok=True)
            summary = {
                "n_requested": len(variants),
                "n_completed": len(results),
                "children": [str(Path(r["save_dir"]).resolve()) for r in results],
            }
            (batch_parent / "batch_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
        if len(results) == 1:
            return results[0]
        return {
            "batch": True,
            "parent": str(batch_parent.resolve()) if batch_parent is not None else None,
            "results": results,
        }

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


_ALLOWED_JOB_PLOTS = frozenset({"training_dashboard.png"})
_ALLOWED_HISTORY_PLOTS = _ALLOWED_JOB_PLOTS


def _full_model_comparative_dataframe(run_dir: Path) -> pd.DataFrame:
    from hybrid_qgnn.analysis.comparative import build_full_model_comparative

    full_p = run_dir / "full_model_comparative.csv"
    if full_p.is_file():
        return pd.read_csv(full_p)
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
    return merged


@router.get("/jobs/{job_id}/live-metrics")
def job_live_metrics(job_id: str, settings: Settings = Depends(get_settings)):
    """Parse ``metrics.csv`` for the job’s run folder (updated every epoch during training)."""
    from hybrid_qgnn.training.plots import metrics_dataframe_to_live_payload

    j = get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    if not j.save_dir:
        return {"ready": False, "reason": "no_save_dir", **metrics_dataframe_to_live_payload(pd.DataFrame())}

    csv_p = (settings.project_root / j.save_dir / "metrics.csv").resolve()
    run_root = (settings.project_root / j.save_dir).resolve()
    if not str(csv_p).startswith(str(run_root)):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not csv_p.is_file():
        return {"ready": False, "reason": "no_csv_yet", **metrics_dataframe_to_live_payload(pd.DataFrame())}

    try:
        df = pd.read_csv(csv_p)
    except Exception:
        return {"ready": False, "reason": "csv_unreadable", **metrics_dataframe_to_live_payload(pd.DataFrame())}

    payload = metrics_dataframe_to_live_payload(df)
    return {"ready": True, **payload}


@router.get("/jobs/{job_id}/plots/{filename}")
def job_plot_file(job_id: str, filename: str, settings: Settings = Depends(get_settings)):
    if filename not in _ALLOWED_JOB_PLOTS:
        raise HTTPException(status_code=400, detail="Unsupported plot file")

    j = get_job(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    if not j.save_dir:
        raise HTTPException(status_code=404, detail="No artifacts for this job yet")

    plot_path = (settings.project_root / j.save_dir / "plots" / filename).resolve()
    run_root = (settings.project_root / j.save_dir).resolve()
    if not str(plot_path).startswith(str(run_root)) or not plot_path.is_file():
        raise HTTPException(status_code=404, detail="Plot not available yet")
    return FileResponse(plot_path, media_type="image/png", filename=filename)


@router.get("/history", response_model=List[RunSummary])
def history(settings: Settings = Depends(get_settings)):
    runs_dir = settings.project_root / "runs"
    return list_runs(runs_dir)


@router.post("/history/clear")
def clear_experiment_history(settings: Settings = Depends(get_settings)):
    """Delete everything under ``runs/`` and wipe the in-memory job list. Not allowed while a job is queued or running."""
    alive = [j for j in list_jobs() if j.status in (JobStatus.queued, JobStatus.running)]
    if alive:
        raise HTTPException(
            status_code=409,
            detail="Cannot clear history while a job is queued or running. Cancel it first.",
        )
    runs_dir = settings.project_root / "runs"
    removed_children = clear_runs_directory(runs_dir)
    removed_jobs = clear_all_job_records()
    return {
        "removed_run_children": removed_children,
        "removed_job_records": removed_jobs,
    }


@router.delete("/history/{run_id}")
def delete_history_run(run_id: str, settings: Settings = Depends(get_settings)):
    """Remove a single folder under ``runs/<run_id>``. Blocked if a queued/running job writes to that path."""
    try:
        run_dir = resolved_run_dir(settings.project_root, run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run id")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found")

    for j in list_jobs():
        if j.status not in (JobStatus.queued, JobStatus.running):
            continue
        if not j.save_dir:
            continue
        sd = j.save_dir.strip().replace("\\", "/")
        if sd.startswith("./"):
            sd = sd[2:]
        try:
            job_out = (settings.project_root / sd).resolve()
        except OSError:
            continue
        if job_output_overlaps_run_dir(job_out, run_dir):
            raise HTTPException(
                status_code=409,
                detail="Cannot delete this run while a job is writing to this folder.",
            )

    delete_run_directory(settings.project_root, run_id)
    return {"ok": True, "run_id": run_id}


@router.post("/history/{run_id}/score-pairs", response_model=ScorePairsResponse)
def score_pairs_on_run(run_id: str, body: ScorePairsBody, settings: Settings = Depends(get_settings)):
    """Forward-pass hybrid checkpoint on (user, item) pairs — uses saved training graph (graph_context.npz)."""
    import numpy as np

    from hybrid_qgnn.inference.scoring import parse_pairs_lines, score_hybrid_pairs

    try:
        if body.pairs_text is not None and str(body.pairs_text).strip():
            arr = parse_pairs_lines(body.pairs_text)
        else:
            arr = np.array(body.pairs, dtype=np.int64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if arr.shape[0] > 50_000:
        raise HTTPException(status_code=400, detail="At most 50_000 pairs per request")

    try:
        scores, meta = score_hybrid_pairs(
            settings.project_root,
            run_id,
            arr,
            micro_bs=body.micro_bs,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    pair_rows = [(int(a), int(b)) for a, b in arr.tolist()]
    return ScorePairsResponse(
        scores=scores,
        pairs=pair_rows,
        n_pairs=int(meta["n_pairs"]),
        run_id=str(meta["run_id"]),
        n_users=int(meta["n_users"]),
        n_items=int(meta["n_items"]),
        hybrid_backbone=str(meta["hybrid_backbone"]),
        graph_context=str(meta["graph_context"]),
    )


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
    run_dir = settings.project_root / "runs" / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")

    df = _full_model_comparative_dataframe(run_dir)
    dfo = df.astype(object).where(pd.notnull(df), None)
    return dfo.to_dict(orient="records")


@router.get("/history/{run_id}/download/metrics.csv")
def download_metrics(run_id: str, settings: Settings = Depends(get_settings)):
    run_dir = settings.project_root / "runs" / run_id
    csv_path = run_dir / "metrics.csv"
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail="metrics.csv not found")
    return FileResponse(csv_path, filename=f"{run_id}_metrics.csv", media_type="text/csv")


@router.get("/history/{run_id}/summary")
def run_summary_text(run_id: str, settings: Settings = Depends(get_settings)):
    run_dir = settings.project_root / "runs" / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    rel = f"runs/{run_id}/summary.txt"
    p = run_dir / "summary.txt"
    if not p.is_file():
        return {"text": None, "relative_path": rel}
    return {
        "text": p.read_text(encoding="utf-8", errors="replace"),
        "relative_path": rel,
    }


@router.get("/history/{run_id}/phase-timings")
def run_phase_timings(run_id: str, settings: Settings = Depends(get_settings)):
    run_dir = settings.project_root / "runs" / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    p = run_dir / "phase_timings.json"
    if not p.is_file():
        return {"phases": None, "relative_path": f"runs/{run_id}/phase_timings.json"}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"phases": None, "relative_path": f"runs/{run_id}/phase_timings.json"}
    if isinstance(data, list):
        return {"phases": data, "relative_path": f"runs/{run_id}/phase_timings.json"}
    return {"phases": data, "relative_path": f"runs/{run_id}/phase_timings.json"}


@router.get("/history/{run_id}/plots/{filename}")
def history_plot_file(run_id: str, filename: str, settings: Settings = Depends(get_settings)):
    if filename not in _ALLOWED_HISTORY_PLOTS:
        raise HTTPException(status_code=400, detail="Unsupported plot file")
    run_dir = (settings.project_root / "runs" / run_id).resolve()
    proj = settings.project_root.resolve()
    if not str(run_dir).startswith(str(proj)) or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    plot_path = (run_dir / "plots" / filename).resolve()
    if not str(plot_path).startswith(str(run_dir)) or not plot_path.is_file():
        raise HTTPException(status_code=404, detail="Plot not available")
    return FileResponse(plot_path, media_type="image/png", filename=filename)


@router.get("/history/{run_id}/download/summary.txt")
def download_summary_txt(run_id: str, settings: Settings = Depends(get_settings)):
    run_dir = settings.project_root / "runs" / run_id
    p = run_dir / "summary.txt"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="summary.txt not found")
    return FileResponse(p, filename=f"{run_id}_summary.txt", media_type="text/plain; charset=utf-8")


@router.get("/history/{run_id}/download/run_config.json")
def download_run_config_json(run_id: str, settings: Settings = Depends(get_settings)):
    run_dir = settings.project_root / "runs" / run_id
    p = run_dir / "run_config.json"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="run_config.json not found")
    return FileResponse(p, filename=f"{run_id}_run_config.json", media_type="application/json")


@router.get("/history/{run_id}/download/full_model_comparative.csv")
def download_full_model_comparative_csv(run_id: str, settings: Settings = Depends(get_settings)):
    run_dir = settings.project_root / "runs" / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    df = _full_model_comparative_dataframe(run_dir)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_full_model_comparative.csv"'},
    )
