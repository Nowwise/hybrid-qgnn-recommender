from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from hybrid_qgnn.inference.scoring import GRAPH_CONTEXT_FILENAME

from app.schemas.experiment import RunSummary


def _parse_summary_auc(text: str) -> tuple[Optional[float], Optional[float]]:
    lg = hy = None
    m1 = re.search(r"LightGCN best val AUC:\s*([\d.]+)", text)
    m2 = re.search(r"HybridQGNN best val AUC:\s*([\d.]+)", text)
    if m1:
        lg = float(m1.group(1))
    if m2:
        hy = float(m2.group(1))
    return lg, hy


def _count_baseline_best_checkpoints(run_dir: Path) -> int:
    n = 0
    for p in run_dir.glob("*_best.pt"):
        if p.name == "hyb_best.pt":
            continue
        n += 1
    return n


def list_runs(runs_root: Path) -> List[RunSummary]:
    if not runs_root.is_dir():
        return []
    out: List[RunSummary] = []
    for child in sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        summary_p = child / "summary.txt"
        metrics_p = child / "metrics.csv"
        cfg_p = child / "run_config.json"
        lg = hy = None
        exp_name = None
        if summary_p.is_file():
            lg, hy = _parse_summary_auc(summary_p.read_text())
        data_dir = hybrid_bb = None
        q_v = L_v = d_v = K_v = el = eh = None
        if cfg_p.is_file():
            try:
                data = json.loads(cfg_p.read_text())
                raw = data.get("experiment_name")
                if isinstance(raw, str) and raw.strip():
                    exp_name = raw.strip()
                if isinstance(data.get("data_dir"), str):
                    data_dir = data["data_dir"].strip() or None
                if isinstance(data.get("hybrid_backbone"), str):
                    hybrid_bb = data["hybrid_backbone"].strip().lower() or None
                if isinstance(data.get("q"), int):
                    q_v = data["q"]
                if isinstance(data.get("L"), int):
                    L_v = data["L"]
                if isinstance(data.get("d"), int):
                    d_v = data["d"]
                if isinstance(data.get("K"), int):
                    K_v = data["K"]
                if isinstance(data.get("epochs_lg"), int):
                    el = data["epochs_lg"]
                if isinstance(data.get("epochs_hyb"), int):
                    eh = data["epochs_hyb"]
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                pass
        try:
            mtime = datetime.fromtimestamp(child.stat().st_mtime, tz=timezone.utc)
        except OSError:
            mtime = None
        has_hyb = (child / "hyb_best.pt").is_file()
        n_bl = _count_baseline_best_checkpoints(child)
        has_gc = (child / GRAPH_CONTEXT_FILENAME).is_file()
        out.append(
            RunSummary(
                run_id=child.name,
                path=str(child),
                has_metrics=metrics_p.is_file(),
                has_summary=summary_p.is_file(),
                experiment_name=exp_name,
                best_lightgcn_auc=lg,
                best_hybrid_auc=hy,
                data_dir=data_dir,
                hybrid_backbone=hybrid_bb,
                q=q_v,
                L=L_v,
                d=d_v,
                K=K_v,
                epochs_lg=el,
                epochs_hyb=eh,
                has_hybrid_checkpoint=has_hyb,
                n_baseline_checkpoints=n_bl,
                modified_at=mtime,
                has_graph_context=has_gc,
            )
        )
    return out


def clear_runs_directory(runs_root: Path) -> int:
    """Delete every direct child under ``runs_root`` (run folders and stray files). Returns count removed."""
    root = runs_root.resolve()
    if not root.is_dir():
        return 0
    n = 0
    for child in list(root.iterdir()):
        try:
            cr = child.resolve()
        except OSError:
            continue
        try:
            if cr.parent.resolve() != root:
                continue
        except OSError:
            continue
        if cr.is_dir():
            shutil.rmtree(cr, ignore_errors=True)
            n += 1
        elif cr.is_file():
            if cr.name in (".gitkeep",):
                continue
            try:
                cr.unlink()
                n += 1
            except OSError:
                pass
    return n


def validate_run_id_segment(run_id: str) -> None:
    """Ensure ``run_id`` is a single relative path segment (no traversal). Raises ValueError."""
    if not run_id or "\x00" in run_id or len(run_id) > 512:
        raise ValueError("invalid run_id")
    p = Path(run_id)
    if p.is_absolute() or ".." in p.parts or len(p.parts) != 1:
        raise ValueError("invalid run_id")


def resolved_run_dir(project_root: Path, run_id: str) -> Path:
    """Return resolved ``project_root/runs/<run_id>`` if it exists and is a direct child of ``runs``."""
    validate_run_id_segment(run_id)
    root = (project_root / "runs").resolve()
    if not root.is_dir():
        raise FileNotFoundError("runs directory missing")
    target = (root / run_id).resolve()
    try:
        if target.parent.resolve() != root:
            raise ValueError("invalid run path")
    except OSError as e:
        raise ValueError("invalid run path") from e
    if not target.is_dir():
        raise FileNotFoundError("run not found")
    return target


def delete_run_directory(project_root: Path, run_id: str) -> None:
    """Remove ``project_root/runs/<run_id>``. Raises FileNotFoundError, ValueError, OSError."""
    target = resolved_run_dir(project_root, run_id)
    shutil.rmtree(target)


def job_output_overlaps_run_dir(job_save_resolved: Path, run_dir_resolved: Path) -> bool:
    """True if the job's output path equals the run folder or lies inside it (active write)."""
    if job_save_resolved == run_dir_resolved:
        return True
    try:
        job_save_resolved.relative_to(run_dir_resolved)
        return True
    except ValueError:
        return False


def read_run_config(run_dir: Path) -> Optional[dict]:
    p = run_dir / "run_config.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())
