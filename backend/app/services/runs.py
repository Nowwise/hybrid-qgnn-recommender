from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

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


def list_runs(runs_root: Path) -> List[RunSummary]:
    if not runs_root.is_dir():
        return []
    out: List[RunSummary] = []
    for child in sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        summary_p = child / "summary.txt"
        metrics_p = child / "metrics.csv"
        lg = hy = None
        if summary_p.is_file():
            lg, hy = _parse_summary_auc(summary_p.read_text())
        out.append(
            RunSummary(
                run_id=child.name,
                path=str(child),
                has_metrics=metrics_p.is_file(),
                has_summary=summary_p.is_file(),
                best_lightgcn_auc=lg,
                best_hybrid_auc=hy,
            )
        )
    return out


def read_run_config(run_dir: Path) -> Optional[dict]:
    p = run_dir / "run_config.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())
