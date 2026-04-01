"""Post-process metrics.csv into comparative tables (notebook parity)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def build_comparative_from_metrics_dir(save_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    metrics_csv = save_dir / "metrics.csv"
    if not metrics_csv.exists():
        return None, None
    df = pd.read_csv(metrics_csv)
    want_metrics = ["AUC", "MAE", "MSE", "RMSE", "MAPE", "WMAPE"]
    val = df[(df["split"] == "val") & (df["metric"].isin(want_metrics))].copy()
    if val.empty:
        return None, None

    best_rows = []
    for model, g in val.groupby("model"):
        auc_g = g[g["metric"] == "AUC"]
        if auc_g.empty:
            continue
        best_ep = auc_g.sort_values("value", ascending=False).iloc[0]["epoch"]
        snap = g[g["epoch"] == best_ep].pivot_table(index="model", columns="metric", values="value", aggfunc="first")
        snap["epoch_best_auc"] = best_ep
        best_rows.append(snap)
    if not best_rows:
        return None, None

    best_table = pd.concat(best_rows).reset_index()
    cols = ["model", "epoch_best_auc"] + want_metrics
    for c in cols:
        if c not in best_table.columns:
            best_table[c] = np.nan
    best_table = best_table[cols]

    disp = best_table.copy()
    disp["epoch_best_auc"] = pd.to_numeric(disp["epoch_best_auc"], errors="coerce").round(0)
    for c in want_metrics:
        disp[c] = disp[c].astype(float).round(6)

    models_present = set(disp["model"].unique())
    if {"LightGCN", "HybridQGNN"}.issubset(models_present):
        lg = disp[disp["model"] == "LightGCN"].iloc[0]
        hy = disp[disp["model"] == "HybridQGNN"].iloc[0]
        delta: Dict[str, Any] = {"model": "Δ (Hybrid − LightGCN)", "epoch_best_auc": np.nan}
        for c in want_metrics:
            delta[c] = float(hy[c]) - float(lg[c])
        delta_row = pd.DataFrame([delta])
        disp_comp = pd.concat([disp, delta_row], ignore_index=True)
    else:
        disp_comp = disp

    wide = (
        val.pivot_table(index=["model", "epoch"], columns="metric", values="value", aggfunc="first")
        .reset_index()
        .sort_values(["model", "epoch"])
    )
    return disp_comp, wide


def write_comparative_tables(save_dir: Path) -> Optional[List[Path]]:
    save_dir = Path(save_dir)
    disp_comp, wide = build_comparative_from_metrics_dir(save_dir)
    if disp_comp is None or wide is None:
        return None
    p1 = save_dir / "val_best_comparative.csv"
    p2 = save_dir / "val_metrics_per_epoch.csv"
    disp_comp.to_csv(p1, index=False)
    wide.to_csv(p2, index=False)
    return [p1, p2]
