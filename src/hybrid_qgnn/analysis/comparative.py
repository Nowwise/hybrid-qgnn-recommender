"""Post-process metrics.csv into comparative tables (notebook parity)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

VAL_CLASS_METRICS = ["AUC", "MAE", "MSE", "RMSE", "MAPE", "WMAPE"]


def _val_best_snapshot(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """One row per model: best val epoch by AUC with all classification metrics at that epoch."""
    val = df[(df["split"] == "val") & (df["metric"].isin(VAL_CLASS_METRICS))].copy()
    if val.empty:
        return None

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
        return None

    best_table = pd.concat(best_rows).reset_index()
    cols = ["model", "epoch_best_auc"] + VAL_CLASS_METRICS
    for c in cols:
        if c not in best_table.columns:
            best_table[c] = np.nan
    best_table = best_table[cols]

    out = best_table.copy()
    out["epoch_best_auc"] = pd.to_numeric(out["epoch_best_auc"], errors="coerce").round(0)
    for c in VAL_CLASS_METRICS:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(6)
    return out


def _add_delta_hybrid_minus_lightgcn(disp: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    models_present = set(disp["model"].astype(str).unique())
    if not {"LightGCN", "HybridQGNN"}.issubset(models_present):
        return disp
    lg = disp[disp["model"] == "LightGCN"].iloc[0]
    hy = disp[disp["model"] == "HybridQGNN"].iloc[0]
    delta: Dict[str, Any] = {"model": "Δ (Hybrid − LightGCN)"}
    for c in metric_cols:
        if c == "epoch_best_auc":
            delta[c] = np.nan
            continue
        if c not in disp.columns:
            delta[c] = np.nan
            continue
        try:
            v_lg = float(lg[c]) if pd.notna(lg[c]) else np.nan
            v_hy = float(hy[c]) if pd.notna(hy[c]) else np.nan
            delta[c] = v_hy - v_lg if pd.notna(v_lg) and pd.notna(v_hy) else np.nan
        except (TypeError, ValueError):
            delta[c] = np.nan
    return pd.concat([disp, pd.DataFrame([delta])], ignore_index=True)


def _ranking_pivot_prefixed(df: pd.DataFrame, split: str, prefix: str) -> Optional[pd.DataFrame]:
    sub = df[df["split"] == split].copy()
    if sub.empty:
        return None
    w = sub.pivot_table(index="model", columns="metric", values="value", aggfunc="first")
    w = w.rename(columns={c: f"{prefix}{c}" for c in w.columns})
    out = w.reset_index()
    for c in out.columns:
        if c == "model":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").round(6)
    return out


def build_full_model_comparative(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Single wide table: val@best-epoch classification metrics + val_rank_* + test_rank_* + Δ row.
    Works for partial data (e.g. ranking-only if val classification rows are missing).
    """
    if df.empty:
        return None

    base = _val_best_snapshot(df)
    rv = _ranking_pivot_prefixed(df, "val_ranking", "val_rank_")
    rt = _ranking_pivot_prefixed(df, "test_ranking", "test_rank_")

    parts = [p for p in (base, rv, rt) if p is not None]
    if not parts:
        return None

    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p, on="model", how="outer")

    order_pref = [
        "LightGCN",
        "HybridQGNN",
        "HybridQGNN (ablation classical head)",
    ]
    mstr = merged["model"].astype(str)
    merged["_ord"] = mstr.map(lambda m: order_pref.index(m) if m in order_pref else 1000)
    merged["_name"] = mstr
    merged = merged.sort_values(["_ord", "_name"]).drop(columns=["_ord", "_name"])

    metric_cols = [c for c in merged.columns if c != "model"]
    merged = _add_delta_hybrid_minus_lightgcn(merged, metric_cols)

    return merged


def build_comparative_from_metrics_dir(save_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    metrics_csv = save_dir / "metrics.csv"
    if not metrics_csv.exists():
        return None, None
    df = pd.read_csv(metrics_csv)
    disp = _val_best_snapshot(df)
    if disp is None:
        return None, None

    val = df[(df["split"] == "val") & (df["metric"].isin(VAL_CLASS_METRICS))].copy()
    disp_comp = _add_delta_hybrid_minus_lightgcn(disp.copy(), VAL_CLASS_METRICS + ["epoch_best_auc"])

    wide = (
        val.pivot_table(index=["model", "epoch"], columns="metric", values="value", aggfunc="first")
        .reset_index()
        .sort_values(["model", "epoch"])
    )
    return disp_comp, wide


def write_ranking_comparative(save_dir: Path) -> Optional[Path]:
    """Pivot val_ranking / test_ranking rows from metrics.csv into a wide table."""
    save_dir = Path(save_dir)
    metrics_csv = save_dir / "metrics.csv"
    if not metrics_csv.exists():
        return None
    df = pd.read_csv(metrics_csv)
    sub = df[df["split"].isin(("val_ranking", "test_ranking"))].copy()
    if sub.empty:
        return None
    wide = sub.pivot_table(index=["model", "split"], columns="metric", values="value", aggfunc="first")
    wide = wide.reset_index()
    p = save_dir / "ranking_comparative.csv"
    wide.to_csv(p, index=False)
    return p


def write_full_model_comparative(save_dir: Path) -> Optional[Path]:
    save_dir = Path(save_dir)
    metrics_csv = save_dir / "metrics.csv"
    if not metrics_csv.is_file():
        return None
    df = pd.read_csv(metrics_csv)
    full = build_full_model_comparative(df)
    if full is None or full.empty:
        return None
    p = save_dir / "full_model_comparative.csv"
    full.to_csv(p, index=False)
    return p


def write_comparative_tables(save_dir: Path) -> Optional[List[Path]]:
    save_dir = Path(save_dir)
    disp_comp, wide = build_comparative_from_metrics_dir(save_dir)
    out: List[Path] = []
    if disp_comp is not None and wide is not None:
        p1 = save_dir / "val_best_comparative.csv"
        p2 = save_dir / "val_metrics_per_epoch.csv"
        disp_comp.to_csv(p1, index=False)
        wide.to_csv(p2, index=False)
        out.extend([p1, p2])
    rk = write_ranking_comparative(save_dir)
    if rk is not None:
        out.append(rk)
    full_p = write_full_model_comparative(save_dir)
    if full_p is not None:
        out.append(full_p)
    return out if out else None
