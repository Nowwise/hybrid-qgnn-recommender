"""Matplotlib training dashboards (pandas + matplotlib only; no torch)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _series_val_auc(df: pd.DataFrame) -> Dict[str, List[Tuple[int, float]]]:
    out: Dict[str, List[Tuple[int, float]]] = {}
    sub = df[(df["split"] == "val") & (df["metric"] == "AUC")].copy()
    if sub.empty:
        return out
    for model, g in sub.groupby("model"):
        g2 = g.sort_values("epoch")
        pts = list(zip(g2["epoch"].astype(int).tolist(), g2["value"].astype(float).tolist()))
        out[str(model)] = pts
    return out


def _series_train_loss(df: pd.DataFrame) -> Dict[str, List[Tuple[int, float]]]:
    out: Dict[str, List[Tuple[int, float]]] = {}
    sub = df[(df["split"] == "train") & (df["metric"].isin(["BCE", "BPR"]))].copy()
    if sub.empty:
        return out
    for model, g in sub.groupby("model"):
        g2 = g.sort_values("epoch")
        pts = list(zip(g2["epoch"].astype(int).tolist(), g2["value"].astype(float).tolist()))
        out[str(model)] = pts
    return out


def _series_p_quantum(df: pd.DataFrame) -> List[Tuple[int, float]]:
    sub = df[(df["split"] == "meta") & (df["metric"] == "p_quantum") & (df["model"] == "HybridQGNN")].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("epoch")
    return list(zip(sub["epoch"].astype(int).tolist(), sub["value"].astype(float).tolist()))


def _series_val_by_metric(df: pd.DataFrame, metric: str) -> Dict[str, List[Tuple[int, float]]]:
    out: Dict[str, List[Tuple[int, float]]] = {}
    sub = df[(df["split"] == "val") & (df["metric"] == metric)].copy()
    if sub.empty:
        return out
    for model, g in sub.groupby("model"):
        g2 = g.sort_values("epoch")
        pts = list(zip(g2["epoch"].astype(int).tolist(), g2["value"].astype(float).tolist()))
        out[str(model)] = pts
    return out


def _json_float(v: Any) -> Any:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _points_json(pts: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for a, b in pts:
        jv = _json_float(b)
        if jv is None:
            continue
        out.append({"epoch": int(a), "value": jv})
    return out


def _latest_val_by_model(df: pd.DataFrame) -> Dict[str, Any]:
    from hybrid_qgnn.analysis.comparative import VAL_CLASS_METRICS

    val = df[(df["split"] == "val") & (df["metric"].isin(VAL_CLASS_METRICS))].copy()
    if val.empty:
        return {}
    out: Dict[str, Any] = {}
    for model, g in val.groupby("model"):
        ep_max = int(g["epoch"].max())
        snap = g[g["epoch"] == ep_max]
        metrics: Dict[str, Any] = {}
        for _, r in snap.iterrows():
            mname = str(r["metric"])
            metrics[mname] = _json_float(r["value"])
        out[str(model)] = {"epoch": ep_max, "metrics": metrics}
    return out


def _best_val_auc_by_model(df: pd.DataFrame) -> Dict[str, Any]:
    sub = df[(df["split"] == "val") & (df["metric"] == "AUC")].copy()
    if sub.empty:
        return {}
    out: Dict[str, Any] = {}
    for model, g in sub.groupby("model"):
        g2 = g.sort_values("epoch").dropna(subset=["value"])
        if g2.empty:
            continue
        idx = g2["value"].idxmax()
        row = g2.loc[idx]
        auc = _json_float(row["value"])
        if auc is None:
            continue
        out[str(model)] = {"epoch": int(row["epoch"]), "auc": auc}
    return out


def _ranking_rows_for_split(df: pd.DataFrame, split: str) -> List[Dict[str, Any]]:
    sub = df[df["split"] == split].copy()
    if sub.empty:
        return []
    sub = sub.sort_values(["model", "metric"])
    rows: List[Dict[str, Any]] = []
    for _, r in sub.iterrows():
        rows.append(
            {
                "model": str(r["model"]),
                "metric": str(r["metric"]),
                "value": _json_float(r["value"]),
            }
        )
    return rows


def metrics_dataframe_to_live_payload(df: pd.DataFrame) -> Dict[str, Any]:
    """Structure for `/live-metrics` and for quick checks."""
    return {
        "val_auc": {k: _points_json(v) for k, v in _series_val_auc(df).items()},
        "val_rmse": {k: _points_json(v) for k, v in _series_val_by_metric(df, "RMSE").items()},
        "val_mae": {k: _points_json(v) for k, v in _series_val_by_metric(df, "MAE").items()},
        "train_loss": {k: _points_json(v) for k, v in _series_train_loss(df).items()},
        "p_quantum": _points_json(_series_p_quantum(df)),
        "latest_val": _latest_val_by_model(df),
        "best_val_auc": _best_val_auc_by_model(df),
        "ranking_val": _ranking_rows_for_split(df, "val_ranking"),
        "ranking_test": _ranking_rows_for_split(df, "test_ranking"),
        "row_count": int(len(df)),
    }


def build_training_dashboard_figure(
    df: pd.DataFrame,
    *,
    title: str = "Training dashboard",
    suptitle_suffix: str = "",
) -> "Figure":
    """
    Build the 2×2 overview figure (validation AUC, train loss, p_quantum, summary).
    Does not save or close the figure — use in notebooks with ``plt.show()`` or save manually.
    """
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.patch.set_facecolor("#0c0d10")
    for ax in axes.flat:
        ax.set_facecolor("#12141a")

    colors = {"LightGCN": "#e8a838", "HybridQGNN": "#8b7cf8"}
    auc_data = _series_val_auc(df)

    ax = axes[0, 0]
    for name, pts in auc_data.items():
        if not pts:
            continue
        ep, y = zip(*pts)
        ax.plot(ep, y, "o-", label=name, color=colors.get(name, "#94a3b8"), lw=2, ms=4)
    ax.set_title("Validation AUC", fontsize=11, color="#e2e8f0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.2)

    ax = axes[0, 1]
    loss_data = _series_train_loss(df)
    for name, pts in loss_data.items():
        if not pts:
            continue
        ep, y = zip(*pts)
        ax.plot(ep, y, "s-", label=name, color=colors.get(name, "#94a3b8"), lw=2, ms=3)
    ax.set_title("Training loss (BCE / BPR)", fontsize=11, color="#e2e8f0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.2)

    ax = axes[1, 0]
    pq = _series_p_quantum(df)
    if pq:
        ep, y = zip(*pq)
        ax.plot(ep, y, "^-", color="#34d399", lw=2, ms=5)
    ax.set_title("Hybrid · p_quantum schedule", fontsize=11, color="#e2e8f0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("p_quantum")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    ax.axis("off")
    lines = []
    for name in ["LightGCN", "HybridQGNN"]:
        if name not in auc_data or not auc_data[name]:
            continue
        best_ep, best_v = max(auc_data[name], key=lambda t: t[1])
        lines.append(f"{name}: best val AUC {best_v:.4f} @ ep {best_ep}")
    if not lines:
        lines.append("No validation AUC rows yet.")
    ax.text(
        0.05,
        0.85,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        color="#cbd5e1",
        fontfamily="monospace",
    )
    ax.text(
        0.05,
        0.35,
        "Artifacts:\n  metrics.csv\n  plots/training_dashboard.png",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        color="#64748b",
        fontfamily="monospace",
    )

    st = title if not suptitle_suffix else f"{title} {suptitle_suffix}"
    fig.suptitle(st, fontsize=13, color="#f8fafc", y=1.02)
    return fig


def refresh_training_plots(save_dir: Path | str) -> Path | None:
    """
    Write ``plots/training_dashboard.png``. Returns path if written, else None.
    Safe to call on empty / missing CSV.
    """
    root = Path(save_dir)
    csv_path = root / "metrics.csv"
    if not csv_path.is_file():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty or "metric" not in df.columns:
        return None

    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "training_dashboard.png"

    fig = build_training_dashboard_figure(df, title="Training dashboard", suptitle_suffix="(live)")
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
