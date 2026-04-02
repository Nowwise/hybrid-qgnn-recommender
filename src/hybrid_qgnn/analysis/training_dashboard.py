"""Matplotlib training dashboards (pandas + matplotlib only; no torch)."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Match dashboard / LiveTrainingMonitor accent colors where names align.
_MODEL_COLORS: Dict[str, str] = {
    "LightGCN": "#e8a838",
    "UltraGCN": "#38bdf8",
    "SGL": "#f472b6",
    "NCL": "#4ade80",
    "XSimGCL": "#a78bfa",
    "HybridQGNN": "#8b7cf8",
    "HybridQGNN (ablation classical head)": "#c4b5fd",
}


def _ranking_ks_in_df(df: pd.DataFrame) -> List[int]:
    if df.empty or "split" not in df.columns or "metric" not in df.columns:
        return []
    sub = df.loc[df["split"] == "val_ranking", "metric"]
    if sub.empty:
        return []
    ks: set[int] = set()
    for raw in sub.astype(str).unique():
        m = re.match(r"^Recall@(\d+)$", raw)
        if m:
            ks.add(int(m.group(1)))
    return sorted(ks)


def _pick_ranking_k(df: pd.DataFrame, prefer: int = 10) -> Optional[int]:
    ks = _ranking_ks_in_df(df)
    if not ks:
        return None
    if prefer in ks:
        return prefer
    return ks[len(ks) // 2]


def _series_val_ranking_at_k(df: pd.DataFrame, k: int, kind: str) -> List[Tuple[str, float]]:
    """kind: 'Recall' or 'NDCG' -> metric column Recall@k / NDCG@k."""
    metric = f"{kind}@{k}"
    sub = df[(df["split"] == "val_ranking") & (df["metric"] == metric)].copy()
    if sub.empty:
        return []
    out: List[Tuple[str, float]] = []
    for _, r in sub.iterrows():
        v = _json_float(r["value"])
        if v is None:
            continue
        out.append((str(r["model"]), float(v)))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


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
    Build the 2×3 overview figure: validation AUC, train loss, sampled Recall@K / NDCG@K (val_ranking),
    p_quantum schedule, and text summary. Ranking bars appear when ``metrics.csv`` contains ``split=val_ranking``.
    Does not save or close the figure — use in notebooks with ``plt.show()`` or save manually.
    """
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig.patch.set_facecolor("#0c0d10")
    for ax in axes.flat:
        ax.set_facecolor("#12141a")

    auc_data = _series_val_auc(df)
    rk = _pick_ranking_k(df, prefer=10)
    recall_series = _series_val_ranking_at_k(df, rk, "Recall") if rk is not None else []
    ndcg_series = _series_val_ranking_at_k(df, rk, "NDCG") if rk is not None else []

    ax = axes[0, 0]
    for name, pts in auc_data.items():
        if not pts:
            continue
        ep, y = zip(*pts)
        ax.plot(ep, y, "o-", label=name, color=_MODEL_COLORS.get(name, "#94a3b8"), lw=2, ms=4)
    ax.set_title("Validation AUC", fontsize=11, color="#e2e8f0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend(framealpha=0.3, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)

    ax = axes[0, 1]
    loss_data = _series_train_loss(df)
    for name, pts in loss_data.items():
        if not pts:
            continue
        ep, y = zip(*pts)
        ax.plot(ep, y, "s-", label=name, color=_MODEL_COLORS.get(name, "#94a3b8"), lw=2, ms=3)
    ax.set_title("Training loss (BCE / BPR)", fontsize=11, color="#e2e8f0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(framealpha=0.3, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    ax = axes[0, 2]
    if recall_series:
        names_r, vals_r = zip(*recall_series)
        y_pos = np.arange(len(names_r))
        ax.barh(
            y_pos,
            vals_r,
            color=[_MODEL_COLORS.get(n, "#94a3b8") for n in names_r],
            height=0.65,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names_r, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Recall")
        ax.set_title(f"Val Recall@{rk} (sampled)", fontsize=11, color="#e2e8f0")
        ax.grid(True, axis="x", alpha=0.2)
    else:
        ax.text(0.5, 0.5, "No val_ranking rows\n(enable eval_ranking)", ha="center", va="center", color="#64748b")
        ax.set_axis_off()

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
    if ndcg_series:
        names_n, vals_n = zip(*ndcg_series)
        y_pos = np.arange(len(names_n))
        ax.barh(
            y_pos,
            vals_n,
            color=[_MODEL_COLORS.get(n, "#94a3b8") for n in names_n],
            height=0.65,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names_n, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("NDCG")
        ax.set_title(f"Val NDCG@{rk} (sampled)", fontsize=11, color="#e2e8f0")
        ax.grid(True, axis="x", alpha=0.2)
    else:
        ax.text(0.5, 0.5, "No val_ranking rows\n(enable eval_ranking)", ha="center", va="center", color="#64748b")
        ax.set_axis_off()

    ax = axes[1, 2]
    ax.axis("off")
    lines = []
    for name in sorted(auc_data.keys()):
        pts = auc_data[name]
        if not pts:
            continue
        best_ep, best_v = max(pts, key=lambda t: t[1])
        lines.append(f"{name}: best val AUC {best_v:.4f} @ ep {best_ep}")
    if not lines:
        lines.append("No validation AUC rows yet.")
    if rk is not None and recall_series:
        top_r = recall_series[0]
        lines.append(f"Best Recall@{rk} (sampled): {top_r[0]} {top_r[1]:.4f}")
    if rk is not None and ndcg_series:
        top_n = ndcg_series[0]
        lines.append(f"Best NDCG@{rk} (sampled): {top_n[0]} {top_n[1]:.4f}")
    ax.text(
        0.05,
        0.92,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        color="#cbd5e1",
        fontfamily="monospace",
    )
    ax.text(
        0.05,
        0.28,
        "Artifacts:\n  metrics.csv\n  plots/training_dashboard.png",
        transform=ax.transAxes,
        fontsize=9,
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
