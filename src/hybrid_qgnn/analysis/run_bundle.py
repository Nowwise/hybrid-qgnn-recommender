"""
Load completed (or in-progress) experiment folders for thesis / notebook analysis.

Typical notebook usage::

    from pathlib import Path
    from hybrid_qgnn.analysis.run_bundle import (
        default_project_root,
        discover_runs,
        resolve_run_dir,
        load_thesis_run_bundle,
    )

    PROJECT_ROOT = default_project_root()
    RUN_EXPERIMENT_NAME = "my amazon baseline"  # matches experiment_name in run_config.json

    run_dir = resolve_run_dir(PROJECT_ROOT, experiment_name=RUN_EXPERIMENT_NAME)
    bundle = load_thesis_run_bundle(run_dir)
    display(bundle.full_comparative)
    bundle.figure_training_dashboard().show()
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from hybrid_qgnn.analysis.comparative import (
    build_comparative_from_metrics_dir,
    build_full_model_comparative,
    write_comparative_tables,
)


def default_project_root() -> Path:
    """Environment ``QGNN_PROJECT_ROOT``, else cwd."""
    raw = os.environ.get("QGNN_PROJECT_ROOT")
    if raw:
        return Path(raw).resolve()
    return Path.cwd().resolve()


def _iter_run_directories(runs_root: Path) -> List[Path]:
    """``runs/<id>/`` and one batch level ``runs/<batch>/<run_id>/`` with ``run_config.json``."""
    out: List[Path] = []
    if not runs_root.is_dir():
        return out
    for child in sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        if (child / "run_config.json").is_file():
            out.append(child)
            continue
        try:
            for sub in sorted(child.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if sub.is_dir() and (sub / "run_config.json").is_file():
                    out.append(sub)
        except OSError:
            continue
    return out


def discover_runs(project_root: Path | str | None = None) -> pd.DataFrame:
    """
    Table of runs under ``<project>/runs/`` with folder name, experiment_name, and path.
    """
    root = Path(project_root).resolve() if project_root else default_project_root()
    runs_root = root / "runs"
    rows: List[Dict[str, Any]] = []
    for d in _iter_run_directories(runs_root):
        cfg_path = d / "run_config.json"
        name = ""
        cfg: Dict[str, Any] = {}
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                raw = cfg.get("experiment_name")
                if isinstance(raw, str) and raw.strip():
                    name = raw.strip()
            except (json.JSONDecodeError, OSError):
                cfg = {}
        try:
            rel = d.relative_to(root)
        except ValueError:
            rel = d
        rows.append(
            {
                "run_folder": d.name,
                "path_posix": str(rel).replace("\\", "/"),
                "experiment_name": name,
                "save_dir_config": cfg.get("save_dir", ""),
                "mtime": d.stat().st_mtime,
                "abs_path": str(d.resolve()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["run_folder", "path_posix", "experiment_name", "save_dir_config", "mtime", "abs_path"]
        )
    return pd.DataFrame(rows).sort_values("mtime", ascending=False).reset_index(drop=True)


def resolve_run_dir(
    project_root: Path | str | None = None,
    *,
    experiment_name: Optional[str] = None,
    run_folder: Optional[str] = None,
    match: Literal["exact", "contains", "folder"] = "contains",
    prefer_latest: bool = True,
) -> Path:
    """
    Resolve a single run directory.

    Parameters
    ----------
    experiment_name
        Matched against ``experiment_name`` in ``run_config.json`` (case-insensitive for
        ``exact``/``contains``).
    run_folder
        Folder name under ``runs/`` (e.g. ``my-exp_20250401_120000``) or ``runs/child`` for nested batch runs.
    match
        ``folder``: only ``run_folder`` is used. ``exact``: full string equality on experiment_name.
        ``contains``: experiment_name must contain the query string (after strip).
    prefer_latest
        If several runs match, return the one with the newest directory mtime.

    Raises
    ------
    FileNotFoundError
        If no run matches or paths are missing.
    ValueError
        If both or neither selectors are ambiguous / missing.
    """
    root = Path(project_root).resolve() if project_root else default_project_root()
    runs_root = root / "runs"

    if run_folder and experiment_name:
        raise ValueError("Pass only one of experiment_name or run_folder.")

    if run_folder:
        # allow "runs/foo" or "foo" or "foo/bar"
        raw = run_folder.strip().replace("\\", "/")
        if raw.startswith("runs/"):
            raw = raw[5:]
        cand = (runs_root / raw).resolve()
        if not cand.is_dir() or not (cand / "run_config.json").is_file():
            raise FileNotFoundError(f"No run with run_config.json at: {cand}")
        return cand

    if not experiment_name or not str(experiment_name).strip():
        raise ValueError("Provide experiment_name=... or run_folder=...")

    query = str(experiment_name).strip()
    df = discover_runs(root)
    if df.empty:
        raise FileNotFoundError(f"No runs found under {runs_root}")

    names = df["experiment_name"].fillna("").astype(str)
    qlow = query.lower()

    if match == "exact":
        mask = names.str.lower() == qlow
    else:
        mask = names.str.lower().str.contains(qlow, regex=False)

    hit = df[mask]
    if hit.empty:
        avail = df[df["experiment_name"].astype(str).str.len() > 0][["experiment_name", "path_posix"]].head(25)
        raise FileNotFoundError(
            f"No run matched experiment_name={query!r} (match={match}).\n"
            f"Recent named runs:\n{avail.to_string(index=False)}"
        )

    if len(hit) > 1 and prefer_latest:
        hit = hit.sort_values("mtime", ascending=False).head(1)

    if len(hit) > 1:
        raise ValueError(
            f"Multiple runs matched {query!r}: use a longer name, set prefer_latest=True, "
            f"or pass run_folder= explicitly.\n{hit[['experiment_name', 'path_posix']].to_string(index=False)}"
        )

    path = Path(hit.iloc[0]["abs_path"]).resolve()
    return path


@dataclass
class ThesisRunBundle:
    """Everything useful for thesis tables / figures from one ``runs/...`` directory."""

    project_root: Path
    run_dir: Path
    config: Dict[str, Any]
    metrics: pd.DataFrame
    summary_text: str
    full_comparative: Optional[pd.DataFrame]
    val_best_comparative: Optional[pd.DataFrame]
    val_metrics_per_epoch: Optional[pd.DataFrame]
    ranking_wide: Optional[pd.DataFrame]
    phase_timings: Optional[pd.DataFrame]
    paths: Dict[str, Optional[Path]] = field(default_factory=dict)

    @property
    def experiment_name(self) -> Optional[str]:
        raw = self.config.get("experiment_name")
        return raw.strip() if isinstance(raw, str) and raw.strip() else None

    def rebuild_export_csvs(self) -> Optional[List[Path]]:
        """Regenerate val_best / full_model / ranking CSVs from ``metrics.csv``."""
        return write_comparative_tables(self.run_dir)

    def figure_training_dashboard(self):
        """Matplotlib figure (2×2); show in notebook with ``plt.show()`` or save with ``savefig``."""
        from hybrid_qgnn.analysis.training_dashboard import build_training_dashboard_figure

        if self.metrics.empty:
            fig = build_training_dashboard_figure(pd.DataFrame())
        else:
            fig = build_training_dashboard_figure(self.metrics, title="Training dashboard", suptitle_suffix="")
        return fig

    def display_dashboard_png(self) -> None:
        """If ``plots/training_dashboard.png`` exists, display in Jupyter / IPython."""
        p = self.paths.get("training_dashboard_png")
        if p is None or not p.is_file():
            print("No plots/training_dashboard.png (run with live_plots or generate from metrics).")
            return
        try:
            from IPython.display import Image, display

            display(Image(filename=str(p)))
        except ImportError:
            print(f"Image on disk: {p}")

    def ranking_metrics_table(self) -> Optional[pd.DataFrame]:
        """Long format: val_ranking / test_ranking rows pivoted for readability."""
        sub = self.metrics[self.metrics["split"].isin(("val_ranking", "test_ranking"))].copy()
        if sub.empty:
            return None
        return sub.sort_values(["split", "model", "metric"]).reset_index(drop=True)


def load_thesis_run_bundle(run_dir: Path | str, project_root: Path | str | None = None) -> ThesisRunBundle:
    """
    Load config, metrics, comparative tables, timings, and artifact paths from a run directory.
    """
    run_dir = Path(run_dir).resolve()
    root = Path(project_root).resolve() if project_root else default_project_root()

    cfg_path = run_dir / "run_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing run_config.json: {run_dir}")

    config = json.loads(cfg_path.read_text(encoding="utf-8"))

    metrics_path = run_dir / "metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()

    summary_path = run_dir / "summary.txt"
    summary_text = summary_path.read_text(encoding="utf-8") if summary_path.is_file() else ""

    full_c = build_full_model_comparative(metrics) if not metrics.empty else None
    disp_comp, wide_ep = build_comparative_from_metrics_dir(run_dir)

    pt_path = run_dir / "phase_timings.json"
    phase_df: Optional[pd.DataFrame] = None
    if pt_path.is_file():
        try:
            phase_df = pd.read_json(pt_path)
        except (ValueError, OSError):
            phase_df = None

    rk_path = run_dir / "ranking_comparative.csv"
    ranking_wide: Optional[pd.DataFrame] = None
    if rk_path.is_file():
        ranking_wide = pd.read_csv(rk_path)

    paths: Dict[str, Optional[Path]] = {
        "run_config": cfg_path,
        "metrics_csv": metrics_path if metrics_path.is_file() else None,
        "summary_txt": summary_path if summary_path.is_file() else None,
        "full_model_comparative_csv": run_dir / "full_model_comparative.csv"
        if (run_dir / "full_model_comparative.csv").is_file()
        else None,
        "val_best_comparative_csv": run_dir / "val_best_comparative.csv"
        if (run_dir / "val_best_comparative.csv").is_file()
        else None,
        "val_metrics_per_epoch_csv": run_dir / "val_metrics_per_epoch.csv"
        if (run_dir / "val_metrics_per_epoch.csv").is_file()
        else None,
        "ranking_comparative_csv": rk_path if rk_path.is_file() else None,
        "phase_timings_json": pt_path if pt_path.is_file() else None,
        "training_dashboard_png": run_dir / "plots" / "training_dashboard.png"
        if (run_dir / "plots" / "training_dashboard.png").is_file()
        else None,
        "lg_best_pt": run_dir / "lg_best.pt" if (run_dir / "lg_best.pt").is_file() else None,
        "hyb_best_pt": run_dir / "hyb_best.pt" if (run_dir / "hyb_best.pt").is_file() else None,
        "batch_summary_json": (run_dir.parent / "batch_summary.json")
        if (run_dir.parent / "batch_summary.json").is_file()
        else None,
    }

    return ThesisRunBundle(
        project_root=root,
        run_dir=run_dir,
        config=config,
        metrics=metrics,
        summary_text=summary_text,
        full_comparative=full_c,
        val_best_comparative=disp_comp,
        val_metrics_per_epoch=wide_ep,
        ranking_wide=ranking_wide,
        phase_timings=phase_df,
        paths=paths,
    )


def load_bundle_by_experiment_name(
    experiment_name: str,
    project_root: Path | str | None = None,
    **kwargs: Any,
) -> ThesisRunBundle:
    """``resolve_run_dir`` + ``load_thesis_run_bundle``."""
    root = Path(project_root).resolve() if project_root else default_project_root()
    rd = resolve_run_dir(root, experiment_name=experiment_name, **kwargs)
    return load_thesis_run_bundle(rd, project_root=root)
