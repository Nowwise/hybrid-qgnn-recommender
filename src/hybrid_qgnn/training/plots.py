"""Matplotlib training dashboards — re-export from ``analysis.training_dashboard`` (torch-free)."""

from hybrid_qgnn.analysis.training_dashboard import (
    build_training_dashboard_figure,
    metrics_dataframe_to_live_payload,
    refresh_training_plots,
)

__all__ = [
    "build_training_dashboard_figure",
    "metrics_dataframe_to_live_payload",
    "refresh_training_plots",
]
