#!/usr/bin/env python3
"""Load ExperimentConfig from JSON and run the full training pipeline.

Used by Main_lab.ipynb via:
  docker compose … run --rm api python /app/scripts/run_experiment_json.py --config /app/runs/….json

Expects QGNN_PROJECT_ROOT=/app inside Docker; falls back to repo root next to scripts/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_qgnn.config import ExperimentConfig
from hybrid_qgnn.training.experiment import run_experiment


def main() -> int:
    p = argparse.ArgumentParser(description="Run run_experiment() from a JSON config file.")
    p.add_argument("--config", type=Path, required=True, help="Path to ExperimentConfig JSON")
    args = p.parse_args()

    cfg = ExperimentConfig.from_json_path(args.config)
    root = Path(os.environ.get("QGNN_PROJECT_ROOT", str(ROOT))).resolve()

    result = run_experiment(cfg, project_root=root, show_progress=True)
    print("save_dir:", result.get("save_dir"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
