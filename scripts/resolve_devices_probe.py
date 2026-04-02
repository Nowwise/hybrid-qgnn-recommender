#!/usr/bin/env python3
"""Print JSON device resolution (torch + PennyLane) for ExperimentConfig — run inside Docker or locally."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_qgnn.config import ExperimentConfig
from hybrid_qgnn.device import resolve_quantum_backend, resolve_training_device


def main() -> int:
    p = argparse.ArgumentParser(description="Resolve training + quantum devices from config JSON.")
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args()

    cfg = ExperimentConfig.from_json_path(args.config)
    device, dev_meta = resolve_training_device(cfg.device)
    q_name, q_meta = resolve_quantum_backend(cfg.backend, device)
    payload = {
        "where": "container" if os.environ.get("QGNN_PROJECT_ROOT") == "/app" else "host",
        "torch_device": str(device),
        "device_meta": dev_meta,
        "pennylane_device": q_name,
        "quantum_meta": q_meta,
    }
    # stdout only — easy to json.loads from subprocess
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
