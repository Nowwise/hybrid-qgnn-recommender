#!/usr/bin/env python3
"""
Minimal HybridQGNN forward pass to verify PyTorch (and optionally lightning.gpu) on the active device.

Usage (from repo root):
  python scripts/verify_gpu_smoke.py
  python scripts/verify_gpu_smoke.py --device cpu
  QGNN_DEVICE=cuda python scripts/verify_gpu_smoke.py

Exits 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from hybrid_qgnn.data.graph import build_norm_adj_from_train_pairs
from hybrid_qgnn.device import resolve_quantum_backend, resolve_training_device
from hybrid_qgnn.models import HybridQGNN


def main() -> int:
    p = argparse.ArgumentParser(description="Tiny HybridQGNN forward pass smoke test.")
    p.add_argument(
        "--device",
        default=None,
        help='Torch device preference (default: env QGNN_DEVICE or "cuda").',
    )
    args = p.parse_args()

    pref = args.device
    if pref is None:
        pref = os.environ.get("QGNN_DEVICE", "cuda")

    torch.manual_seed(0)
    np.random.seed(0)

    n_users, n_items = 12, 10
    train_pairs = np.array([[0, 0], [0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    A_norm = build_norm_adj_from_train_pairs(n_users, n_items, train_pairs)

    device, dmeta = resolve_training_device(pref)
    print("torch device:", device, "|", dmeta.get("note", "").strip())

    q_req = "lightning.gpu" if device.type == "cuda" else "lightning.qubit"
    q_back, qmeta = resolve_quantum_backend(q_req, device)
    print("quantum backend:", q_back, "|", qmeta.get("note", "").strip())

    model = HybridQGNN(
        n_users,
        n_items,
        d=8,
        K=1,
        A_norm=A_norm,
        q=2,
        L=1,
        p_quantum=1.0,
        dev_name=q_back,
    ).to(device)
    model.eval()

    u = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
    i = torch.tensor([1, 2, 3], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(u, i, micro_bs=8)
    if out.shape != (3,):
        print("FAIL: bad output shape", out.shape)
        return 1
    # torch.device("cuda") vs cuda:0 compare unequal but both are valid GPU tensors
    same = out.device.type == device.type and (
        device.type != "cuda" or out.is_cuda
    )
    if not same:
        print("FAIL: output on wrong device", out.device, "expected", device)
        return 1

    print("forward OK | shape", tuple(out.shape), "| out device", out.device)
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
