"""Load a saved hybrid run and score user–item pairs without retraining."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from hybrid_qgnn.config import ExperimentConfig
from hybrid_qgnn.data.graph import build_norm_adj_from_train_pairs
from hybrid_qgnn.device import resolve_quantum_backend, resolve_training_device
from hybrid_qgnn.models import HybridQGNN
from hybrid_qgnn.models.graph_encoders import create_graph_encoder


GRAPH_CONTEXT_FILENAME = "graph_context.npz"


def _validate_run_id_segment(run_id: str) -> None:
    if not run_id or "\x00" in run_id or len(run_id) > 512:
        raise ValueError("invalid run_id")
    p = Path(run_id)
    if p.is_absolute() or ".." in p.parts or len(p.parts) != 1:
        raise ValueError("invalid run_id")


def resolved_run_dir(project_root: Path, run_id: str) -> Path:
    _validate_run_id_segment(run_id)
    root = (project_root / "runs").resolve()
    if not root.is_dir():
        raise FileNotFoundError("runs directory missing")
    target = (root / run_id).resolve()
    if target.parent.resolve() != root or not target.is_dir():
        raise FileNotFoundError("run not found")
    return target


def experiment_config_from_run_json(run_dir: Path) -> ExperimentConfig:
    p = run_dir / "run_config.json"
    if not p.is_file():
        raise FileNotFoundError("run_config.json missing")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("run_config must be a JSON object")
    allowed = {f.name for f in fields(ExperimentConfig)}
    filtered = {k: v for k, v in raw.items() if k in allowed}
    return ExperimentConfig(**{**asdict(ExperimentConfig()), **filtered})


def load_graph_context(run_dir: Path) -> Tuple[np.ndarray, int, int]:
    npz_path = run_dir / GRAPH_CONTEXT_FILENAME
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"Missing {GRAPH_CONTEXT_FILENAME}. Re-train this run once with the current codebase "
            "so the exact training graph is saved for inference."
        )
    z = np.load(npz_path)
    train_pos = np.asarray(z["train_pos"], dtype=np.int64)
    n_users = int(z["n_users"])
    n_items = int(z["n_items"])
    if train_pos.ndim != 2 or train_pos.shape[1] != 2:
        raise ValueError("corrupt graph_context: train_pos must be (N, 2)")
    return train_pos, n_users, n_items


def score_hybrid_pairs(
    project_root: Path,
    run_id: str,
    pairs: np.ndarray,
    *,
    micro_bs: int = 256,
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Return logits for each (user, item) pair using ``hyb_best.pt``.

    ``pairs`` must use the same 0-based ID space as the benchmark (train.txt / test.txt).
    """
    run_dir = resolved_run_dir(project_root, run_id)
    cfg = experiment_config_from_run_json(run_dir)
    train_pos, n_users, n_items = load_graph_context(run_dir)

    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (N, 2)")
    if len(pairs) == 0:
        return [], {"run_id": run_id, "n_users": n_users, "n_items": n_items, "n_pairs": 0}

    u = pairs[:, 0]
    i = pairs[:, 1]
    if (u < 0).any() or (u >= n_users).any() or (i < 0).any() or (i >= n_items).any():
        raise ValueError(
            f"pair indices out of range for this run (expect user in [0,{n_users - 1}], "
            f"item in [0,{n_items - 1}])"
        )

    hy_path = run_dir / "hyb_best.pt"
    if not hy_path.is_file():
        raise FileNotFoundError("hyb_best.pt not found for this run")

    device, _ = resolve_training_device(cfg.device)
    q_backend, _ = resolve_quantum_backend(cfg.backend, device)

    try:
        ckpt = torch.load(hy_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(hy_path, map_location=device)
    ex = ckpt.get("extra") or {}
    p_q = float(ex.get("p_quantum", cfg.p_quantum_end))

    A_norm = build_norm_adj_from_train_pairs(n_users, n_items, train_pos)
    model = HybridQGNN(
        n_users,
        n_items,
        d=cfg.d,
        K=cfg.K,
        A_norm=A_norm,
        encoder=create_graph_encoder(
            cfg.hybrid_backbone.strip().lower(), n_users, n_items, cfg.d, cfg.K, A_norm
        ),
        q=cfg.q,
        L=cfg.L,
        p_quantum=p_q,
        dev_name=q_backend,
        quantum_entangle=bool(cfg.quantum_entangle),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    out: List[float] = []
    u_t = torch.as_tensor(u, device=device, dtype=torch.long)
    i_t = torch.as_tensor(i, device=device, dtype=torch.long)
    bs = max(1, int(micro_bs))
    with torch.no_grad():
        for s in range(0, len(u_t), bs):
            e = min(s + bs, len(u_t))
            logits = model(u_t[s:e], i_t[s:e], micro_bs=bs)
            out.extend(logits.detach().float().cpu().numpy().tolist())

    meta: Dict[str, Any] = {
        "run_id": run_id,
        "n_users": n_users,
        "n_items": n_items,
        "n_pairs": len(pairs),
        "hybrid_backbone": cfg.hybrid_backbone,
        "graph_context": GRAPH_CONTEXT_FILENAME,
    }
    return out, meta


def parse_pairs_lines(text: str) -> np.ndarray:
    """Parse lines ``u i`` (0-based ints) into shape (N, 2)."""
    rows: List[Tuple[int, int]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = re.split(r"[\s,;]+", line)
        toks = [t for t in toks if t]
        if len(toks) < 2:
            continue
        rows.append((int(toks[0]), int(toks[1])))
    if not rows:
        raise ValueError("no valid pairs (expected lines like: 0 42)")
    return np.array(rows, dtype=np.int64)
