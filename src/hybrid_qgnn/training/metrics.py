from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from hybrid_qgnn.models.hybrid import HybridQGNN
from hybrid_qgnn.exceptions import ExperimentCancelled


class MetricsLogger:
    def __init__(self, save_dir):
        self.rows: List[Dict[str, Any]] = []
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def log(self, **kv):
        self.rows.append({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in kv.items()})

    def dataframe(self):
        return pd.DataFrame(self.rows)

    def save_csv(self, name="metrics.csv"):
        p = self.save_dir / name
        self.dataframe().to_csv(p, index=False)
        return p

    def save_json(self, name="metrics.json"):
        p = self.save_dir / name
        with open(p, "w") as f:
            json.dump(self.rows, f, indent=2)
        return p


def eval_regression_metrics(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    non_zero_mask = np.abs(y_true) > eps
    if non_zero_mask.any():
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = float("nan")
    wmape = 100 * np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + eps)
    return dict(MAE=mae, MSE=mse, RMSE=rmse, MAPE=mape, WMAPE=wmape)


@torch.no_grad()
def eval_metrics(
    model,
    loader,
    device: torch.device,
    micro_bs=32,
    logger: Optional[MetricsLogger] = None,
    epoch=None,
    model_name=None,
    split="val",
    use_amp: bool = True,
    on_val_batch: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[dict, float]:
    import time

    model.eval()
    ys, ps = [], []
    t0 = time.time()
    amp_enabled = use_amp and device.type == "cuda"
    n_batches = len(loader)
    for bi, (u, i, y) in enumerate(loader, start=1):
        if cancel_check and cancel_check():
            raise ExperimentCancelled()
        u = u.to(device, non_blocking=True) if device.type == "cuda" else u.to(device)
        i = i.to(device, non_blocking=True) if device.type == "cuda" else i.to(device)
        y = y.to(device, non_blocking=True) if device.type == "cuda" else y.to(device)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logit = model(u, i, micro_bs=micro_bs) if isinstance(model, HybridQGNN) else model(u, i)
            prob = torch.sigmoid(logit).cpu().numpy()
        ys.append(y.cpu().numpy())
        ps.append(prob)
        if on_val_batch:
            on_val_batch(bi, n_batches)

    if not ys:
        dur = time.time() - t0
        metrics = {"AUC": float("nan"), **{k: float("nan") for k in ["MAE", "MSE", "RMSE", "MAPE", "WMAPE"]}}
        if logger:
            for k, v in metrics.items():
                logger.log(model=model_name, split=split, metric=k, epoch=epoch, value=v, seconds=dur)
        return metrics, dur

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    try:
        auc = roc_auc_score(ys, ps)
    except Exception:
        auc = float(np.mean((ps > 0.5) == ys))
    reg = eval_regression_metrics(ys, ps)
    metrics = {"AUC": auc, **reg}
    dur = time.time() - t0
    if logger:
        for k, v in metrics.items():
            logger.log(model=model_name, split=split, metric=k, epoch=epoch, value=v, seconds=dur)
    return metrics, dur
