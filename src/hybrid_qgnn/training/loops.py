from __future__ import annotations

import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from hybrid_qgnn.models.hybrid import HybridQGNN
from hybrid_qgnn.training.metrics import MetricsLogger


def train_epoch(
    model,
    loader,
    opt,
    device: torch.device,
    micro_bs=32,
    desc="train",
    logger: Optional[MetricsLogger] = None,
    epoch=None,
    model_name=None,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = True,
    show_progress: bool = True,
) -> Tuple[float, float]:
    model.train()
    crit = nn.BCEWithLogitsLoss()
    total = 0.0
    n = 0
    t0 = time.time()
    amp_enabled = use_amp and device.type == "cuda"
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    iterator = tqdm(loader, desc=desc, leave=False, disable=not show_progress)
    for u, i, y in iterator:
        u = u.to(device, non_blocking=True) if device.type == "cuda" else u.to(device)
        i = i.to(device, non_blocking=True) if device.type == "cuda" else i.to(device)
        y = y.to(device, non_blocking=True) if device.type == "cuda" else y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logit = model(u, i, micro_bs=micro_bs) if isinstance(model, HybridQGNN) else model(u, i)
            loss = crit(logit, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item() * u.size(0)
        n += u.size(0)
        iterator.set_postfix(loss=f"{loss.item():.4f}")
    dur = time.time() - t0
    avg = total / max(1, n)
    if logger:
        logger.log(model=model_name, split="train", metric="BCE", epoch=epoch, value=avg, seconds=dur)
    return avg, dur
