from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from hybrid_qgnn.models.hybrid import HybridQGNN
from hybrid_qgnn.exceptions import ExperimentCancelled
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
    on_batch: Optional[Callable[[int, int, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[float, float]:
    model.train()
    crit = nn.BCEWithLogitsLoss()
    total = 0.0
    n = 0
    t0 = time.time()
    amp_enabled = use_amp and device.type == "cuda"
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    n_batches = len(loader)
    iterator = tqdm(loader, desc=desc, leave=False, disable=not show_progress)
    for bi, (u, i, y) in enumerate(iterator, start=1):
        if cancel_check and cancel_check():
            raise ExperimentCancelled()
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
        if on_batch:
            on_batch(bi, n_batches, float(loss.item()))
    dur = time.time() - t0
    avg = total / max(1, n)
    if logger:
        logger.log(model=model_name, split="train", metric="BCE", epoch=epoch, value=avg, seconds=dur)
    return avg, dur


def train_epoch_bpr(
    model,
    loader,
    opt,
    device: torch.device,
    micro_bs=32,
    desc="train bpr",
    logger: Optional[MetricsLogger] = None,
    epoch=None,
    model_name=None,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = True,
    show_progress: bool = True,
    on_batch: Optional[Callable[[int, int, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[float, float]:
    model.train()
    total = 0.0
    n = 0
    t0 = time.time()
    amp_enabled = use_amp and device.type == "cuda"
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    n_batches = len(loader)
    iterator = tqdm(loader, desc=desc, leave=False, disable=not show_progress)
    for bi, (u, i_pos, i_neg) in enumerate(iterator, start=1):
        if cancel_check and cancel_check():
            raise ExperimentCancelled()
        u = u.to(device, non_blocking=True) if device.type == "cuda" else u.to(device)
        i_pos = i_pos.to(device, non_blocking=True) if device.type == "cuda" else i_pos.to(device)
        i_neg = i_neg.to(device, non_blocking=True) if device.type == "cuda" else i_neg.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            s_pos = model(u, i_pos, micro_bs=micro_bs) if isinstance(model, HybridQGNN) else model(u, i_pos)
            s_neg = model(u, i_neg, micro_bs=micro_bs) if isinstance(model, HybridQGNN) else model(u, i_neg)
            loss = -F.logsigmoid(s_pos - s_neg).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item() * u.size(0)
        n += u.size(0)
        iterator.set_postfix(loss=f"{loss.item():.4f}")
        if on_batch:
            on_batch(bi, n_batches, float(loss.item()))
    dur = time.time() - t0
    avg = total / max(1, n)
    if logger:
        logger.log(model=model_name, split="train", metric="BPR", epoch=epoch, value=avg, seconds=dur)
    return avg, dur


@torch.no_grad()
def eval_bce_val_loss(
    model,
    loader,
    device: torch.device,
    micro_bs=32,
    use_amp: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> float:
    model.eval()
    crit = nn.BCEWithLogitsLoss(reduction="sum")
    amp_enabled = use_amp and device.type == "cuda"
    total, n = 0.0, 0
    for u, i, y in loader:
        if cancel_check and cancel_check():
            raise ExperimentCancelled()
        u = u.to(device, non_blocking=True) if device.type == "cuda" else u.to(device)
        i = i.to(device, non_blocking=True) if device.type == "cuda" else i.to(device)
        y = y.to(device, non_blocking=True) if device.type == "cuda" else y.to(device)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logit = model(u, i, micro_bs=micro_bs) if isinstance(model, HybridQGNN) else model(u, i)
            total += crit(logit, y).item()
            n += int(y.numel())
    return total / max(1, n)


@torch.no_grad()
def eval_bpr_val_loss(
    model,
    loader,
    device: torch.device,
    micro_bs=32,
    use_amp: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> float:
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    total, n = 0.0, 0
    for u, i_pos, i_neg in loader:
        if cancel_check and cancel_check():
            raise ExperimentCancelled()
        u = u.to(device, non_blocking=True) if device.type == "cuda" else u.to(device)
        i_pos = i_pos.to(device, non_blocking=True) if device.type == "cuda" else i_pos.to(device)
        i_neg = i_neg.to(device, non_blocking=True) if device.type == "cuda" else i_neg.to(device)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            s_pos = model(u, i_pos, micro_bs=micro_bs) if isinstance(model, HybridQGNN) else model(u, i_pos)
            s_neg = model(u, i_neg, micro_bs=micro_bs) if isinstance(model, HybridQGNN) else model(u, i_neg)
            loss = -F.logsigmoid(s_pos - s_neg).mean()
        total += loss.item() * u.size(0)
        n += u.size(0)
    return total / max(1, n)
