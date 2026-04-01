from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from hybrid_qgnn.config import ExperimentConfig
from hybrid_qgnn.data import (
    build_norm_adj_from_train_pairs,
    load_amazon_book_dir,
    make_loaders,
    make_small_implicit_split,
)
from hybrid_qgnn.models import HybridQGNN, LightGCNLite
from hybrid_qgnn.training.loops import train_epoch
from hybrid_qgnn.training.metrics import MetricsLogger, eval_metrics
from hybrid_qgnn.training.progress import make_emit


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float32)


def save_ckpt(p, model, opt, extra):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "extra": extra}, p)


def save_best(p, model, opt, metric_name, val, payload=None):
    payload = payload or {}
    payload.update({"best_metric": float(val), "metric_name": metric_name})
    save_ckpt(p, model, opt, payload)


@dataclass
class TrainCfg:
    epochs_lg: int
    epochs_hyb: int
    lr: float
    wd: float
    batch_size: int
    micro_bs: int
    eval_every: int
    p_quantum_start: float
    p_quantum_end: float


def run_experiment(
    cfg: ExperimentConfig,
    project_root: Optional[Path] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_phase: Optional[Callable[[str, Optional[str]], None]] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Full LightGCN + Hybrid QGNN training pipeline. Writes artifacts under cfg.save_dir.

    on_progress: optional callback with keys progress_pct, steps[{id,label,status}], phase, detail.
    on_phase: legacy (phase, detail); called in addition when provided.
    """

    def _legacy(phase: str, detail: Optional[str]) -> None:
        if on_phase:
            on_phase(phase, detail)
    _set_seed(cfg.seed)
    root = project_root or Path.cwd()
    data_path = root / cfg.data_dir
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    save_dir = Path(cfg.save_dir)
    if not save_dir.is_absolute():
        save_dir = root / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    emit = make_emit(cfg.epochs_lg, cfg.epochs_hyb, on_progress)
    emit(
        0,
        "Prepare",
        "Loading dataset & building graph",
        prepare="running",
        lightgcn="pending",
        hybrid_warmup="pending",
        hybrid_train="pending",
        analysis="pending",
    )
    _legacy("data", None)

    (u_tr, i_tr), (u_te, i_te), n_users, n_items = load_amazon_book_dir(str(data_path))
    (Xtr, ytr), (Xva, yva) = make_small_implicit_split(
        u_tr,
        i_tr,
        u_te,
        i_te,
        n_users,
        n_items,
        max_users=cfg.max_users,
        max_pos_per_user=cfg.max_pos_per_user,
        neg_per_pos=cfg.neg_per_pos,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )
    A_norm = build_norm_adj_from_train_pairs(n_users, n_items, Xtr[ytr == 1])
    train_loader, val_loader = make_loaders(
        (Xtr, ytr), (Xva, yva), batch_size=cfg.batch_size, num_workers=0, device=device
    )

    logger = MetricsLogger(save_dir)
    run_cfg = cfg.to_dict()
    with open(save_dir / "run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    tcfg = TrainCfg(
        epochs_lg=cfg.epochs_lg,
        epochs_hyb=cfg.epochs_hyb,
        lr=cfg.lr,
        wd=cfg.wd,
        batch_size=cfg.batch_size,
        micro_bs=cfg.micro_bs,
        eval_every=cfg.eval_every,
        p_quantum_start=cfg.p_quantum_start,
        p_quantum_end=cfg.p_quantum_end,
    )

    _legacy("lightgcn", None)

    lg = LightGCNLite(n_users, n_items, d=cfg.d, K=cfg.K, A_norm=A_norm).to(device)
    opt_lg = torch.optim.Adam(lg.parameters(), lr=tcfg.lr, weight_decay=tcfg.wd)
    best_lg = {"auc": -1.0, "ep": 0}

    for ep in range(1, tcfg.epochs_lg + 1):
        emit(
            1 + (ep - 1),
            "LightGCN",
            f"Epoch {ep}/{tcfg.epochs_lg}",
            prepare="done",
            lightgcn="running",
            hybrid_warmup="pending",
            hybrid_train="pending",
            analysis="pending",
        )
        _legacy("lightgcn_epoch", f"{ep}/{tcfg.epochs_lg}")
        train_epoch(
            lg,
            train_loader,
            opt_lg,
            device,
            micro_bs=tcfg.micro_bs,
            logger=logger,
            epoch=ep,
            model_name="LightGCN",
            scaler=scaler,
            show_progress=show_progress,
        )
        m, _ = eval_metrics(
            lg,
            val_loader,
            device,
            micro_bs=tcfg.micro_bs,
            logger=logger,
            epoch=ep,
            model_name="LightGCN",
            split="val",
        )
        save_ckpt(save_dir / f"lg_ep{ep}.pt", lg, opt_lg, {"epoch": ep, "val_auc": m["AUC"]})
        if m["AUC"] > best_lg["auc"]:
            best_lg.update({"auc": m["AUC"], "ep": ep})
            save_best(save_dir / "lg_best.pt", lg, opt_lg, "val_auc", m["AUC"], {"epoch": ep})

        emit(
            1 + ep,
            "LightGCN",
            f"Epoch {ep}/{tcfg.epochs_lg} · val AUC {m['AUC']:.4f}",
            prepare="done",
            lightgcn="running" if ep < tcfg.epochs_lg else "done",
            hybrid_warmup="pending" if ep < tcfg.epochs_lg else "running",
            hybrid_train="pending",
            analysis="pending",
        )

    _legacy("hybrid", None)

    hyb = HybridQGNN(
        n_users,
        n_items,
        d=cfg.d,
        K=cfg.K,
        A_norm=A_norm,
        q=cfg.q,
        L=cfg.L,
        p_quantum=tcfg.p_quantum_start,
        dev_name=cfg.backend,
    ).to(device)
    opt_hyb = torch.optim.Adam(
        hyb.parameters(), lr=tcfg.lr * cfg.hybrid_lr_mult, weight_decay=tcfg.wd
    )
    best_hyb = {"auc": -1.0, "ep": 0}

    for p in hyb.encoder.parameters():
        p.requires_grad = False
    train_epoch(
        hyb,
        train_loader,
        opt_hyb,
        device,
        micro_bs=tcfg.micro_bs,
        logger=logger,
        epoch=0,
        model_name="HybridQGNN",
        scaler=scaler,
        show_progress=show_progress,
    )
    eval_metrics(
        hyb,
        val_loader,
        device,
        micro_bs=tcfg.micro_bs,
        logger=logger,
        epoch=0,
        model_name="HybridQGNN",
        split="val",
    )
    for p in hyb.encoder.parameters():
        p.requires_grad = True

    emit(
        2 + tcfg.epochs_lg,
        "Hybrid QGNN",
        f"Epoch 1/{tcfg.epochs_hyb}",
        prepare="done",
        lightgcn="done",
        hybrid_warmup="done",
        hybrid_train="running",
        analysis="pending",
    )

    for ep in range(1, tcfg.epochs_hyb + 1):
        _legacy("hybrid_epoch", f"{ep}/{tcfg.epochs_hyb}")
        cur_p = tcfg.p_quantum_start + (tcfg.p_quantum_end - tcfg.p_quantum_start) * (ep - 1) / max(
            1, tcfg.epochs_hyb - 1
        )
        hyb.set_p_quantum(cur_p)
        train_epoch(
            hyb,
            train_loader,
            opt_hyb,
            device,
            micro_bs=tcfg.micro_bs,
            desc=f"Hybrid ep{ep}/{tcfg.epochs_hyb} p_q={cur_p:.2f}",
            logger=logger,
            epoch=ep,
            model_name="HybridQGNN",
            scaler=scaler,
            show_progress=show_progress,
        )
        m, _ = eval_metrics(
            hyb,
            val_loader,
            device,
            micro_bs=tcfg.micro_bs,
            logger=logger,
            epoch=ep,
            model_name="HybridQGNN",
            split="val",
        )
        logger.log(model="HybridQGNN", split="meta", metric="p_quantum", epoch=ep, value=cur_p)
        save_ckpt(
            save_dir / f"hyb_ep{ep}.pt",
            hyb,
            opt_hyb,
            {"epoch": ep, "val_auc": m["AUC"], "p_quantum": cur_p},
        )
        if m["AUC"] > best_hyb["auc"]:
            best_hyb.update({"auc": m["AUC"], "ep": ep})
            save_best(
                save_dir / "hyb_best.pt",
                hyb,
                opt_hyb,
                "val_auc",
                m["AUC"],
                {"epoch": ep, "p_quantum": cur_p},
            )

        emit(
            2 + tcfg.epochs_lg + ep,
            "Hybrid QGNN",
            f"Epoch {ep}/{tcfg.epochs_hyb} · val AUC {m['AUC']:.4f} · p_q {cur_p:.2f}",
            prepare="done",
            lightgcn="done",
            hybrid_warmup="done",
            hybrid_train="running" if ep < tcfg.epochs_hyb else "done",
            analysis="pending" if ep < tcfg.epochs_hyb else "running",
        )

    logger.save_csv("metrics.csv")
    logger.save_json("metrics.json")
    summary_text = (
        f"LightGCN best val AUC: {best_lg['auc']:.4f} (epoch {best_lg['ep']})\n"
        f"HybridQGNN best val AUC: {best_hyb['auc']:.4f} (epoch {best_hyb['ep']})\n"
    )
    (save_dir / "summary.txt").write_text(summary_text)

    emit(
        2 + tcfg.epochs_lg + tcfg.epochs_hyb,
        "Analysis",
        "Writing comparative tables",
        prepare="done",
        lightgcn="done",
        hybrid_warmup="done",
        hybrid_train="done",
        analysis="running",
    )
    _legacy("analysis", None)
    from hybrid_qgnn.analysis.comparative import write_comparative_tables

    write_comparative_tables(save_dir)

    total_u = 1 + tcfg.epochs_lg + 1 + tcfg.epochs_hyb + 1
    emit(
        total_u,
        "Complete",
        "All steps finished",
        prepare="done",
        lightgcn="done",
        hybrid_warmup="done",
        hybrid_train="done",
        analysis="done",
    )
    _legacy("done", None)

    return {
        "save_dir": str(save_dir),
        "best_lightgcn_auc": best_lg["auc"],
        "best_lightgcn_epoch": best_lg["ep"],
        "best_hybrid_auc": best_hyb["auc"],
        "best_hybrid_epoch": best_hyb["ep"],
        "device": str(device),
    }
