from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import torch

from hybrid_qgnn.config import ExperimentConfig
from hybrid_qgnn.device import resolve_quantum_backend, resolve_training_device
from hybrid_qgnn.data import (
    build_norm_adj_from_train_pairs,
    load_lightgcn_interaction_dir,
    make_bpr_loader,
    make_loaders,
    make_small_implicit_split,
)
from hybrid_qgnn.models import HybridQGNN
from hybrid_qgnn.models.graph_encoders import (
    BASELINE_DISPLAY_NAMES,
    BASELINE_MODEL_IDS,
    baseline_checkpoint_stem,
    create_graph_encoder,
)
from hybrid_qgnn.exceptions import ExperimentCancelled
from hybrid_qgnn.training.loops import (
    eval_bce_val_loss,
    eval_bpr_val_loss,
    train_epoch,
    train_epoch_bpr,
)
from hybrid_qgnn.training.metrics import MetricsLogger, eval_metrics
from hybrid_qgnn.training.plots import refresh_training_plots
from hybrid_qgnn.training.progress import (
    PipelineReporter,
    segment_train_val_position,
    should_emit_batch,
)
from hybrid_qgnn.inference.scoring import GRAPH_CONTEXT_FILENAME
from hybrid_qgnn.training.ranking import ranking_metrics_sampled


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


@contextmanager
def _timed(phase_timings: List[Dict[str, Any]], name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        phase_timings.append({"phase": name, "seconds": round(time.perf_counter() - t0, 4)})


def _log_ranking_dict(logger: MetricsLogger, model_name: str, split: str, metrics: Dict[str, float]) -> None:
    for k, v in metrics.items():
        logger.log(model=model_name, split=split, metric=k, epoch=0, value=float(v))


def _run_ranking_evaluations(
    cfg: ExperimentConfig,
    save_dir: Path,
    device: torch.device,
    n_users: int,
    n_items: int,
    d: int,
    K: int,
    q: int,
    L: int,
    A_norm,
    backend: str,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    u_te: np.ndarray,
    i_te: np.ndarray,
    micro_bs: int,
    logger: MetricsLogger,
    baseline_ids: List[str],
) -> None:
    """Load best checkpoints and log sampled Recall@K / NDCG@K / HitRatio@K (val + optional test + ablation)."""
    ks = tuple(int(k) for k in cfg.ranking_ks)
    train_pos = Xtr[ytr == 1]
    val_pos = Xva[yva == 1]
    if len(train_pos) == 0 or len(val_pos) == 0:
        return

    hy_path = save_dir / "hyb_best.pt"
    if not hy_path.is_file():
        return

    try:
        sd_hy = torch.load(hy_path, map_location=device, weights_only=False)
    except TypeError:
        sd_hy = torch.load(hy_path, map_location=device)

    ex = sd_hy.get("extra") or {}
    p_q_ckpt = float(ex.get("p_quantum", cfg.p_quantum_end))
    hy_eval = HybridQGNN(
        n_users,
        n_items,
        d=d,
        K=K,
        A_norm=A_norm,
        encoder=create_graph_encoder(cfg.hybrid_backbone.lower(), n_users, n_items, d, K, A_norm),
        q=q,
        L=L,
        p_quantum=p_q_ckpt,
        dev_name=backend,
        quantum_entangle=bool(cfg.quantum_entangle),
    ).to(device)
    hy_eval.load_state_dict(sd_hy["model"])
    hy_eval.eval()

    seed = cfg.seed
    rk_kw = dict(
        n_users=n_users,
        n_items=n_items,
        train_pos=train_pos,
        device=device,
        micro_bs=micro_bs,
        ks=ks,
        max_users=cfg.ranking_max_users,
        n_negatives=cfg.ranking_negatives,
        seed=seed,
    )

    for bid in baseline_ids:
        stem = baseline_checkpoint_stem(bid)
        path = save_dir / f"{stem}_best.pt"
        if not path.is_file():
            continue
        try:
            sd = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            sd = torch.load(path, map_location=device)
        label = BASELINE_DISPLAY_NAMES.get(bid, bid)
        m_eval = create_graph_encoder(bid, n_users, n_items, d=d, K=K, A_norm=A_norm).to(device)
        m_eval.load_state_dict(sd["model"])
        m_eval.eval()
        m_b = ranking_metrics_sampled(
            m_eval, label, eval_pos=val_pos, force_hybrid_classical=False, **rk_kw
        )
        _log_ranking_dict(logger, label, "val_ranking", m_b)

    m_hy = ranking_metrics_sampled(
        hy_eval, "HybridQGNN", eval_pos=val_pos, force_hybrid_classical=False, **rk_kw
    )
    _log_ranking_dict(logger, "HybridQGNN", "val_ranking", m_hy)

    if cfg.eval_hybrid_ablation:
        m_ab = ranking_metrics_sampled(
            hy_eval, "HybridQGNN_ablation", eval_pos=val_pos, force_hybrid_classical=True, **rk_kw
        )
        _log_ranking_dict(logger, "HybridQGNN (ablation classical head)", "val_ranking", m_ab)

    if cfg.eval_test_ranking and len(u_te) > 0 and len(i_te) > 0:
        train_u = np.unique(Xtr[:, 0])
        te = np.stack([u_te.astype(np.int64), i_te.astype(np.int64)], axis=1)
        test_pos = te[np.isin(te[:, 0], train_u)]
        if len(test_pos) > 0:
            for bid in baseline_ids:
                stem = baseline_checkpoint_stem(bid)
                path = save_dir / f"{stem}_best.pt"
                if not path.is_file():
                    continue
                try:
                    sd = torch.load(path, map_location=device, weights_only=False)
                except TypeError:
                    sd = torch.load(path, map_location=device)
                label = BASELINE_DISPLAY_NAMES.get(bid, bid)
                m_eval = create_graph_encoder(bid, n_users, n_items, d=d, K=K, A_norm=A_norm).to(device)
                m_eval.load_state_dict(sd["model"])
                m_eval.eval()
                m_b_te = ranking_metrics_sampled(
                    m_eval, f"{label}_test", eval_pos=test_pos, force_hybrid_classical=False, **rk_kw
                )
                _log_ranking_dict(logger, label, "test_ranking", m_b_te)
            m_hy_te = ranking_metrics_sampled(
                hy_eval, "HybridQGNN_test", eval_pos=test_pos, force_hybrid_classical=False, **rk_kw
            )
            _log_ranking_dict(logger, "HybridQGNN", "test_ranking", m_hy_te)
            if cfg.eval_hybrid_ablation:
                m_ab_te = ranking_metrics_sampled(
                    hy_eval,
                    "HybridQGNN_ablation_test",
                    eval_pos=test_pos,
                    force_hybrid_classical=True,
                    **rk_kw,
                )
                _log_ranking_dict(logger, "HybridQGNN (ablation classical head)", "test_ranking", m_ab_te)


@dataclass
class TrainCfg:
    epochs_lg: int
    epochs_hyb: int
    lightgcn_lr: float
    hybrid_lr: float
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
    cancel_event: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Full LightGCN + Hybrid QGNN training pipeline. Writes artifacts under cfg.save_dir.

    on_progress: optional callback with keys progress_pct, steps[{id,label,status}], phase, detail,
    activity (optional), event (optional).
    cancel_event: optional threading.Event; when set, raises ExperimentCancelled at safe points.
    """

    def _legacy(phase: str, detail: Optional[str]) -> None:
        if on_phase:
            on_phase(phase, detail)

    def cancelled() -> bool:
        return bool(cancel_event is not None and cancel_event.is_set())

    if cancelled():
        raise ExperimentCancelled()

    _set_seed(cfg.seed)
    root = project_root or Path.cwd()
    data_path = root / cfg.data_dir
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    device, device_meta = resolve_training_device(cfg.device)
    q_backend, q_meta = resolve_quantum_backend(cfg.backend, device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    save_dir = Path(cfg.save_dir)
    if not save_dir.is_absolute():
        save_dir = root / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    bb = cfg.hybrid_backbone.strip().lower()
    if bb not in BASELINE_MODEL_IDS:
        raise ValueError(f"hybrid_backbone must be one of {list(BASELINE_MODEL_IDS)}; got {bb!r}")
    baseline_ids = cfg.ordered_enabled_baselines()

    rep = PipelineReporter(baseline_ids, cfg.epochs_lg, cfg.epochs_hyb, on_progress)
    phase_timings: List[Dict[str, Any]] = []
    rep.push(
        rep.base_prepare(),
        "Prepare",
        "Loading dataset & building graph",
        prepare="running",
        hybrid_warmup="pending",
        hybrid_train="pending",
        analysis="pending",
        event_message="Starting data load",
    )
    _legacy("data", None)

    if cancelled():
        raise ExperimentCancelled()

    with _timed(phase_timings, "prepare_data"):
        (u_tr, i_tr), (u_te, i_te), n_users, n_items = load_lightgcn_interaction_dir(str(data_path))
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
        train_pos_graph = Xtr[ytr == 1].astype(np.int64)
        A_norm = build_norm_adj_from_train_pairs(n_users, n_items, train_pos_graph)
        np.savez_compressed(
            save_dir / GRAPH_CONTEXT_FILENAME,
            train_pos=train_pos_graph,
            n_users=np.int32(n_users),
            n_items=np.int32(n_items),
        )
        train_loader, val_loader = make_loaders(
            (Xtr, ytr), (Xva, yva), batch_size=cfg.batch_size, num_workers=0, device=device
        )
        train_pos_only = Xtr[ytr == 1]
        val_pos_only = Xva[yva == 1]
        val_bpr_loader = None
        if cfg.training_loss == "bpr":
            train_loader = make_bpr_loader(
                train_pos_only,
                n_items,
                cfg.batch_size,
                0,
                device,
                cfg.seed,
                shuffle=True,
            )
            if cfg.early_stopping and cfg.early_stopping_monitor == "val_training_loss":
                val_bpr_loader = make_bpr_loader(
                    val_pos_only,
                    n_items,
                    cfg.batch_size,
                    0,
                    device,
                    cfg.seed + 7,
                    shuffle=False,
                )
    nt, nv = len(train_loader), len(val_loader)

    logger = MetricsLogger(save_dir)
    run_cfg = cfg.to_dict()
    run_cfg["device_info"] = {"torch_device": str(device), **device_meta}
    run_cfg["quantum_backend_info"] = {"pennylane_device": q_backend, **q_meta}
    with open(save_dir / "run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    def _emit_save_dir_for_ui() -> None:
        if on_progress:
            try:
                rel = save_dir.resolve().relative_to(root.resolve())
                sd = rel.as_posix()
            except ValueError:
                sd = save_dir.as_posix()
            on_progress({"save_dir": sd})

    def flush_live_artifacts() -> None:
        logger.save_csv("metrics.csv")
        if cfg.live_plots:
            refresh_training_plots(save_dir)

    _emit_save_dir_for_ui()

    tcfg = TrainCfg(
        epochs_lg=cfg.epochs_lg,
        epochs_hyb=cfg.epochs_hyb,
        lightgcn_lr=cfg.resolved_lightgcn_lr(),
        hybrid_lr=cfg.resolved_hybrid_lr(),
        wd=cfg.wd,
        batch_size=cfg.batch_size,
        micro_bs=cfg.micro_bs,
        eval_every=cfg.eval_every,
        p_quantum_start=cfg.p_quantum_start,
        p_quantum_end=cfg.p_quantum_end,
    )

    rep.set_baseline_active(0)
    rep.push(
        1.0,
        "Prepare",
        f"{n_users} users · {n_items} items · batches train/val {nt}/{nv}",
        prepare="done",
        hybrid_warmup="pending",
        hybrid_train="pending",
        analysis="pending",
        event_message="Graph and loaders ready",
    )

    _legacy("baselines", None)

    higher_es = cfg.early_stopping_monitor == "val_auc"
    train_supervised = train_epoch_bpr if cfg.training_loss == "bpr" else train_epoch
    best_baselines: Dict[str, Dict[str, Any]] = {bid: {"auc": -1.0, "ep": 0} for bid in baseline_ids}

    for bi, bid in enumerate(baseline_ids):
        if cancelled():
            raise ExperimentCancelled()
        rep.set_baseline_active(bi)
        display = BASELINE_DISPLAY_NAMES.get(bid, bid)
        stem = baseline_checkpoint_stem(bid)
        model = create_graph_encoder(bid, n_users, n_items, d=cfg.d, K=cfg.K, A_norm=A_norm).to(device)
        opt_b = torch.optim.Adam(model.parameters(), lr=tcfg.lightgcn_lr, weight_decay=tcfg.wd)
        best_b = best_baselines[bid]
        es_best_b: Optional[float] = None
        es_stale_b = 0

        for ep in range(1, tcfg.epochs_lg + 1):
            if cancelled():
                raise ExperimentCancelled()
            base = rep.base_baseline_epoch(bi, ep)
            rep.push(
                base,
                display,
                f"Epoch {ep}/{tcfg.epochs_lg} · training",
                prepare="done",
                hybrid_warmup="pending",
                hybrid_train="pending",
                analysis="pending",
                event_message=f"{display} epoch {ep}/{tcfg.epochs_lg} · train",
            )
            _legacy("baseline_epoch", f"{display} {ep}/{tcfg.epochs_lg}")

            def on_batch_b(bi_bt: int, B: int, loss: float) -> None:
                if not should_emit_batch(bi_bt, B):
                    return
                pos = segment_train_val_position(base, bi_bt, B, 0, nv, in_train=True)
                rep.push_fine(
                    pos,
                    display,
                    f"Epoch {ep}/{tcfg.epochs_lg} · train batch {bi_bt}/{B}",
                    {
                        "model": display,
                        "split": "train",
                        "epoch": ep,
                        "batch": bi_bt,
                        "total_batches": B,
                        "loss": round(loss, 5),
                    },
                )

            def on_val_b(bi_bt: int, B: int) -> None:
                if not should_emit_batch(bi_bt, B):
                    return
                pos = segment_train_val_position(base, nt, nt, bi_bt, B, in_train=False)
                rep.push_fine(
                    pos,
                    display,
                    f"Epoch {ep}/{tcfg.epochs_lg} · val batch {bi_bt}/{B}",
                    {
                        "model": display,
                        "split": "val",
                        "epoch": ep,
                        "batch": bi_bt,
                        "total_batches": B,
                    },
                )

            with _timed(phase_timings, f"{bid}_epoch_{ep}"):
                train_supervised(
                    model,
                    train_loader,
                    opt_b,
                    device,
                    micro_bs=tcfg.micro_bs,
                    logger=logger,
                    epoch=ep,
                    model_name=display,
                    scaler=scaler,
                    show_progress=show_progress,
                    on_batch=on_batch_b,
                    cancel_check=cancelled,
                )

                m, _ = eval_metrics(
                    model,
                    val_loader,
                    device,
                    micro_bs=tcfg.micro_bs,
                    logger=logger,
                    epoch=ep,
                    model_name=display,
                    split="val",
                    on_val_batch=on_val_b,
                    cancel_check=cancelled,
                )
                save_ckpt(
                    save_dir / f"{stem}_ep{ep}.pt", model, opt_b, {"epoch": ep, "val_auc": m["AUC"]}
                )
                if m["AUC"] > best_b["auc"]:
                    best_b.update({"auc": m["AUC"], "ep": ep})
                    save_best(
                        save_dir / f"{stem}_best.pt", model, opt_b, "val_auc", m["AUC"], {"epoch": ep}
                    )
                    if bid == "lightgcn":
                        save_best(
                            save_dir / "lg_best.pt", model, opt_b, "val_auc", m["AUC"], {"epoch": ep}
                        )

                if cfg.early_stopping:
                    if cfg.early_stopping_monitor == "val_auc":
                        cur_mon = m["AUC"]
                        mon_name = "AUC"
                    elif cfg.training_loss == "bpr":
                        if val_bpr_loader is None:
                            val_bpr_loader = make_bpr_loader(
                                val_pos_only,
                                n_items,
                                cfg.batch_size,
                                0,
                                device,
                                cfg.seed + 7,
                                shuffle=False,
                            )
                        cur_mon = eval_bpr_val_loss(
                            model,
                            val_bpr_loader,
                            device,
                            micro_bs=tcfg.micro_bs,
                            cancel_check=cancelled,
                        )
                        mon_name = "BPR_loss"
                    else:
                        cur_mon = eval_bce_val_loss(
                            model,
                            val_loader,
                            device,
                            micro_bs=tcfg.micro_bs,
                            cancel_check=cancelled,
                        )
                        mon_name = "BCE_loss"
                    logger.log(
                        model=display,
                        split="meta",
                        metric=f"early_stop_mon_{mon_name}",
                        epoch=ep,
                        value=float(cur_mon),
                    )
                    if es_best_b is None:
                        es_best_b = float(cur_mon)
                        es_stale_b = 0
                    elif (higher_es and cur_mon > es_best_b + cfg.early_stopping_min_delta) or (
                        not higher_es and cur_mon < es_best_b - cfg.early_stopping_min_delta
                    ):
                        es_best_b = float(cur_mon)
                        es_stale_b = 0
                    else:
                        es_stale_b += 1
                    if es_stale_b >= cfg.early_stopping_patience:
                        logger.log(
                            model=display,
                            split="meta",
                            metric="early_stopped",
                            epoch=ep,
                            value=1.0,
                        )
                        more_in_this = ep < tcfg.epochs_lg
                        more_baselines = bi < len(baseline_ids) - 1
                        if not more_in_this and not more_baselines:
                            rep.all_baselines_finished()
                        rep.push(
                            base + 1.0,
                            display,
                            f"Early stop epoch {ep} · val AUC {m['AUC']:.4f}",
                            prepare="done",
                            hybrid_warmup="running" if not more_in_this and not more_baselines else "pending",
                            hybrid_train="pending",
                            analysis="pending",
                            event_message=f"{display} early stop at ep {ep}",
                        )
                        flush_live_artifacts()
                        break

                more_in_this = ep < tcfg.epochs_lg
                more_baselines = bi < len(baseline_ids) - 1
                if not more_in_this and not more_baselines:
                    rep.all_baselines_finished()
                rep.push(
                    base + 1.0,
                    display,
                    f"Epoch {ep}/{tcfg.epochs_lg} · val AUC {m['AUC']:.4f}",
                    prepare="done",
                    hybrid_warmup="running" if not more_in_this and not more_baselines else "pending",
                    hybrid_train="pending",
                    analysis="pending",
                    event_message=f"{display} ep {ep} val AUC={m['AUC']:.4f}",
                )
                flush_live_artifacts()

        rep.mark_baseline_done(bi)

    rep.all_baselines_finished()

    if cancelled():
        raise ExperimentCancelled()

    _legacy("hybrid", None)

    enc_h = create_graph_encoder(cfg.hybrid_backbone.lower(), n_users, n_items, d=cfg.d, K=cfg.K, A_norm=A_norm)
    hyb = HybridQGNN(
        n_users,
        n_items,
        d=cfg.d,
        K=cfg.K,
        A_norm=A_norm,
        encoder=enc_h,
        q=cfg.q,
        L=cfg.L,
        p_quantum=tcfg.p_quantum_start,
        dev_name=q_backend,
        quantum_entangle=bool(cfg.quantum_entangle),
    ).to(device)
    hb_stem = baseline_checkpoint_stem(cfg.hybrid_backbone.lower())
    hb_ckpt = save_dir / f"{hb_stem}_best.pt"
    if hb_ckpt.is_file():
        try:
            sd_hb = torch.load(hb_ckpt, map_location=device, weights_only=False)
        except TypeError:
            sd_hb = torch.load(hb_ckpt, map_location=device)
        hyb.encoder.load_state_dict(sd_hb["model"], strict=True)
    opt_hyb = torch.optim.Adam(hyb.parameters(), lr=tcfg.hybrid_lr, weight_decay=tcfg.wd)
    best_hyb = {"auc": -1.0, "ep": 0}
    es_best_hy: Optional[float] = None
    es_stale_hy = 0

    w_base = rep.base_hybrid_warmup()
    rep.push(
        w_base,
        "Hybrid QGNN",
        "Warmup (encoder frozen) · training",
        prepare="done",
        hybrid_warmup="running",
        hybrid_train="pending",
        analysis="pending",
        event_message="Hybrid warmup · encoder frozen",
    )

    for p in hyb.encoder.parameters():
        p.requires_grad = False

    def on_batch_w(bi: int, B: int, loss: float) -> None:
        if not should_emit_batch(bi, B):
            return
        pos = segment_train_val_position(w_base, bi, B, 0, nv, in_train=True)
        rep.push_fine(
            pos,
            "Hybrid QGNN",
            f"Warmup · train batch {bi}/{B}",
            {
                "model": "HybridQGNN",
                "split": "train",
                "phase": "warmup",
                "epoch": 0,
                "batch": bi,
                "total_batches": B,
                "loss": round(loss, 5),
            },
        )

    def on_val_w(bi: int, B: int) -> None:
        if not should_emit_batch(bi, B):
            return
        pos = segment_train_val_position(w_base, nt, nt, bi, B, in_train=False)
        rep.push_fine(
            pos,
            "Hybrid QGNN",
            f"Warmup · val batch {bi}/{B}",
            {
                "model": "HybridQGNN",
                "split": "val",
                "phase": "warmup",
                "epoch": 0,
                "batch": bi,
                "total_batches": B,
            },
        )

    with _timed(phase_timings, "hybrid_warmup"):
        train_supervised(
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
            desc="hybrid warmup",
            on_batch=on_batch_w,
            cancel_check=cancelled,
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
            on_val_batch=on_val_w,
            cancel_check=cancelled,
        )
        flush_live_artifacts()
    for p in hyb.encoder.parameters():
        p.requires_grad = True

    rep.push(
        w_base + 1.0,
        "Hybrid QGNN",
        f"Epoch 1/{tcfg.epochs_hyb} · full training",
        prepare="done",
        hybrid_warmup="done",
        hybrid_train="running",
        analysis="pending",
        event_message="Warmup done · encoder unfrozen",
    )

    for ep in range(1, tcfg.epochs_hyb + 1):
        if cancelled():
            raise ExperimentCancelled()
        _legacy("hybrid_epoch", f"{ep}/{tcfg.epochs_hyb}")
        cur_p = tcfg.p_quantum_start + (tcfg.p_quantum_end - tcfg.p_quantum_start) * (ep - 1) / max(
            1, tcfg.epochs_hyb - 1
        )
        hyb.set_p_quantum(cur_p)
        h_base = rep.base_hybrid_epoch(ep)
        rep.push(
            h_base,
            "Hybrid QGNN",
            f"Epoch {ep}/{tcfg.epochs_hyb} · p_q={cur_p:.2f} · training",
            prepare="done",
            hybrid_warmup="done",
            hybrid_train="running",
            analysis="pending",
            event_message=f"Hybrid epoch {ep}/{tcfg.epochs_hyb} · p_quantum={cur_p:.2f}",
        )

        def on_batch_h(bi: int, B: int, loss: float) -> None:
            if not should_emit_batch(bi, B):
                return
            pos = segment_train_val_position(h_base, bi, B, 0, nv, in_train=True)
            rep.push_fine(
                pos,
                "Hybrid QGNN",
                f"Epoch {ep}/{tcfg.epochs_hyb} · train {bi}/{B}",
                {
                    "model": "HybridQGNN",
                    "split": "train",
                    "phase": "train",
                    "epoch": ep,
                    "batch": bi,
                    "total_batches": B,
                    "p_quantum": round(cur_p, 4),
                    "loss": round(loss, 5),
                },
            )

        def on_val_h(bi: int, B: int) -> None:
            if not should_emit_batch(bi, B):
                return
            pos = segment_train_val_position(h_base, nt, nt, bi, B, in_train=False)
            rep.push_fine(
                pos,
                "Hybrid QGNN",
                f"Epoch {ep}/{tcfg.epochs_hyb} · val {bi}/{B}",
                {
                    "model": "HybridQGNN",
                    "split": "val",
                    "phase": "train",
                    "epoch": ep,
                    "batch": bi,
                    "total_batches": B,
                    "p_quantum": round(cur_p, 4),
                },
            )

        with _timed(phase_timings, f"hybrid_epoch_{ep}"):
            train_supervised(
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
                on_batch=on_batch_h,
                cancel_check=cancelled,
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
                on_val_batch=on_val_h,
                cancel_check=cancelled,
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

            if cfg.early_stopping:
                if cfg.early_stopping_monitor == "val_auc":
                    cur_mon_h = m["AUC"]
                    mon_h = "AUC"
                elif cfg.training_loss == "bpr":
                    if val_bpr_loader is None:
                        val_bpr_loader = make_bpr_loader(
                            val_pos_only,
                            n_items,
                            cfg.batch_size,
                            0,
                            device,
                            cfg.seed + 7,
                            shuffle=False,
                        )
                    cur_mon_h = eval_bpr_val_loss(
                        hyb,
                        val_bpr_loader,
                        device,
                        micro_bs=tcfg.micro_bs,
                        cancel_check=cancelled,
                    )
                    mon_h = "BPR_loss"
                else:
                    cur_mon_h = eval_bce_val_loss(
                        hyb,
                        val_loader,
                        device,
                        micro_bs=tcfg.micro_bs,
                        cancel_check=cancelled,
                    )
                    mon_h = "BCE_loss"
                logger.log(
                    model="HybridQGNN",
                    split="meta",
                    metric=f"early_stop_mon_{mon_h}",
                    epoch=ep,
                    value=float(cur_mon_h),
                )
                if es_best_hy is None:
                    es_best_hy = float(cur_mon_h)
                    es_stale_hy = 0
                elif (higher_es and cur_mon_h > es_best_hy + cfg.early_stopping_min_delta) or (
                    not higher_es and cur_mon_h < es_best_hy - cfg.early_stopping_min_delta
                ):
                    es_best_hy = float(cur_mon_h)
                    es_stale_hy = 0
                else:
                    es_stale_hy += 1
                if es_stale_hy >= cfg.early_stopping_patience:
                    logger.log(
                        model="HybridQGNN",
                        split="meta",
                        metric="early_stopped",
                        epoch=ep,
                        value=1.0,
                    )
                    rep.push(
                        h_base + 1.0,
                        "Hybrid QGNN",
                        f"Early stop epoch {ep} · val AUC {m['AUC']:.4f}",
                        prepare="done",
                        hybrid_warmup="done",
                        hybrid_train="done",
                        analysis="pending",
                        event_message=f"Hybrid early stop at ep {ep}",
                    )
                    flush_live_artifacts()
                    break

            rep.push(
                h_base + 1.0,
                "Hybrid QGNN",
                f"Epoch {ep}/{tcfg.epochs_hyb} · val AUC {m['AUC']:.4f} · p_q {cur_p:.2f}",
                prepare="done",
                hybrid_warmup="done",
                hybrid_train="running" if ep < tcfg.epochs_hyb else "done",
                # Keep "Metrics & tables" pending until ranking / CSV export — not while ranking is still running.
                analysis="pending",
                event_message=f"Hybrid ep {ep} val AUC={m['AUC']:.4f}",
            )
            flush_live_artifacts()

    if cancelled():
        raise ExperimentCancelled()

    if cfg.eval_ranking:
        rep.push(
            rep.base_analysis(),
            "Analysis",
            "Ranking evaluation (Recall@K / NDCG@K) — can take several minutes on larger settings",
            prepare="done",
            hybrid_warmup="done",
            hybrid_train="done",
            analysis="running",
            event_message="Starting sampled ranking metrics",
        )
        with _timed(phase_timings, "ranking_evaluation"):
            _run_ranking_evaluations(
                cfg,
                save_dir,
                device,
                n_users,
                n_items,
                cfg.d,
                cfg.K,
                cfg.q,
                cfg.L,
                A_norm,
                q_backend,
                Xtr,
                ytr,
                Xva,
                yva,
                u_te,
                i_te,
                tcfg.micro_bs,
                logger,
                baseline_ids,
            )

    loss_line = f"Training loss: {cfg.training_loss.upper()} (BCE = pointwise labels, BPR = pairwise ranking).\n"
    es_line = ""
    if cfg.early_stopping:
        es_line = (
            f"Early stopping: on (patience={cfg.early_stopping_patience}, "
            f"min_delta={cfg.early_stopping_min_delta}, monitor={cfg.early_stopping_monitor}).\n"
        )
    else:
        es_line = "Early stopping: off (fixed epoch count per phase).\n"
    baseline_summary = "\n".join(
        f"{BASELINE_DISPLAY_NAMES.get(bid, bid)} best val AUC: {best_baselines[bid]['auc']:.4f} "
        f"(epoch {best_baselines[bid]['ep']})"
        for bid in baseline_ids
    )
    summary_text = (
        baseline_summary
        + "\n"
        + f"HybridQGNN best val AUC: {best_hyb['auc']:.4f} (epoch {best_hyb['ep']})\n"
        "\n"
        + loss_line
        + es_line
        + "Checkpoints: <baseline>_best.pt per graph model (lightgcn also lg_best.pt), plus hyb_best.pt.\n"
        "Sampled Recall@K / NDCG@K / HitRatio@K (metrics.csv splits val_ranking and test_ranking) "
        "use one held-out positive per query plus random negatives (not full catalog).\n"
        "Official test.txt interactions (train-overlap users) are used only when eval_test_ranking is true.\n"
    )
    (save_dir / "summary.txt").write_text(summary_text)

    a_base = rep.base_analysis()
    rep.push(
        a_base,
        "Analysis",
        "Writing comparative tables",
        prepare="done",
        hybrid_warmup="done",
        hybrid_train="done",
        analysis="running",
        event_message="Exporting comparative tables",
    )
    _legacy("analysis", None)
    # Comparative tables read metrics.csv from disk — must flush logger first (was only in memory until now).
    logger.save_csv("metrics.csv")
    logger.save_json("metrics.json")

    from hybrid_qgnn.analysis.comparative import write_comparative_tables

    with _timed(phase_timings, "analysis_export"):
        write_comparative_tables(save_dir)

    if cfg.log_phase_timings:
        (save_dir / "phase_timings.json").write_text(json.dumps(phase_timings, indent=2))
        for row in phase_timings:
            logger.log(
                model="pipeline",
                split="timing",
                metric=str(row["phase"]),
                epoch=0,
                value=float(row["seconds"]),
            )
        logger.save_csv("metrics.csv")
        logger.save_json("metrics.json")

    if cfg.live_plots:
        refresh_training_plots(save_dir)

    rep.push(
        rep.M,
        "Complete",
        "All steps finished",
        prepare="done",
        hybrid_warmup="done",
        hybrid_train="done",
        analysis="done",
        event_message="Run complete",
    )
    _legacy("done", None)

    lg_best = best_baselines.get("lightgcn") or {"auc": -1.0, "ep": 0}
    return {
        "save_dir": str(save_dir),
        "best_baselines": {bid: dict(best_baselines[bid]) for bid in baseline_ids},
        "best_lightgcn_auc": lg_best["auc"],
        "best_lightgcn_epoch": lg_best["ep"],
        "best_hybrid_auc": best_hyb["auc"],
        "best_hybrid_epoch": best_hyb["ep"],
        "hybrid_backbone": cfg.hybrid_backbone.lower(),
        "device": str(device),
        "device_info": device_meta,
        "quantum_backend": q_backend,
        "quantum_backend_info": q_meta,
    }
