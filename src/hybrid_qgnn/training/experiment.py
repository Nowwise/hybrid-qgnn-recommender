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
from hybrid_qgnn.data import (
    build_norm_adj_from_train_pairs,
    load_lightgcn_interaction_dir,
    make_loaders,
    make_small_implicit_split,
)
from hybrid_qgnn.models import HybridQGNN, LightGCNLite
from hybrid_qgnn.exceptions import ExperimentCancelled
from hybrid_qgnn.training.loops import train_epoch
from hybrid_qgnn.training.metrics import MetricsLogger, eval_metrics
from hybrid_qgnn.training.progress import (
    PipelineReporter,
    segment_train_val_position,
    should_emit_batch,
)
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
) -> None:
    """Load best checkpoints and log sampled Recall@K / NDCG@K (val + optional test + ablation)."""
    ks = tuple(int(k) for k in cfg.ranking_ks)
    train_pos = Xtr[ytr == 1]
    val_pos = Xva[yva == 1]
    if len(train_pos) == 0 or len(val_pos) == 0:
        return

    lg_path = save_dir / "lg_best.pt"
    hy_path = save_dir / "hyb_best.pt"
    if not lg_path.is_file() or not hy_path.is_file():
        return

    try:
        sd_lg = torch.load(lg_path, map_location=device, weights_only=False)
        sd_hy = torch.load(hy_path, map_location=device, weights_only=False)
    except TypeError:
        sd_lg = torch.load(lg_path, map_location=device)
        sd_hy = torch.load(hy_path, map_location=device)

    lg_eval = LightGCNLite(n_users, n_items, d=d, K=K, A_norm=A_norm).to(device)
    lg_eval.load_state_dict(sd_lg["model"])
    lg_eval.eval()

    ex = sd_hy.get("extra") or {}
    p_q_ckpt = float(ex.get("p_quantum", cfg.p_quantum_end))
    hy_eval = HybridQGNN(
        n_users,
        n_items,
        d=d,
        K=K,
        A_norm=A_norm,
        q=q,
        L=L,
        p_quantum=p_q_ckpt,
        dev_name=backend,
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

    m_lg = ranking_metrics_sampled(
        lg_eval, "LightGCN", eval_pos=val_pos, force_hybrid_classical=False, **rk_kw
    )
    _log_ranking_dict(logger, "LightGCN", "val_ranking", m_lg)

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
            m_lg_te = ranking_metrics_sampled(
                lg_eval, "LightGCN_test", eval_pos=test_pos, force_hybrid_classical=False, **rk_kw
            )
            _log_ranking_dict(logger, "LightGCN", "test_ranking", m_lg_te)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    save_dir = Path(cfg.save_dir)
    if not save_dir.is_absolute():
        save_dir = root / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    rep = PipelineReporter(cfg.epochs_lg, cfg.epochs_hyb, on_progress)
    phase_timings: List[Dict[str, Any]] = []
    rep.push(
        rep.base_prepare(),
        "Prepare",
        "Loading dataset & building graph",
        prepare="running",
        lightgcn="pending",
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
        A_norm = build_norm_adj_from_train_pairs(n_users, n_items, Xtr[ytr == 1])
        train_loader, val_loader = make_loaders(
            (Xtr, ytr), (Xva, yva), batch_size=cfg.batch_size, num_workers=0, device=device
        )
    nt, nv = len(train_loader), len(val_loader)

    logger = MetricsLogger(save_dir)
    run_cfg = cfg.to_dict()
    with open(save_dir / "run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

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

    rep.push(
        1.0,
        "Prepare",
        f"{n_users} users · {n_items} items · batches train/val {nt}/{nv}",
        prepare="done",
        lightgcn="running",
        hybrid_warmup="pending",
        hybrid_train="pending",
        analysis="pending",
        event_message="Graph and loaders ready",
    )

    _legacy("lightgcn", None)

    lg = LightGCNLite(n_users, n_items, d=cfg.d, K=cfg.K, A_norm=A_norm).to(device)
    opt_lg = torch.optim.Adam(lg.parameters(), lr=tcfg.lightgcn_lr, weight_decay=tcfg.wd)
    best_lg = {"auc": -1.0, "ep": 0}

    for ep in range(1, tcfg.epochs_lg + 1):
        if cancelled():
            raise ExperimentCancelled()
        base = rep.base_lightgcn_epoch(ep)
        rep.push(
            base,
            "LightGCN",
            f"Epoch {ep}/{tcfg.epochs_lg} · training",
            prepare="done",
            lightgcn="running",
            hybrid_warmup="pending",
            hybrid_train="pending",
            analysis="pending",
            event_message=f"LightGCN epoch {ep}/{tcfg.epochs_lg} · train",
        )
        _legacy("lightgcn_epoch", f"{ep}/{tcfg.epochs_lg}")

        def on_batch_lg(bi: int, B: int, loss: float) -> None:
            if not should_emit_batch(bi, B):
                return
            pos = segment_train_val_position(base, bi, B, 0, nv, in_train=True)
            rep.push_fine(
                pos,
                "LightGCN",
                f"Epoch {ep}/{tcfg.epochs_lg} · train batch {bi}/{B}",
                {
                    "model": "LightGCN",
                    "split": "train",
                    "epoch": ep,
                    "batch": bi,
                    "total_batches": B,
                    "loss": round(loss, 5),
                },
            )

        def on_val_lg(bi: int, B: int) -> None:
            if not should_emit_batch(bi, B):
                return
            pos = segment_train_val_position(base, nt, nt, bi, B, in_train=False)
            rep.push_fine(
                pos,
                "LightGCN",
                f"Epoch {ep}/{tcfg.epochs_lg} · val batch {bi}/{B}",
                {
                    "model": "LightGCN",
                    "split": "val",
                    "epoch": ep,
                    "batch": bi,
                    "total_batches": B,
                },
            )

        with _timed(phase_timings, f"lightgcn_epoch_{ep}"):
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
                on_batch=on_batch_lg,
                cancel_check=cancelled,
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
                on_val_batch=on_val_lg,
                cancel_check=cancelled,
            )
            save_ckpt(save_dir / f"lg_ep{ep}.pt", lg, opt_lg, {"epoch": ep, "val_auc": m["AUC"]})
            if m["AUC"] > best_lg["auc"]:
                best_lg.update({"auc": m["AUC"], "ep": ep})
                save_best(save_dir / "lg_best.pt", lg, opt_lg, "val_auc", m["AUC"], {"epoch": ep})

            rep.push(
                base + 1.0,
                "LightGCN",
                f"Epoch {ep}/{tcfg.epochs_lg} · val AUC {m['AUC']:.4f}",
                prepare="done",
                lightgcn="running" if ep < tcfg.epochs_lg else "done",
                hybrid_warmup="pending" if ep < tcfg.epochs_lg else "running",
                hybrid_train="pending",
                analysis="pending",
                event_message=f"LightGCN ep {ep} val AUC={m['AUC']:.4f}",
            )

    if cancelled():
        raise ExperimentCancelled()

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
    opt_hyb = torch.optim.Adam(hyb.parameters(), lr=tcfg.hybrid_lr, weight_decay=tcfg.wd)
    best_hyb = {"auc": -1.0, "ep": 0}

    w_base = rep.base_hybrid_warmup()
    rep.push(
        w_base,
        "Hybrid QGNN",
        "Warmup (encoder frozen) · training",
        prepare="done",
        lightgcn="done",
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
    for p in hyb.encoder.parameters():
        p.requires_grad = True

    rep.push(
        w_base + 1.0,
        "Hybrid QGNN",
        f"Epoch 1/{tcfg.epochs_hyb} · full training",
        prepare="done",
        lightgcn="done",
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
            lightgcn="done",
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

            rep.push(
                h_base + 1.0,
                "Hybrid QGNN",
                f"Epoch {ep}/{tcfg.epochs_hyb} · val AUC {m['AUC']:.4f} · p_q {cur_p:.2f}",
                prepare="done",
                lightgcn="done",
                hybrid_warmup="done",
                hybrid_train="running" if ep < tcfg.epochs_hyb else "done",
                analysis="pending" if ep < tcfg.epochs_hyb else "running",
                event_message=f"Hybrid ep {ep} val AUC={m['AUC']:.4f}",
            )

    if cancelled():
        raise ExperimentCancelled()

    if cfg.eval_ranking:
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
                cfg.backend,
                Xtr,
                ytr,
                Xva,
                yva,
                u_te,
                i_te,
                tcfg.micro_bs,
                logger,
            )

    summary_text = (
        f"LightGCN best val AUC: {best_lg['auc']:.4f} (epoch {best_lg['ep']})\n"
        f"HybridQGNN best val AUC: {best_hyb['auc']:.4f} (epoch {best_hyb['ep']})\n"
        "\n"
        "Protocol: validation AUC is used for checkpointing during training. "
        "Sampled Recall@K / NDCG@K (see metrics.csv splits val_ranking and test_ranking) "
        "use one held-out positive per user plus random negatives — standard implicit-feedback sanity check. "
        "Official test.txt interactions (users overlapping train) are used only when eval_test_ranking is true.\n"
    )
    (save_dir / "summary.txt").write_text(summary_text)

    a_base = rep.base_analysis()
    rep.push(
        a_base,
        "Analysis",
        "Writing comparative tables",
        prepare="done",
        lightgcn="done",
        hybrid_warmup="done",
        hybrid_train="done",
        analysis="running",
        event_message="Exporting comparative tables",
    )
    _legacy("analysis", None)
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

    rep.push(
        rep.M,
        "Complete",
        "All steps finished",
        prepare="done",
        lightgcn="done",
        hybrid_warmup="done",
        hybrid_train="done",
        analysis="done",
        event_message="Run complete",
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
