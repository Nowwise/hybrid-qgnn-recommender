"""Default experiment configuration (matches notebook GLOBAL_CFG intent)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


@dataclass
class ExperimentConfig:
    data_dir: str = "dataset/amazon-book"
    save_dir: str = "./runs/default"
    # Optional friendly label for this run; stored in run_config.json (UI / org). Only affects save_dir when API merges a name with an empty save_dir override.
    experiment_name: Optional[str] = None

    max_users: int = 6000
    max_pos_per_user: int = 10
    neg_per_pos: int = 1
    val_ratio: float = 0.10

    # Shared embedding / graph depth (LightGCN encoder inside hybrid uses the same)
    d: int = 64
    K: int = 2

    # Hybrid quantum head
    q: int = 4
    L: int = 2
    quantum_entangle: bool = True

    backend: str = "lightning.qubit"
    micro_bs: int = 48

    epochs_lg: int = 8
    epochs_hyb: int = 6
    batch_size: int = 1536

    # Default LR when lightgcn_lr / hybrid_lr are None
    lr: float = 1.5e-3
    lightgcn_lr: Optional[float] = None
    hybrid_lr: Optional[float] = None
    hybrid_lr_mult: float = 0.7

    wd: float = 1e-6
    eval_every: int = 1

    p_quantum_start: float = 0.4
    p_quantum_end: float = 1.0

    seed: int = 42

    # Training objective: "bce" = pointwise BCE on labeled pairs; "bpr" = Bayesian Personalized Ranking triplets.
    training_loss: Literal["bce", "bpr"] = "bce"
    # Stop an epoch phase early when the monitored validation signal stops improving.
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    # val_auc = ROC-AUC on validation pairs; val_training_loss = BCE or BPR loss on validation (matches training_loss).
    early_stopping_monitor: Literal["val_auc", "val_training_loss"] = "val_auc"

    # Training compute: None → env QGNN_DEVICE or "auto". Use "cpu", "cuda", "cuda:0", "auto".
    device: Optional[str] = None

    # API/UI only: "auto" | "cpu" | "gpu" — echoed in run_config; server applies before run if set.
    compute_mode: Optional[str] = None

    # Sampled ranking metrics (Recall@K / NDCG@K / HitRatio@K) — thesis-style implicit ranking eval
    eval_ranking: bool = True
    ranking_max_users: int = 512
    ranking_negatives: int = 99
    ranking_ks: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    eval_test_ranking: bool = True
    eval_hybrid_ablation: bool = True
    log_phase_timings: bool = True

    # Write metrics.csv + plots/training_dashboard.png after each epoch (for dashboard live view).
    live_plots: bool = True

    # Graph baselines (trained sequentially; each uses epochs_lg and lightgcn_lr). Hybrid encoder is chosen separately.
    train_baseline_lightgcn: bool = True
    train_baseline_ultragcn: bool = True
    train_baseline_sgl: bool = True
    train_baseline_ncl: bool = True
    train_baseline_xsimgcl: bool = True
    # Which baseline’s trained weights initialize HybridQGNN.encoder (must be one of the graph encoder ids).
    hybrid_backbone: str = "lightgcn"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_path(cls, path: Path | str) -> "ExperimentConfig":
        """Load from JSON object on disk; unknown keys raise. Merges onto defaults (same as CLI scripts)."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("config JSON must be an object")
        allowed = {f.name for f in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown ExperimentConfig keys: {sorted(unknown)}")
        return cls(**{**asdict(cls()), **raw})

    def resolved_lightgcn_lr(self) -> float:
        return self.lightgcn_lr if self.lightgcn_lr is not None else self.lr

    def resolved_hybrid_lr(self) -> float:
        if self.hybrid_lr is not None:
            return self.hybrid_lr
        return self.lr * self.hybrid_lr_mult

    def ordered_enabled_baselines(self) -> List[str]:
        """Baselines to train this run, in fixed order. Always includes ``hybrid_backbone``."""
        from hybrid_qgnn.models.graph_encoders import BASELINE_MODEL_IDS

        selected = {bid for bid in BASELINE_MODEL_IDS if getattr(self, f"train_baseline_{bid}", False)}
        bb = self.hybrid_backbone.strip().lower()
        selected.add(bb)
        return [b for b in BASELINE_MODEL_IDS if b in selected]

    @classmethod
    def lightweight(cls) -> "ExperimentConfig":
        """Small fast run (UI smoke tests)."""
        return cls(
            save_dir="./runs/lightweight_demo",
            max_users=400,
            max_pos_per_user=5,
            neg_per_pos=1,
            val_ratio=0.15,
            d=32,
            K=1,
            q=3,
            L=1,
            micro_bs=16,
            epochs_lg=2,
            epochs_hyb=2,
            batch_size=512,
            lr=2e-3,
            lightgcn_lr=None,
            hybrid_lr=None,
            hybrid_lr_mult=0.7,
            ranking_max_users=96,
            # Need >= max(K) so top-K metrics (e.g. @50) are well-defined (1 pos + n_neg candidates).
            ranking_negatives=99,
        )

    @classmethod
    def notebook_balanced(cls) -> "ExperimentConfig":
        """Matches Main.ipynb GLOBAL_CFG (balanced_cpu profile)."""
        return cls(
            save_dir="./runs/balanced_cpu",
            max_users=6000,
            max_pos_per_user=10,
            neg_per_pos=1,
            val_ratio=0.10,
            d=64,
            K=2,
            q=4,
            L=2,
            backend="lightning.qubit",
            micro_bs=48,
            epochs_lg=8,
            epochs_hyb=6,
            batch_size=1536,
            lr=1.5e-3,
            lightgcn_lr=None,
            hybrid_lr=None,
            hybrid_lr_mult=0.7,
            wd=1e-6,
            eval_every=1,
            p_quantum_start=0.4,
            p_quantum_end=1.0,
            seed=42,
        )

    @classmethod
    def large(cls) -> "ExperimentConfig":
        """Heavier thesis run: more users, wider embeddings, deeper GCN and quantum stack."""
        return cls(
            save_dir="./runs/large",
            max_users=16000,
            max_pos_per_user=15,
            neg_per_pos=1,
            val_ratio=0.10,
            d=128,
            K=3,
            q=5,
            L=3,
            backend="lightning.qubit",
            micro_bs=32,
            epochs_lg=12,
            epochs_hyb=8,
            batch_size=1280,
            lr=1e-3,
            lightgcn_lr=None,
            hybrid_lr=None,
            hybrid_lr_mult=0.7,
            wd=1e-6,
            eval_every=1,
            p_quantum_start=0.4,
            p_quantum_end=1.0,
            seed=42,
            ranking_max_users=768,
            ranking_negatives=99,
        )

    @classmethod
    def extra_large(cls) -> "ExperimentConfig":
        """Long, high-coverage run; enable early stopping and less frequent validation by default."""
        return cls(
            save_dir="./runs/extra_large",
            max_users=30000,
            max_pos_per_user=25,
            neg_per_pos=1,
            val_ratio=0.10,
            d=128,
            K=3,
            q=6,
            L=3,
            backend="lightning.qubit",
            micro_bs=20,
            epochs_lg=24,
            epochs_hyb=12,
            batch_size=768,
            lr=1e-3,
            lightgcn_lr=None,
            hybrid_lr=None,
            hybrid_lr_mult=0.7,
            wd=1e-6,
            eval_every=2,
            p_quantum_start=0.4,
            p_quantum_end=1.0,
            seed=42,
            early_stopping=True,
            early_stopping_patience=3,
            early_stopping_min_delta=1e-4,
            early_stopping_monitor="val_auc",
            ranking_max_users=512,
            ranking_negatives=99,
        )

    @classmethod
    def quick_demo(cls) -> "ExperimentConfig":
        """Alias for lightweight (backward compatibility)."""
        return cls.lightweight()
