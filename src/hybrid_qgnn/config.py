"""Default experiment configuration (matches notebook GLOBAL_CFG intent)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentConfig:
    data_dir: str = "dataset/amazon-book"
    save_dir: str = "./runs/default"

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def resolved_lightgcn_lr(self) -> float:
        return self.lightgcn_lr if self.lightgcn_lr is not None else self.lr

    def resolved_hybrid_lr(self) -> float:
        if self.hybrid_lr is not None:
            return self.hybrid_lr
        return self.lr * self.hybrid_lr_mult

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
    def quick_demo(cls) -> "ExperimentConfig":
        """Alias for lightweight (backward compatibility)."""
        return cls.lightweight()
