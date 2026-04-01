"""Default experiment configuration (matches notebook GLOBAL_CFG intent)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class ExperimentConfig:
    data_dir: str = "amazon-book"
    save_dir: str = "./runs/default"

    max_users: int = 6000
    max_pos_per_user: int = 10
    neg_per_pos: int = 1
    val_ratio: float = 0.10

    d: int = 64
    K: int = 2
    q: int = 4
    L: int = 2

    backend: str = "lightning.qubit"
    micro_bs: int = 48

    epochs_lg: int = 8
    epochs_hyb: int = 6
    batch_size: int = 1536
    lr: float = 1.5e-3
    wd: float = 1e-6
    eval_every: int = 1

    p_quantum_start: float = 0.4
    p_quantum_end: float = 1.0

    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def quick_demo(cls) -> "ExperimentConfig":
        """Small run for UI smoke tests (not for publication numbers)."""
        return cls(
            save_dir="./runs/quick_demo",
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
        )
