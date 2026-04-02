from __future__ import annotations

import torch
import torch.nn as nn

from hybrid_qgnn.models.lightgcn import LightGCNLite
from hybrid_qgnn.models.quantum import QuantumBlock


class HybridQGNN(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        d=32,
        K=1,
        A_norm=None,
        q=3,
        L=1,
        p_quantum=1.0,
        dev_name="lightning.qubit",
    ):
        super().__init__()
        self.encoder = LightGCNLite(n_users, n_items, d=d, K=K, A_norm=A_norm)
        self.quantum = QuantumBlock(q=q, L=L, in_dim=2 * d, dev_name=dev_name)
        self.head = nn.Sequential(nn.Linear(q, 64), nn.ReLU(), nn.Linear(64, 1))
        self.fallback = nn.Linear(2 * d, q)
        self.p_quantum = float(p_quantum)

    def set_p_quantum(self, p):
        self.p_quantum = float(p)

    def forward(self, u, i, micro_bs=32, force_classical: bool = False):
        all_emb = self.encoder.propagate()
        x = torch.cat([all_emb[u], all_emb[self.encoder.n_users + i]], dim=-1)
        # Encoder is fp32; AMP makes Linear/quantum outputs fp16 — align before masked writes and the head.
        out_dtype = x.dtype
        if force_classical:
            zq = self.fallback(x).to(out_dtype)
        elif self.training and self.p_quantum < 1.0:
            mask = torch.rand(x.size(0), device=x.device) < self.p_quantum
            zq = torch.empty(x.size(0), self.quantum.q, device=x.device, dtype=out_dtype)
            if mask.any():
                zq[mask] = self.quantum(x[mask], micro_bs=micro_bs).to(out_dtype)
            if (~mask).any():
                zq[~mask] = self.fallback(x[~mask]).to(out_dtype)
        else:
            zq = self.quantum(x, micro_bs=micro_bs).to(out_dtype)
        return self.head(zq).squeeze(-1)
