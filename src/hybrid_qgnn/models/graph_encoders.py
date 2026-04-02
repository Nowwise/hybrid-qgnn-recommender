"""
Pluggable collaborative filtering graph encoders (LightGCN-style propagation).

Includes **simplified thesis baselines** inspired by UltraGCN, SGL, NCL, and XSimGCL: same
BCE/BPR training loop as LightGCN, but distinct propagation / forward behavior. They are not
full paper re-implementations; see class docstrings.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Type

import numpy as np
import torch
import torch.nn as nn

from hybrid_qgnn.models.lightgcn import LightGCNLite, sparse_mm_fp32_safe


class UltraGCNLite(LightGCNLite):
    """Deeper-layer–weighted propagation (emphasizes high-order neighbors vs uniform mean)."""

    def propagate(self):
        A = self.A.to(self.E.weight.device)
        w = self.E.weight
        embs = [w]
        x = w
        amp_off = torch.amp.autocast("cuda", enabled=False) if w.is_cuda else nullcontext()
        with amp_off:
            for _ in range(self.K):
                x = sparse_mm_fp32_safe(A, x)
                embs.append(x)
            weights = torch.linspace(0.5, 1.5, len(embs), device=w.device, dtype=w.dtype)
            weights = weights / weights.sum()
            out = sum(e * wt for e, wt in zip(embs, weights))
        return out


class SGLite(LightGCNLite):
    """SGL-style **training-time edge dropout** on the normalized adjacency (self-supervised aug)."""

    def __init__(self, n_users, n_items, d=32, K=1, A_norm=None, edge_dropout: float = 0.1):
        super().__init__(n_users, n_items, d=d, K=K, A_norm=A_norm)
        self.edge_dropout = float(edge_dropout)

    def propagate(self):
        A = self.A.to(self.E.weight.device)
        if self.training and self.edge_dropout > 0:
            idx = A.indices()
            v = A.values()
            keep = torch.rand_like(v) > self.edge_dropout
            if keep.any():
                v = v * keep.float()
            else:
                v = v
            A = torch.sparse_coo_tensor(idx, v, A.size()).coalesce()
        w = self.E.weight
        embs = [w]
        x = w
        amp_off = torch.amp.autocast("cuda", enabled=False) if w.is_cuda else nullcontext()
        with amp_off:
            for _ in range(self.K):
                x = sparse_mm_fp32_safe(A, x)
                embs.append(x)
            out = torch.stack(embs).mean(dim=0)
        return out


class NCLite(LightGCNLite):
    """NCL-style **neighborhood blend**: add a hop of graph-smoothed representations (simplified)."""

    def __init__(self, n_users, n_items, d=32, K=1, A_norm=None, neighbor_mix: float = 0.12):
        super().__init__(n_users, n_items, d=d, K=K, A_norm=A_norm)
        self.neighbor_mix = float(neighbor_mix)

    def propagate(self):
        out = super().propagate()
        if self.training and self.neighbor_mix > 0:
            A = self.A.to(out.device)
            amp_off = torch.amp.autocast("cuda", enabled=False) if out.is_cuda else nullcontext()
            with amp_off:
                neigh = sparse_mm_fp32_safe(A, out.detach())
            out = out + self.neighbor_mix * neigh
        return out


class XSimGCLite(LightGCNLite):
    """XSimGCL-style **stochastic noise** on embeddings at training (similarity / contrastive lite)."""

    def __init__(self, n_users, n_items, d=32, K=1, A_norm=None, noise_std: float = 0.02):
        super().__init__(n_users, n_items, d=d, K=K, A_norm=A_norm)
        self.noise_std = float(noise_std)

    def forward(self, u, i):
        all_emb = self.propagate()
        if self.training and self.noise_std > 0:
            all_emb = all_emb + self.noise_std * torch.randn_like(all_emb)
        uemb = all_emb[u]
        iemb = all_emb[self.n_users + i]
        return (uemb * iemb).sum(dim=-1)


BASELINE_MODEL_IDS = ("lightgcn", "ultragcn", "sgl", "ncl", "xsimgcl")

# Metrics / UI / checkpoint stem -> display name
BASELINE_DISPLAY_NAMES: Dict[str, str] = {
    "lightgcn": "LightGCN",
    "ultragcn": "UltraGCN",
    "sgl": "SGL",
    "ncl": "NCL",
    "xsimgcl": "XSimGCL",
}

ENCODER_REGISTRY: Dict[str, Type[nn.Module]] = {
    "lightgcn": LightGCNLite,
    "ultragcn": UltraGCNLite,
    "sgl": SGLite,
    "ncl": NCLite,
    "xsimgcl": XSimGCLite,
}


def create_graph_encoder(kind: str, n_users: int, n_items: int, d: int, K: int, A_norm) -> nn.Module:
    k = (kind or "lightgcn").strip().lower()
    if k not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown graph encoder {kind!r}; expected one of {list(ENCODER_REGISTRY)}")
    cls = ENCODER_REGISTRY[k]
    return cls(n_users, n_items, d=d, K=K, A_norm=A_norm)


def baseline_checkpoint_stem(baseline_id: str) -> str:
    """File stem for best checkpoint, e.g. lightgcn -> lightgcn_best.pt (lg_best kept for lightgcn compat)."""
    return baseline_id.strip().lower()
