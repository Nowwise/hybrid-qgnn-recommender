"""Sampled ranking metrics (Recall@K, NDCG@K) for implicit recommendation evaluation."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import torch

from hybrid_qgnn.models.hybrid import HybridQGNN


def _build_train_item_sets(n_users: int, train_pos: np.ndarray) -> List[set]:
    sets: List[set] = [set() for _ in range(n_users)]
    for u, i in train_pos:
        u, i = int(u), int(i)
        if 0 <= u < n_users:
            sets[u].add(i)
    return sets


@torch.no_grad()
def _scores_for_candidates(
    model: torch.nn.Module,
    user_id: int,
    item_ids: np.ndarray,
    device: torch.device,
    micro_bs: int,
    force_hybrid_classical: bool,
) -> torch.Tensor:
    u_t = torch.full((len(item_ids),), user_id, dtype=torch.long)
    i_t = torch.from_numpy(item_ids.astype(np.int64))
    out: List[torch.Tensor] = []
    n = len(item_ids)
    for s in range(0, n, micro_bs):
        e = min(s + micro_bs, n)
        ub = u_t[s:e].to(device)
        ib = i_t[s:e].to(device)
        if isinstance(model, HybridQGNN):
            out.append(model(ub, ib, micro_bs=micro_bs, force_classical=force_hybrid_classical).float().cpu())
        else:
            out.append(model(ub, ib).float().cpu())
    return torch.cat(out, dim=0)


def ranking_metrics_sampled(
    model: torch.nn.Module,
    model_label: str,
    n_users: int,
    n_items: int,
    train_pos: np.ndarray,
    eval_pos: np.ndarray,
    device: torch.device,
    micro_bs: int,
    ks: Tuple[int, ...],
    max_users: int,
    n_negatives: int,
    seed: int,
    force_hybrid_classical: bool = False,
) -> Dict[str, float]:
    """
    One positive per sampled user from eval_pos; candidates = 1 positive + n_negatives random negatives
    (not in user's training positives). Averaged Recall@K and NDCG@K over queries.
    """
    if len(eval_pos) == 0 or len(train_pos) == 0:
        return {}

    rng = np.random.default_rng(seed + (hash(model_label) % 10_000))
    train_sets = _build_train_item_sets(n_users, train_pos)
    train_users = set(int(u) for u in np.unique(train_pos[:, 0]))
    users_eval = np.unique(eval_pos[:, 0])
    users_eval = np.array(
        [int(u) for u in users_eval if int(u) in train_users and 0 <= int(u) < n_users],
        dtype=np.int64,
    )
    if len(users_eval) == 0:
        return {}
    if len(users_eval) > max_users:
        users_eval = rng.choice(users_eval, size=max_users, replace=False)

    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    model.eval()

    for u in users_eval:
        u = int(u)
        pos_for_u = eval_pos[eval_pos[:, 0] == u][:, 1]
        if len(pos_for_u) == 0:
            continue
        i_pos = int(rng.choice(pos_for_u))
        seen = set(train_sets[u])
        seen.add(i_pos)
        negs: List[int] = []
        guard = 0
        while len(negs) < n_negatives and guard < n_negatives * 50:
            guard += 1
            j = int(rng.integers(0, n_items))
            if j not in seen:
                negs.append(j)
                seen.add(j)
        if len(negs) < n_negatives:
            continue
        items = np.array([i_pos] + negs, dtype=np.int64)
        scores = _scores_for_candidates(model, u, items, device, micro_bs, force_hybrid_classical)
        pos_score = scores[0]
        rank = int((scores > pos_score).sum().item())
        for kk in ks:
            recalls[kk].append(1.0 if rank < kk else 0.0)
            ndcgs[kk].append((1.0 / math.log2(rank + 2)) if rank < kk else 0.0)

    out: Dict[str, float] = {}
    for kk in ks:
        r = recalls[kk]
        n = ndcgs[kk]
        if r:
            out[f"Recall@{kk}"] = float(np.mean(r))
            out[f"NDCG@{kk}"] = float(np.mean(n))
    return out
