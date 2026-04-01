from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.sparse import coo_matrix, diags
from sklearn.model_selection import train_test_split


def make_small_implicit_split(
    u_train,
    i_train,
    u_test,
    i_test,
    n_users: int,
    n_items: int,
    max_users: int = 30000,
    max_pos_per_user: int = 30,
    neg_per_pos: int = 1,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    unique_users = np.unique(u_train)
    users_subset = rng.choice(unique_users, size=min(max_users, len(unique_users)), replace=False)

    user_pos = {}
    for u, i in zip(u_train, i_train):
        if u in users_subset:
            lst = user_pos.setdefault(u, [])
            if len(lst) < max_pos_per_user:
                lst.append(int(i))

    pos_pairs = np.array([(u, i) for u, items in user_pos.items() for i in items], dtype=np.int64)
    y_pos = np.ones(len(pos_pairs), dtype=np.float32)

    neg_pairs = []
    for (u, _) in pos_pairs:
        seen = set(user_pos.get(int(u), []))
        while True:
            j = int(rng.integers(0, n_items))
            if j not in seen:
                neg_pairs.append((u, j))
                break
    neg_pairs = np.array(neg_pairs, dtype=np.int64)
    y_neg = np.zeros(len(neg_pairs), dtype=np.float32)

    X = np.vstack([pos_pairs, neg_pairs])
    y = np.concatenate([y_pos, y_neg])

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=val_ratio, stratify=y, random_state=seed)
    return (Xtr, ytr), (Xva, yva)


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, labels):
        self.u = torch.as_tensor(pairs[:, 0], dtype=torch.int64)
        self.i = torch.as_tensor(pairs[:, 1], dtype=torch.int64)
        self.y = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.y[idx]


def make_loaders(train, val, batch_size: int = 4096, num_workers: int = 0, device=None):
    tr, va = PairDataset(*train), PairDataset(*val)
    pin = device is not None and device.type == "cuda"
    loader_args = dict(pin_memory=pin, num_workers=num_workers, persistent_workers=False)
    train_loader = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(va, batch_size=batch_size, shuffle=False, **loader_args)
    return train_loader, val_loader


def build_norm_adj_from_train_pairs(n_users: int, n_items: int, train_pairs: np.ndarray):
    """Symmetric normalized adjacency for LightGCN propagation (positives only)."""
    N = n_users + n_items
    rows, cols = train_pairs[:, 0], train_pairs[:, 1] + n_users
    A_ui = coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(N, N))
    A = A_ui + A_ui.T
    deg = np.array(A.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    D_inv = diags(1.0 / np.sqrt(deg))
    return (D_inv @ A @ D_inv).tocsr()
