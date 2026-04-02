from __future__ import annotations

from typing import Dict, Tuple

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


def _user_positive_sets(train_pos: np.ndarray) -> Dict[int, set]:
    m: Dict[int, set] = {}
    for row in train_pos:
        u, i = int(row[0]), int(row[1])
        m.setdefault(u, set()).add(i)
    return m


class BPRTripletDataset(torch.utils.data.Dataset):
    """One positive + one random negative per user (implicit BPR)."""

    def __init__(self, pos_pairs: np.ndarray, n_items: int, seed: int):
        self.pos = np.ascontiguousarray(pos_pairs, dtype=np.int64)
        self.n_items = int(n_items)
        self.user_pos = _user_positive_sets(self.pos)
        self._rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return len(self.pos)

    def __getitem__(self, idx: int):
        u, i_pos = self.pos[idx % len(self.pos)]
        u_i, i_pos_i = int(u), int(i_pos)
        seen = self.user_pos[u_i]
        for _ in range(64):
            j = int(self._rng.integers(0, self.n_items))
            if j not in seen:
                return u_i, i_pos_i, j
        for j in range(self.n_items):
            if j not in seen:
                return u_i, i_pos_i, j
        return u_i, i_pos_i, (i_pos_i + 1) % self.n_items


def make_bpr_loader(
    pos_pairs: np.ndarray,
    n_items: int,
    batch_size: int,
    num_workers: int,
    device,
    seed: int,
    *,
    shuffle: bool,
):
    ds = BPRTripletDataset(pos_pairs, n_items, seed)
    pin = device is not None and getattr(device, "type", None) == "cuda"
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=False,
    )


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
