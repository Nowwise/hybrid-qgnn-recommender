"""Load user–item interaction splits in LightGCN list format (train.txt / test.txt)."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np


def load_lightgcn_interaction_dir(
    data_dir: str,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], int, int]:
    """
    Read LightGCN-format train/test files under ``data_dir``:
    each line ``u i`` or ``u i1 i2 ...`` (user id then one or more item ids, 0-based).
    """
    train_path = os.path.join(data_dir, "train.txt")
    test_path = os.path.join(data_dir, "test.txt")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(f"Missing train.txt or test.txt in {data_dir}")

    def read_user_items(path: str) -> Tuple[np.ndarray, np.ndarray]:
        users, items = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                toks = line.strip().split()
                if not toks:
                    continue
                u = int(toks[0])
                if len(toks) == 2:
                    users.append(u)
                    items.append(int(toks[1]))
                else:
                    for t in toks[1:]:
                        users.append(u)
                        items.append(int(t))
        return np.array(users, dtype=np.int64), np.array(items, dtype=np.int64)

    u_train, i_train = read_user_items(train_path)
    u_test, i_test = read_user_items(test_path)

    n_users = int(max(u_train.max(initial=0), u_test.max(initial=0)) + 1)
    n_items = int(max(i_train.max(initial=0), i_test.max(initial=0)) + 1)
    return (u_train, i_train), (u_test, i_test), n_users, n_items
