from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LightGCNLite(nn.Module):
    def __init__(self, n_users, n_items, d=32, K=1, A_norm=None):
        super().__init__()
        self.n_users, self.n_items, self.K = n_users, n_items, K
        self.E = nn.Embedding(n_users + n_items, d)
        nn.init.normal_(self.E.weight, std=0.05)
        assert A_norm is not None
        self.A = self._to_torch_sparse(A_norm)

    @staticmethod
    def _to_torch_sparse(A_csr):
        A = A_csr.tocoo()
        idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
        val = torch.tensor(A.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, torch.Size(A.shape)).coalesce()

    def propagate(self):
        embs = [self.E.weight]
        x = self.E.weight
        for _ in range(self.K):
            x = torch.sparse.mm(self.A.to(x.device), x)
            embs.append(x)
        return torch.stack(embs).mean(dim=0)

    def forward(self, u, i):
        all_emb = self.propagate()
        uemb = all_emb[u]
        iemb = all_emb[self.n_users + i]
        return (uemb * iemb).sum(dim=-1)
