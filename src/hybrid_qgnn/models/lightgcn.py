from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn


class _SparseDenseMM(torch.autograd.Function):
    """Sparse @ dense matmul in float32; AMP cannot use Half in CUDA sparse backward."""

    @staticmethod
    def forward(ctx, A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not A.is_sparse:
            raise ValueError("_SparseDenseMM expects sparse A")
        ctx.save_for_backward(A, x)
        return torch.sparse.mm(A, x.float())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if grad_output is None:
            return None, None
        A, x = ctx.saved_tensors
        go = grad_output.float()
        At = A.transpose(0, 1).coalesce()
        grad_x = torch.sparse.mm(At, go)
        return None, grad_x.to(dtype=x.dtype, device=x.device)


def sparse_mm_fp32_safe(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sparse @ dense matmul in float32 (safe under AMP). Public for graph encoder variants."""
    return _SparseDenseMM.apply(A, x)


# Backward-compatible alias
_sparse_mm_fp32_safe = sparse_mm_fp32_safe


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
        A = self.A.to(self.E.weight.device)
        w = self.E.weight
        embs = [w]
        x = w
        # Nested off: keeps sparse ops out of autocast; custom Function fixes backward half-grads from AMP above.
        amp_off = torch.amp.autocast("cuda", enabled=False) if w.is_cuda else nullcontext()
        with amp_off:
            for _ in range(self.K):
                x = sparse_mm_fp32_safe(A, x)
                embs.append(x)
            out = torch.stack(embs).mean(dim=0)
        return out

    def forward(self, u, i):
        all_emb = self.propagate()
        uemb = all_emb[u]
        iemb = all_emb[self.n_users + i]
        return (uemb * iemb).sum(dim=-1)
