from __future__ import annotations

import math
from typing import Any, List, Sequence

import pennylane as qml
import torch
import torch.nn as nn


def _stack_z_expvals(raw: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Normalize batched or single QNode outputs to shape (batch, q)."""
    if isinstance(raw, torch.Tensor):
        t = raw.to(device=device, dtype=dtype)
        if t.ndim == 1:
            return t.unsqueeze(-1)
        return t
    seq: Sequence[Any] = raw
    tensors: List[torch.Tensor] = []
    for t in seq:
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, device=device, dtype=dtype)
        else:
            t = t.to(device=device, dtype=dtype)
        tensors.append(t.reshape(t.shape[0]))
    return torch.stack(tensors, dim=-1)


class QuantumBlock(nn.Module):
    """
    Variational quantum layer. Forward uses PennyLane ``batch_input`` when possible so each
    ``micro_bs`` chunk runs as **one** batched device execution instead of a per-sample Python loop
    (large practical speedup for training). Falls back to the legacy loop if the transform or
    ``adjoint`` combo is unsupported for the active device.
    """

    def __init__(self, q=3, L=1, in_dim=64, dev_name="lightning.qubit", entangle: bool = True):
        super().__init__()
        self.q, self.L = q, L
        self.entangle = bool(entangle)
        self.proj = nn.Linear(in_dim, q)
        try:
            self.dev = qml.device(dev_name, wires=q)
        except Exception:
            self.dev = qml.device("lightning.qubit", wires=q)
        self.weights = nn.Parameter(torch.randn(L, q, 2, dtype=torch.float32) * 0.1)

        ent = self.entangle
        nq = self.q
        nL = self.L
        dev = self.dev

        def _body(x, w):
            qml.templates.AngleEmbedding(x, wires=range(nq), rotation="Y")
            for ell in range(nL):
                if ent:
                    for wi in range(nq):
                        qml.CNOT(wires=[wi, (wi + 1) % nq])
                for wi in range(nq):
                    qml.RX(w[ell, wi, 0], wires=wi)
                    qml.RY(w[ell, wi, 1], wires=wi)
            return [qml.expval(qml.PauliZ(wi)) for wi in range(nq)]

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def _single_circuit(x, w):
            return _body(x, w)

        self._single_circuit = _single_circuit
        self._batched_circuit = None
        if hasattr(qml, "batch_input"):
            try:
                batched = qml.batch_input(argnum=0)(_single_circuit)
                with torch.no_grad():
                    probe = torch.zeros(2, nq, dtype=torch.float32)
                    raw = batched(probe, self.weights)
                    _ = _stack_z_expvals(raw, dtype=torch.float32, device=probe.device)
                self._batched_circuit = batched
            except Exception:
                self._batched_circuit = None

    def _forward_chunk_batched(self, xb: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        assert self._batched_circuit is not None
        raw = self._batched_circuit(xb, self.weights)
        return _stack_z_expvals(raw, dtype=dtype, device=xb.device)

    def _forward_chunk_loop(self, xb: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        rows: List[torch.Tensor] = []
        for b in range(xb.shape[0]):
            out_b = self._single_circuit(xb[b], self.weights)
            rows.append(torch.stack(out_b).to(dtype))
        return torch.stack(rows, dim=0)

    def forward(self, x, micro_bs=32):
        xq = torch.tanh(self.proj(x)) * math.pi / 2.0
        use_batch = self._batched_circuit is not None
        outs: List[torch.Tensor] = []
        for s in range(0, xq.shape[0], micro_bs):
            xb = xq[s : s + micro_bs]
            if use_batch:
                try:
                    outs.append(self._forward_chunk_batched(xb, x.dtype))
                    continue
                except Exception:
                    use_batch = False
                    self._batched_circuit = None
            outs.append(self._forward_chunk_loop(xb, x.dtype))
        return torch.cat(outs, dim=0)
