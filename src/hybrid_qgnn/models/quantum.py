from __future__ import annotations

import math

import pennylane as qml
import torch
import torch.nn as nn


class QuantumBlock(nn.Module):
    def __init__(self, q=3, L=1, in_dim=64, dev_name="lightning.qubit"):
        super().__init__()
        self.q, self.L = q, L
        self.proj = nn.Linear(in_dim, q)
        try:
            self.dev = qml.device(dev_name, wires=q)
        except Exception:
            self.dev = qml.device("lightning.qubit", wires=q)
        self.weights = nn.Parameter(torch.randn(L, q, 2, dtype=torch.float32) * 0.1)

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(x, w):
            qml.templates.AngleEmbedding(x, wires=range(self.q), rotation="Y")
            for l in range(self.L):
                for wi in range(self.q):
                    qml.CNOT(wires=[wi, (wi + 1) % self.q])
                for wi in range(self.q):
                    qml.RX(w[l, wi, 0], wires=wi)
                    qml.RY(w[l, wi, 1], wires=wi)
            return [qml.expval(qml.PauliZ(wi)) for wi in range(self.q)]

        self.circuit = circuit

    def forward(self, x, micro_bs=32):
        xq = torch.tanh(self.proj(x)) * math.pi / 2.0
        outs = []
        for s in range(0, xq.shape[0], micro_bs):
            xb = xq[s : s + micro_bs]
            for b in range(xb.shape[0]):
                out_b = self.circuit(xb[b], self.weights)
                outs.append(torch.stack(out_b).to(x.dtype))
        return torch.stack(outs, dim=0)
