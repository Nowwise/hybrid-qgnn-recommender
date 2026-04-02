"""Training device selection: CUDA when available, with explicit CPU/GPU overrides."""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional, Tuple

import torch

ENV_DEVICE = "QGNN_DEVICE"


def _probe_penny_lane_device(name: str, wires: int = 1) -> Tuple[bool, Optional[str]]:
    """Try to construct a PennyLane device; return (ok, error snippet)."""
    try:
        import pennylane as qml

        _ = qml.device(name, wires=wires)
        return True, None
    except Exception as e:
        return False, str(e)[:240]


def compute_device_summary() -> Dict[str, Any]:
    """Introspection for APIs or notebooks (PyTorch + PennyLane simulators)."""
    cuda = torch.cuda.is_available()
    devices: list[Dict[str, Any]] = []
    if cuda:
        for i in range(torch.cuda.device_count()):
            devices.append(
                {
                    "id": f"cuda:{i}",
                    "kind": "cuda",
                    "name": torch.cuda.get_device_name(i),
                }
            )
    devices.append({"id": "cpu", "kind": "cpu", "name": "CPU"})
    q_ok, q_err = _probe_penny_lane_device("lightning.qubit")
    g_ok, g_err = _probe_penny_lane_device("lightning.gpu")
    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda,
        "cuda_device_count": torch.cuda.device_count() if cuda else 0,
        "cuda_version": torch.version.cuda,
        "devices": devices,
        "quantum_simulators": {
            "lightning.qubit": {"available": q_ok, **({} if q_ok else {"error": q_err})},
            "lightning.gpu": {"available": g_ok, **({} if g_ok else {"error": g_err})},
        },
    }


def resolve_quantum_backend(requested: str, torch_device: torch.device) -> Tuple[str, Dict[str, Any]]:
    """
    Resolve PennyLane device string for :class:`~hybrid_qgnn.models.quantum.QuantumBlock`.

    ``lightning.gpu`` is only used when PyTorch runs on CUDA and the plugin loads; otherwise
    falls back to ``lightning.qubit``.
    """
    meta: Dict[str, Any] = {"requested": requested}
    if requested != "lightning.gpu":
        return requested, {**meta, "resolved": requested, "note": "PennyLane backend as requested"}

    if torch_device.type != "cuda":
        return "lightning.qubit", {
            **meta,
            "resolved": "lightning.qubit",
            "note": "lightning.gpu needs PyTorch on CUDA; using lightning.qubit",
            "fallback_reason": "torch_not_cuda",
        }

    ok, err = _probe_penny_lane_device("lightning.gpu")
    if ok:
        return "lightning.gpu", {**meta, "resolved": "lightning.gpu", "note": "PennyLane Lightning GPU"}

    return "lightning.qubit", {
        **meta,
        "resolved": "lightning.qubit",
        "note": f"lightning.gpu unavailable ({err or 'error'})",
        "fallback_reason": "lightning_gpu_unavailable",
    }


def resolve_training_device(
    preference: Optional[str] = None,
) -> Tuple[torch.device, Dict[str, Any]]:
    """
    Choose ``torch.device`` for training with safe CPU fallback.

    Precedence: explicit ``preference`` (e.g. :attr:`ExperimentConfig.device`)
    → env :envvar:`QGNN_DEVICE` → ``auto``.

    - ``auto``: CUDA if :func:`torch.cuda.is_available` else CPU.
    - ``cpu``: always CPU.
    - ``cuda`` / ``cuda:N``: that GPU when CUDA is available and index is valid;
      otherwise CPU with a warning (so runs never fail solely due to GPU config).
    """
    raw_pref = preference if preference is not None else os.environ.get(ENV_DEVICE)
    if raw_pref is None or str(raw_pref).strip() == "":
        requested = "auto"
        token = "auto"
    else:
        requested = str(raw_pref).strip()
        token = requested.lower()

    cuda_ok = torch.cuda.is_available()
    n_gpu = torch.cuda.device_count() if cuda_ok else 0

    meta: Dict[str, Any] = {
        "requested": requested,
        "cuda_available": cuda_ok,
        "cuda_device_count": n_gpu,
    }

    def with_cpu(note: str, **extra: Any) -> Tuple[torch.device, Dict[str, Any]]:
        return torch.device("cpu"), {**meta, "resolved": "cpu", "note": note, **extra}

    def with_cuda(resolved: str, note: str) -> Tuple[torch.device, Dict[str, Any]]:
        return torch.device(resolved), {**meta, "resolved": resolved, "note": note}

    if token in ("auto", "default"):
        if cuda_ok:
            return with_cuda("cuda", "auto: CUDA available — using GPU")
        return with_cpu("auto: CUDA not available — using CPU")

    if token == "cpu":
        return with_cpu("CPU requested")

    if token == "cuda" or token.startswith("cuda:"):
        if not cuda_ok:
            warnings.warn(
                f"Device {requested!r} was requested but CUDA is not available "
                "(install a CUDA-enabled PyTorch build or set device to 'cpu'); using CPU.",
                UserWarning,
                stacklevel=2,
            )
            return with_cpu(
                "CUDA requested but not available — using CPU",
                fallback_reason="cuda_unavailable",
            )

        if token == "cuda":
            return with_cuda("cuda", "CUDA requested — using default GPU")

        try:
            idx = int(requested.split(":", 1)[1])
        except (IndexError, ValueError):
            warnings.warn(
                f"Invalid CUDA index in {requested!r}; using cuda:0.",
                UserWarning,
                stacklevel=2,
            )
            idx = 0
        if idx < 0 or idx >= n_gpu:
            warnings.warn(
                f"CUDA device index {idx} is out of range (0..{n_gpu - 1}); using CPU.",
                UserWarning,
                stacklevel=2,
            )
            return with_cpu(
                "Invalid GPU index — using CPU",
                fallback_reason="invalid_cuda_index",
            )
        return with_cuda(f"cuda:{idx}", f"Using GPU {idx}")

    warnings.warn(
        f"Unknown {ENV_DEVICE} / device preference {requested!r}; behaving like 'auto'.",
        UserWarning,
        stacklevel=2,
    )
    if cuda_ok:
        return with_cuda("cuda", "unknown preference — auto-selected CUDA")
    return with_cpu("unknown preference — auto-selected CPU")
