"""Vercel ASGI entrypoint: exports the FastAPI app from backend/app (see backend/app/main.py)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_BACKEND = _ROOT / "backend"
_SRC = _ROOT / "src"
for path in (_BACKEND, _SRC):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from app.main import app  # noqa: E402

__all__ = ["app"]
