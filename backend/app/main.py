"""Uvicorn entry: run from `backend/` with: uvicorn app.main:app --reload --port 8000"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _BACKEND_ROOT.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.settings import get_settings
from app.routers import datasets, experiments, health

settings = get_settings()

app = FastAPI(
    title="Hybrid QGNN Research API",
    description="Backend for LightGCN vs Hybrid QGNN training runs and metrics.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")
app.include_router(experiments.router, prefix="/api")
