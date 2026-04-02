#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --build \
  -e PYTHONPATH=/app/src \
  api python /app/scripts/verify_gpu_smoke.py
