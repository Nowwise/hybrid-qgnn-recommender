#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
echo ""
echo "GPU API stack (NVIDIA Container Toolkit required on host)"
echo "Dashboard   http://localhost:8080"
echo "API /docs   http://localhost:8080/docs"
echo "GET         http://localhost:8080/api/experiments/device"
echo ""
echo "Logs:  docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f"
echo "Stop:  docker compose -f docker-compose.yml -f docker-compose.gpu.yml down"
