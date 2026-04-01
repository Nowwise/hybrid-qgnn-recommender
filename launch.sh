#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker compose up --build -d
echo ""
echo "Dashboard   http://localhost:8080"
echo "API /docs   http://localhost:8080/docs"
echo ""
echo "Logs:  docker compose logs -f"
echo "Stop:  docker compose down"
