# Build/run the Linux GPU API image and run verify_gpu_smoke.py inside it (quantum lightning.gpu works there).
# Prerequisites: Docker Desktop (Linux containers, WSL2) + GPU enabled + NVIDIA driver on Windows.
# From repo root:
#   powershell -ExecutionPolicy Bypass -File scripts/docker-gpu-verify.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$compose = @(
    "compose",
    "-f", "docker-compose.yml",
    "-f", "docker-compose.gpu.yml",
    "run", "--rm", "--build",
    "-e", "PYTHONPATH=/app/src",
    "api",
    "python", "/app/scripts/verify_gpu_smoke.py"
)

Write-Host "Building GPU image (first time is slow) and running smoke test in container..."
& docker @compose
exit $LASTEXITCODE
