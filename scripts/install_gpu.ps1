# Install dependencies with CUDA-enabled PyTorch (NVIDIA driver required).
# Two pip steps: PyPI for all deps, then PyTorch CUDA wheels only (pip merges indexes badly if combined).
# Run:  powershell -ExecutionPolicy Bypass -File scripts/install_gpu.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "Creating venv..."
    python -m venv venv
}

& ".\venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\venv\Scripts\pip.exe" install -r requirements-base.txt
& ".\venv\Scripts\pip.exe" install -r requirements-gpu-torch.txt

Write-Host ""
Write-Host "Installing pennylane-lightning-gpu (optional; may be unavailable on Windows)..."
$oldEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& ".\venv\Scripts\pip.exe" install -r requirements-gpu-quantum.txt
if ($LASTEXITCODE -ne 0) {
    Write-Warning "pennylane-lightning-gpu install failed — quantum sim will use lightning.qubit only."
}
$ErrorActionPreference = $oldEap

Write-Host ""
Write-Host "Checking PyTorch CUDA..."
& ".\venv\Scripts\python.exe" -c "import torch; print('torch', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

Write-Host ""
Write-Host "Running GPU smoke test (tiny HybridQGNN forward)..."
& ".\venv\Scripts\python.exe" "scripts\verify_gpu_smoke.py"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "verify_gpu_smoke.py exited with code $LASTEXITCODE"
}
