# ==============================================================================
# demo.ps1
# Windows PowerShell quick demo script.
# Mirrors demo.sh behavior for review/demo on Windows.
# ==============================================================================

Write-Host "========================================================="
Write-Host "Start quick demo (DRPM-NSCV / HERLN)"
Write-Host "========================================================="

Set-Location -Path "$PSScriptRoot\src"

Write-Host ">>> Running demo model (Dataset: ICEWS14s, Epochs: 20)..."
python main.py `
    -d ICEWS14s `
    --self-loop `
    --layer-norm `
    --weight 0.5 `
    --theta 1 `
    --relation-prediction `
    --relation-evaluation `
    --task-weight 0.0 `
    --gpu 0 `
    --freq-reg 5e-4 `
    --alpha 10 `
    --n-epochs 20 `
    --temporal-gating `
    --time-embedding-alpha 0.05

if ($LASTEXITCODE -ne 0) {
    Write-Error "Demo failed, exit code: $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "========================================================="
Write-Host "Demo finished. Check checkpoints for outputs."
Write-Host "========================================================="
