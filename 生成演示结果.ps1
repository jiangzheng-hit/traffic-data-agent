$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (Test-Path -LiteralPath "$projectRoot\.venv\Scripts\python.exe") {
    $python = "$projectRoot\.venv\Scripts\python.exe"
} else {
    $python = "python"
}

$env:PYTHONPATH = "$projectRoot\src"
$env:PYTHONDONTWRITEBYTECODE = "1"
Set-Location -LiteralPath $projectRoot
& $python -m traffic_data_agent.cli `
    --input "data\raw\traffic_ml_homework_dataset.csv" `
    --target "is_congested" `
    --split "time" `
    --output "outputs\latest"

Write-Host "演示结果已生成：$projectRoot\outputs\latest" -ForegroundColor Green
