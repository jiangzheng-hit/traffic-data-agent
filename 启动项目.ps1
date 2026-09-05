$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (Test-Path -LiteralPath "$projectRoot\.venv\Scripts\python.exe") {
    $python = "$projectRoot\.venv\Scripts\python.exe"
} else {
    $python = "python"
}

$env:PYTHONPATH = "$projectRoot\src"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
Set-Location -LiteralPath $projectRoot
& $python -m streamlit run app.py --browser.gatherUsageStats false --server.showEmailPrompt false
