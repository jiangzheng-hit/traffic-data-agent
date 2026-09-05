@echo off
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
set "PYTHON_EXE=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
echo Generating demo outputs...
"%PYTHON_EXE%" -m traffic_data_agent.cli --input "data\raw\traffic_ml_homework_dataset.csv" --target "is_congested" --split "time" --output "outputs\latest"
if errorlevel 1 (
    echo.
    echo Generation failed. Please open README.md for help.
    pause
) else (
    echo.
    echo Done. Results are in outputs\latest
    pause
)
