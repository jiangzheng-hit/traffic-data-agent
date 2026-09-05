@echo off
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "PYTHON_EXE=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
echo Starting Traffic Data Agent...
echo The browser will open at http://localhost:8501
"%PYTHON_EXE%" -m streamlit run app.py --browser.gatherUsageStats false --server.showEmailPrompt false
if errorlevel 1 (
    echo.
    echo Startup failed. Please open README.md for help.
    pause
)
