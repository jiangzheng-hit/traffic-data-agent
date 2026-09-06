@echo off
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "PYTHON_EXE=python"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo First-time setup: creating the project Python environment...
    python -m venv "%~dp0.venv"
    if errorlevel 1 (
        echo Could not create .venv. Please make sure Python 3 is installed.
        pause
        exit /b 1
    )
)
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
"%PYTHON_EXE%" -c "import sklearn, streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing missing project libraries. This only happens on first setup or after an incomplete install.
    "%PYTHON_EXE%" -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo Library installation failed. Please check your network and run this file again.
        pause
        exit /b 1
    )
)
echo Starting Traffic Data Agent...
echo The browser will open at http://localhost:8501
"%PYTHON_EXE%" -m streamlit run app.py --browser.gatherUsageStats false --server.showEmailPrompt false
if errorlevel 1 (
    echo.
    echo Startup failed. Please open README.md for help.
    pause
)
