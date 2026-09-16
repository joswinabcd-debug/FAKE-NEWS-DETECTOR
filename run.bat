@echo off
setlocal enabledelayedexpansion

:: Move to script directory dynamically without hardcoded paths
cd /d "%~dp0"

echo ===================================================
echo             TRUTHSCAN AI - LAUNCHER
echo   Deep Learning Fake News Detection (CNN & LSTM)
echo ===================================================
echo.

:: Check if Python is available in PATH
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your PATH.
    echo Please install Python 3.9+ and ensure "Add Python to PATH" is checked.
    echo.
    pause
    exit /b 1
)

:: Check or create virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Creating virtual environment in .\venv...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created successfully.
)

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Check if requirements need to be installed
echo [INFO] Verifying installed dependencies...
python -c "import torch, streamlit, sklearn, matplotlib, pandas, tqdm, openpyxl" >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Installing required packages from requirements.txt...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Package installation encountered an issue.
        pause
        exit /b 1
    )
    echo [INFO] All packages installed successfully.
) else (
    echo [INFO] Dependencies verified.
)

:: Starting application
echo.
echo ===================================================
echo Starting TRUTHSCAN AI...
echo Opening the application at http://localhost:8501...
echo ===================================================
echo.

streamlit run app.py --server.port 8501 --server.headless false

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Streamlit exited with an error code: %errorlevel%
    pause
)
