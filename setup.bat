@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo   MIRA setup - Mandala-Inspired Memory Arch
echo ============================================

rem --- 1. verify Python ---
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found on PATH. Install Python 3.11 from python.org
    exit /b 1
)
python --version

rem --- 2. create .venv (never touches the system Python) ---
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create .venv
        exit /b 1
    )
)

rem --- 3. install dependencies (idempotent) ---
set "TMP=%~dp0.tmp"
set "TEMP=%~dp0.tmp"
if not exist ".tmp" mkdir ".tmp"
echo Installing dependencies ^(cached, safe to re-run^)...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install --cache-dir "%~dp0.pipcache" -r requirements.txt
if errorlevel 1 (
    echo [WARN] Some optional dependencies failed - continuing with core stack.
)

rem --- 4. NVIDIA / CUDA check (informational only) ---
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >nul 2>nul
if errorlevel 1 (
    echo [INFO] No NVIDIA GPU visible - MIRA will run CPU-only.
) else (
    echo [INFO] NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
)

rem --- 5. create directories + initialize SQLite ---
".venv\Scripts\python.exe" -c "from config.auto_config import ensure_dirs; ensure_dirs(); from storage.sqlite_store import SQLiteStore; from config.auto_config import DB_PATH; s=SQLiteStore(DB_PATH); s.close(); print('SQLite initialized at', DB_PATH)"
if errorlevel 1 (
    echo [ERROR] Initialization failed - check the error above.
    exit /b 1
)

rem --- 6. validate installation ---
".venv\Scripts\python.exe" -c "import numpy, faiss, networkx, streamlit, plotly, pandas; print('Core imports OK')"
if errorlevel 1 (
    echo [ERROR] Validation failed - re-run setup.bat.
    exit /b 1
)

echo.
echo ============================================
echo   Setup complete. Run:  run.bat
echo ============================================
