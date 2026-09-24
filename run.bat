@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No virtual environment found. Run setup.bat first.
    pause
    exit /b 1
)

set "TMP=%~dp0.tmp"
set "TEMP=%~dp0.tmp"
if not exist ".tmp" mkdir ".tmp"

rem default: web console.  "run.bat streamlit" launches the Streamlit UI instead.
if /i "%~1"=="streamlit" goto streamlit

set "HF_HOME=%~dp0.hf_cache"
echo Starting MIRA web console at http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
".venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8000
goto end

:streamlit
echo Starting MIRA Streamlit UI...
".venv\Scripts\python.exe" -m streamlit run app.py --browser.gatherUsageStats=false

:end
if errorlevel 1 (
    echo [ERROR] MIRA exited with an error.
    pause
    exit /b 1
)
