@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found. Follow the setup instructions in README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -B -m src.gui %*
set "run_exit=%errorlevel%"
if not "%run_exit%"=="0" pause
exit /b %run_exit%
