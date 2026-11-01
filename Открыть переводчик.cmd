@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Сначала выполните установку по инструкции в README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -B -m src.gui
if errorlevel 1 pause
