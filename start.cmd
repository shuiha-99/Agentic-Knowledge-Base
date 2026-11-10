@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set "PYTHONPATH=%~dp0src"
if not exist ".venv\Scripts\python.exe" (
  echo 请先按照 README.md 准备项目环境。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m agentic_kb start-ollama
if errorlevel 1 (
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m agentic_kb serve --open-browser
pause
