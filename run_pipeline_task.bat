@echo off
REM ============================================================================
REM  run_pipeline_task.bat — Wrapper for scheduled pipeline run.
REM  Called by Windows Task Scheduler. Handles path quoting correctly.
REM ============================================================================
setlocal
set PYTHON_EXE=%~dp0.venv\Scripts\python.exe

if not exist "%~dp0logs\scheduler\" mkdir "%~dp0logs\scheduler\"

"%PYTHON_EXE%" -u "%~dp0run_pipeline.py" --lightweight --skip-value-bets >> "%~dp0logs\scheduler\pipeline.log" 2>&1
