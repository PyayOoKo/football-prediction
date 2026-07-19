@echo off
REM ============================================================================
REM  run_pipeline_task.bat — Wrapper for scheduled pipeline run.
REM  Called by Windows Task Scheduler. Handles path quoting correctly.
REM ============================================================================
setlocal
set PYTHON_EXE=C:\Users\dell\AppData\Local\Python\pythoncore-3.14-64\python.exe

if not exist "%~dp0logs\scheduler\" mkdir "%~dp0logs\scheduler\"

"%PYTHON_EXE%" -u "%~dp0run_pipeline.py" --lightweight >> "%~dp0logs\scheduler\pipeline.log" 2>&1
