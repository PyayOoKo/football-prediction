@echo off
REM ============================================================================
REM  run_value_bets_task.bat — Wrapper for scheduled value bets run.
REM  Called by Windows Task Scheduler. Handles path quoting correctly.
REM ============================================================================
setlocal
set PYTHON_EXE=C:\Users\dell\AppData\Local\Python\pythoncore-3.14-64\python.exe

if not exist "%~dp0logs\scheduler\" mkdir "%~dp0logs\scheduler\"

"%PYTHON_EXE%" -u "%~dp0today_value_bets_live.py" --quiet --days 3 >> "%~dp0logs\scheduler\value_bets.log" 2>&1
