@echo off
REM ============================================================================
REM  run_value_bets_task.bat — Wrapper for scheduled league value bets run.
REM  Called by Windows Task Scheduler. Handles path quoting correctly.
REM  Uses today_league_value_bets.py for per-league models with DC blend.
REM  *** Top 5 leagues: E0 +11.67%, D1 +17.12%, F1 +19.32% Over ROI ***
REM  *** Waiting for August 2026 season start for fixtures ***
REM ============================================================================
setlocal
set PYTHON_EXE=%~dp0.venv\Scripts\python.exe

if not exist "%~dp0logs\scheduler\" mkdir "%~dp0logs\scheduler\"

"%PYTHON_EXE%" -u "%~dp0today_league_value_bets.py" --leagues E0 D1 F1 --calibrate hybrid --kelly 0.25 --min-ev 0.05 --ou --btts --quiet >> "%~dp0logs\scheduler\value_bets.log" 2>&1
