@echo off
REM ============================================================================
REM  setup_value_bets_scheduler.bat — Daily value bets automated run
REM
REM  IMPORTANT: Right-click this file and select "Run as Administrator".
REM  Task Scheduler requires admin privileges to create scheduled tasks.
REM
REM  Runs today_value_bets_live.py every morning at 7:00 AM, generating
REM  live value bets with Dixon-Coles features, live odds, and Platt
REM  calibration. Results are saved to reports/value_bets/latest.csv
REM  for the Streamlit dashboard to pick up.
REM ============================================================================

setlocal
set PROJECT_DIR=%~dp0
set PYTHON_EXE=C:\Users\dell\AppData\Local\Python\pythoncore-3.14-64\python.exe
set SCRIPT=%PROJECT_DIR%today_value_bets_live.py
set LOG_FILE=%PROJECT_DIR%reports\value_bets\daily_run.log

echo.
echo ========================================================
echo   Installing Daily Value Bets Task
echo ========================================================
echo.
echo Project:  %PROJECT_DIR%
echo Python:   %PYTHON_EXE%
echo Script:   %SCRIPT%
echo Log:      %LOG_FILE%
echo.

REM ---- Check if running as Administrator ----
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!!] This script must be run as Administrator.
    echo      Right-click setup_value_bets_scheduler.bat and select
    echo      "Run as Administrator", then try again.
    echo.
    pause
    exit /b 1
)

REM ---- Check Python ----
if not exist "%PYTHON_EXE%" (
    echo [WARNING] Python not found at %PYTHON_EXE%
    echo Will try "python" from PATH instead.
    set PYTHON_EXE=python
)

REM ---- Check script exists ----
if not exist "%SCRIPT%" (
    echo [ERROR] Script not found at:
    echo          %SCRIPT%
    echo.
    pause
    exit /b 1
)

REM ---- Delete existing task if any ----
schtasks /delete /tn "FootballValueBets" /f >nul 2>&1

REM ---- Create scheduled task ----
REM Runs daily at 7:00 AM. The script handles its own logging
REM via --quiet and --log-file flags (same pattern as refresh_worldcup.py).
schtasks /create ^
    /tn "FootballValueBets" ^
    /tr "\"%PYTHON_EXE%\" -u \"%SCRIPT%\" --quiet --log-file \"%LOG_FILE%\"" ^
    /sc daily ^
    /st 07:00 ^
    /ru "%USERNAME%" ^
    /rl highest ^
    /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Task "FootballValueBets" created successfully.
    echo      Runs daily at 07:00.
    echo.
) else (
    echo.
    echo [ERROR] Failed to create task (error %ERRORLEVEL%).
    echo.
    pause
    exit /b 1
)

REM ---- Quick test ----
echo Testing the value bets script (press Ctrl+C to skip)...
echo.
"%PYTHON_EXE%" -u "%SCRIPT%" --fast --calibrate none --no-save --days 1
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Smoke test passed.
) else (
    echo.
    echo [WARNING] Smoke test returned exit code %ERRORLEVEL%.
    echo           Check %LOG_FILE% for details.
)

echo.
echo ========================================================
echo  All set! Value bets will be generated daily at 07:00.
echo ========================================================
echo.
echo  Useful commands:
echo    Run now:        schtasks /run /tn "FootballValueBets"
echo    View status:    schtasks /query /tn "FootballValueBets"
echo    View log:       type "%LOG_FILE%"
echo    Delete task:    schtasks /delete /tn "FootballValueBets" /f
echo.
pause
