@echo off
REM ============================================================================
REM  setup_scheduler.bat — Install World Cup automated refresh in Task Scheduler
REM
REM  IMPORTANT: Right-click this file and select "Run as Administrator".
REM  Task Scheduler requires admin privileges to create scheduled tasks.
REM ============================================================================

setlocal
set PROJECT_DIR=%~dp0
set PYTHON_EXE=C:\Users\dell\AppData\Local\Python\pythoncore-3.14-64\python.exe
set SCRIPT=%PROJECT_DIR%refresh_worldcup.py
set LOG_FILE=%PROJECT_DIR%refresh.log

echo.
echo ========================================================
echo   Installing World Cup Auto-Refresh Task
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
    echo      Right-click setup_scheduler.bat and select
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

REM ---- Create scheduled task ----
REM Runs every 6 hours during the tournament (at 02:00, 08:00, 14:00, 20:00)
REM Adjust /mo to change frequency: /mo 12 for every 12 hours, /mo 24 for daily
schtasks /delete /tn "WorldCupRefresh" /f >nul 2>&1

schtasks /create ^
    /tn "WorldCupRefresh" ^
    /tr "\"%PYTHON_EXE%\" -u \"%SCRIPT%\" --quiet --log-file \"%LOG_FILE%\"" ^
    /sc hourly ^
    /mo 6 ^
    /st 08:00 ^
    /ru "%USERNAME%" ^
    /rl highest ^
    /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Task "WorldCupRefresh" created successfully.
    echo      Runs every 6 hours (02:00, 08:00, 14:00, 20:00).
    echo.
) else (
    echo.
    echo [ERROR] Failed to create task (error %ERRORLEVEL%).
    echo.
    pause
    exit /b 1
)

REM ---- Quick smoke test ----
echo Testing the refresh script (data only, skip training)...
"%PYTHON_EXE%" -u "%SCRIPT%" --skip-train --quiet
if %ERRORLEVEL% EQU 0 (
    echo [OK] Smoke test passed — refresh script works.
) else (
    echo [WARNING] Smoke test failed — check Python/log.
)

echo.
echo ========================================================
echo  All set! The task will automatically refresh World Cup
echo  predictions every 6 hours.
echo ========================================================
echo.
echo  Useful commands:
echo    Run now:        schtasks /run /tn "WorldCupRefresh"
echo    View status:    schtasks /query /tn "WorldCupRefresh"
echo    Delete task:    schtasks /delete /tn "WorldCupRefresh" /f
echo    View log:       type "%LOG_FILE%"
echo.
pause
