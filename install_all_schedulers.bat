@echo off
REM ============================================================================
REM  install_all_schedulers.bat — One-click installer for both schedulers.
REM
REM  Installs:
REM    1. FootballPipeline  — runs run_pipeline.py --lightweight every 6 hours
REM    2. FootballValueBets — runs today_value_bets_live.py daily at 07:00
REM
REM  IMPORTANT: Right-click this file and select "Run as Administrator".
REM ============================================================================

setlocal
set PROJECT_DIR=%~dp0
set PYTHON_EXE=C:\Users\dell\AppData\Local\Python\pythoncore-3.14-64\python.exe

echo ========================================================
echo   Installing Both Schedulers
echo ========================================================
echo.
echo Project:  %PROJECT_DIR%
echo Python:   %PYTHON_EXE%
echo.

REM ---- Check if running as Administrator ----
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!!] This script must be run as Administrator.
    echo      Right-click install_all_schedulers.bat and select
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

REM ========================================================
REM  SCHEDULER 1: FootballPipeline (every 6 hours)
REM ========================================================
echo.
echo --- [1/2] Installing Pipeline Scheduler (every 6 hours) ---
echo.

schtasks /delete /tn "FootballPipeline" /f >nul 2>&1

schtasks /create ^
    /tn "FootballPipeline" ^
    /tr "%PROJECT_DIR%run_pipeline_task.bat" ^
    /sc hourly ^
    /mo 6 ^
    /st 08:00 ^
    /rl highest ^
    /f

if %ERRORLEVEL% EQU 0 (
    echo [OK] Task "FootballPipeline" created.
    echo      Runs every 6 hours (02:00, 08:00, 14:00, 20:00).
    echo      Script: %PROJECT_DIR%run_pipeline_task.bat
) else (
    echo [ERROR] Failed to create "FootballPipeline" (error %ERRORLEVEL%).
)

REM ========================================================
REM  SCHEDULER 2: FootballValueBets (daily at 07:00)
REM ========================================================
echo.
echo --- [2/2] Installing Value Bets Scheduler (daily at 07:00) ---
echo.

schtasks /delete /tn "FootballValueBets" /f >nul 2>&1

schtasks /create ^
    /tn "FootballValueBets" ^
    /tr "%PROJECT_DIR%run_value_bets_task.bat" ^
    /sc daily ^
    /st 07:00 ^
    /rl highest ^
    /f

if %ERRORLEVEL% EQU 0 (
    echo [OK] Task "FootballValueBets" created.
    echo      Runs daily at 07:00.
    echo      Script: %PROJECT_DIR%run_value_bets_task.bat
) else (
    echo [ERROR] Failed to create "FootballValueBets" (error %ERRORLEVEL%).
)

REM ========================================================
REM  VERIFICATION
REM ========================================================
echo.
echo ========================================================
echo  Verification
echo ========================================================
echo.

schtasks /query /tn "FootballPipeline" /fo LIST /v 2>nul | findstr /i "TaskName|Task To Run|Schedule|Next Run" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Pipeline scheduler is installed.
    schtasks /query /tn "FootballPipeline" /fo LIST /v | findstr /i "TaskName|Task To Run|Schedule"
) else (
    echo [WARNING] Pipeline scheduler not found — may need admin rights to verify.
)

echo.

schtasks /query /tn "FootballValueBets" /fo LIST /v 2>nul | findstr /i "TaskName|Task To Run|Schedule|Next Run" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Value bets scheduler is installed.
    schtasks /query /tn "FootballValueBets" /fo LIST /v | findstr /i "TaskName|Task To Run|Schedule"
) else (
    echo [WARNING] Value bets scheduler not found — may need admin rights to verify.
)

REM ========================================================
REM  SUMMARY
REM ========================================================
echo.
echo ========================================================
echo  All done!
echo ========================================================
echo.
echo  Installed:
echo    FootballPipeline    - Every 6 hours (02, 08, 14, 20)
echo    FootballValueBets   - Daily at 07:00
echo.
echo  Run manually:
echo    schtasks /run /tn "FootballPipeline"
echo    schtasks /run /tn "FootballValueBets"
echo.
pause
