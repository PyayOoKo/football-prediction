@echo off
REM ============================================================================
REM  setup_auto_commit.bat — Schedule auto-commit + push to GitHub
REM
REM  IMPORTANT: Right-click this file and select "Run as Administrator".
REM  Task Scheduler requires admin privileges to create scheduled tasks.
REM
REM  Installs a task that runs scripts\auto_commit.ps1 every hour,
REM  staging, committing, and pushing any file changes to GitHub.
REM ============================================================================

setlocal
set PROJECT_DIR=%~dp0
set PS_SCRIPT=%PROJECT_DIR%scripts\auto_commit.ps1
set LOG_FILE=%PROJECT_DIR%auto_commit.log

echo.
echo ========================================================
echo   Installing Auto-Commit to GitHub Task
echo ========================================================
echo.
echo Project:  %PROJECT_DIR%
echo Script:   %PS_SCRIPT%
echo Log:      %LOG_FILE%
echo.

REM ---- Check if running as Administrator ----
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!!] This script must be run as Administrator.
    echo      Right-click setup_auto_commit.bat and select
    echo      "Run as Administrator", then try again.
    echo.
    pause
    exit /b 1
)

REM ---- Check PowerShell script exists ----
if not exist "%PS_SCRIPT%" (
    echo [ERROR] PowerShell script not found at:
    echo          %PS_SCRIPT%
    echo.
    pause
    exit /b 1
)

REM ---- Delete existing task if any ----
schtasks /delete /tn "FootballPredictionAutoCommit" /f >nul 2>&1

REM ---- Create scheduled task ----
REM Runs hourly, using the logged-in user's credentials.
REM The PowerShell execution policy is bypassed for this single script.
schtasks /create ^
    /tn "FootballPredictionAutoCommit" ^
    /tr "powershell.exe -ExecutionPolicy Bypass -File \"%PS_SCRIPT%\" -Quiet" ^
    /sc hourly ^
    /mo 1 ^
    /st 00:00 ^
    /du 23:59 ^
    /ru "%USERNAME%" ^
    /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Task "FootballPredictionAutoCommit" created successfully.
    echo      Runs every hour, auto-committing and pushing changes.
    echo.
) else (
    echo.
    echo [ERROR] Failed to create task (error %ERRORLEVEL%).
    echo.
    pause
    exit /b 1
)

REM ---- Quick test ----
echo Testing the auto-commit script...
echo.
powershell.exe -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -Quiet

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Auto-commit script works.
) else (
    echo.
    echo [WARNING] Auto-commit script returned exit code %ERRORLEVEL%.
    echo           Check %LOG_FILE% for details.
)

echo.
echo ========================================================
echo  All set! Changes will auto-commit and push to GitHub
echo  every hour.
echo ========================================================
echo.
echo  Useful commands:
echo    Run now:        schtasks /run /tn "FootballPredictionAutoCommit"
echo    View status:    schtasks /query /tn "FootballPredictionAutoCommit"
echo    View log:       type "%LOG_FILE%"
echo    Delete task:    schtasks /delete /tn "FootballPredictionAutoCommit" /f
echo.
pause
