@echo off
REM ============================================================
REM  Daily Football Data Collection
REM  Runs every day to download fresh match data from
REM  football-data.co.uk for all configured leagues.
REM
REM  Called by Windows Task Scheduler (setup_football_data_scheduler.ps1)
REM  Logs written to logs/daily_collection_YYYYMMDD.log
REM ============================================================
setlocal enabledelayedexpansion

SET SCRIPT_DIR=%~dp0
SET LOG_DIR=%SCRIPT_DIR%logs

REM Build a safe date-stamped log filename (locale-independent)
REM Uses PowerShell if available, then wmic, then %DATE% fallback
set DT=
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd" 2^>nul') do set DT=%%i
if "%DT%"=="" (
    for /f "tokens=2 delims==." %%a in ('wmic os get LocalDateTime /value 2^>nul') do set DT=%%a
)
if "%DT%"=="" (
    set DT=%DATE:~-4%%DATE:~3,2%%DATE:~0,2%
)
set LOG_FILE=%LOG_DIR%\daily_collection_%DT:~0,8%.log

REM Create logs directory if it doesn't exist
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo [%DATE% %TIME%] Starting football data collection... >> "%LOG_FILE%"

REM Activate venv and run the collector
cd /d "%SCRIPT_DIR%"
if not exist "%SCRIPT_DIR%.venv\Scripts\activate.bat" (
    echo [%DATE% %TIME%] ERROR: Virtual environment not found at .venv\ >> "%LOG_FILE%"
    exit /b 1
)
call .venv\Scripts\activate.bat

REM Run the collector (only football-data, skip weather)
python -m football_data.scheduler.update_daily --skip-weather >> "%LOG_FILE%" 2>&1

SET EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE%==0 (
    echo [%DATE% %TIME%] Collection completed successfully >> "%LOG_FILE%"
) else (
    echo [%DATE% %TIME%] Collection FAILED with exit code %EXIT_CODE% >> "%LOG_FILE%"
)

REM Keep only the last 14 log files (2 weeks)
for /f "skip=14" %%f in ('dir "%LOG_DIR%\daily_collection_*.log" /b /o-d 2^>nul') do (
    del "%LOG_DIR%\%%f" 2>nul
)

echo [%DATE% %TIME%] Done (exit code %EXIT_CODE%). >> "%LOG_FILE%"
exit /b %EXIT_CODE%
