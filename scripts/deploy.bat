@echo off
REM =========================================================
REM  Football Prediction System - Windows Deployment Script
REM  Usage: scripts\deploy.bat [command]
REM
REM  Commands:
REM    build        Build Docker images
REM    push         Push images to registry
REM    up           Start all services
REM    down         Stop all services
REM    restart      Restart all services
REM    logs         View service logs
REM    status       Check service status
REM    backup       Backup the database
REM    restore      Restore the database from latest backup
REM    migrate      Run database migrations
REM    setup        Full setup: build -> migrate -> up
REM    health       Check health of all services
REM    help         Show this help message
REM =========================================================

setlocal enabledelayedexpansion

REM -- Check Docker --
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker is not installed.  Install it from: https://docs.docker.com/get-docker/
    exit /b 1
)

set CMD=%1
if "%CMD%"=="" set CMD=help

if /I "%CMD%"=="build"   goto :build
if /I "%CMD%"=="push"    goto :push
if /I "%CMD%"=="up"      goto :up
if /I "%CMD%"=="down"    goto :down
if /I "%CMD%"=="restart" goto :restart
if /I "%CMD%"=="logs"    goto :logs
if /I "%CMD%"=="status"  goto :status
if /I "%CMD%"=="backup"  goto :backup
if /I "%CMD%"=="restore" goto :restore
if /I "%CMD%"=="migrate" goto :migrate
if /I "%CMD%"=="setup"   goto :setup
if /I "%CMD%"=="health"  goto :health
if /I "%CMD%"=="help"    goto :help

echo [ERROR] Unknown command: %CMD%
echo   Usage: %0 [command]
echo   Run '%0 help' for available commands.
exit /b 1

:build
echo [INFO] Building Docker images...
docker compose -f docker-compose.yml build
if %ERRORLEVEL% equ 0 ( echo [OK] Build complete ) else ( echo [ERROR] Build failed & exit /b 1 )
goto :end

:push
echo [INFO] Pushing image to registry...
if "%DOCKER_REGISTRY%"=="" (set REGISTRY=ghcr.io) else (set REGISTRY=%DOCKER_REGISTRY%)
if "%DOCKER_IMAGE%"=="" (set IMG_NAME=football-prediction) else (set IMG_NAME=%DOCKER_IMAGE%)
if "%DOCKER_TAG%"=="" (set IMG_TAG=latest) else (set IMG_TAG=%DOCKER_TAG%)
docker tag football-prediction:latest %REGISTRY%/%IMG_NAME%:%IMG_TAG%
docker push %REGISTRY%/%IMG_NAME%:%IMG_TAG%
echo [OK] Push complete
goto :end

:up
echo [INFO] Starting all services...
if exist .env ( echo [INFO] Using .env ) else ( echo [WARN] No .env file found. Using defaults. )
docker compose -f docker-compose.yml up -d
if %ERRORLEVEL% equ 0 ( echo [OK] Services started ) else ( echo [ERROR] Failed to start & exit /b 1 )
goto :end

:down
echo [INFO] Stopping all services...
docker compose -f docker-compose.yml down
echo [OK] Services stopped
goto :end

:restart
call :down
call :up
goto :end

:logs
shift
docker compose -f docker-compose.yml logs -f %*
goto :end

:status
echo.
echo ========================================
echo   SERVICE STATUS
echo ========================================
docker compose -f docker-compose.yml ps
echo.
echo ========================================
echo   DISK USAGE
echo ========================================
for %%d in (data models logs) do (
    if exist %%d (
        for /f "tokens=3" %%s in ('dir /-c /w "%%d" 2^>nul ^| findstr /b "  "') do set SIZ=%%s
        echo   %%d: !SIZ! bytes
    )
)
goto :end

:backup
echo [INFO] Backing up database...
set BACKUP_DIR=data\backups
if not exist %BACKUP_DIR% mkdir %BACKUP_DIR%

REM PowerShell for reliable timestamp (locale-independent)
for /f %%i in ('powershell -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"') do set TS=%%i
set BACKUP_FILE=%BACKUP_DIR%\football_db_%TS%.sql

docker compose exec db pg_dump --no-owner --no-acl -U postgres football_prediction > "%BACKUP_FILE%"
if %ERRORLEVEL% equ 0 (
    echo [OK] Backup created: %BACKUP_FILE%
    REM Keep last 7 backups
    for /f "skip=7" %%b in ('dir /b /o-d "%BACKUP_DIR%\football_db_*.sql" 2^>nul') do (
        del "%BACKUP_DIR%\%%b"
        echo [INFO] Removed old backup: %%b
    )
) else (
    echo [ERROR] Backup failed
    exit /b 1
)
goto :end

:restore
set BACKUP_DIR=data\backups
set RESTORE_FILE=%2
if "%RESTORE_FILE%"=="" set RESTORE_FILE=%BACKUP_DIR%\football_db_latest.sql
if not exist "%RESTORE_FILE%" (
    echo [ERROR] Backup file not found: %RESTORE_FILE%
    echo   Usage: %0 restore [backup-file]
    echo   Available:
    dir /b "%BACKUP_DIR%\football_db_*.sql" 2>nul
    exit /b 1
)
echo [WARN] About to restore database from: %RESTORE_FILE%
echo   This will OVERWRITE the current database!
set /p CONFIRM="  Are you sure? (y/N): "
if /I not "!CONFIRM!"=="y" ( echo [INFO] Restore cancelled. & goto :end )
REM Decompress if gzipped
set RESTORE_FILE_EXT=%RESTORE_FILE:~-3%
if /I "%RESTORE_FILE_EXT%"==".gz" (
    where gzip >nul 2>nul
    if !ERRORLEVEL! equ 0 (
        gzip -d -c "%RESTORE_FILE%" | docker compose exec -T db psql -U postgres -d football_prediction
    ) else (
        echo [ERROR] gzip not found.  Decompress the file manually first.
        exit /b 1
    )
) else (
    docker compose exec -T db psql -U postgres -d football_prediction < "%RESTORE_FILE%"
)
echo [OK] Database restored from: %RESTORE_FILE%
goto :end

:migrate
echo [INFO] Running database migrations...
docker compose run --rm migrate
if %ERRORLEVEL% equ 0 ( echo [OK] Migrations complete ) else ( echo [ERROR] Migrations failed & exit /b 1 )
goto :end

:setup
echo [INFO] === Full Setup ===
if not exist .env (
    if exist .env.example (
        copy .env.example .env
        echo [WARN] Created .env from .env.example. Edit it with your settings.
    )
)
call :build
docker compose up -d db
echo [INFO] Waiting for database...
timeout /t 5 /nobreak >nul
call :migrate
call :up
echo [OK] === Setup Complete ===
goto :end

:health
echo.
echo ========================================
echo   HEALTH CHECK
echo ========================================
for %%s in (app db) do (
    docker compose ps --services --filter "status=running" 2>nul | findstr /i "%%s" >nul
    if !ERRORLEVEL! equ 0 ( echo [OK] %%s: running ) else ( echo [ERROR] %%s: not running )
)
echo.
for /f "tokens=1-3" %%a in ('powershell -Command "Get-PSDrive C ^| Select-Object Used,Free"') do (
    echo   Disk: %%a used
)
goto :end

:help
echo Football Prediction System - Windows Deployment Script
echo.
echo Usage: %0 [command]
echo.
echo Commands:
echo   build        Build Docker images
echo   push         Push images to registry
echo   up           Start all services
echo   down         Stop all services
echo   restart      Restart all services
echo   logs         View service logs
echo   status       Check service status
echo   backup       Backup the database
echo   restore      Restore the database from latest backup
echo   migrate      Run database migrations
echo   setup        Full setup: build -^> migrate -^> up
echo   health       Check health of all services
echo   help         Show this help message
goto :end

:end
endlocal
