<#
.SYNOPSIS
    Sets up a Windows Scheduled Task to run daily football data collection.

.DESCRIPTION
    Creates a scheduled task that runs run_football_data_collection.bat
    every day at a configurable time (default: 6:00 AM).

    Run this from an Administrator PowerShell prompt:
        powershell -ExecutionPolicy Bypass -File setup_football_data_scheduler.ps1

.PARAMETER TaskTime
    Time of day to run (24h format). Default: "06:00" (6:00 AM).
.PARAMETER TaskName
    Name for the scheduled task. Default: "FootballDataDailyCollection".
#>

param(
    [string]$TaskTime = "06:00",
    [string]$TaskName = "FootballDataDailyCollection"
)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Football Data Daily Collection Scheduler" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Get the script's directory (project root)
$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$BatchFile = Join-Path $ScriptPath "run_football_data_collection.bat"

# Validate the batch file exists
if (-not (Test-Path $BatchFile)) {
    Write-Host "ERROR: $BatchFile not found!" -ForegroundColor Red
    Write-Host "Make sure you run this script from the project root directory." -ForegroundColor Red
    exit 1
}

Write-Host "Project root: $ScriptPath" -ForegroundColor Gray
Write-Host "Batch script:  $BatchFile" -ForegroundColor Gray
Write-Host ""

# Check if running as Administrator
$isAdmin = ([System.Security.Principal.WindowsPrincipal] [System.Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "WARNING: Not running as Administrator!" -ForegroundColor Yellow
    Write-Host "Task Scheduler registration requires admin privileges." -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne "y") {
        Write-Host "Aborted. Please re-run as Administrator." -ForegroundColor Red
        exit 1
    }
}

# Parse the task time
try {
    $TaskTimeObj = [DateTime]::ParseExact($TaskTime, "HH:mm", $null)
} catch {
    Write-Host "ERROR: Invalid time format '$TaskTime'. Use HH:mm (e.g., '06:00' or '18:30')." -ForegroundColor Red
    exit 1
}

Write-Host "Task name:   $TaskName" -ForegroundColor White
Write-Host "Run time:    $TaskTime daily" -ForegroundColor White
Write-Host ""

# ── Remove existing task if present ──
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task '$TaskName'..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# ── Create the scheduled task ──
$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"$BatchFile`"" `
    -WorkingDirectory $ScriptPath

$Trigger = New-ScheduledTaskTrigger `
    -Daily `
    -At $TaskTimeObj

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
    -Priority 5 `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew

$Principal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest

Write-Host "Registering scheduled task..." -ForegroundColor Cyan
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description "Daily football data collection from football-data.co.uk. Downloads fresh match results and odds for all configured leagues."

    Write-Host ""
    Write-Host "SUCCESS! Task '$TaskName' registered." -ForegroundColor Green
    Write-Host ""
    Write-Host "Details:" -ForegroundColor White
    Write-Host "  Runs at:    $TaskTime daily" -ForegroundColor Gray
    Write-Host "  Runs as:    SYSTEM" -ForegroundColor Gray
    Write-Host "  Time limit: 1 hour" -ForegroundColor Gray
    Write-Host "  Logs:       logs/daily_collection_*.log" -ForegroundColor Gray
    Write-Host ""
    Write-Host "To test immediately, run:" -ForegroundColor Yellow
    Write-Host "  Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To remove:" -ForegroundColor Yellow
    Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false" -ForegroundColor Yellow

} catch {
    Write-Host "ERROR: Failed to register task: $_" -ForegroundColor Red
    exit 1
}
