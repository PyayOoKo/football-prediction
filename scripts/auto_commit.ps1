<#
.SYNOPSIS
    Auto-commit any uncommitted changes and push to GitHub.

.DESCRIPTION
    Checks for file changes in the football prediction project, stages them,
    creates a commit with a timestamp summary, and pushes to GitHub.
    Designed to run on a schedule via Windows Task Scheduler.

    Logs activity to a timestamped log file in the project root.

.NOTES
    Schedule this via Task Scheduler to run every hour (or your preferred interval).
    Setup: .\setup_auto_commit.bat
#>

param(
    [switch]$Quiet
)

# ── Configuration ───────────────────────────────────────
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogFile = Join-Path $ProjectRoot "auto_commit.log"
$GitExe = "git"
$MaxCommitsPerRun = 5   # safety limit to avoid runaway loops

# ── Logging helpers ─────────────────────────────────────
$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

function Write-Log {
    param([string]$Message)
    $line = "$Timestamp  $Message"
    Add-Content -Path $LogFile -Value $line
    if (-not $Quiet) { Write-Host $line }
}

Write-Log "=== Auto-commit starting ==="

# ── 1. Change to project root ──────────────────────────
try {
    Set-Location -Path $ProjectRoot -ErrorAction Stop
    Write-Log "Changed to project root: $ProjectRoot"
}
catch {
    Write-Log "ERROR: Could not change to project root: $_"
    exit 1
}

# ── 2. Check for changes ───────────────────────────────
$status = & $GitExe status --porcelain 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Log "ERROR: Git status failed: $status"
    exit 1
}

if (-not $status) {
    Write-Log "No changes to commit. Exiting."
    exit 0
}

# ── 3. Summarise the changes ───────────────────────────
$changedFiles = @($status -split "`n" | Where-Object { $_ -ne "" })
$fileCount = $changedFiles.Count
Write-Log "Found $fileCount changed file(s):"

# Categorise changes
$added = 0; $modified = 0; $deleted = 0; $renamed = 0
foreach ($line in $changedFiles) {
    $prefix = $line.Substring(0, 2).Trim()
    switch -Wildcard ($prefix) {
        "A?"  { $added++ }
        "M?"  { $modified++ }
        "D?"  { $deleted++ }
        "R?"  { $renamed++ }
        "??"  { $added++ }
        default { $modified++ }
    }
    Write-Log "  $line"
}

# ── 4. Stage all changes ───────────────────────────────
& $GitExe add -A 2>&1 | ForEach-Object { Write-Log "  git add: $_" }
if ($LASTEXITCODE -ne 0) {
    Write-Log "ERROR: Git add failed"
    exit 1
}
Write-Log "Staged all changes."

# ── 5. Build commit message ────────────────────────────
$dateStr = Get-Date -Format "yyyy-MM-dd HH:mm"
$parts = @()
if ($added -gt 0)    { $parts += "+$added" }
if ($modified -gt 0) { $parts += "~$modified" }
if ($deleted -gt 0)  { $parts += "-$deleted" }
$summary = $parts -join " "
$commitMessage = "Auto-commit: $dateStr [$fileCount files $summary]"

# ── 6. Commit ──────────────────────────────────────────
& $GitExe commit -m $commitMessage 2>&1 | ForEach-Object { Write-Log "  git commit: $_" }
if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 1) {
    Write-Log "ERROR: Git commit failed with code $LASTEXITCODE"
    exit 1
}
Write-Log "Committed: $commitMessage"

# ── 7. Push to GitHub ─────────────────────────────────
Write-Log "Pushing to GitHub (origin master)..."
& $GitExe push origin master 2>&1 | ForEach-Object { Write-Log "  git push: $_" }
if ($LASTEXITCODE -ne 0) {
    Write-Log "ERROR: Git push failed with code $LASTEXITCODE"
    Write-Log "  Check: is your remote configured? Any merge conflicts?"
    exit 1
}
Write-Log "Pushed successfully."

Write-Log "=== Auto-commit complete ==="
exit 0
