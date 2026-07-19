"""
install_scheduler.py — Install Football Pipeline as Windows Scheduled Task.

Run this script as Administrator:
    python install_scheduler.py

Creates a task that runs run_pipeline.py --lightweight every 6 hours.
"""

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    python_exe = sys.executable
    script = project_root / "run_pipeline.py"
    log_dir = project_root / "logs" / "scheduler"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    task_name = "FootballPipeline"
    # Task runs every 6 hours starting at 8:00 AM
    cmd = (
        f'schtasks /create /tn "{task_name}" '
        f'/tr "\'{python_exe}\' -u \'{script}\' --lightweight >> \'{log_file}\' 2>&1" '
        f"/sc hourly /mo 6 /st 08:00 "
        f"/rl highest /f"
    )

    print("=" * 60)
    print("  Installing Football Pipeline Scheduler")
    print("=" * 60)
    print(f"  Task name:  {task_name}")
    print(f"  Schedule:   Every 6 hours (starting 08:00)")
    print(f"  Python:     {python_exe}")
    print(f"  Script:     {script}")
    print(f"  Log:        {log_file}")
    print()

    # Delete existing task if any
    subprocess.run(
        ["schtasks", "/delete", "/tn", task_name, "/f"],
        capture_output=True, text=True,
    )

    # Create the task
    result = subprocess.run(
        [
            "schtasks", "/create",
            "/tn", task_name,
            "/tr", f'"{python_exe}" -u "{script}" --lightweight',
            "/sc", "hourly",
            "/mo", "6",
            "/st", "08:00",
            "/rl", "highest",
            "/f",
        ],
        capture_output=True, text=True,
    )

    if result.returncode == 0:
        print("  OK Task created successfully!")
        print(f"  Next runs: 02:00, 08:00, 14:00, 20:00")
    else:
        print(f"  FAIL Failed (error {result.returncode})")
        print(f"     {result.stderr.strip()}")
        print()
        print("  Tasks requiring Administrator privileges may fail if")
        print("  this script is not run as Administrator.")
        print()
        print("  To run as Admin:")
        print(f"    1. Right-click on cmd.exe -> 'Run as Administrator'")
        print(f"    2. cd {project_root}")
        print(f"    3. python install_scheduler.py")

    print()
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
