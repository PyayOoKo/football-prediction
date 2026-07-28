"""
install_scheduler.py — Install Football Pipeline as Windows Scheduled Task.

Run this script as Administrator:
    python install_scheduler.py

Creates a task that runs run_pipeline.py --lightweight every 6 hours
(02:00, 08:00, 14:00, 20:00) via the project's virtual environment.
"""

import os
import subprocess
import sys
from pathlib import Path


def _venv_python(project_root: Path) -> str:
    """Return the project's virtual environment Python path."""
    candidates = [
        project_root / ".venv" / "Scripts" / "python.exe",
        project_root / "venv" / "Scripts" / "python.exe",
        project_root / ".venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    # Fallback to whatever Python is currently running
    return sys.executable


def main() -> int:
    project_root = Path(__file__).resolve().parent
    python_exe = _venv_python(project_root)
    bat_script = project_root / "run_pipeline_task.bat"
    log_dir = project_root / "logs" / "scheduler"
    log_dir.mkdir(parents=True, exist_ok=True)

    task_name = "FootballPipeline"

    print("=" * 60)
    print("  Installing Football Pipeline Scheduler")
    print("=" * 60)
    print(f"  Task name:  {task_name}")
    print(f"  Schedule:   Every 6 hours (02:00, 08:00, 14:00, 20:00)")
    print(f"  Python:     {python_exe}")
    print(f"  Batch:      {bat_script}")
    print()

    # Delete existing task if any
    subprocess.run(
        ["schtasks", "/delete", "/tn", task_name, "/f"],
        capture_output=True, text=True,
    )

    # Create the task using the batch wrapper (more robust)
    result = subprocess.run(
        [
            "schtasks", "/create",
            "/tn", task_name,
            "/tr", f'"{bat_script}"',
            "/sc", "hourly",
            "/mo", "6",
            "/st", "08:00",
            "/rl", "highest",
            "/f",
        ],
        capture_output=True, text=True,
    )

    if result.returncode == 0:
        print("  [+] Task created successfully!")
        print(f"  [+] Runs every 6 hours: 02:00, 08:00, 14:00, 20:00")
        print(f"  [+] Uses: {bat_script.name}")
        print()
        print("  Test now:    schtasks /run /tn FootballPipeline")
        print("  View logs:   type logs\\scheduler\\pipeline.log")
    else:
        print(f"  [!] Failed (error {result.returncode})")
        print(f"      {result.stderr.strip()}")
        print()
        print("  [!] Administrator privileges required.")
        print()
        print("  To run as Admin:")
        print(f"     1. Right-click on cmd.exe -> 'Run as Administrator'")
        print(f"     2. cd {project_root}")
        print(f"     3. python install_scheduler.py")

    print()
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
