"""
install_value_bets_scheduler.py — Install Daily Value Bets as Windows Scheduled Task.

Run this script as Administrator:
    python install_value_bets_scheduler.py

Creates a task that runs today_value_bets_live.py daily at 7:00 AM.
"""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    python_exe = sys.executable
    script = project_root / "today_value_bets_live.py"
    log_dir = project_root / "logs" / "scheduler"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "value_bets.log"

    task_name = "FootballValueBets"

    print("=" * 60)
    print("  Installing Daily Value Bets Scheduler")
    print("=" * 60)
    print(f"  Task name:  {task_name}")
    print(f"  Schedule:   Daily at 07:00")
    print(f"  Python:     {python_exe}")
    print(f"  Script:     {script}")
    print(f"  Log:        {log_file}")
    print()

    # Delete existing task if any
    subprocess.run(
        ["schtasks", "/delete", "/tn", task_name, "/f"],
        capture_output=True, text=True,
    )

    # Create the task - runs daily at 7:00 AM
    # Uses --quiet for silent operation with logging to file
    # Results saved to reports/value_bets/latest.csv for dashboard
    task_cmd = (
        f'"{python_exe}" -u "{script}" '
        f'--quiet --days 1'
    )

    result = subprocess.run(
        [
            "schtasks", "/create",
            "/tn", task_name,
            "/tr", task_cmd,
            "/sc", "daily",
            "/st", "07:00",
            "/rl", "highest",
            "/f",
        ],
        capture_output=True, text=True,
    )

    if result.returncode == 0:
        print("  [+] Task created successfully!")
        print(f"  [+] Runs daily at 07:00")
        print(f"  [+] Log file: {log_file}")
        print()
        print("  Use 'schtasks /run /tn FootballValueBets' to test it now.")
    else:
        print(f"  [!] Failed (error {result.returncode})")
        print(f"      {result.stderr.strip()}")
        print()
        print("  [!] Administrator privileges may be required.")
        print()
        print("  To run as Admin:")
        print(f"     1. Right-click on cmd.exe -> 'Run as Administrator'")
        print(f"     2. cd {project_root}")
        print(f"     3. python install_value_bets_scheduler.py")

    print()
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
