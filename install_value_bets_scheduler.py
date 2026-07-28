"""
install_value_bets_scheduler.py — Install Daily League Value Bets as Windows Scheduled Task.

Run this script as Administrator:
    python install_value_bets_scheduler.py

Creates a task that runs today_league_value_bets.py daily at 7:00 AM
on profitable top 5 European leagues (E0, D1, F1) with RF O/U model,
BTTS implied model, hybrid calibration, and O/U + BTTS value betting.

Leagues selected based on backtest profitability:
- E0 (EPL): +11.67% Over ROI (2023-2024)
- D1 (Bundesliga): +17.12% Over ROI (2023-2024)
- F1 (Ligue 1): +19.32% Over ROI (2023-2024)
- SP1 (La Liga) and I1 (Serie A) excluded due to negative Over ROI
- SE1 was removed due to -0.87% ROI (breakeven)

Waiting for August 2026 season start for new fixtures to appear.
"""

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
    return sys.executable


def main() -> int:
    project_root = Path(__file__).resolve().parent
    bat_script = project_root / "run_value_bets_task.bat"

    task_name = "FootballValueBets"

    print("=" * 60)
    print("  Installing Daily League Value Bets Scheduler")
    print("=" * 60)
    print(f"  Task name:  {task_name}")
    print(f"  Schedule:   Daily at 07:00")
    print(f"  Script:     {bat_script.name}")
    print(f"  Leagues:    SE1, SWE, NOR, FI")
    print()

    # Delete existing task if any
    subprocess.run(
        ["schtasks", "/delete", "/tn", task_name, "/f"],
        capture_output=True, text=True,
    )

    # Create the task using the batch wrapper
    result = subprocess.run(
        [
            "schtasks", "/create",
            "/tn", task_name,
            "/tr", f'"{bat_script}"',
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
        print(f"  [+] Uses: {bat_script.name}")
        print()
        print("  Test now:    schtasks /run /tn FootballValueBets")
        print("  View logs:   type logs\\scheduler\\value_bets.log")
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
