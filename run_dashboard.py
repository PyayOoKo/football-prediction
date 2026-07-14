"""
Streamlit Dashboard — Football Match Predictor.

Launch with::

    streamlit run run_dashboard.py

Or::

    python run_dashboard.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the Streamlit dashboard."""
    project_root = Path(__file__).resolve().parent
    app_path = project_root / "src" / "app" / "dashboard.py"

    if not app_path.exists():
        print(f"✗ Dashboard file not found at {app_path}")
        sys.exit(1)

    print("=" * 60)
    print("  FOOTBALL PREDICTION DASHBOARD")
    print("=" * 60)
    print(f"\n  Starting Streamlit from: {app_path.parent}")
    print("  Open your browser to the URL below.\n")

    # Ensure the project root is on sys.path so dashboard pages
    # can ``from config import config`` and ``from src import …``
    existing_pp = os.environ.get("PYTHONPATH", "")
    paths = [str(project_root)] + [p for p in existing_pp.split(os.pathsep) if p]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--browser.serverAddress", "localhost",
        "--server.runOnSave", "true",
    ]

    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
    except subprocess.CalledProcessError as exc:
        print(f"\n  ✗ Streamlit exited with code {exc.returncode}")
        sys.exit(exc.returncode)
    except FileNotFoundError:
        print("\n  ✗ Streamlit not found. Install with: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
