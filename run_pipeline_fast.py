"""
run_pipeline_fast.py — Run the full pipeline with optimised defaults.

With the pipeline optimisation changes (Phase 2 — feature caching and
DC refit tuning) now baked into ``config.py`` (refit_every=2000) and
``run_pipeline.py``, this script still provides a convenient shortcut:
- Overrides DC refit_every to 4000 for extra speed (fewer MLE fits)
- Skips download and value bets by default

Usage:
    python run_pipeline_fast.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    # ── Speed up Dixon-Coles further (config now defaults to 2000) ──
    from config import config
    original_refit = config.dixon_coles.refit_every
    # 2000 → 4000: cuts MLE refits from ~9 to ~4 over 18k rows
    config.dixon_coles.refit_every = 4000
    print(f"[FAST] Dixon-Coles refit_every={config.dixon_coles.refit_every} (default was {original_refit})")

    # ── Run pipeline in-process ──────────────────────────
    from run_pipeline import main as pipeline_main
    print(f"[RUN] pipeline_main(['--skip-download', '--skip-value-bets'])")
    ret = pipeline_main(["--skip-download", "--skip-value-bets"])

    # ── Restore ──────────────────────────────────────────
    config.dixon_coles.refit_every = original_refit
    print(f"[DONE] Dixon-Coles refit_every restored to {original_refit}")
    return ret


if __name__ == "__main__":
    sys.exit(main())
