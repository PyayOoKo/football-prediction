"""
Shared utilities for the feature-engineering sub-package.

Provides:
    _get_target_columns — columns to drop from the feature matrix
    _PREFIX_HOME / _PREFIX_AWAY — column name prefixes
    _match_points — convert result to points for a team
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# ── Column name constants ───────────────────────────────
_PREFIX_HOME = "h_"
_PREFIX_AWAY = "a_"


def _get_target_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that should be dropped from the feature matrix."""
    drop_cols = [
        "target",
        "result",
        "home_goals",
        "away_goals",
        "goal_diff",
        "total_goals",
        "date",
        "season",
        "league",
        "source",
        "downloaded_at",
    ]
    # Also drop any match_id auxiliary columns
    drop_cols.extend([c for c in df.columns if c.endswith("_match_id")])
    # Drop internal row ID (re-attached later by build_features for prediction)
    drop_cols.append("_row_id")
    return drop_cols


def _match_points(result: Any, is_home: bool) -> int:
    """Convert match result to points for a given team.

    Parameters
    ----------
    result : str
        ``"H"``, ``"D"``, or ``"A"``.
    is_home : bool
        Whether the team in question was the home team.

    Returns
    -------
    int
        3 for win, 1 for draw, 0 for loss.
    """
    if result == "H":
        return 3 if is_home else 0
    if result == "A":
        return 0 if is_home else 3
    if result == "D":
        return 1
    return 0
