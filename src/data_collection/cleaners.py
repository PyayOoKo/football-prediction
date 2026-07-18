"""
Cleaners — remove duplicates, handle missing values, and validate data.

Every user-facing cleaning function returns a new DataFrame and logs
the number of rows affected so you can track what happened.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that together identify a unique match (for dedup)
MATCH_KEY_COLS = ["date", "home_team", "away_team", "league"]

# Columns whose values are numeric match stats that should be filled with 0
_ZEROABLE_STATS = [
    "home_goals",
    "away_goals",
    "home_goals_ht",
    "away_goals_ht",
    "home_shots",
    "away_shots",
    "home_shots_target",
    "away_shots_target",
    "home_corners",
    "away_corners",
    "home_fouls",
    "away_fouls",
    "home_yellow",
    "away_yellow",
    "home_red",
    "away_red",
]

# ── Public API ──────────────────────────────────────────


def deduplicate(df: pd.DataFrame, key: list[str] | None = None) -> pd.DataFrame:
    """Remove duplicate rows, keeping the most recently downloaded copy.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    key : list[str], optional
        Columns used to identify duplicates.  Defaults to
        ``[date, home_team, away_team, league]``.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame with duplicates removed.
    """
    if key is None:
        key = MATCH_KEY_COLS

    before = len(df)
    # Only use columns that actually exist in the DataFrame
    existing_key = [col for col in key if col in df.columns]

    if not existing_key:
        logger.warning("No deduplication key columns found — skipping dedup")
        return df

    # Sort so the most recent download wins when duplicates exist
    if "downloaded_at" in df.columns:
        df = df.sort_values("downloaded_at", ascending=False)

    df = df.drop_duplicates(subset=existing_key, keep="first")
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate rows (%.1f%%)", removed, removed / before * 100)
    else:
        logger.info("No duplicates found")
    return df.reset_index(drop=True)


def handle_missing_values(
    df: pd.DataFrame,
    strategy: Literal["drop", "fill_zero", "fill_median"] = "fill_zero",
    max_missing_pct: float = 50.0,
) -> pd.DataFrame:
    """Handle missing values across the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    strategy : str
        - ``"drop"`` — drop rows with any missing values in essential columns.
        - ``"fill_zero"`` — fill match-stat columns with 0, forward-fill
          team names, leave result as-is.
        - ``"fill_median"`` — fill numeric columns with median, drop if result
          is missing.
    max_missing_pct : float
        If a column has more than this percentage of missing values, the
        entire column is dropped (default 50%).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = df.copy()
    before = len(df)

    # ── Drop overly sparse columns ─────────────────
    null_pct = df.isnull().mean() * 100
    cols_to_drop = null_pct[null_pct > max_missing_pct].index.tolist()
    if cols_to_drop:
        logger.warning(
            "Dropping %d columns with >%.0f%% missing: %s",
            len(cols_to_drop),
            max_missing_pct,
            cols_to_drop,
        )
        df.drop(columns=cols_to_drop, inplace=True)

    # ── Apply strategy ─────────────────────────────
    if strategy == "drop":
        before_drop = len(df)
        essential = MATCH_KEY_COLS + ["result"]
        essential_cols = [c for c in essential if c in df.columns]
        df.dropna(subset=essential_cols, inplace=True)
        logger.info("Dropped %d rows with missing essential values", before_drop - len(df))

    elif strategy in ("fill_zero", "fill_median"):

        # Fill match-stat columns
        stat_cols = [c for c in _ZEROABLE_STATS if c in df.columns]
        if strategy == "fill_zero":
            df[stat_cols] = df[stat_cols].fillna(0)
        else:  # fill_median
            df[stat_cols] = df[stat_cols].fillna(df[stat_cols].median())

        # Forward-fill team names (should rarely be missing)
        for col in ["home_team", "away_team"]:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].ffill()

        # Drop rows where the result is still missing (unplayable match)
        if "result" in df.columns:
            missing_result = df["result"].isnull().sum()
            if missing_result:
                df.dropna(subset=["result"], inplace=True)
                logger.info("Dropped %d rows with missing result", missing_result)

    logger.info(
        "Missing-value handling complete: %d rows (%.1f%% kept)",
        len(df),
        len(df) / before * 100 if before else 0,
    )
    return df.reset_index(drop=True)


def standardise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a consistent, predictable column set.

    Adds any missing standard columns as ``NaN`` and reorders columns
    into a logical grouping.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.

    Returns
    -------
    pd.DataFrame
        Schema-standardised DataFrame.
    """
    preferred_order = [
        "season",
        "date",
        "league",
        "home_team",
        "away_team",
        "result",
        "home_goals",
        "away_goals",
        "home_goals_ht",
        "away_goals_ht",
        "result_ht",
        "home_shots",
        "away_shots",
        "home_shots_target",
        "away_shots_target",
        "home_corners",
        "away_corners",
        "home_fouls",
        "away_fouls",
        "home_yellow",
        "away_yellow",
        "home_red",
        "away_red",
        "source",
        "downloaded_at",
    ]

    # Add any missing columns as NaN
    for col in preferred_order:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder: put preferred columns first, then any extras
    extra_cols = [c for c in df.columns if c not in preferred_order]
    ordered = [c for c in preferred_order if c in df.columns] + extra_cols
    df = df[ordered]

    return df


def validate_data(df: pd.DataFrame) -> dict[str, Any]:
    """Run basic validation checks and return a report.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.

    Returns
    -------
    dict[str, Any]
        Validation report with keys ``"is_valid"``, ``"warnings"``, and
        ``"stats"``.
    """
    warnings: list[str] = []
    stats: dict[str, Any] = {}

    stats["rows"] = len(df)
    stats["columns"] = len(df.columns)
    stats["missing_cells"] = int(df.isnull().sum().sum())
    stats["missing_pct"] = round(df.isnull().mean().mean() * 100, 2)

    # Check essential columns
    for col in MATCH_KEY_COLS:
        if col not in df.columns:
            warnings.append(f"Missing required column: '{col}'")

    # Check date range
    if "date" in df.columns and not df["date"].isnull().all():
        valid_dates = df["date"].dropna()
        stats["date_min"] = str(valid_dates.min())
        stats["date_max"] = str(valid_dates.max())
        stats["date_range_days"] = (valid_dates.max() - valid_dates.min()).days
    else:
        warnings.append("No valid dates found")

    # Check result distribution
    if "result" in df.columns:
        stats["result_distribution"] = (
            df["result"].value_counts(normalize=True).to_dict()
        )

    # Check for any all-null columns
    all_null = [c for c in df.columns if df[c].isnull().all()]
    if all_null:
        warnings.append(f"Entirely null columns: {all_null}")

    # Check for extreme values
    for col in ["home_goals", "away_goals"]:
        if col in df.columns:
            max_val = df[col].max()
            if not pd.isna(max_val) and max_val >= 15:
                warnings.append(f"Suspiciously high value in '{col}': {max_val}")

    is_valid = len(warnings) == 0
    report = {"is_valid": is_valid, "warnings": warnings, "stats": stats}

    if is_valid:
        logger.info("Validation passed — %d rows, %d columns", stats["rows"], stats["columns"])
    else:
        logger.warning("Validation found %d issues", len(warnings))
        for w in warnings:
            logger.warning("  ⚠ %s", w)

    return report
