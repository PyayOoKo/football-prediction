"""
Coverage Analyzer — computes data coverage across key football dimensions.

Measures what fraction of the dataset has non-null values for critical
columns: odds, expected goals (xG), league classification, season
coverage, and schema completeness.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_quality.models import CoverageMetrics

logger = logging.getLogger(__name__)

# ── Standard column patterns for auto-detection ─────────
_ODDS_PATTERNS = [
    "bbavh", "bbavd", "bbava",
    "b365h", "b365d", "b365a",
    "psh", "psd", "psa",
    "maxh", "maxd", "maxa",
    "avh", "avd", "ava",
    "home_odds", "draw_odds", "away_odds",
    "odds_home", "odds_draw", "odds_away",
]

_XG_PATTERNS = [
    "xg", "expected_goals", "x_g", "xga", "x g home",
    "x_g_home", "x_g_away", "xg_home", "xg_away",
    "home_xg", "away_xg", "expected_goals_home",
    "expected_goals_away",
]

_EXPECTED_SCHEMA = [
    "date", "season", "league",
    "home_team", "away_team",
    "home_goals", "away_goals", "result",
]


def _match_pattern(col: str, patterns: list[str]) -> bool:
    """Check if a column name matches any pattern."""
    cl = col.lower().strip()
    return any(p in cl for p in patterns)


class CoverageAnalyzer:
    """Analyzes dataset coverage across key football data dimensions.

    Parameters
    ----------
    expected_schema : list[str], optional
        Columns that should exist in every valid dataset.
    odds_patterns : list[str], optional
        Column name substrings identifying odds columns.
    xg_patterns : list[str], optional
        Column name substrings identifying xG columns.
    """

    def __init__(
        self,
        expected_schema: list[str] | None = None,
        odds_patterns: list[str] | None = None,
        xg_patterns: list[str] | None = None,
    ) -> None:
        self.expected_schema = expected_schema or _EXPECTED_SCHEMA
        self.odds_patterns = odds_patterns or _ODDS_PATTERNS
        self.xg_patterns = xg_patterns or _XG_PATTERNS

    def analyze(self, df: pd.DataFrame) -> CoverageMetrics:
        """Run all coverage analyses on the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            The football match dataset.

        Returns
        -------
        CoverageMetrics
            All coverage statistics.
        """
        metrics = CoverageMetrics()
        if df.empty:
            return metrics

        # ── 1. Odds Coverage ──────────────────────────────
        odds_cols = [c for c in df.columns if _match_pattern(c, self.odds_patterns)]
        if odds_cols:
            # Any row that has at least one non-null odds column
            has_odds = df[odds_cols].notna().any(axis=1)
            metrics.odds_coverage_pct = float(
                has_odds.sum() / len(df) * 100
            )
        else:
            metrics.odds_coverage_pct = 0.0

        # ── 2. xG Coverage ────────────────────────────────
        xg_cols = [c for c in df.columns if _match_pattern(c, self.xg_patterns)]
        if xg_cols:
            has_xg = df[xg_cols].notna().any(axis=1)
            metrics.xg_coverage_pct = float(
                has_xg.sum() / len(df) * 100
            )
        else:
            metrics.xg_coverage_pct = 0.0

        # ── 3. League Coverage ────────────────────────────
        league_cols = [c for c in ["league", "division", "competition", "league_name"]
                       if c in df.columns]
        if league_cols:
            league_col = league_cols[0]
            df[league_col] = df[league_col].fillna("UNKNOWN")
            metrics.league_coverage = {
                str(k): int(v)
                for k, v in df[league_col].value_counts().items()
            }
            # Percentage mapped to known leagues (not UNKNOWN)
            known = df[league_col] != "UNKNOWN"
            metrics.league_coverage_pct = float(
                known.sum() / len(df) * 100
            ) if len(df) > 0 else 0.0

        # ── 4. Season Coverage ────────────────────────────
        season_cols = [c for c in ["season", "Season", "season_name"]
                       if c in df.columns]
        if season_cols:
            season_col = season_cols[0]
            df[season_col] = df[season_col].fillna("UNKNOWN")
            metrics.season_coverage = {
                str(k): int(v)
                for k, v in df[season_col].value_counts().items()
            }
            metrics.season_count = len(metrics.season_coverage)

        # ── 5. Schema Validation ──────────────────────────
        present = set(df.columns)
        expected = set(self.expected_schema)
        metrics.n_columns_expected = len(self.expected_schema)
        metrics.n_columns_actual = len(df.columns)
        metrics.columns_missing = sorted(expected - present)
        metrics.columns_added = sorted(present - expected)[:20]

        logger.info(
            "Coverage: odds=%.1f%% xg=%.1f%% leagues=%d seasons=%d schema=%d/%d",
            metrics.odds_coverage_pct,
            metrics.xg_coverage_pct,
            len(metrics.league_coverage),
            metrics.season_count,
            metrics.n_columns_actual,
            metrics.n_columns_expected,
        )
        return metrics

    @staticmethod
    def from_csv(
        csv_path: str | Path,
        **kwargs: Any,
    ) -> CoverageMetrics:
        """Load a CSV and analyze coverage in one step.

        Parameters
        ----------
        csv_path : str | Path
            Path to the CSV file.
        **kwargs
            Passed to ``pd.read_csv``.

        Returns
        -------
        CoverageMetrics
        """
        df = pd.read_csv(csv_path, low_memory=False, **kwargs)
        analyzer = CoverageAnalyzer()
        return analyzer.analyze(df)

    @staticmethod
    def from_parquet(
        parquet_path: str | Path,
        **kwargs: Any,
    ) -> CoverageMetrics:
        """Load a Parquet file and analyze coverage in one step.

        Parameters
        ----------
        parquet_path : str | Path
            Path to the Parquet file.
        **kwargs
            Passed to ``pd.read_parquet``.

        Returns
        -------
        CoverageMetrics
        """
        df = pd.read_parquet(parquet_path, **kwargs)
        analyzer = CoverageAnalyzer()
        return analyzer.analyze(df)
