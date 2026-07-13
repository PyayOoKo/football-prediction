"""
Head-to-Head (H2H) Feature Generator — historical matchup statistics between team pairs.

Computes rolling H2H metrics for every pair of teams that have faced each
other, supporting multiple meeting windows, venue contexts, and optional
stat columns.

Features
--------
+----------------+------------------------------------------+---------+
| Metric         | Description                              | Type    |
+================+==========================================+=========+
| wins           | Team won the H2H meeting                 | binary  |
| draws          | H2H meeting ended in a draw              | binary  |
| losses         | Team lost the H2H meeting                | binary  |
| goals_scored   | Goals the team scored in the H2H meeting | int     |
| goals_conceded | Goals the team conceded in H2H meeting   | int     |
| goal_diff      | goals_scored - goals_conceded            | int     |
| btts           | Both teams scored in the H2H meeting     | binary  |
| over_2.5       | Total goals > 2.5 in the H2H meeting     | binary  |
| clean_sheets   | Team kept a clean sheet in H2H meeting   | binary  |
| xg             | Team's xG in the H2H meeting             | float   |
| xga            | Opponent's xG in the H2H meeting         | float   |
| xgd            | xg - xga                                 | float   |
+----------------+------------------------------------------+---------+

Windows
-------
Default: ``[3, 5, 10]`` — configurable via ``params[\"windows\"]``.

Contexts (3)
------------
- ``overall`` — all H2H meetings regardless of venue
- ``home`` — only H2H meetings where ``team`` was at home
- ``away`` — only H2H meetings where ``team`` was away

Column naming convention
------------------------
Pattern: ``{h|a}_h2h_{context}_{metric}_last{window}``

Examples
~~~~~~~~
- ``h_h2h_overall_wins_last3`` — home team's win rate vs away team in last 3 meetings
- ``a_h2h_away_goals_scored_last5`` — away team's avg goals when playing away vs this opponent
- ``h_h2h_home_clean_sheets_last10`` — home team's clean sheet rate at home vs this opponent

Leakage prevention
------------------
All rolling stats use ``.shift(1)`` so the current match is never included
in its own H2H features. The pair groups are sorted chronologically.

SQL integration
---------------
Supports an optional ``load_fn`` param that, if provided, is called to
load historical H2H data from a database.  The loaded data is merged
with the input DataFrame before computing features.

Usage
-----
::

    from src.feature_framework.features.h2h import H2HTransformer

    t = H2HTransformer(windows=[3, 5, 10])
    t.init()
    result = t.transform(df)

    # With SQL: pass a callable that returns extra historical matches
    def load_db_matches(home_team, away_team):
        with get_session() as session:
            return pd.read_sql(session.query(Match)...)

    t = H2HTransformer(load_fn=load_db_matches)
    t.init()
    result = t.transform(df)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.feature_framework.base import FeatureTransformer
from src.feature_framework.models import TransformContext

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

_DEFAULT_WINDOWS: tuple[int, ...] = (3, 5, 10)
_DEFAULT_CONTEXTS: tuple[str, ...] = ("overall", "home", "away")

# Core and optional metrics for pair-level computation
_H2H_CORE_METRICS: dict[str, dict[str, Any]] = {
    "wins":           {"type": "binary", "description": "Win rate in H2H meetings"},
    "draws":          {"type": "binary", "description": "Draw rate in H2H meetings"},
    "losses":         {"type": "binary", "description": "Loss rate in H2H meetings"},
    "goals_scored":   {"type": "numeric", "description": "Average goals scored"},
    "goals_conceded": {"type": "numeric", "description": "Average goals conceded"},
    "goal_diff":      {"type": "numeric", "description": "Average goal difference"},
    "btts":           {"type": "binary", "description": "Both Teams Scored rate"},
    "over_2.5":       {"type": "binary", "description": "Over 2.5 goals rate"},
    "clean_sheets":   {"type": "binary", "description": "Clean sheet rate"},
}

_H2H_OPTIONAL_METRICS: dict[str, dict[str, Any]] = {
    "xg":  {"type": "numeric", "source_home": "home_xg", "source_away": "away_xg",
            "description": "Average xG in H2H meetings"},
    "xga": {"type": "numeric", "source_home": "away_xg", "source_away": "home_xg",
            "description": "Average xGA conceded in H2H meetings"},
    "xgd": {"type": "numeric", "depends_on": ["xg", "xga"],
            "description": "Average xG difference in H2H meetings"},
}


# ═══════════════════════════════════════════════════════════════
#  H2HTransformer
# ═══════════════════════════════════════════════════════════════


class H2HTransformer(FeatureTransformer):
    """Compute head-to-head historical matchup features between team pairs.

    For each match, computes rolling statistics over the last N meetings
    between the home and away teams (regardless of which side was home/away).
    """

    name: str = "head_to_head"
    version: int = 1
    description: str = (
        "Head-to-head historical statistics: rolling win/draw/loss rates, "
        "goals, xG, BTTS, and clean sheets across the last 3/5/10 meetings "
        "between team pairs (overall, home, and away contexts)."
    )
    dependencies: list[str] = []
    data_type: str = "float"
    computation_time: str = "medium"
    category: str = "h2h"
    author: str = "system"
    tags: list[str] = ["h2h", "head-to-head", "historical", "pairwise"]
    source: str = "derived"

    # output_columns are computed dynamically in init()
    output_columns: list[str] = []

    _REQUIRED_COLS: frozenset[str] = frozenset({
        "date", "home_team", "away_team", "home_goals", "away_goals", "result",
    })

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._resolved_outputs: list[str] = []
        self._resolved_windows: tuple[int, ...] = ()
        self._resolved_contexts: tuple[str, ...] = ()
        self._active_metrics: dict[str, dict[str, Any]] = {}
        self._available_optional_cols: dict[str, str] = {}

    def init(self, context: TransformContext | None = None) -> None:
        self._resolved_windows = self._resolve_windows()
        self._resolved_contexts = self._resolve_contexts()
        self._resolve_output_columns()
        self.output_columns = list(self._resolved_outputs)
        self._initialized = True
        logger.debug(
            "H2HTransformer initialized: %d windows, %d contexts, ~%d columns",
            len(self._resolved_windows), len(self._resolved_contexts),
            len(self.output_columns),
        )

    # ── Param resolution ──────────────────────────────────

    def _resolve_windows(self) -> tuple[int, ...]:
        windows = self.params.get("windows", _DEFAULT_WINDOWS)
        if isinstance(windows, (list, tuple)):
            resolved = tuple(sorted(set(int(w) for w in windows if w > 0)))
            if not resolved:
                return _DEFAULT_WINDOWS
            return resolved
        return _DEFAULT_WINDOWS

    def _resolve_contexts(self) -> tuple[str, ...]:
        contexts = self.params.get("contexts", _DEFAULT_CONTEXTS)
        if isinstance(contexts, (list, tuple)):
            valid = {"overall", "home", "away"}
            return tuple(c for c in contexts if c in valid)
        return _DEFAULT_CONTEXTS

    def _resolve_output_columns(self) -> None:
        windows = self._resolved_windows
        contexts = self._resolved_contexts
        metrics = self._get_active_metrics()

        outputs: list[str] = []
        for metric_name in metrics:
            for ctx in contexts:
                for w in windows:
                    outputs.append(f"h_h2h_{ctx}_{metric_name}_last{w}")
                    outputs.append(f"a_h2h_{ctx}_{metric_name}_last{w}")

        self._resolved_outputs = sorted(set(outputs))

    # ── Input validation ──────────────────────────────────

    def validate_input(self, df: pd.DataFrame) -> list[str]:
        errors: list[str] = []
        for col in self._REQUIRED_COLS:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        return errors

    # ── Core transform ────────────────────────────────────

    def transform(
        self,
        df: pd.DataFrame,
        context: TransformContext | None = None,
    ) -> pd.DataFrame:
        """Compute H2H features and add them to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: ``date``, ``home_team``, ``away_team``,
            ``home_goals``, ``away_goals``, ``result``.
            May contain: ``home_xg``, ``away_xg``.
        context : TransformContext, optional
            Pipeline context (ignored in this implementation).

        Returns
        -------
        pd.DataFrame
            Input DataFrame with H2H feature columns added.
        """
        df = df.copy()

        if "date" in df.columns and self.params.get("sort_by_date", True):
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values(["date", "home_team"], inplace=True)
            df.reset_index(drop=True, inplace=True)

        logger.debug("H2H: transforming %d rows", len(df))

        # Detect optional columns (xG sources)
        self._available_optional_cols = self._detect_optional_columns(df)
        self._active_metrics = self._get_active_metrics()
        self._resolve_output_columns()

        # Build pair stats and compute rolling H2H
        pair_stats = self._build_pair_stats(df)
        pair_rolling = self._compute_rolling_h2h(pair_stats)

        # Empty handling
        if df.empty:
            for col in self._resolved_outputs:
                df[col] = np.nan
            return df

        df = self._merge_features(df, pair_rolling)

        # Ensure all resolved output columns exist (fill NaN for pairs with
        # fewer than 2 meetings, which produce no rolling features)
        for col in self._resolved_outputs:
            if col not in df.columns:
                df[col] = np.nan

        added = [c for c in self._resolved_outputs if c in df.columns]
        logger.debug(
            "H2H: added %d / %d possible columns",
            len(added), len(self._resolved_outputs),
        )

        return df

    # ══════════════════════════════════════════════════════
    #  Internal: column detection
    # ══════════════════════════════════════════════════════

    def _detect_optional_columns(self, df: pd.DataFrame) -> dict[str, str]:
        """Detect xG source columns in the DataFrame.

        Returns mapping from canonical name (``home_xg``, ``away_xg``)
        to actual column name found.
        """
        col_lower = {c.lower(): c for c in df.columns}
        found: dict[str, str] = {}
        patterns = {
            "home_xg": ["home_xg", "h_xg", "hxg", "xg_home", "expected_goals_home"],
            "away_xg": ["away_xg", "a_xg", "axg", "xg_away", "expected_goals_away"],
        }
        for canonical, candidates in patterns.items():
            for p in candidates:
                if p in col_lower:
                    found[canonical] = col_lower[p]
                    break
        return found

    def _get_active_metrics(self) -> dict[str, dict[str, Any]]:
        """Combine core and optional metrics based on available columns."""
        metrics: dict[str, dict[str, Any]] = {}
        metrics.update(_H2H_CORE_METRICS)

        opts = self._available_optional_cols
        if "home_xg" in opts and "away_xg" in opts:
            for name, meta in _H2H_OPTIONAL_METRICS.items():
                if "depends_on" not in meta:
                    metrics[name] = meta
            # Add xga and xgd if xg is available
            metrics["xga"] = _H2H_OPTIONAL_METRICS["xga"]
            metrics["xgd"] = _H2H_OPTIONAL_METRICS["xgd"]

        # Filter by params
        include_xg = self.params.get("include_xg", True)
        if not include_xg:
            for k in ("xg", "xga", "xgd"):
                metrics.pop(k, None)

        return metrics

    # ══════════════════════════════════════════════════════
    #  Internal: pair stats construction
    # ══════════════════════════════════════════════════════

    def _build_pair_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build per-pair per-match stats DataFrame (2 rows per match).

        Each match produces:
        - 1 row for the home team's perspective vs the away team
        - 1 row for the away team's perspective vs the home team

        Result columns:
            team, opponent, date, match_id, is_home,
            goals_for, goals_against, goal_diff,
            is_win, is_draw, is_loss,
            btts, over_2.5, clean_sheets,
            xg, xga (optional)
        """
        n = len(df)

        # ── Home perspective ──────────────────────────────
        home_persp = pd.DataFrame({
            "team": df["home_team"].values,
            "opponent": df["away_team"].values,
            "date": pd.to_datetime(df["date"]).values if "date" in df.columns else pd.NaT,
            "match_id": df.index.values,
            "is_home": np.ones(n, dtype=np.int8),
            "goals_for": df["home_goals"].values.astype(float),
            "goals_against": df["away_goals"].values.astype(float),
        })

        # ── Away perspective ──────────────────────────────
        away_persp = pd.DataFrame({
            "team": df["away_team"].values,
            "opponent": df["home_team"].values,
            "date": pd.to_datetime(df["date"]).values if "date" in df.columns else pd.NaT,
            "match_id": df.index.values,
            "is_home": np.zeros(n, dtype=np.int8),
            "goals_for": df["away_goals"].values.astype(float),
            "goals_against": df["home_goals"].values.astype(float),
        })

        pair_stats = pd.concat([home_persp, away_persp], ignore_index=True)

        # Result-derived indicators
        result = df["result"].values
        result_series = pd.Series(result)
        result_upper = result_series.astype(str).str.upper().fillna("").values
        home_is_win = (result_upper == "H")
        away_is_win = (result_upper == "A")
        is_draw = (result_upper == "D")

        h_win = np.concatenate([home_is_win, away_is_win])
        h_draw = np.concatenate([is_draw, is_draw])
        h_loss = ~h_win & ~h_draw

        pair_stats["is_win"] = h_win.astype(np.int8)
        pair_stats["is_draw"] = h_draw.astype(np.int8)
        pair_stats["is_loss"] = h_loss.astype(np.int8)

        gf = pair_stats["goals_for"].values
        ga = pair_stats["goals_against"].values

        pair_stats["goal_diff"] = gf - ga
        pair_stats["btts"] = ((gf > 0) & (ga > 0)).astype(np.int8)
        pair_stats["over_2.5"] = ((gf + ga) > 2.5).astype(np.int8)
        pair_stats["clean_sheets"] = (ga == 0).astype(np.int8)

        # Optional xG columns
        opts = self._available_optional_cols
        if "home_xg" in opts and "away_xg" in opts:
            home_xg_vals = pd.to_numeric(df[opts["home_xg"]], errors="coerce").values
            away_xg_vals = pd.to_numeric(df[opts["away_xg"]], errors="coerce").values

            pair_stats["xg"] = np.concatenate([home_xg_vals, away_xg_vals])
            pair_stats["xga"] = np.concatenate([away_xg_vals, home_xg_vals])
            pair_stats["xgd"] = pair_stats["xg"] - pair_stats["xga"]

        # Sort by (team, opponent, date) for chronological pair grouping
        pair_stats.sort_values(["team", "opponent", "date"], inplace=True)
        pair_stats.reset_index(drop=True, inplace=True)

        # Optional SQL integration: load extra historical data
        load_fn: Callable | None = self.params.get("load_fn")
        if load_fn is not None:
            try:
                extra = load_fn()
                if extra is not None and not extra.empty:
                    extra["date"] = pd.to_datetime(extra["date"])
                    pair_stats = pd.concat(
                        [extra, pair_stats], ignore_index=True
                    )
                    pair_stats.sort_values(["team", "opponent", "date"], inplace=True)
                    pair_stats.reset_index(drop=True, inplace=True)
            except Exception as exc:
                logger.warning("H2H: SQL load_fn failed: %s", exc)

        return pair_stats

    # ══════════════════════════════════════════════════════
    #  Internal: rolling H2H computation
    # ══════════════════════════════════════════════════════

    def _compute_rolling_h2h(
        self,
        pair_stats: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute rolling H2H metrics per (team, opponent) pair.

        Returns a DataFrame with match_id, is_home, and all H2H feature
        columns for each context and window.
        """
        if pair_stats.empty:
            return pd.DataFrame(columns=["match_id"])

        windows = self._resolved_windows
        contexts = self._resolved_contexts
        metrics = self._active_metrics

        # Map metric names to source columns
        source_map: dict[str, str] = {
            "wins": "is_win",
            "draws": "is_draw",
            "losses": "is_loss",
            "goals_scored": "goals_for",
            "goals_conceded": "goals_against",
            "goal_diff": "goal_diff",
            "btts": "btts",
            "over_2.5": "over_2.5",
            "clean_sheets": "clean_sheets",
        }
        xg_metrics = ("xg", "xga", "xgd")

        result_frames: list[pd.DataFrame] = []

        for (_team, _opponent), grp in pair_stats.groupby(
            ["team", "opponent"], sort=False
        ):
            grp = grp.sort_values("date").copy()
            grp.reset_index(drop=True, inplace=True)

            for ctx in contexts:
                # Select rows for this context
                if ctx == "overall":
                    ctx_mask = slice(None)  # All rows
                elif ctx == "home":
                    ctx_mask = grp["is_home"] == 1
                elif ctx == "away":
                    ctx_mask = grp["is_home"] == 0
                else:
                    continue

                compute_on = grp[ctx_mask].sort_values("date")
                if len(compute_on) < 2:
                    continue

                # Compute rolling features on the filtered subset
                rolling_cols: dict[str, np.ndarray] = {}
                for metric_name, src_col in source_map.items():
                    if metric_name not in metrics:
                        continue
                    if src_col not in grp.columns:
                        continue
                    values = compute_on[src_col].values.astype(float)
                    for w in windows:
                        col_name = f"h2h_{ctx}_{metric_name}_last{w}"
                        rolling_cols[col_name] = self._rolling_last_n(values, w)

                # xG metrics
                for metric_name in xg_metrics:
                    if metric_name in metrics and metric_name in grp.columns:
                        values = compute_on[metric_name].values.astype(float)
                        for w in windows:
                            col_name = f"h2h_{ctx}_{metric_name}_last{w}"
                            rolling_cols[col_name] = self._rolling_last_n(values, w)

                # Map rolling values back to the full group by match_id
                match_ids = compute_on["match_id"].values
                for col_name, arr in rolling_cols.items():
                    val_map = dict(zip(match_ids, arr))
                    grp[col_name] = grp["match_id"].map(val_map)

            result_frames.append(grp)

        if not result_frames:
            return pd.DataFrame(columns=["match_id"])

        result = pd.concat(result_frames, ignore_index=True)

        # Keep only needed columns
        keep = {"match_id", "is_home", "team"}
        for c in result.columns:
            if c.startswith("h2h_"):
                keep.add(c)

        return result[list(keep)]

    @staticmethod
    def _rolling_last_n(values: np.ndarray, n: int) -> np.ndarray:
        """Compute rolling mean over last N values, shifted by 1 for leakage prevention.

        For position i: mean of values[i-n:i] (previous n values, NOT including i).
        If there are fewer than n previous values, uses all available.
        First entry always gets NaN (no history).

        Parameters
        ----------
        values : np.ndarray
            Float array of values.
        n : int
            Window size.

        Returns
        -------
        np.ndarray
            Rolling means with shift(1) applied.
        """
        m = len(values)
        result = np.full(m, np.nan, dtype=np.float64)
        if m < 2:
            return result

        for i in range(1, m):
            start = max(0, i - n)
            window = values[start:i]
            result[i] = float(np.mean(window))
        return result

    # ══════════════════════════════════════════════════════
    #  Internal: merge features back to original DF
    # ══════════════════════════════════════════════════════

    def _merge_features(
        self,
        df: pd.DataFrame,
        pair_rolling: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge H2H rolling features back onto the original DataFrame.

        Uses (team, match_id) as the lookup key, matching the home team's
        H2H stats to ``h_`` prefix columns and the away team's to ``a_``.
        """
        if pair_rolling.empty:
            return df

        df_result = df.copy()

        # Collect all H2H feature columns
        feat_cols = [
            c for c in pair_rolling.columns
            if c.startswith("h2h_")
        ]
        if not feat_cols:
            return df_result

        lookup_df = pair_rolling.set_index(["team", "match_id"])

        for col in feat_cols:
            h_col = f"h_{col}"
            a_col = f"a_{col}"
            values = lookup_df[col].to_dict()

            df_result[h_col] = [
                values.get((df_result.at[idx, "home_team"], idx))
                for idx in df_result.index
            ]
            df_result[a_col] = [
                values.get((df_result.at[idx, "away_team"], idx))
                for idx in df_result.index
            ]

        return df_result

    # ── Validation ──────────────────────────────────────

    def validate_output(self, df: pd.DataFrame) -> list[str]:
        errors: list[str] = []
        if not self.output_columns:
            return errors
        # Check a sample of expected columns
        sample_cols = self.output_columns[:10]  # Check first 10
        for col in sample_cols:
            if col not in df.columns:
                errors.append(f"Missing output column: {col}")
        return errors


# ═══════════════════════════════════════════════════════════════
#  Convenience factory
# ═══════════════════════════════════════════════════════════════


def create_h2h_transformer(
    windows: tuple[int, ...] = _DEFAULT_WINDOWS,
    contexts: tuple[str, ...] = _DEFAULT_CONTEXTS,
    include_xg: bool = True,
    **kwargs: Any,
) -> H2HTransformer:
    """Create a configured H2HTransformer with explicit params.

    Parameters
    ----------
    windows : tuple[int, ...]
        Number of meetings to look back (default (3, 5, 10)).
    contexts : tuple[str, ...]
        Venue contexts (default (\"overall\", \"home\", \"away\")).
    include_xg : bool
        Include xG/xGA/xGD metrics when data available (default True).

    Returns
    -------
    H2HTransformer
        Pre-configured transformer instance.
    """
    return H2HTransformer(
        windows=list(windows),
        contexts=list(contexts),
        include_xg=include_xg,
        **kwargs,
    )
