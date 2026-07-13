"""
Team Form Feature Generator — rolling team form statistics.

Computes windowed rolling averages for a comprehensive set of team
performance metrics, broken down by venue context (overall, home, away).

This is the **first concrete football feature** built on the feature
engineering framework (``src.feature_framework``).  It replaces and
extends the legacy ``_add_rolling_features()`` in
``src.feature_engineering``.

Windows
-------
Default: ``[3, 5, 10, 20]`` — configurable via ``params["windows"]``.

Metrics (20 total)
------------------
All metrics are computed as **rolling means** (proportions for binary
indicators, averages for numeric).  Each metric is binary (0/1) unless
otherwise noted.

+-------------------------+-----------------------------+---------+
| Metric                  | Source / Formula            | Type    |
+=========================+=============================+=========+
| points                  | 3 (win), 1 (draw), 0 (loss) | float   |
| wins                    | result == team win           | binary  |
| draws                   | result == draw               | binary  |
| losses                  | result == team loss          | binary  |
| goals_scored            | home_goals / away_goals      | int     |
| goals_conceded          | opponent's goals             | int     |
| goal_diff               | scored - conceded            | int     |
| clean_sheets            | goals_conceded == 0          | binary  |
| btts                    | both teams scored > 0        | binary  |
| over_2.5                | total goals > 2.5            | binary  |
| under_2.5               | total goals <= 2.5           | binary  |
| xg                      | home_xg / away_xg            | float   |
| xga                     | opponent's xg                | float   |
| xgd                     | xg - xga                     | float   |
| shots                   | home_shots / away_shots      | int     |
| shots_on_target         | home_sot / away_sot          | int     |
| possession              | home_possession / away_poss   | float   |
| corners                 | home_corners / away_corners   | int     |
| yellow_cards            | home_yc / away_yc             | int     |
| red_cards               | home_rc / away_rc             | int     |

Contexts (3)
------------
- ``overall`` — all matches the team has played
- ``home`` — only home matches for the team
- ``away`` — only away matches for the team

Leakage prevention
------------------
All rolling statistics use ``.shift(1)`` so the current match's data
is never included in its own feature values.  The DataFrame is assumed
to be sorted chronologically (or will be sorted by ``date`` at the
start of ``transform()``).

Column naming convention
------------------------
Pattern: ``{h|a}_{context}_{metric}_avg{window}``

Examples
~~~~~~~~
- ``h_overall_points_avg5`` — home team's avg points/last 5 matches
- ``a_home_goals_scored_avg3`` — away team's avg goals/last 3 home matches
- ``h_away_clean_sheets_avg10`` — home team's clean sheet rate/last 10 away

Integration with FeaturePipeline
--------------------------------
::

    # Register the transformer class
    pipeline = FeaturePipeline(config_path="features.yaml")
    pipeline.plugins.register(TeamFormTransformer)

    # Or use auto-discovery
    pipeline = FeaturePipeline(config_dict=...)
    pipeline.run(entity_type="dataframe", df=matches_df)

YAML config example::

    features:
      - name: team_form_features
        type: team_form
        category: form
        version: 1
        data_type: float
        computation_time: medium
        output_columns: []
        dependencies: []
        enabled: true
        params:
          windows: [3, 5, 10, 20]
          contexts: [overall, home, away]
          league_specific: true
          sort_by_date: true
          include_goals: true
          include_xg: true
          include_shots: true
          include_possession: false
          include_cards: false
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

from src.feature_framework.base import FeatureTransformer
from src.feature_framework.models import TransformContext

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  Column pattern detection (case-insensitive, first match wins)
# ═══════════════════════════════════════════════════════════════

# Each optional metric maps home-patterns and away-patterns
_COLUMN_PATTERNS: dict[str, list[str]] = {
    "home_shots": [
        "home_shots", "h_shots", "hshots", "hs", "shots_home",
    ],
    "away_shots": [
        "away_shots", "a_shots", "ashots", "as", "shots_away",
    ],
    "home_shots_on_target": [
        "home_shots_on_target", "home_shots_target", "h_sot",
        "h_sht", "hst", "shots_target_home",
    ],
    "away_shots_on_target": [
        "away_shots_on_target", "away_shots_target", "a_sot",
        "a_sht", "ast", "shots_target_away",
    ],
    "home_xg": [
        "home_xg", "h_xg", "hxg", "xg_home", "expected_goals_home",
    ],
    "away_xg": [
        "away_xg", "a_xg", "axg", "xg_away", "expected_goals_away",
    ],
    "home_possession": [
        "home_possession", "h_possession", "hposs", "possession_home",
    ],
    "away_possession": [
        "away_possession", "a_possession", "aposs", "possession_away",
    ],
    "home_corners": [
        "home_corners", "h_corners", "hcorners", "hc",
    ],
    "away_corners": [
        "away_corners", "a_corners", "acorners", "ac",
    ],
    "home_yellow_cards": [
        "home_yellow_cards", "home_yellow", "h_yellow_cards",
        "h_yellow", "hy", "yellow_cards_home",
    ],
    "away_yellow_cards": [
        "away_yellow_cards", "away_yellow", "a_yellow_cards",
        "a_yellow", "ay", "yellow_cards_away",
    ],
    "home_red_cards": [
        "home_red_cards", "home_red", "h_red_cards",
        "h_red", "hr", "red_cards_home",
    ],
    "away_red_cards": [
        "away_red_cards", "away_red", "a_red_cards",
        "a_red", "ar", "red_cards_away",
    ],
}

# ═══════════════════════════════════════════════════════════════
#  Metric definitions
# ═══════════════════════════════════════════════════════════════

# Metrics always available (derived from result + goals).
# is_binary: True = rolling mean gives decimal rate (e.g. 0.600),
#            False = rolling mean of raw values (e.g. 1.4 goals).
_CORE_METRICS: OrderedDict[str, dict[str, Any]] = OrderedDict({
    "points":       {"is_binary": False, "description": "Points per match (3/1/0)"},
    "wins":         {"is_binary": True,  "description": "Win rate (proportion)"},
    "draws":        {"is_binary": True,  "description": "Draw rate (proportion)"},
    "losses":       {"is_binary": True,  "description": "Loss rate (proportion)"},
    "goals_scored":  {"is_binary": False, "description": "Average goals scored per match"},
    "goals_conceded":{"is_binary": False, "description": "Average goals conceded per match"},
    "goal_diff":    {"is_binary": False, "description": "Average goal difference per match"},
    "clean_sheets": {"is_binary": True,  "description": "Clean sheet rate (proportion)"},
    "btts":         {"is_binary": True,  "description": "Both Teams Scored rate"},
    "over_2.5":     {"is_binary": True,  "description": "Over 2.5 goals rate"},
    "under_2.5":    {"is_binary": True,  "description": "Under 2.5 goals rate"},
})

# Optional metrics — only computed when source columns exist
_OPTIONAL_METRICS: OrderedDict[str, dict[str, Any]] = OrderedDict({
    "xg":            {"is_binary": False, "description": "Average xG per match",
                      "source_home": "home_xg", "source_away": "away_xg"},
    "xga":           {"is_binary": False, "description": "Average xGA conceded per match",
                      "source_home": "away_xg", "source_away": "home_xg"},
    "xgd":           {"is_binary": False, "description": "Average xG difference per match",
                      "depends_on": ["xg", "xga"]},
    "shots":         {"is_binary": False, "description": "Average shots per match",
                      "source_home": "home_shots", "source_away": "away_shots"},
    "shots_on_target":{"is_binary": False, "description": "Average shots on target per match",
                       "source_home": "home_shots_on_target",
                       "source_away": "away_shots_on_target"},
    "possession":    {"is_binary": False, "description": "Average possession % per match",
                      "source_home": "home_possession", "source_away": "away_possession"},
    "corners":       {"is_binary": False, "description": "Average corners per match",
                      "source_home": "home_corners", "source_away": "away_corners"},
    "yellow_cards":  {"is_binary": False, "description": "Average yellow cards per match",
                      "source_home": "home_yellow_cards",
                      "source_away": "away_yellow_cards"},
    "red_cards":     {"is_binary": False, "description": "Average red cards per match",
                      "source_home": "home_red_cards",
                      "source_away": "away_red_cards"},
})

# Default params
_DEFAULT_WINDOWS = (3, 5, 10, 20)
_DEFAULT_CONTEXTS = ("overall", "home", "away")


# ═══════════════════════════════════════════════════════════════
#  TeamFormTransformer
# ═══════════════════════════════════════════════════════════════


class TeamFormTransformer(FeatureTransformer):
    """Compute rolling team form features for every context and window.

    This is the flagship concrete feature in the framework.  Given a
    match DataFrame with minimal required columns (``date``, ``home_team``,
    ``away_team``, ``home_goals``, ``away_goals``, ``result``), it produces
    hundreds of leakage-free rolling averages.

    Optional match-stat columns are auto-detected and included when present.
    """

    name: str = "team_form"
    version: int = 1
    description: str = (
        "Rolling team form features: points, goals, xG, shots, and more "
        "for overall/home/away contexts across multiple windows."
    )
    dependencies: list[str] = []
    data_type: str = "float"
    computation_time: str = "medium"
    category: str = "form"
    author: str = "system"
    tags: list[str] = ["form", "rolling", "team", "stats"]
    source: str = "derived"

    # output_columns are computed dynamically in init() based on
    # available data columns and configured params.
    output_columns: list[str] = []

    # ── Required columns ─────────────────────────────────

    _REQUIRED_COLS: frozenset[str] = frozenset({
        "date", "home_team", "away_team", "home_goals", "away_goals", "result",
    })

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._team_stats: pd.DataFrame | None = None
        self._available_optional_cols: dict[str, str] = {}
        self._active_metrics: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._resolved_outputs: list[str] = []
        self._resolved_windows: tuple[int, ...] = ()
        self._resolved_contexts: tuple[str, ...] = ()

    def init(self, context: TransformContext | None = None) -> None:
        """Pre-compute the output column names from params and data.

        This allows ``validate_output()`` to work correctly even though
        the exact columns depend on which optional data columns exist.
        """
        self._resolved_windows = self._resolve_windows()
        self._resolved_contexts = self._resolve_contexts()
        self._resolve_output_columns()
        self.output_columns = list(self._resolved_outputs)
        self._initialized = True
        logger.debug(
            "TeamFormTransformer initialized: %d windows, %d contexts, ~%d columns",
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
        """Generate all output column names from params."""
        windows = self._resolved_windows
        contexts = self._resolved_contexts
        active = self._active_metrics or _CORE_METRICS

        outputs: list[str] = []
        for metrics_dict in [active, self._get_optional_metrics()]:
            for metric_name in metrics_dict:
                for ctx in contexts:
                    for w in windows:
                        outputs.append(f"h_{ctx}_{metric_name}_avg{w}")
                        outputs.append(f"a_{ctx}_{metric_name}_avg{w}")

        self._resolved_outputs = sorted(set(outputs))
        self.output_columns = self._resolved_outputs

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
        """Compute rolling team form features and add them to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: ``date``, ``home_team``, ``away_team``,
            ``home_goals``, ``away_goals``, ``result``.
            May also contain optional stat columns (xG, shots, etc.).
        context : TransformContext, optional
            Pipeline context (ignored in this implementation).

        Returns
        -------
        pd.DataFrame
            Input DataFrame with rolling feature columns added.
        """
        df = df.copy()

        # ── 1. Sort chronologically ───────────────────────
        if "date" in df.columns and self.params.get("sort_by_date", True):
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values(["date", "home_team"], inplace=True)
            df.reset_index(drop=True, inplace=True)

        logger.debug("TeamForm: transforming %d rows", len(df))

        # ── 2. Detect optional columns ────────────────────
        self._available_optional_cols = self._detect_optional_columns(df)
        self._active_metrics = self._get_active_metrics()
        self._resolve_output_columns()

        # ── 3. Build per-team stats DataFrame ─────────────
        team_stats = self._build_team_stats(df)

        # ── 4. Compute rolling features per context ───────
        windows = self._resolved_windows
        contexts = self._resolved_contexts
        league_specific = self.params.get("league_specific", True)

        # Track which stats we need to merge back
        context_dfs: dict[str, pd.DataFrame] = {}

        for ctx in contexts:
            rolling_df = self._compute_rolling(
                team_stats, context_name=ctx,
                windows=windows,
                league_specific=league_specific,
            )
            context_dfs[ctx] = rolling_df

        # ── 5. Merge all context features onto original DF ─
        df = self._merge_features(df, context_dfs, team_stats)

        n_new = len([c for c in df.columns if c not in df.columns  # already in original
                     if c not in (
                         set(df.columns) - set(self._resolved_outputs))])
        # Simpler: count how many of our output columns are now in df
        added = [c for c in self._resolved_outputs if c in df.columns]
        logger.debug(
            "TeamForm: added %d / %d possible columns (%d windows, %d contexts)",
            len(added), len(self._resolved_outputs),
            len(windows), len(contexts),
        )

        return df

    # ══════════════════════════════════════════════════════
    #  Internal: column detection
    # ══════════════════════════════════════════════════════

    def _detect_optional_columns(self, df: pd.DataFrame) -> dict[str, str]:
        """Detect which optional stat columns exist in the DataFrame.

        Returns
        -------
        dict[str, str]
            Mapping from canonical column name (e.g. ``home_xg``) to the
            actual column name found in the DataFrame.
        """
        col_lower = {c.lower(): c for c in df.columns}
        found: dict[str, str] = {}

        for canonical, patterns in _COLUMN_PATTERNS.items():
            for pattern in patterns:
                if pattern in col_lower:
                    found[canonical] = col_lower[pattern]
                    break

        return found

    def _detect_home_away_col(self, metric: str) -> tuple[str | None, str | None]:
        """Find home and away source column names for an optional metric.

        Returns (home_col, away_col) or (None, None) if not found.
        """
        opts = self._available_optional_cols
        home_key = f"home_{metric}"
        away_key = f"away_{metric}"
        return opts.get(home_key), opts.get(away_key)

    def _get_optional_metrics(self) -> OrderedDict[str, dict[str, Any]]:
        """Determine which optional metrics can be computed.

        Checks which source columns exist and returns only the
        metrics that can be computed with available data.

        Handles two categories:
        1. **Direct metrics** — have their own source columns (xg, shots, etc.)
        2. **Derived metrics** — computed from other metrics (xga, xgd)
        """
        available: OrderedDict[str, dict[str, Any]] = OrderedDict()

        # Check which metric categories are enabled via params
        enabled_xg = self.params.get("include_xg", True)
        enabled_shots = self.params.get("include_shots", True)
        enabled_poss = self.params.get("include_possession", True)
        enabled_cards = self.params.get("include_cards", True)

        for name, meta in _OPTIONAL_METRICS.items():
            # Check category enablement
            if name in ("xg", "xga", "xgd") and not enabled_xg:
                continue
            if name in ("shots", "shots_on_target") and not enabled_shots:
                continue
            if name == "possession" and not enabled_poss:
                continue
            if name in ("yellow_cards", "red_cards") and not enabled_cards:
                continue

            # Check column availability
            if "depends_on" in meta:
                # Derived metric (e.g. xgd depends on xg, xga)
                deps = meta["depends_on"]
                if all(d in available for d in deps):
                    available[name] = meta
            else:
                home_col, away_col = self._detect_home_away_col(name)
                if home_col is not None and away_col is not None:
                    available[name] = meta

        # ── Auto-derive xga when xg columns are detected ───────────
        # xga doesn't have its own columns — it's the opponent's xG
        # Detect via: if "home_xg" and "away_xg" columns exist, xga is available
        if "xg" in available and enabled_xg:
            # Check that xg was found via column detection (not randomly added)
            if "home_xg" in self._available_optional_cols and "away_xg" in self._available_optional_cols:
                available["xga"] = _OPTIONAL_METRICS["xga"]
                # xgd depends on xg + xga, so it's always available when both are
                available["xgd"] = _OPTIONAL_METRICS["xgd"]

        return available

    def _get_active_metrics(self) -> OrderedDict[str, dict[str, Any]]:
        """Combine core and optional metrics."""
        metrics: OrderedDict[str, dict[str, Any]] = OrderedDict()
        metrics.update(_CORE_METRICS)

        for name, meta in self._get_optional_metrics().items():
            metrics[name] = meta

        return metrics

    # ══════════════════════════════════════════════════════
    #  Internal: team stats construction
    # ══════════════════════════════════════════════════════

    def _build_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build per-team per-match stats DataFrame (2 rows per match).

        Each match produces:
        - 1 row for the home team (is_home=1)
        - 1 row for the away team (is_home=0)

        Result DataFrame columns:
            team, date, season, league, opponent, is_home,
            match_id, is_win, is_draw, is_loss, points,
            goals_scored, goals_conceded, goal_diff, gd,
            clean_sheets, btts, over_2.5, under_2.5,
            xg, xga, shots, shots_on_target, ... (optional)
        """
        # ── Home team rows ────────────────────────────────
        home_df = pd.DataFrame({
            "team": df["home_team"].values,
            "date": pd.to_datetime(df["date"]).values if "date" in df.columns else pd.NaT,
            "opponent": df["away_team"].values,
            "is_home": np.ones(len(df), dtype=np.int8),
            "match_id": df.index.values,
            "goals_scored": df["home_goals"].values.astype(float),
            "goals_conceded": df["away_goals"].values.astype(float),
        })

        # ── Away team rows ────────────────────────────────
        away_df = pd.DataFrame({
            "team": df["away_team"].values,
            "date": pd.to_datetime(df["date"]).values if "date" in df.columns else pd.NaT,
            "opponent": df["home_team"].values,
            "is_home": np.zeros(len(df), dtype=np.int8),
            "match_id": df.index.values,
            "goals_scored": df["away_goals"].values.astype(float),
            "goals_conceded": df["home_goals"].values.astype(float),
        })

        # ── Season and league ─────────────────────────────
        for col in ("season", "league"):
            if col in df.columns:
                home_df[col] = df[col].values
                away_df[col] = df[col].values

        # ── Combine ───────────────────────────────────────
        team_stats = pd.concat([home_df, away_df], ignore_index=True)

        # ── Result-derived indicators ─────────────────────
        result = df["result"].values
        n = len(df)
        is_home_arr = np.concatenate([np.ones(n), np.zeros(n)]).astype(bool)

        # Map result strings to binary indicators
        # Use pandas Series.str for robust NaN handling
        result_series = pd.Series(result)
        result_upper = result_series.astype(str).str.upper().fillna("").values
        home_is_win = (result_upper == "H")
        away_is_win = (result_upper == "A")
        is_draw = (result_upper == "D")

        h_win = np.concatenate([home_is_win, away_is_win])
        h_draw = np.concatenate([is_draw, is_draw])
        h_loss = ~h_win & ~h_draw

        team_stats["is_win"] = h_win.astype(np.int8)
        team_stats["is_draw"] = h_draw.astype(np.int8)
        team_stats["is_loss"] = h_loss.astype(np.int8)

        # 3/1/0 points
        team_stats["points"] = (h_win.astype(float) * 3.0 +
                                h_draw.astype(float) * 1.0)

        # ── Goal-derived indicators ───────────────────────
        gs = team_stats["goals_scored"].values
        gc = team_stats["goals_conceded"].values

        team_stats["goal_diff"] = gs - gc
        team_stats["clean_sheets"] = (gc == 0).astype(np.int8)
        team_stats["btts"] = ((gs > 0) & (gc > 0)).astype(np.int8)
        total_goals = gs + gc
        team_stats["over_2.5"] = (total_goals > 2.5).astype(np.int8)
        team_stats["under_2.5"] = (total_goals <= 2.5).astype(np.int8)

        # ── Optional stat columns ─────────────────────────
        optional = self._available_optional_cols
        opts_active = self._get_optional_metrics()

        for name, meta in _OPTIONAL_METRICS.items():
            if name not in opts_active:
                continue
            if "depends_on" in meta:
                continue  # Derived metrics computed later

            home_key = f"home_{name}"
            away_key = f"away_{name}"
            home_col = optional.get(home_key)
            away_col = optional.get(away_key)
            if home_col is None or away_col is None:
                continue

            home_vals = pd.to_numeric(df[home_col], errors="coerce").values
            away_vals = pd.to_numeric(df[away_col], errors="coerce").values

            team_stats[name] = np.concatenate([home_vals, away_vals])

        # ── Derived optional metrics (xga, xgd) ────────────
        # xga is the opponent's xG — swap the source arrays
        if "xg" in team_stats.columns and "xga" in opts_active:
            home_xg_col_actual = optional.get(f"home_xg")
            away_xg_col_actual = optional.get(f"away_xg")
            if home_xg_col_actual is not None and away_xg_col_actual is not None:
                home_xg_vals = pd.to_numeric(df[home_xg_col_actual], errors="coerce").values
                away_xg_vals = pd.to_numeric(df[away_xg_col_actual], errors="coerce").values
                # Home team xga = away team's xg (goals conceded expected)
                # Away team xga = home team's xg
                team_stats["xga"] = np.concatenate([away_xg_vals, home_xg_vals])

        # xgd = xg - xga
        if "xg" in team_stats.columns and "xga" in team_stats.columns and "xgd" in opts_active:
            team_stats["xgd"] = team_stats["xg"] - team_stats["xga"]

        # ── Sort by team then date ────────────────────────
        team_stats.sort_values(["team", "date"], inplace=True)
        team_stats.reset_index(drop=True, inplace=True)

        self._team_stats = team_stats
        return team_stats

    # ══════════════════════════════════════════════════════
    #  Internal: rolling computation
    # ══════════════════════════════════════════════════════

    def _compute_rolling(
        self,
        team_stats: pd.DataFrame,
        context_name: str,
        windows: tuple[int, ...],
        league_specific: bool = True,
    ) -> pd.DataFrame:
        """Compute rolling means for a single context.

        Parameters
        ----------
        team_stats : pd.DataFrame
            Per-team stats from ``_build_team_stats``.
        context_name : str
            One of ``overall``, ``home``, ``away``.
        windows : tuple[int, ...]
            Windows to compute (e.g. (3, 5, 10, 20)).
        league_specific : bool
            If True, reset rolling stats per league/season combo.

        Returns
        -------
        pd.DataFrame
            With columns: match_id, team, is_home,
            and all ``{context}_{metric}_avg{w}`` for both prefixes.
        """
        # ── Filter by context ────────────────────────────
        if context_name == "home":
            subset = team_stats[team_stats["is_home"] == 1].copy()
        elif context_name == "away":
            subset = team_stats[team_stats["is_home"] == 0].copy()
        else:
            subset = team_stats.copy()

        if len(subset) == 0:
            # Return empty scaffold
            return pd.DataFrame(columns=["match_id"])

        # ── Determine group keys ─────────────────────────
        group_keys = ["team"]
        if league_specific and "league" in subset.columns:
            group_keys.append("league")

        # ── Metrics to compute ───────────────────────────
        active = self._active_metrics

        # Map metric names to source columns in team_stats
        source_cols: dict[str, str] = {}
        for metric_name in active:
            if metric_name == "points":
                source_cols[metric_name] = "points"
            elif metric_name == "wins":
                source_cols[metric_name] = "is_win"
            elif metric_name == "draws":
                source_cols[metric_name] = "is_draw"
            elif metric_name == "losses":
                source_cols[metric_name] = "is_loss"
            elif metric_name == "goal_diff":
                source_cols[metric_name] = "goal_diff"
            elif metric_name == "goals_scored":
                source_cols[metric_name] = "goals_scored"
            elif metric_name == "goals_conceded":
                source_cols[metric_name] = "goals_conceded"
            elif metric_name == "clean_sheets":
                source_cols[metric_name] = "clean_sheets"
            elif metric_name == "btts":
                source_cols[metric_name] = "btts"
            elif metric_name == "over_2.5":
                source_cols[metric_name] = "over_2.5"
            elif metric_name == "under_2.5":
                source_cols[metric_name] = "under_2.5"
            elif metric_name in subset.columns:
                # Optional metrics (xg, xga, shots, etc.)
                source_cols[metric_name] = metric_name

        # ── Compute rolling means per group ─────────────
        def _compute_group_rolling(grp: pd.DataFrame) -> pd.DataFrame:
            grp = grp.sort_values("date").copy()
            for metric_name, src_col in source_cols.items():
                if src_col not in grp.columns:
                    continue
                values = grp[src_col]
                for w in windows:
                    rolling_name = f"{context_name}_{metric_name}_avg{w}"
                    grp[rolling_name] = (
                        values.rolling(w, min_periods=1).mean().shift(1)
                    )
            return grp

        # Use explicit loop instead of groupby.apply to avoid pandas version
        # differences in how group key columns are handled
        result_frames: list[pd.DataFrame] = []
        for _team_name, grp in subset.groupby(group_keys, sort=False):
            result_frames.append(_compute_group_rolling(grp))

        if not result_frames:
            return pd.DataFrame(columns=["match_id"])

        result = pd.concat(result_frames, ignore_index=True)

        # Keep only match_id + is_home + rolling columns
        rolling_cols = [
            c for c in result.columns
            if c in ("match_id", "is_home", "team")
            or "_avg" in c
        ]
        return result[rolling_cols]

    # ══════════════════════════════════════════════════════
    #  Internal: merge features back to original DF
    # ══════════════════════════════════════════════════════

    def _merge_features(
        self,
        df: pd.DataFrame,
        context_dfs: dict[str, pd.DataFrame],
        team_stats: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge rolling features back onto the original DataFrame.

        Uses team-based lookup: for each match, looks up the home team's
        rolling stats and away team's rolling stats by (team, match_id).

        This correctly handles all context types (overall, home, away)
        because it resolves by team name, not by ``is_home`` flag.
        """
        df_result = df.copy()

        for ctx_name, rolling_df in context_dfs.items():
            if rolling_df.empty:
                continue

            col_prefix = f"{ctx_name}_"
            feat_cols = [
                c for c in rolling_df.columns
                if c.startswith(col_prefix)
                and c not in ("match_id", "is_home", "team")
            ]
            if not feat_cols:
                continue

            # Build lookup: (team, match_id) -> row dict for each feature
            # Using set_index + to_dict for vectorized lookup generation
            lookup_df = rolling_df.set_index(["team", "match_id"])

            for col in feat_cols:
                h_col = f"h_{col}"
                a_col = f"a_{col}"

                # Build (team, match_id) -> value dict
                values = lookup_df[col].to_dict()

                # Home team: look up by (home_team_name, match_index)
                df_result[h_col] = [
                    values.get((df_result.at[idx, "home_team"], idx))
                    for idx in df_result.index
                ]

                # Away team: look up by (away_team_name, match_index)
                df_result[a_col] = [
                    values.get((df_result.at[idx, "away_team"], idx))
                    for idx in df_result.index
                ]

        return df_result

    # ── Validation ──────────────────────────────────────

    def validate_output(self, df: pd.DataFrame) -> list[str]:
        """Validate that computed output columns exist.

        Since columns vary by data availability, we check a sample
        of expected columns rather than the full set.
        """
        errors: list[str] = []
        if not self.output_columns:
            return errors

        # Sample check: first few contexts + windows
        check_windows = self._resolved_windows[:2]  # first 2 windows
        check_contexts = self._resolved_contexts[:1]  # first context
        active = self._active_metrics or _CORE_METRICS
        metrics_sample = list(active.keys())[:3]  # first 3 metrics

        for metric in metrics_sample:
            for ctx in check_contexts:
                for w in check_windows:
                    for prefix in ("h_", "a_"):
                        col = f"{prefix}{ctx}_{metric}_avg{w}"
                        if col not in df.columns:
                            errors.append(f"Missing output column: {col}")

        return errors


# ═══════════════════════════════════════════════════════════════
#  Convenience factory
# ═══════════════════════════════════════════════════════════════


def create_team_form_transformer(
    windows: tuple[int, ...] = _DEFAULT_WINDOWS,
    contexts: tuple[str, ...] = _DEFAULT_CONTEXTS,
    league_specific: bool = True,
    include_xg: bool = True,
    include_shots: bool = True,
    include_possession: bool = True,
    include_cards: bool = True,
    **kwargs: Any,
) -> TeamFormTransformer:
    """Create a configured TeamFormTransformer with explicit params.

    Parameters
    ----------
    windows : tuple[int, ...]
        Rolling windows (default (3, 5, 10, 20)).
    contexts : tuple[str, ...]
        Form contexts (default (\"overall\", \"home\", \"away\")).
    league_specific : bool
        Reset rolls per league (default True).
    include_xg, include_shots, include_possession, include_cards : bool
        Enable/disable optional metric categories.

    Returns
    -------
    TeamFormTransformer
        Pre-configured transformer instance.
    """
    return TeamFormTransformer(
        windows=list(windows),
        contexts=list(contexts),
        league_specific=league_specific,
        include_xg=include_xg,
        include_shots=include_shots,
        include_possession=include_possession,
        include_cards=include_cards,
        **kwargs,
    )
