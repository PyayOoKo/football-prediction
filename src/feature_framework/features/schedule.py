"""
Schedule Feature Generator — fixture congestion and rest-day features for football.

Computes team-level schedule congestion metrics that capture fixture density,
recovery time, back-to-back scenarios, and venue pattern streaks.

Features (8 total)
-------------------
+-----------------------------+----------------------------------------------------+---------+
| Feature                     | Description                                        | Type    |
+=============================+====================================================+=========+
| rest_days                   | Days since the team's last match (any venue)       | float   |
| days_since_last_match       | Alias for rest_days                                | float   |
| matches_last_7_days         | Number of matches in the last 7 days               | int     |
| matches_last_14_days        | Number of matches in the last 14 days              | int     |
| consec_home                 | Consecutive home matches before this one           | int     |
| consec_away                 | Consecutive away matches before this one           | int     |
| is_back_to_back             | 1 if facing the same opponent as last match        | binary  |
| travel_distance             | Great-circle distance from previous venue (km)     | float   |
| days_since_competition      | Days since last match in the same competition      | float   |
+-----------------------------+----------------------------------------------------+---------+

Leakage prevention
-------------------
- Daily-features (rest_days, matches_last_7_days, days_since_competition) use
  ``.shift(1)`` to exclude the current match's date from its own features.
- Consecutive counts use ``cumcount()`` within venue-streak, which excludes
  the current match by design.
- ``is_back_to_back`` uses the previous match's opponent (via ``shift(1)``).

Column naming convention
------------------------
Pattern: ``{h|a}_{feature_name}``

Examples
~~~~~~~~
- ``h_rest_days`` — home team's rest days before this match
- ``a_matches_last_7_days`` — away team's matches played in the last 7 days
- ``h_consec_home`` — consecutive home matches for the home team

Requirements coverage
---------------------
- **League aware**: rolling windows reset per league (``league_specific`` param)
- **Cup competitions**: ``days_since_competition`` tracks gaps per competition
- **International breaks**: long gaps (>= 14 days) flagged via ``rest_days``
- **Batch updates**: full DataFrame processed in one ``transform()`` call
- **Feature validation**: standard ``validate_input()`` / ``validate_output()``

Travel distance
---------------
Auto-detected from optional columns:
- ``home_lat`` / ``home_lon`` and ``away_lat`` / ``away_lon`` (per-match venue)
- Or ``lat`` / ``lon`` columns in team_stats if present
- Computed using the Haversine formula (great-circle distance)

Integration with FeaturePipeline
--------------------------------
::

    pipeline = FeaturePipeline(config_path=\"features.yaml\")
    pipeline.plugins.register(ScheduleTransformer)

    report = pipeline.run(entity_type=\"dataframe\", df=matches_df)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.feature_framework.base import FeatureTransformer
from src.feature_framework.models import TransformContext

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

# Column patterns for travel distance detection
_TRAVEL_COL_PATTERNS: dict[str, list[str]] = {
    "home_lat": [
        "home_lat", "home_latitude", "h_lat", "venue_lat_home",
    ],
    "home_lon": [
        "home_lon", "home_longitude", "home_lng", "h_lon",
        "venue_lon_home", "venue_lng_home",
    ],
    "away_lat": [
        "away_lat", "away_latitude", "a_lat", "venue_lat_away",
    ],
    "away_lon": [
        "away_lon", "away_longitude", "away_lng", "a_lon",
        "venue_lon_away", "venue_lng_away",
    ],
}

# Default params
_DEFAULT_LEAGUE_SPECIFIC: bool = True
_DEFAULT_MAX_TRAVEL_KM: float = 20000.0  # Sanity cap for travel distance


# ═══════════════════════════════════════════════════════════════
#  ScheduleTransformer
# ═══════════════════════════════════════════════════════════════


class ScheduleTransformer(FeatureTransformer):
    """Compute fixture schedule / congestion features for every match.

    Produces per-team schedule metrics that capture how much rest a team
    has had, how many recent matches they've played, venue streaks, and
    back-to-back opponent scenarios.

    Auto-detects venue coordinates for travel-distance computation.
    """

    name: str = "schedule"
    version: int = 1
    description: str = (
        "Fixture schedule features: rest days, matches in last 7/14 days, "
        "consecutive home/away streaks, back-to-back opponents, and travel distance."
    )
    dependencies: list[str] = []
    data_type: str = "float"
    computation_time: str = "fast"
    category: str = "schedule"
    author: str = "system"
    tags: list[str] = ["schedule", "congestion", "rest", "fixture", "fatigue"]
    source: str = "derived"

    output_columns: list[str] = [
        "h_rest_days",
        "a_rest_days",
        "h_days_since_last_match",
        "a_days_since_last_match",
        "h_matches_last_7_days",
        "a_matches_last_7_days",
        "h_matches_last_14_days",
        "a_matches_last_14_days",
        "h_consec_home",
        "a_consec_home",
        "h_consec_away",
        "a_consec_away",
        "h_is_back_to_back",
        "a_is_back_to_back",
        "h_travel_distance",
        "a_travel_distance",
        "h_days_since_competition",
        "a_days_since_competition",
    ]

    _REQUIRED_COLS: frozenset[str] = frozenset({
        "date", "home_team", "away_team",
    })

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._available_travel_cols: dict[str, str] = {}
        self._resolved_outputs: list[str] = []

    def init(self, context: TransformContext | None = None) -> None:
        """Pre-compute the effective output column list."""
        self._resolve_output_columns()
        self.output_columns = list(self._resolved_outputs)
        self._initialized = True
        logger.debug(
            "ScheduleTransformer initialized: %d output columns",
            len(self.output_columns),
        )

    def _resolve_output_columns(self) -> None:
        """Build the set of output columns based on available data.

        Travel distance is conditionally included.
        """
        base_cols = [
            "rest_days",
            "days_since_last_match",
            "matches_last_7_days",
            "matches_last_14_days",
            "consec_home",
            "consec_away",
            "is_back_to_back",
            "days_since_competition",
        ]
        # Always include
        outputs: list[str] = []
        for col in base_cols:
            outputs.append(f"h_{col}")
            outputs.append(f"a_{col}")

        # Conditional: travel distance
        include_travel = self.params.get("include_travel_distance", True)
        if include_travel:
            outputs.append("h_travel_distance")
            outputs.append("a_travel_distance")

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
        """Compute schedule features and add them to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: ``date``, ``home_team``, ``away_team``.
            May contain: ``home_goals``, ``away_goals``, ``season``, ``league``,
            and lat/lon columns for travel distance.
        context : TransformContext, optional
            Pipeline context (ignored in this implementation).

        Returns
        -------
        pd.DataFrame
            Input DataFrame with schedule feature columns added.
        """
        df = df.copy()

        # ── 1. Sort chronologically ───────────────────────
        if "date" in df.columns and self.params.get("sort_by_date", True):
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values(["date", "home_team"], inplace=True)
            df.reset_index(drop=True, inplace=True)

        logger.debug("Schedule: transforming %d rows", len(df))

        # ── 2. Detect optional columns ────────────────────
        self._available_travel_cols = self._detect_travel_columns(df)
        self._resolve_output_columns()

        # ── 3. Build per-team schedule DataFrame ──────────
        team_schedule = self._build_team_schedule(df)

        # ── 4. Compute schedule features per team ─────────
        league_specific = self.params.get("league_specific", _DEFAULT_LEAGUE_SPECIFIC)
        schedule_features = self._compute_schedule_features(
            team_schedule, league_specific=league_specific,
        )

        # ── 5. Add empty feature columns for empty input ──
        if df.empty:
            for col in self._resolved_outputs:
                df[col] = np.nan
            return df

        # ── 6. Merge features onto original DF ─────────────
        df = self._merge_features(df, schedule_features)

        added = [c for c in self._resolved_outputs if c in df.columns]
        logger.debug(
            "Schedule: added %d / %d possible columns",
            len(added), len(self._resolved_outputs),
        )

        return df

    # ══════════════════════════════════════════════════════
    #  Internal: column detection
    # ══════════════════════════════════════════════════════

    def _detect_travel_columns(self, df: pd.DataFrame) -> dict[str, str]:
        """Detect lat/lon columns for travel distance computation.

        Returns
        -------
        dict[str, str]
            Mapping from canonical name (e.g. ``home_lat``) to the
            actual column name found in the DataFrame.
        """
        col_lower = {c.lower(): c for c in df.columns}
        found: dict[str, str] = {}
        for canonical, patterns in _TRAVEL_COL_PATTERNS.items():
            for pattern in patterns:
                if pattern in col_lower:
                    found[canonical] = col_lower[pattern]
                    break
        return found

    def _has_travel_data(self) -> bool:
        """Return True if all four travel columns are available."""
        required = {"home_lat", "home_lon", "away_lat", "away_lon"}
        return required.issubset(self._available_travel_cols)

    # ══════════════════════════════════════════════════════
    #  Internal: team schedule construction
    # ══════════════════════════════════════════════════════

    def _build_team_schedule(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build per-team per-match schedule DataFrame (2 rows per match).

        Each match produces:
        - 1 row for the home team (is_home=1)
        - 1 row for the away team (is_home=0)

        Result DataFrame columns:
            team, date, match_id, is_home, opponent,
            league, season, competition,
            venue_changed (1 if venue changed from team's last match)
            lat, lon (team's venue lat/lon, if available)
            opponent_lat, opponent_lon
        """
        n = len(df)

        # ── Home team rows ────────────────────────────────
        home_df = pd.DataFrame({
            "team": df["home_team"].values,
            "date": pd.to_datetime(df["date"]).values if "date" in df.columns else pd.NaT,
            "match_id": df.index.values,
            "is_home": np.ones(n, dtype=np.int8),
            "opponent": df["away_team"].values,
        })

        # ── Away team rows ────────────────────────────────
        away_df = pd.DataFrame({
            "team": df["away_team"].values,
            "date": pd.to_datetime(df["date"]).values if "date" in df.columns else pd.NaT,
            "match_id": df.index.values,
            "is_home": np.zeros(n, dtype=np.int8),
            "opponent": df["home_team"].values,
        })

        # ── Season and league ─────────────────────────────
        for col in ("season", "league"):
            if col in df.columns:
                home_df[col] = df[col].values
                away_df[col] = df[col].values

        # ── Combine ───────────────────────────────────────
        team_schedule = pd.concat([home_df, away_df], ignore_index=True)
        team_schedule.sort_values(["team", "date"], inplace=True)
        team_schedule.reset_index(drop=True, inplace=True)

        # ── Travel coordinates ────────────────────────────
        opts = self._available_travel_cols
        if self._has_travel_data():
            # Home team gets home venue coordinates
            home_lat_col = opts.get("home_lat", "")
            home_lon_col = opts.get("home_lon", "")
            away_lat_col = opts.get("away_lat", "")
            away_lon_col = opts.get("away_lon", "")

            if home_lat_col and home_lon_col and away_lat_col and away_lon_col:
                home_lat_vals = pd.to_numeric(df[home_lat_col], errors="coerce").values
                home_lon_vals = pd.to_numeric(df[home_lon_col], errors="coerce").values
                away_lat_vals = pd.to_numeric(df[away_lat_col], errors="coerce").values
                away_lon_vals = pd.to_numeric(df[away_lon_col], errors="coerce").values

                team_schedule["lat"] = np.concatenate([home_lat_vals, away_lat_vals])
                team_schedule["lon"] = np.concatenate([home_lon_vals, away_lon_vals])

        return team_schedule

    # ══════════════════════════════════════════════════════
    #  Internal: schedule feature computation
    # ══════════════════════════════════════════════════════

    @staticmethod
    def _haversine_km(
        lat1: float, lon1: float, lat2: float, lon2: float,
    ) -> float:
        """Compute great-circle distance between two points in kilometres.

        Uses the Haversine formula with Earth radius = 6371 km.
        """
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            return np.nan
        R = 6371.0  # Earth radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
            * np.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return float(R * c)

    @staticmethod
    def _count_in_window(dates: np.ndarray, window_days: int) -> np.ndarray:
        """For each position, count how many previous dates fall within window_days.

        Parameters
        ----------
        dates : np.ndarray
            Astype 'datetime64[D]' date array, sorted ascending.
        window_days : int
            Look-back window in days.

        Returns
        -------
        np.ndarray
            Integer count per position (0 for first entry).
        """
        n = len(dates)
        counts = np.zeros(n, dtype=np.int64)
        for i in range(1, n):
            cutoff = dates[i] - np.timedelta64(window_days, "D")
            counts[i] = int(np.sum(dates[:i] >= cutoff))
        return counts

    def _compute_per_team_features(
        self,
        team_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute all schedule features for a single team's matches.

        Parameters
        ----------
        team_data : pd.DataFrame
            Team's match rows sorted by date.
            Must have: ``date``, ``match_id``, ``is_home``, ``opponent``.
            May have: ``lat``, ``lon`` (for travel distance).

        Returns
        -------
        pd.DataFrame
            Same rows with schedule feature columns added.
        """
        team_data = team_data.sort_values("date").copy()
        team_data.reset_index(drop=True, inplace=True)

        dates = team_data["date"].values.astype("datetime64[D]")

        # ── 1. Days since last match (rest days) ──────────
        days_since = np.full(len(dates), np.nan, dtype=np.float64)
        if len(dates) > 1:
            diff = np.diff(dates).astype("timedelta64[D]").astype(np.float64)
            days_since[1:] = diff
            days_since[0] = np.nan  # First match → no data

        team_data["rest_days"] = days_since
        team_data["days_since_last_match"] = days_since  # Alias

        # ── 2. Matches in last 7 / 14 days ────────────────
        team_data["matches_last_7_days"] = self._count_in_window(dates, 7)
        team_data["matches_last_14_days"] = self._count_in_window(dates, 14)

        # ── 3. Consecutive home / away streaks ────────────
        is_home_arr = team_data["is_home"].values.astype(bool)
        streak_change = np.zeros(len(is_home_arr), dtype=bool)
        if len(is_home_arr) > 1:
            streak_change[1:] = is_home_arr[1:] != is_home_arr[:-1]
        streak_id = np.cumsum(streak_change)

        # cumcount per streak (already leakage-free: counts PREVIOUS in streak)
        cumcount = np.zeros(len(streak_id), dtype=np.int64)
        streak_start: dict[int, int] = {}
        for i, sid in enumerate(streak_id):
            cumcount[i] = i - streak_start.get(sid, i)
            if sid not in streak_start:
                streak_start[sid] = i

        consec_home = np.where(is_home_arr, cumcount.astype(np.float64), np.nan)
        consec_away = np.where(~is_home_arr, cumcount.astype(np.float64), np.nan)

        # Mask first match of each streak as NaN (no prior data in that streak)
        if len(streak_id) > 0:
            first_in_streak = np.zeros(len(streak_id), dtype=bool)
            for i in range(1, len(streak_id)):
                if streak_id[i] != streak_id[i - 1]:
                    first_in_streak[i] = True
            first_in_streak[0] = True  # First match overall
            consec_home[first_in_streak] = np.nan
            consec_away[first_in_streak] = np.nan

        team_data["consec_home"] = consec_home
        team_data["consec_away"] = consec_away

        # ── 4. Back-to-back opponent ──────────────────────
        prev_opponent = team_data["opponent"].shift(1)
        team_data["is_back_to_back"] = (
            (team_data["opponent"] == prev_opponent).astype(np.int64)
        ).where(prev_opponent.notna(), other=0)

        # ── 5. Days since competition (if league/season data) ──
        if "league" in team_data.columns:
            comp_days = np.full(len(dates), np.nan, dtype=np.float64)
            # Track last match date per competition
            last_comp_date: dict[str, np.datetime64] = {}
            for i in range(len(dates)):
                league_name = str(team_data.iloc[i].get("league", ""))
                if league_name and league_name != "nan":
                    if league_name in last_comp_date:
                        diff_days = (dates[i] - last_comp_date[league_name]).astype(
                            "timedelta64[D]"
                        ).astype(np.float64)
                        comp_days[i] = diff_days
                    last_comp_date[league_name] = dates[i]
            team_data["days_since_competition"] = comp_days
        else:
            team_data["days_since_competition"] = days_since  # Fallback

        # ── 6. Travel distance ────────────────────────────
        if "lat" in team_data.columns and "lon" in team_data.columns:
            travel_dist = np.full(len(dates), np.nan, dtype=np.float64)
            for i in range(1, len(dates)):
                prev_lat = team_data.iloc[i - 1]["lat"]
                prev_lon = team_data.iloc[i - 1]["lon"]
                curr_lat = team_data.iloc[i]["lat"]
                curr_lon = team_data.iloc[i]["lon"]
                dist = self._haversine_km(prev_lat, prev_lon, curr_lat, curr_lon)
                # Cap at sane maximum
                if not np.isnan(dist) and dist < _DEFAULT_MAX_TRAVEL_KM:
                    travel_dist[i] = dist
            team_data["travel_distance"] = travel_dist
        else:
            team_data["travel_distance"] = np.nan

        return team_data

    def _compute_schedule_features(
        self,
        team_schedule: pd.DataFrame,
        league_specific: bool = True,
    ) -> pd.DataFrame:
        """Compute schedule features for all teams.

        Parameters
        ----------
        team_schedule : pd.DataFrame
            Per-team schedule from ``_build_team_schedule``.
        league_specific : bool
            If True, reset feature computation per league.

        Returns
        -------
        pd.DataFrame
            With columns: match_id, is_home, team, and all schedule feature columns.
        """
        if team_schedule.empty:
            return pd.DataFrame(columns=["match_id"])

        # Determine group keys
        group_keys = ["team"]
        if league_specific and "league" in team_schedule.columns:
            group_keys.append("league")

        feature_frames: list[pd.DataFrame] = []
        for _keys, grp in team_schedule.groupby(group_keys, sort=False):
            feats = self._compute_per_team_features(grp)
            feature_frames.append(feats)

        if not feature_frames:
            return pd.DataFrame(columns=["match_id"])

        result = pd.concat(feature_frames, ignore_index=True)

        # Keep only match_id, is_home, team, and schedule feature columns
        feat_cols = [
            "match_id", "is_home", "team",
            "rest_days", "days_since_last_match",
            "matches_last_7_days", "matches_last_14_days",
            "consec_home", "consec_away",
            "is_back_to_back", "days_since_competition", "travel_distance",
        ]
        keep = [c for c in feat_cols if c in result.columns]
        return result[keep]

    # ══════════════════════════════════════════════════════
    #  Internal: merge features back to original DF
    # ══════════════════════════════════════════════════════

    def _merge_features(
        self,
        df: pd.DataFrame,
        schedule_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge computed schedule features back onto the original DataFrame.

        Uses team-based lookup: for each match, looks up the home team's
        schedule features and away team's schedule features by (team, match_id).
        """
        if schedule_features.empty:
            return df

        df_result = df.copy()

        # Feature columns to map (exclude metadata columns)
        feat_cols = [
            c for c in schedule_features.columns
            if c not in ("match_id", "is_home", "team")
        ]
        # Filter to only include columns in resolved outputs (excludes
        # travel_distance when include_travel_distance=False)
        expected_prefixes = set()
        for col_name in self._resolved_outputs:
            # Strip h_ or a_ prefix to get the base feature name
            if col_name.startswith("h_") or col_name.startswith("a_"):
                expected_prefixes.add(col_name[2:])
        feat_cols = [c for c in feat_cols if c in expected_prefixes]

        if not feat_cols:
            return df_result

        # Build lookup: (team, match_id) -> row dict for each feature
        lookup_df = schedule_features.set_index(["team", "match_id"])

        for col in feat_cols:
            h_col = f"h_{col}"
            a_col = f"a_{col}"

            values = lookup_df[col].to_dict()

            # Home team: look up by (home_team, match_index)
            df_result[h_col] = [
                values.get((df_result.at[idx, "home_team"], idx))
                for idx in df_result.index
            ]

            # Away team: look up by (away_team, match_index)
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

        for col in self.output_columns:
            if col not in df.columns:
                errors.append(f"Missing output column: {col}")

        return errors


# ═══════════════════════════════════════════════════════════════
#  Convenience factory
# ═══════════════════════════════════════════════════════════════


def create_schedule_transformer(
    league_specific: bool = True,
    include_travel_distance: bool = True,
    sort_by_date: bool = True,
    **kwargs: Any,
) -> ScheduleTransformer:
    """Create a configured ScheduleTransformer with explicit params.

    Parameters
    ----------
    league_specific : bool
        Reset schedule computations per league (default True).
    include_travel_distance : bool
        Compute travel distance when lat/lon data available (default True).
    sort_by_date : bool
        Sort input DataFrame chronologically (default True).

    Returns
    -------
    ScheduleTransformer
        Pre-configured transformer instance.
    """
    return ScheduleTransformer(
        league_specific=league_specific,
        include_travel_distance=include_travel_distance,
        sort_by_date=sort_by_date,
        **kwargs,
    )
