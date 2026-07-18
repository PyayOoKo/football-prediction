"""
Data cleaners — source-specific data cleaning and normalisation.

Each function takes a raw DataFrame from a specific source
and returns a cleaned, standardised version with consistent
column names and types.

Standard output columns:
    season, date, league, home_team, away_team,
    result (H/D/A), home_goals, away_goals,
    home_goals_ht, away_goals_ht,
    home_xg, away_xg, source, downloaded_at

Season derivation convention:
    - August–December 2023 → 2023 (numeric start year, representing 2023/24)
    - January–May 2024     → 2023 (same season, 2023/24)
    - June–July 2024       → 2023 (end of season; included in prior season)
    - An explicit June/July note: these months are treated as the tail
      of the season that started in the previous August.  If the league
      uses a calendar-year season (e.g. some Nordic leagues), the
      season column should be provided explicitly by the caller.
    - Existing ``season`` column is preserved when present.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _parse_boolean(val: Any, col_name: str = "") -> bool:
    """Convert a variety of representations to a boolean.

    Supported inputs:
        True / False (bool)
        "true" / "false" (str, case-insensitive)
        "yes" / "no" (str, case-insensitive)
        "y" / "n" (str, case-insensitive)
        "1" / "0" (str)
        1 / 0 (int / float)
        "" (empty string) → False
        None / NaN → False

    Parameters
    ----------
    val : Any
        Value to convert.
    col_name : str
        Column name for logging (optional).

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        If *val* is a string that does not match any known pattern.
    """
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return False

    if isinstance(val, (int, float)):
        if val == 1:
            return True
        if val == 0:
            return False

    s = str(val).strip().lower()
    if s in ("true", "yes", "y", "1"):
        return True
    if s in ("false", "no", "n", "0", ""):
        return False

    logger.warning(
        "Unexpected boolean value %r in column %r — treating as False",
        val,
        col_name,
    )
    return False


def _derive_season(dt_series: pd.Series) -> pd.Series:
    """Derive a football-season start year from a datetime series.

    Convention
    ----------
    * Aug–Dec → year (e.g. 2023-08 → 2023)
    * Jan–May → year - 1 (e.g. 2024-01 → 2023)
    * Jun–Jul → year - 1 (treated as tail of prior season; e.g. 2024-06 → 2023)

    Returns an ``Int64`` series (allows NA).
    """
    mask = dt_series.notna()
    result = pd.Series(pd.NA, index=dt_series.index, dtype="Int64")
    result[mask] = (
        dt_series[mask]
        .dt.year.where(
            dt_series[mask].dt.month >= 8,
            dt_series[mask].dt.year - 1,
        )
        .astype("Int64")
    )
    return result


# ── Column mapping: football-data.co.uk → standard ─────
_FOOTBALL_DATA_COLUMNS: dict[str, tuple[str, str | type]] = {
    "Div": ("league", str),
    "Date": ("date", str),
    "Time": ("time", str),
    "HomeTeam": ("home_team", str),
    "AwayTeam": ("away_team", str),
    "FTHG": ("home_goals", float),
    "FTAG": ("away_goals", float),
    "FTR": ("result", str),
    "HTHG": ("home_goals_ht", float),
    "HTAG": ("away_goals_ht", float),
    "HTR": ("result_ht", str),
    "HS": ("home_shots", float),
    "AS": ("away_shots", float),
    "HST": ("home_shots_on_target", float),
    "AST": ("away_shots_on_target", float),
    "HF": ("home_fouls", float),
    "AF": ("away_fouls", float),
    "HC": ("home_corners", float),
    "AC": ("away_corners", float),
    "HY": ("home_yellow", float),
    "AY": ("away_yellow", float),
    "HR": ("home_red", float),
    "AR": ("away_red", float),
    "B365H": ("odds_home_b365", float),
    "B365D": ("odds_draw_b365", float),
    "B365A": ("odds_away_b365", float),
    "BbAvH": ("odds_home_avg", float),
    "BbAvD": ("odds_draw_avg", float),
    "BbAvA": ("odds_away_avg", float),
    "BbMxH": ("odds_home_max", float),
    "BbMxD": ("odds_draw_max", float),
    "BbMxA": ("odds_away_max", float),
    "BbOU": ("odds_over_under_market", str),
    "BbAv>2.5": ("odds_over_avg", float),
    "BbAv<2.5": ("odds_under_avg", float),
    "home_xg": ("home_xg", float),
    "away_xg": ("away_xg", float),
}


class DataCleaner:
    """Collection of data cleaning methods for different sources."""

    # ── football-data.co.uk ─────────────────────────────

    @staticmethod
    def football_data_co_uk(df: pd.DataFrame) -> pd.DataFrame:
        """Clean a DataFrame from football-data.co.uk.

        Handles column renaming, type coercion, missing values,
        and date parsing for the standard football-data.co.uk CSV format.

        Parameters
        ----------
        df : pd.DataFrame
            Raw data as downloaded from football-data.co.uk.

        Returns
        -------
        pd.DataFrame
            Cleaned data with standardised column names and types.
        """
        logger.info("Cleaning football-data.co.uk data: %d rows", len(df))
        df = df.copy()

        # ── 1. Rename columns ───────────────────────────
        renamed: dict[str, str] = {}
        col_types: dict[str, type] = {}
        for raw_col, (std_name, dtype) in _FOOTBALL_DATA_COLUMNS.items():
            if raw_col in df.columns:
                renamed[raw_col] = std_name
                col_types[std_name] = dtype
        df = df.rename(columns=renamed)

        # ── 2. Coerce types ─────────────────────────────
        for col, dtype in col_types.items():
            if col not in df.columns:
                continue
            if dtype is float:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype is str and not pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].fillna("").astype(str)

        # ── 3. Parse dates ──────────────────────────────
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

        # ── 4. Create result from FTR if present ────────
        if "result" in df.columns:
            df["result"] = df["result"].str.strip().str.upper()
            df["result"] = df["result"].map({"H": "H", "D": "D", "A": "A"}).fillna("")

        # ── 5. Add metadata columns ─────────────────────
        df["source"] = "football-data.co.uk"
        df["downloaded_at"] = datetime.now().isoformat()

        # ── 6. Add season column if missing ─────────────
        if "season" not in df.columns and "date" in df.columns:
            df["season"] = _derive_season(df["date"])

        # ── 7. Sort chronologically ─────────────────────
        if "date" in df.columns:
            df = df.sort_values(["date", "home_team"]).reset_index(drop=True)

        logger.info(
            "Cleaned football-data.co.uk: %d rows × %d cols",
            len(df),
            len(df.columns),
        )
        return df

    # ── football-data.org API ───────────────────────────

    @staticmethod
    def football_data_org(df: pd.DataFrame) -> pd.DataFrame:
        """Clean a DataFrame from the football-data.org API.

        Handles the API response format which typically returns
        nested match objects with ``score.fullTime.home`` etc.
        Automatically flattens nested dicts using ``json_normalize``.

        Parameters
        ----------
        df : pd.DataFrame
            Raw data from football-data.org API (list of matches).

        Returns
        -------
        pd.DataFrame
            Cleaned, flattened data with standard column names.
        """
        logger.info("Cleaning football-data.org data: %d rows", len(df))
        df = df.copy()

        # ── 0. Flatten nested dict columns ──────────────
        # If columns look nested (tuples or multi-index), use json_normalize
        has_nested = any(isinstance(c, tuple) for c in df.columns) or any(
            isinstance(df[col].iloc[0], dict) if len(df) > 0 else False
            for col in df.columns
            if df[col].dtype == "object"
        )
        if has_nested:
            records = df.to_dict(orient="records")
            try:
                df = pd.json_normalize(records, max_level=2)
                logger.info("Flattened nested JSON columns via json_normalize")
            except Exception as exc:
                logger.warning(
                    "Failed to flatten nested columns: %s",
                    exc,
                )

        # ── 1. Normalise column names ───────────────────
        col_map: dict[str, str] = {
            "id": "match_id",
            "competition.name": "league",
            "utcDate": "date",
            "status": "match_status",
            "matchday": "matchday",
            "stage": "round",
            "group": "group",
            "homeTeam.name": "home_team",
            "awayTeam.name": "away_team",
            "score.fullTime.home": "home_goals",
            "score.fullTime.away": "away_goals",
            "score.halfTime.home": "home_goals_ht",
            "score.halfTime.away": "away_goals_ht",
        }

        # Handle nested columns (dot-separated paths)
        flat_cols: dict[str, str] = {}
        for col in df.columns:
            # Try direct match first
            if col in col_map:
                flat_cols[col] = col_map[col]
            else:
                # Try matching the flattened version
                key = str(col).strip().lower().replace(" ", ".")
                if key in col_map:
                    flat_cols[col] = col_map[key]
                else:
                    # Keep as-is but clean the name
                    flat_cols[col] = str(col).strip().lower().replace(" ", "_")

        df = df.rename(columns=flat_cols)

        # ── 2. Parse dates ──────────────────────────────
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # ── 3. Coerce goal columns to numeric ───────────
        for col in ["home_goals", "away_goals", "home_goals_ht", "away_goals_ht"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ── 4. Create result column ─────────────────────
        if "home_goals" in df.columns and "away_goals" in df.columns:
            df["result"] = np.where(
                df["home_goals"].notna() & df["away_goals"].notna(),
                np.select(
                    [
                        df["home_goals"] > df["away_goals"],
                        df["home_goals"] < df["away_goals"],
                    ],
                    ["H", "A"],
                    default="D",
                ),
                "",
            )
        elif "result" in df.columns:
            df["result"] = df["result"].fillna("")

        # ── 5. Add metadata ─────────────────────────────
        df["source"] = "football-data.org"
        df["downloaded_at"] = datetime.now().isoformat()

        if "season" not in df.columns and "date" in df.columns:
            df["season"] = _derive_season(df["date"])

        if "date" in df.columns:
            df = df.sort_values(["date", "home_team"]).reset_index(drop=True)

        logger.info(
            "Cleaned football-data.org: %d rows × %d cols",
            len(df),
            len(df.columns),
        )
        return df

    # ── Transfermarkt ──────────────────────────────────

    @staticmethod
    def transfermarkt(df: pd.DataFrame) -> pd.DataFrame:
        """Clean a DataFrame scraped from Transfermarkt.

        Handles the typical Transfermarkt squad/player CSV format,
        normalising column names and data types.

        Parameters
        ----------
        df : pd.DataFrame
            Raw scraped data from Transfermarkt.

        Returns
        -------
        pd.DataFrame
            Cleaned data with standardised column names.
        """
        logger.info("Cleaning Transfermarkt data: %d rows", len(df))
        df = df.copy()

        # ── 1. Normalise column names ───────────────────
        col_map: dict[str, str] = {
            "player_name": "player_name",
            "name": "player_name",
            "player": "player_name",
            "team": "team",
            "club": "team",
            "squad": "team",
            "position": "position",
            "pos": "position",
            "age": "age",
            "date_of_birth": "date_of_birth",
            "dob": "date_of_birth",
            "birth": "date_of_birth",
            "nationality": "nationality",
            "nation": "nationality",
            "market_value": "market_value",
            "value": "market_value",
            "price": "market_value",
            "estimated_value": "market_value",
            "injured": "injured",
            "injury": "injured",
            "suspended": "suspended",
            "suspension": "suspended",
            "goals": "goals_scored",
            "goals_scored": "goals_scored",
            "assists": "assists",
            "appearances": "appearances",
            "matches": "appearances",
            "minutes_played": "minutes_played",
            "minutes": "minutes_played",
            "yellow_cards": "yellow_cards",
            "yellow": "yellow_cards",
            "red_cards": "red_cards",
            "red": "red_cards",
            "rating": "rating",
            "average_rating": "rating",
        }

        renamed: dict[str, str] = {}
        for col in df.columns:
            key = col.strip().lower().replace(" ", "_")
            if key in col_map:
                renamed[col] = col_map[key]
        df = df.rename(columns=renamed)

        # ── 2. Coerce numeric columns ───────────────────
        for col in [
            "age",
            "market_value",
            "goals_scored",
            "assists",
            "appearances",
            "minutes_played",
            "yellow_cards",
            "red_cards",
            "rating",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ── 3. Parse date of birth ──────────────────────
        if "date_of_birth" in df.columns:
            df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")

        # ── 4. Fill missing values ──────────────────────
        if "injured" not in df.columns:
            df["injured"] = False
        else:
            df["injured"] = df["injured"].apply(lambda x: _parse_boolean(x, "injured"))

        if "suspended" not in df.columns:
            df["suspended"] = False
        else:
            df["suspended"] = df["suspended"].apply(
                lambda x: _parse_boolean(x, "suspended")
            )

        if "position" in df.columns:
            df["position"] = (
                df["position"].fillna("").astype(str).str.strip().str.upper()
            )

        # ── 5. Add metadata ─────────────────────────────
        df["source"] = "transfermarkt"
        df["downloaded_at"] = datetime.now().isoformat()

        logger.info(
            "Cleaned Transfermarkt: %d rows × %d cols",
            len(df),
            len(df.columns),
        )
        return df
