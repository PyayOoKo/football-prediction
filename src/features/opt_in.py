"""
Optional external-data features — weather, referee, schedule/congestion, transfer impact.

Each feature group is opt-in via its corresponding config flag.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from config import config as _global_config

logger = logging.getLogger(__name__)

# Lazy import for extended feature transformers
_EXTENDED_FEATURES_AVAILABLE = False
try:
    from src.feature_framework.features.schedule import ScheduleTransformer

    _EXTENDED_FEATURES_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════
#  1.  Weather features (temperature, humidity, wind, pitch)
# ═══════════════════════════════════════════════════════════


def _add_weather_features(
    df: pd.DataFrame,
    config: Any | None = None,
) -> pd.DataFrame:
    """Add weather-related features from database or CSV.

    Reads weather data from ``weather.csv`` (external dir) or fills
    neutral placeholders when no weather data is available.

    Features added: ``{h,a}_temperature_celsius``, ``{h,a}_humidity_pct``,
    ``{h,a}_wind_speed_kmh``, ``{h,a}_precipitation_mm``,
    ``{h,a}_pitch_condition_encoded``, ``{h,a}_weather_severity``.

    Parameters
    ----------
    df : pd.DataFrame
        Match data.
    config : Any, optional
        Injected config object.  Falls back to global ``config`` when
        ``None`` (default).
    """
    cfg = config or _global_config
    if not cfg.weather.enabled:
        return df

    _weather_csv = cfg.paths.external / "weather.csv"
    has_weather = _weather_csv.exists()

    if not has_weather:
        if cfg.weather.warn_missing:
            logger.warning(
                "Weather features enabled but %s not found — using placeholders. "
                "Collect weather data and save to: %s",
                _weather_csv,
                _weather_csv,
            )
        defaults = {
            "temperature_celsius": cfg.weather.default_temp,
            "humidity_pct": 50.0,
            "wind_speed_kmh": 10.0,
            "precipitation_mm": 0.0,
            "pitch_condition_encoded": 0.0,
            "weather_severity": 0.0,
        }
        for col, val in defaults.items():
            df[f"h_{col}"] = val
            df[f"a_{col}"] = val
        return df

    try:
        weather_df = pd.read_csv(_weather_csv)
        logger.info("Loaded %d weather records from %s", len(weather_df), _weather_csv)

        col_map: dict[str, str] = {}
        norm_targets = {
            "match_id": "match_id",
            "temperature": "temperature_celsius",
            "temp": "temperature_celsius",
            "humidity": "humidity_pct",
            "wind": "wind_speed_kmh",
            "precipitation": "precipitation_mm",
            "pitch": "pitch_condition_encoded",
            "condition": "condition_str",
        }
        for c in weather_df.columns:
            cl = c.lower().strip().replace(" ", "_")
            if cl in norm_targets:
                col_map[c] = norm_targets[cl]
        weather_df.rename(columns=col_map, inplace=True)

        if "match_id" in weather_df.columns:
            weather_df.set_index("match_id", inplace=True)

        if "condition_str" in weather_df.columns:
            cond_map = {"dry": 0, "wet": 1, "waterlogged": 2, "frozen": 3}
            weather_df["pitch_condition_encoded"] = (
                weather_df["condition_str"]
                .astype(str)
                .str.lower()
                .map(cond_map)
                .fillna(0)
            )

        severity = pd.Series(0.0, index=weather_df.index)
        for col, weight in [("precipitation_mm", 0.4), ("wind_speed_kmh", 0.3)]:
            if col in weather_df.columns:
                norm_val = weather_df[col].fillna(0) / (
                    weather_df[col].max() if weather_df[col].max() > 0 else 1
                )
                severity += weight * norm_val
        if "temperature_celsius" in weather_df.columns:
            temp = weather_df["temperature_celsius"].fillna(15.0)
            temp_extreme = (temp - 15.0).abs() / 20.0
            severity += 0.3 * temp_extreme.clip(0, 1)
        weather_df["weather_severity"] = severity.clip(0, 1)

        for col in [
            "temperature_celsius",
            "humidity_pct",
            "wind_speed_kmh",
            "precipitation_mm",
            "pitch_condition_encoded",
            "weather_severity",
        ]:
            if col in weather_df.columns:
                vals = weather_df[col].values
                if len(vals) >= len(df):
                    vals = vals[: len(df)]
                else:
                    logger.warning(
                        "Weather CSV has %d rows but match DF has %d — extending with placeholders",
                        len(vals),
                        len(df),
                    )
                    vals = list(vals) + [cfg.weather.placeholder_value] * (
                        len(df) - len(vals)
                    )
                df["h_" + col] = vals
                df["a_" + col] = vals

        logger.info(
            "Added weather features (%d columns) from %s",
            len([c for c in df.columns if "temperature" in c or "humidity" in c]),
            _weather_csv,
        )

    except Exception as exc:
        logger.error("Failed to load weather data: %s — using placeholders", exc)
        defaults = {
            "temperature_celsius": cfg.weather.default_temp,
            "humidity_pct": 50.0,
            "wind_speed_kmh": 10.0,
            "precipitation_mm": 0.0,
            "pitch_condition_encoded": 0.0,
            "weather_severity": 0.0,
        }
        for col, val in defaults.items():
            df[f"h_{col}"] = val
            df[f"a_{col}"] = val

    return df


# ═══════════════════════════════════════════════════════════
#  2.  Referee statistics (card rates, foul rates, home bias)
# ═══════════════════════════════════════════════════════════


def _add_referee_features(
    df: pd.DataFrame,
    config: Any | None = None,
) -> pd.DataFrame:
    """Add referee-based features from database or CSV.

    Features: ``referee_home_yellow_rate``, ``referee_away_yellow_rate``,
    ``referee_home_win_rate``, ``referee_card_total_avg``.

    Parameters
    ----------
    df : pd.DataFrame
        Match data.
    config : Any, optional
        Injected config object.  Falls back to global ``config`` when
        ``None`` (default).
    """
    cfg = config or _global_config
    if not cfg.referee.enabled:
        return df

    _referee_csv = cfg.paths.external / "referees.csv"
    has_referee_data = _referee_csv.exists()

    if not has_referee_data:
        if cfg.referee.warn_missing:
            logger.warning(
                "Referee features enabled but %s not found — using placeholders. "
                "Save referee data to: %s",
                _referee_csv,
                _referee_csv,
            )
        df["referee_home_yellow_rate"] = cfg.referee.placeholder_value
        df["referee_away_yellow_rate"] = cfg.referee.placeholder_value
        df["referee_home_win_rate"] = 0.5
        df["referee_card_total_avg"] = cfg.referee.placeholder_value
        return df

    try:
        ref_df = pd.read_csv(_referee_csv)
        logger.info("Loaded %d referee records from %s", len(ref_df), _referee_csv)

        col_map = {}
        for c in ref_df.columns:
            cl = c.lower().strip()
            if cl in ("referee", "referee_name", "name", "full_name"):
                col_map[c] = "referee_name"
            elif cl in ("home_yellow", "home_yellow_cards", "h_yellow", "h_yc"):
                col_map[c] = "home_yellow_cards"
            elif cl in ("away_yellow", "away_yellow_cards", "a_yellow", "a_yc"):
                col_map[c] = "away_yellow_cards"
            elif cl in ("home_red", "home_red_cards", "h_red", "h_rc"):
                col_map[c] = "home_red_cards"
            elif cl in ("away_red", "away_red_cards", "a_red", "a_rc"):
                col_map[c] = "away_red_cards"
            elif cl in ("match_id", "id", "matchid"):
                col_map[c] = "match_id"
            elif cl in ("home_fouls", "h_fouls"):
                col_map[c] = "home_fouls"
            elif cl in ("away_fouls", "a_fouls"):
                col_map[c] = "away_fouls"
            elif cl in ("result", "winner"):
                col_map[c] = "result"
            elif cl in ("date", "match_date"):
                col_map[c] = "date"
        ref_df.rename(columns=col_map, inplace=True)

        if "referee_name" not in ref_df.columns:
            yellow_h = ref_df.get("home_yellow_cards", pd.Series()).mean() or 0
            yellow_a = ref_df.get("away_yellow_cards", pd.Series()).mean() or 0
            red_h = ref_df.get("home_red_cards", pd.Series()).mean() or 0
            red_a = ref_df.get("away_red_cards", pd.Series()).mean() or 0
            df["referee_home_yellow_rate"] = yellow_h
            df["referee_away_yellow_rate"] = yellow_a
            df["referee_card_total_avg"] = yellow_h + yellow_a + red_h + red_a
            df["referee_home_win_rate"] = 0.5
            return df

        if "date" in ref_df.columns:
            ref_df["date"] = pd.to_datetime(ref_df["date"])
            ref_df.sort_values(["referee_name", "date"], inplace=True)

        window = cfg.referee.window

        def _ref_stats(grp: pd.DataFrame) -> pd.DataFrame:
            grp = (
                grp.sort_values("date").copy() if "date" in grp.columns else grp.copy()
            )
            grp["ref_home_yellow_rate"] = (
                grp.get("home_yellow_cards", pd.Series(0, index=grp.index))
                .rolling(window, min_periods=1)
                .mean()
                .shift(1)
            )
            grp["ref_away_yellow_rate"] = (
                grp.get("away_yellow_cards", pd.Series(0, index=grp.index))
                .rolling(window, min_periods=1)
                .mean()
                .shift(1)
            )
            total_cards = (
                grp.get("home_yellow_cards", pd.Series(0, index=grp.index)).fillna(0)
                + grp.get("away_yellow_cards", pd.Series(0, index=grp.index)).fillna(0)
                + grp.get("home_red_cards", pd.Series(0, index=grp.index)).fillna(0)
                + grp.get("away_red_cards", pd.Series(0, index=grp.index)).fillna(0)
            )
            grp["ref_card_total_avg"] = (
                total_cards.rolling(window, min_periods=1).mean().shift(1)
            )

            if "result" in grp.columns:
                grp["ref_home_win_rate"] = (
                    (grp["result"].str.upper() == "H")
                    .rolling(window, min_periods=1)
                    .mean()
                    .shift(1)
                )
            else:
                grp["ref_home_win_rate"] = 0.5
            return grp

        ref_stats = ref_df.groupby("referee_name", group_keys=False).apply(_ref_stats)

        if "match_id" in ref_stats.columns and "match_id" in df.columns:
            merge_cols = [
                "match_id",
                "ref_home_yellow_rate",
                "ref_away_yellow_rate",
                "ref_card_total_avg",
                "ref_home_win_rate",
            ]
            existing = [c for c in merge_cols if c in ref_stats.columns]
            df = df.merge(ref_stats[existing], on="match_id", how="left")
        else:
            logger.warning(
                "Cannot merge referee stats by match_id. Using sequential alignment — "
                "this may misalign data if referee.csv and match DataFrame are not "
                "in the same order or have different numbers of rows."
            )
            for col in [
                "ref_home_yellow_rate",
                "ref_away_yellow_rate",
                "ref_card_total_avg",
                "ref_home_win_rate",
            ]:
                if col in ref_stats.columns:
                    df[col] = (
                        ref_stats[col].iloc[: len(df)].values
                        if len(ref_stats) >= len(df)
                        else cfg.referee.placeholder_value
                    )
                else:
                    df[col] = cfg.referee.placeholder_value

        for col in [
            "ref_home_yellow_rate",
            "ref_away_yellow_rate",
            "ref_card_total_avg",
            "ref_home_win_rate",
        ]:
            if col in df.columns:
                df[col] = df[col].fillna(cfg.referee.placeholder_value)

        logger.info(
            "Added referee features (%d columns)",
            len([c for c in df.columns if "ref_" in c]),
        )

    except Exception as exc:
        logger.error("Failed to load referee data: %s — using placeholders", exc)
        df["referee_home_yellow_rate"] = cfg.referee.placeholder_value
        df["referee_away_yellow_rate"] = cfg.referee.placeholder_value
        df["referee_home_win_rate"] = 0.5
        df["referee_card_total_avg"] = cfg.referee.placeholder_value

    return df


# ═══════════════════════════════════════════════════════════
#  3.  Schedule / congestion features (travel, fatigue, rest)
# ═══════════════════════════════════════════════════════════


def _add_schedule_features(
    df: pd.DataFrame,
    config: Any | None = None,
) -> pd.DataFrame:
    """Add schedule/congestion features using the ScheduleTransformer.

    Features: rest days, matches in last 7/14 days, consecutive home/away
    streaks, back-to-back opponent, travel distance, days since last competition.

    Parameters
    ----------
    df : pd.DataFrame
        Match data.
    config : Any, optional
        Injected config object.  Falls back to global ``config`` when
        ``None`` (default).
    """
    cfg = config or _global_config
    if not cfg.schedule.enabled:
        return df

    try:
        if not _EXTENDED_FEATURES_AVAILABLE:
            logger.warning(
                "Schedule features require the feature framework modules. "
                "Install them or set config.schedule.enabled=False."
            )
            return df

        transformer = ScheduleTransformer(
            include_travel_distance=cfg.schedule.include_travel_distance,
            league_specific=True,
            sort_by_date=False,
        )
        transformer.init()
        df = transformer.transform(df)

        n_added = len(
            [
                c
                for c in df.columns
                if c.startswith(("h_", "a_"))
                and any(
                    kw in c
                    for kw in [
                        "rest_days",
                        "matches_last",
                        "consec_",
                        "back_to_back",
                        "travel_distance",
                        "days_since_competition",
                    ]
                )
            ]
        )
        logger.info("Added %d schedule/congestion feature columns", n_added)

    except Exception as exc:
        logger.error("Failed to compute schedule features: %s", exc)

    return df


# ═══════════════════════════════════════════════════════════
#  4.  Transfer impact features (recent signings, squad turnover)
# ═══════════════════════════════════════════════════════════


def _add_transfer_features(
    df: pd.DataFrame,
    config: Any | None = None,
) -> pd.DataFrame:
    """Add transfer/roster-change impact features.

    Reads transfer data from ``transfers.csv`` (external dir) or fills
    neutral placeholders when no data is available.

    Features: ``{h,a}_signings_count``, ``{h,a}_departures_count``,
    ``{h,a}_net_spend_meur``, ``{h,a}_squad_churn_pct``.

    Parameters
    ----------
    df : pd.DataFrame
        Match data.
    config : Any, optional
        Injected config object.  Falls back to global ``config`` when
        ``None`` (default).
    """
    cfg = config or _global_config
    if not cfg.extended_features.enabled:
        return df

    _transfers_csv = cfg.paths.external / "transfers.csv"
    has_transfers = _transfers_csv.exists()

    if not has_transfers:
        if cfg.player_info.warn_missing:
            logger.info(
                "Transfer features: %s not found — using neutral placeholders. "
                "Save transfer data to enable squad-churn features.",
                _transfers_csv,
            )
        for prefix in ["h_", "a_"]:
            for col in [
                "signings_count",
                "departures_count",
                "net_spend_meur",
                "squad_churn_pct",
            ]:
                df[f"{prefix}{col}"] = 0.0
        return df

    try:
        transfer_df = pd.read_csv(_transfers_csv)
        logger.info(
            "Loaded %d transfer records from %s", len(transfer_df), _transfers_csv
        )

        col_map = {}
        for c in transfer_df.columns:
            cl = c.lower().strip()
            if cl in ("team", "club", "squad"):
                col_map[c] = "team"
            elif cl in ("date", "window_date", "transfer_date"):
                col_map[c] = "date"
            elif cl in ("incoming", "signings", "players_in", "arrivals"):
                col_map[c] = "signings"
            elif cl in ("outgoing", "departures", "players_out", "sales"):
                col_map[c] = "departures"
            elif cl in ("net_spend", "spend_net", "net", "net_spend_meur"):
                col_map[c] = "net_spend"
            elif cl in ("squad_size", "total_players", "size"):
                col_map[c] = "squad_size"
        transfer_df.rename(columns=col_map, inplace=True)

        if "team" not in transfer_df.columns:
            logger.warning("Transfer CSV missing 'team' column — using placeholders")
            for prefix in ["h_", "a_"]:
                for col in [
                    "signings_count",
                    "departures_count",
                    "net_spend_meur",
                    "squad_churn_pct",
                ]:
                    df[f"{prefix}{col}"] = 0.0
            return df

        # Aggregate per team (most recent window)
        if "date" in transfer_df.columns:
            transfer_df["date"] = pd.to_datetime(transfer_df["date"])
            transfer_df = (
                transfer_df.sort_values("date").groupby("team").last().reset_index()
            )

        team_transfer: dict[str, dict[str, float]] = {}
        for _, row in transfer_df.iterrows():
            team_name = str(row.get("team", ""))
            signings = float(row.get("signings", 0))
            departures = float(row.get("departures", 0))
            net_spend = float(row.get("net_spend", 0))
            squad_size = float(row.get("squad_size", 25))
            if squad_size <= 0:
                squad_size = 25
            team_transfer[team_name] = {
                "signings_count": signings,
                "departures_count": departures,
                "net_spend_meur": net_spend,
                "squad_churn_pct": round((signings + departures) / squad_size, 3),
            }

        for prefix, team_col in [("h_", "home_team"), ("a_", "away_team")]:
            for col in [
                "signings_count",
                "departures_count",
                "net_spend_meur",
                "squad_churn_pct",
            ]:
                df[f"{prefix}{col}"] = df[team_col].map(
                    lambda t, _col=col: team_transfer.get(t, {}).get(_col, 0.0)
                )

        logger.info("Added transfer features for %d teams", len(team_transfer))

    except Exception as exc:
        logger.error("Failed to load transfer data: %s — using placeholders", exc)
        for prefix in ["h_", "a_"]:
            for col in [
                "signings_count",
                "departures_count",
                "net_spend_meur",
                "squad_churn_pct",
            ]:
                df[f"{prefix}{col}"] = 0.0

    return df
