"""
OpenWeatherMap — historical weather data collector.

Fetches historical weather conditions for match dates and locations
from the OpenWeatherMap One Call API / History API.

Data source: https://openweathermap.org/
API docs: https://openweathermap.org/api/one-call-api

Output columns
--------------
- ``match_id``         Unique match identifier (from input)
- ``date``             Match date
- ``temperature_celsius``   Temperature in °C
- ``feels_like_celsius``    Feels-like temperature in °C
- ``humidity_pct``          Relative humidity (0-100)
- ``wind_speed_ms``         Wind speed in m/s
- ``precipitation_mm``      Precipitation in mm (rain/snow)
- ``condition``             Weather condition text (e.g. "Clear", "Rain")
- ``condition_code``        OWM weather condition code
- ``pressure_hpa``          Atmospheric pressure in hPa

Usage
-----
    from src.data_collection.sources.weather_api import collect_weather

    df = collect_weather(matches_df, lat_lon_map)
    # df: pd.DataFrame with weather data per match
"""

from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import config

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────

OWM_BASE = "https://api.openweathermap.org"
"""Base URL for OpenWeatherMap API."""

FREE_TIER_DELAY = 1.1
"""Minimum seconds between free-tier API calls (60 calls/min limit)."""

PAID_TIER_DELAY = 0.25
"""Minimum seconds between paid-tier API calls."""

CACHE_DIR = "data/external/weather_cache"
"""Directory for cached weather responses."""

# Default city coordinates (fallback when team location is unknown)
DEFAULT_LAT = 51.5074
DEFAULT_LON = -0.1278  # London


# ── Session ─────────────────────────────────────────────


def _session() -> requests.Session:
    """Create a requests session with retry logic."""
    sess = requests.Session()
    retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 502, 503, 504])
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.headers.update({
        "User-Agent": "FootballPrediction/2.0.0",
    })
    return sess


# ── Public API ──────────────────────────────────────────


def collect_weather(
    matches_df: pd.DataFrame,
    lat_lon_map: dict[str, tuple[float, float]] | None = None,
    api_key: str | None = None,
    use_cache: bool = True,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Fetch historical weather data for a set of matches.

    Parameters
    ----------
    matches_df : pd.DataFrame
        DataFrame with at least ``date`` and ``home_team`` columns.
    lat_lon_map : dict[str, tuple[float, float]], optional
        Mapping from team name → (latitude, longitude) for their home city.
        Defaults to a built-in set of known national team locations.
    api_key : str, optional
        OpenWeatherMap API key. Falls back to env var ``OPENWEATHER_API_KEY``.
    use_cache : bool
        Whether to use cached responses (default True).
    output_path : str, optional
        If provided, save the resulting DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        Weather data with one row per match.
    """
    key = api_key or os.environ.get(config.weather_collector.api_key_env) or ""
    if not key:
        logger.warning(
            "No OpenWeatherMap API key found. Set %s env var. "
            "Returning placeholder data.",
            config.weather_collector.api_key_env,
        )
        return _build_placeholder_df(len(matches_df))

    if lat_lon_map is None:
        lat_lon_map = _nat_team_coords

    delay = FREE_TIER_DELAY
    sess = _session()
    cache_dir = Path(CACHE_DIR)
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    errors = 0

    # Deduplicate by (team, date) to minimise API calls
    seen: dict[tuple[str, str], dict[str, Any]] = {}

    for _, row in matches_df.iterrows():
        date_str = _normalise_date(row.get("date", ""))
        team = str(row.get("home_team", ""))

        if not date_str or not team:
            continue

        cache_key = (team, date_str)
        if cache_key in seen:
            records.append(seen[cache_key].copy())
            continue

        # Get coordinates for the team
        lat, lon = lat_lon_map.get(team, (DEFAULT_LAT, DEFAULT_LON))

        # Check cache
        cache_file = cache_dir / f"{team}_{date_str.replace('-', '')}.json"
        if use_cache and cache_file.exists():
            try:
                import json
                data = json.loads(cache_file.read_text())
                weather_row = _parse_owm_response(data, row)
                records.append(weather_row)
                seen[cache_key] = weather_row
                continue
            except Exception:
                pass  # Cache miss — re-fetch

        try:
            dt = int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())
            url = (
                f"{OWM_BASE}/data/3.0/onecall/timemachine"
                f"?lat={lat}&lon={lon}&dt={dt}&appid={key}&units=metric"
            )
            resp = sess.get(url, timeout=15)

            if resp.status_code == 404:
                # Try history API fallback
                url = (
                    f"{OWM_BASE}/data/2.5/onecall/timemachine"
                    f"?lat={lat}&lon={lon}&dt={dt}&appid={key}&units=metric"
                )
                resp = sess.get(url, timeout=15)

            resp.raise_for_status()
            data = resp.json()

            if use_cache:
                import json
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(data, indent=2))

            weather_row = _parse_owm_response(data, row)
            records.append(weather_row)
            seen[cache_key] = weather_row

        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 401:
                logger.error("Invalid OWM API key — returning placeholder data")
                return _build_placeholder_df(len(matches_df))
            errors += 1
            if errors >= 3:
                logger.warning(
                    "3 consecutive OWM errors — falling back to placeholders"
                )
                break
            time.sleep(delay * 2)
        except Exception as exc:
            logger.debug("Weather fetch error for %s on %s: %s", team, date_str, exc)
            errors += 1

        time.sleep(delay)

    # Build final DataFrame
    if not records:
        logger.warning("No weather data fetched — returning placeholders")
        return _build_placeholder_df(len(matches_df))

    df = pd.DataFrame(records)

    # Fill any missing matches with placeholders
    if len(df) < len(matches_df):
        placeholders = _build_placeholder_df(
            len(matches_df) - len(df)
        )
        df = pd.concat([df, placeholders], ignore_index=True)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved %d weather rows to %s", len(df), output_path)

    return df


def get_team_coordinates(
    team_names: list[str],
    save_path: str | None = None,
) -> dict[str, tuple[float, float]]:
    """Look up GPS coordinates for teams using OpenWeatherMap Geocoding API.

    Uses the built-in national team map as primary source, then
    queries the OWM Geocoding API for any unknown teams.

    Parameters
    ----------
    team_names : list[str]
        Team names to look up.
    save_path : str, optional
        If provided, save the mapping as CSV.

    Returns
    -------
    dict[str, tuple[float, float]]
        Team → (lat, lon) mapping.
    """
    result = {}
    unknown: list[str] = []

    for name in team_names:
        if name in _nat_team_coords:
            result[name] = _nat_team_coords[name]
        else:
            unknown.append(name)

    if unknown:
        geocoded = _geocode_teams(unknown)
        result.update(geocoded)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["team", "latitude", "longitude"])
            for team, (lat, lon) in sorted(result.items()):
                writer.writerow([team, lat, lon])

    return result


# ═══════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════


def _normalise_date(date_val: Any) -> str:
    """Convert a date value to ``YYYY-MM-DD`` string."""
    if isinstance(date_val, str):
        date_val = date_val.strip()
        if " " in date_val:
            date_val = date_val.split()[0]
        return date_val[:10]
    if isinstance(date_val, (int, float)):
        return datetime.fromtimestamp(date_val).strftime("%Y-%m-%d")
    try:
        return pd.Timestamp(date_val).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ""


def _parse_owm_response(
    data: dict[str, Any],
    match_row: pd.Series,
) -> dict[str, Any]:
    """Parse an OpenWeatherMap One Call response into a flat dict.

    Parameters
    ----------
    data : dict
        OWM API response JSON.
    match_row : pd.Series
        Original match row (for match_id, team, etc.).

    Returns
    -------
    dict
        Flat weather record matching the module's output schema.
    """
    current = data.get("data", [{}])[0] if "data" in data else data.get("current", {})

    weather_desc = ""
    weather_code = 0
    weather_list = current.get("weather", [])
    if weather_list:
        weather_desc = weather_list[0].get("description", "")
        weather_code = weather_list[0].get("id", 0)

    # Extract precipitation
    rain = current.get("rain", {})
    snow = current.get("snow", {})
    precip = rain.get("1h", 0) or rain.get("3h", 0) or 0
    if not precip:
        precip = snow.get("1h", 0) or snow.get("3h", 0) or 0

    record = {
        "match_id": match_row.get("match_id", ""),
        "home_team": match_row.get("home_team", ""),
        "date": match_row.get("date", ""),
        "temperature_celsius": current.get("temp", config.weather.default_temp),
        "feels_like_celsius": current.get("feels_like", config.weather.default_temp),
        "humidity_pct": float(current.get("humidity", 50)),
        "wind_speed_ms": float(current.get("wind_speed", 0)),
        "precipitation_mm": float(precip),
        "condition": weather_desc,
        "condition_code": weather_code,
        "pressure_hpa": float(current.get("pressure", 1013)),
    }
    return record


def _geocode_teams(team_names: list[str]) -> dict[str, tuple[float, float]]:
    """Look up GPS coordinates for unknown teams via OWM Geocoding API."""
    api_key = os.environ.get(config.weather_collector.api_key_env, "")
    if not api_key:
        return {name: (DEFAULT_LAT, DEFAULT_LON) for name in team_names}

    sess = _session()
    results: dict[str, tuple[float, float]] = {}

    for name in team_names:
        try:
            url = (
                f"{OWM_BASE}/geo/1.0/direct"
                f"?q={name}&limit=1&appid={api_key}"
            )
            resp = sess.get(url, timeout=10)
            resp.raise_for_status()
            geo = resp.json()
            if geo:
                results[name] = (geo[0]["lat"], geo[0]["lon"])
            else:
                results[name] = (DEFAULT_LAT, DEFAULT_LON)
            time.sleep(0.5)
        except Exception:
            results[name] = (DEFAULT_LAT, DEFAULT_LON)

    return results


def _build_placeholder_df(n: int) -> pd.DataFrame:
    """Build a DataFrame of placeholder weather values."""
    return pd.DataFrame([{
        "match_id": "",
        "home_team": "",
        "date": "",
        "temperature_celsius": config.weather.default_temp,
        "feels_like_celsius": config.weather.default_temp,
        "humidity_pct": 50.0,
        "wind_speed_ms": 0.0,
        "precipitation_mm": 0.0,
        "condition": "",
        "condition_code": 0,
        "pressure_hpa": 1013.0,
    }] * n)


# ── Built-in national team coordinates ──────────────────

_nat_team_coords: dict[str, tuple[float, float]] = {
    # World Cup 2026 teams + common historical teams
    "Algeria":            (36.7538, 3.0588),
    "Argentina":          (-34.6037, -58.3816),
    "Australia":          (-33.8688, 151.2093),
    "Austria":            (48.2082, 16.3738),
    "Belgium":            (50.8503, 4.3517),
    "Brazil":             (-15.7939, -47.8828),
    "Cameroon":           (3.8480, 11.5021),
    "Canada":             (45.4215, -75.6972),
    "Chile":              (-33.4489, -70.6693),
    "China":              (39.9042, 116.4074),
    "Colombia":           (4.7110, -74.0721),
    "Costa Rica":         (9.9281, -84.0907),
    "Croatia":            (45.8150, 15.9819),
    "Czech Republic":     (50.0755, 14.4378),
    "Denmark":            (55.6761, 12.5683),
    "Ecuador":            (-0.2299, -78.5249),
    "Egypt":              (30.0444, 31.2357),
    "England":            (51.5074, -0.1278),
    "France":             (48.8566, 2.3522),
    "Germany":            (52.5200, 13.4050),
    "Ghana":              (5.6037, -0.1870),
    "Greece":             (37.9838, 23.7275),
    "Iran":               (35.6892, 51.3890),
    "Italy":              (41.9028, 12.4964),
    "Ivory Coast":        (5.3600, -4.0083),
    "Japan":              (35.6762, 139.6503),
    "Mexico":             (19.4326, -99.1332),
    "Morocco":            (34.0209, -6.8416),
    "Netherlands":        (52.3676, 4.9041),
    "Nigeria":            (6.5244, 3.3792),
    "Norway":             (59.9139, 10.7522),
    "Poland":             (52.2297, 21.0122),
    "Portugal":           (38.7223, -9.1393),
    "Saudi Arabia":       (24.7136, 46.6753),
    "Senegal":            (14.7167, -17.4677),
    "Serbia":             (44.7866, 20.4489),
    "South Korea":        (37.5665, 126.9780),
    "Spain":              (40.4168, -3.7038),
    "Sweden":             (59.3293, 18.0686),
    "Switzerland":        (46.9480, 7.4474),
    "Tunisia":            (36.8065, 10.1815),
    "Turkey":             (39.9334, 32.8597),
    "USA":                (38.9072, -77.0369),
    "Ukraine":            (50.4501, 30.5234),
    "Uruguay":            (-34.9011, -56.1645),
    "Wales":              (51.4833, -3.1833),
    "Scotland":           (55.8642, -4.2518),
    "Ireland":            (53.3498, -6.2603),
    "Slovakia":           (48.1486, 17.1077),
    "Slovenia":           (46.0569, 14.5058),
    "Bosnia & Herzegovina": (43.8563, 18.4131),
    "DR Congo":           (-4.4419, 15.2663),
    "Honduras":           (14.0723, -87.1921),
    "Iceland":            (64.1466, -21.9426),
    "Paraguay":           (-25.2637, -57.5759),
    "Peru":               (-12.0464, -77.0428),
    "Russia":             (55.7558, 37.6173),
    "South Africa":       (-25.7461, 28.1881),
    "Venezuela":          (10.4806, -66.9036),
}


# ═══════════════════════════════════════════════════════════
#  CLI for testing
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    # Test: build placeholder data
    df = _build_placeholder_df(3)
    print(f"\n  Placeholder weather data ({len(df)} rows):\n")
    print(df.to_string(index=False))

    # Test: get coordinates for a team
    coords = get_team_coordinates(["Brazil", "England"])
    print(f"\n  Team coordinates: {coords}")
