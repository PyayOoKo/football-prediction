"""
Free weather data collector using Open-Meteo.

Open-Meteo (https://open-meteo.com) provides free weather data
without requiring an API key. The free tier allows 10,000 requests/day
with historical data from 1940 onward.

Compared to OpenWeatherMap (which requires a paid subscription for
historical data), Open-Meteo is fully free and more generous.

Source: https://archive-api.open-meteo.com/v1/archive
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import aiohttp

from football_data.config import config

logger = logging.getLogger(__name__)

# Latitude/longitude lookup for key cities in our target leagues
# (used as fallback when stadium coordinates are not available)
CITY_COORDS: dict[str, dict[str, float]] = {
    # Sweden Superettan
    "Stockholm": {"lat": 59.3293, "lon": 18.0686},
    "Gothenburg": {"lat": 57.7089, "lon": 11.9746},
    "Malmö": {"lat": 55.6050, "lon": 13.0038},
    # Norway OBOS
    "Oslo": {"lat": 59.9139, "lon": 10.7522},
    "Bergen": {"lat": 60.3913, "lon": 5.3221},
    "Trondheim": {"lat": 63.4305, "lon": 10.3951},
    # Finland
    "Helsinki": {"lat": 60.1699, "lon": 24.9384},
    "Tampere": {"lat": 61.4978, "lon": 23.7610},
    "Turku": {"lat": 60.4518, "lon": 22.2666},
    # Ireland
    "Dublin": {"lat": 53.3498, "lon": -6.2603},
    "Cork": {"lat": 51.8969, "lon": -8.4863},
    "Galway": {"lat": 53.2707, "lon": -9.0568},
    # Denmark
    "Copenhagen": {"lat": 55.6761, "lon": 12.5683},
    "Aarhus": {"lat": 56.1629, "lon": 10.2039},
    "Odense": {"lat": 55.4038, "lon": 10.4024},
    # Poland
    "Warsaw": {"lat": 52.2297, "lon": 21.0122},
    "Krakow": {"lat": 50.0647, "lon": 19.9450},
    "Gdansk": {"lat": 54.3520, "lon": 18.6466},
    "Wroclaw": {"lat": 51.1079, "lon": 17.0385},
}

# Default coordinates for leagues (center of country)
LEAGUE_DEFAULT_COORDS: dict[str, dict[str, float]] = {
    "SE1": {"lat": 62.0, "lon": 16.0},   # Sweden
    "NO2": {"lat": 62.0, "lon": 10.0},   # Norway
    "FI2": {"lat": 64.0, "lon": 26.0},   # Finland
    "IRL": {"lat": 53.5, "lon": -8.0},   # Ireland
    "D2": {"lat": 56.0, "lon": 10.0},    # Denmark
    "P1": {"lat": 52.0, "lon": 19.0},    # Poland
}


class WeatherCollector:
    """Async collector for historical weather data via Open-Meteo.

    Usage
    -----
    >>> collector = WeatherCollector()
    >>> weather = await collector.collect_for_match(
    ...     "2025-04-15", lat=59.33, lon=18.07
    ... )
    """

    def __init__(self) -> None:
        self.cfg = config.weather
        self._session: aiohttp.ClientSession | None = None
        self._semaphore = asyncio.Semaphore(
            max(1, self.cfg.rate_limit_per_minute // 6)  # ~6 req/10s
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.cfg.request_timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": "FootballDataCollector/1.0"},
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def collect_for_match(
        self,
        date_str: str,
        lat: float,
        lon: float,
    ) -> dict[str, Any] | None:
        """Fetch weather data for a specific date and location.

        Parameters
        ----------
        date_str : str
            Match date in 'YYYY-MM-DD' format.
        lat, lon : float
            Latitude and longitude of the match venue.

        Returns
        -------
        dict | None
            Weather record dict, or None if unavailable.
        """
        url = f"{self.cfg.base_url}"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "daily": ",".join([
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "rain_sum",
                "wind_speed_10m_max",
                "wind_gusts_10m_max",
                "relative_humidity_2m_max",
                "surface_pressure_max",
                "weather_code",
            ]),
            "timezone": "UTC",
        }

        session = await self._get_session()

        for attempt in range(self.cfg.max_retries):
            try:
                async with self._semaphore:
                    async with session.get(url, params=params) as resp:
                        if resp.status == 404:
                            return None
                        resp.raise_for_status()
                        data = await resp.json()

                daily = data.get("daily", {})
                if not daily:
                    return None

                return {
                    "temperature": (
                        daily.get("temperature_2m_max", [None])[0]
                    ),
                    "feels_like": (
                        daily.get("temperature_2m_min", [None])[0]
                    ),
                    "humidity": (
                        daily.get("relative_humidity_2m_max", [None])[0]
                    ),
                    "precipitation": (
                        daily.get("precipitation_sum", [None])[0]
                    ),
                    "rain": (
                        daily.get("rain_sum", [None])[0]
                    ),
                    "wind_speed": (
                        daily.get("wind_speed_10m_max", [None])[0]
                    ),
                    "wind_gust": (
                        daily.get("wind_gusts_10m_max", [None])[0]
                    ),
                    "pressure": (
                        daily.get("surface_pressure_max", [None])[0]
                    ),
                    "condition": str(
                        daily.get("weather_code", [None])[0]
                    ),
                }

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt < self.cfg.max_retries - 1:
                    wait = self.cfg.retry_backoff * (2 ** attempt)
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        "Weather fetch failed for %s at (%.2f, %.2f): %s",
                        date_str, lat, lon, exc,
                    )
                    return None

        return None

    async def collect_for_matches(
        self,
        matches: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Fetch weather for multiple matches in parallel.

        Parameters
        ----------
        matches : list[dict]
            List of match dicts. Each should have 'date', and optionally
            'lat', 'lon' fields.

        Returns
        -------
        list[dict]
            Weather records with match_id references.
        """
        tasks = []
        for match in matches:
            lat = match.get("lat") or LEAGUE_DEFAULT_COORDS.get(
                match.get("league", ""), {}
            ).get("lat", 55.0)
            lon = match.get("lon") or LEAGUE_DEFAULT_COORDS.get(
                match.get("league", ""), {}
            ).get("lon", 10.0)

            tasks.append(self._collect_with_match_id(
                match.get("match_id"),
                match.get("date", ""),
                lat,
                lon,
            ))

        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                results.append(result)

        return results

    async def _collect_with_match_id(
        self,
        match_id: int | None,
        date_str: str,
        lat: float,
        lon: float,
    ) -> dict[str, Any] | None:
        """Collect weather and tag it with a match ID."""
        weather = await self.collect_for_match(date_str, lat, lon)
        if weather:
            weather["match_id"] = match_id
            return weather
        return None
