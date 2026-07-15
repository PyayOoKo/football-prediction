"""
ETL extractors for external data sources.

Each extractor wraps a ``src.data_collection.sources.*`` module's
scraping function in a ``BaseExtractor`` for use in the ETL pipeline.

Usage
-----
    from src.etl.extractors import TransferExtractor, WeatherExtractor

    extractor = TransferExtractor()
    result = extractor.run(team_names=["Brazil", "England"])
    # result.data: list[dict] of transfer records
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.etl.extract import BaseExtractor, RetryWithBackoff

logger = logging.getLogger(__name__)


# ── Transfer Extractor ──────────────────────────────────


class TransferExtractor(BaseExtractor):
    """Extract transfer history data from Transfermarkt.

    Parameters
    ----------
    max_windows : int
        Max transfer windows per team (default 5).
    delay : float
        Seconds between requests (default 1.5).
    save_path : str, optional
        If provided, saves raw data to this CSV path.
    """

    def __init__(
        self,
        max_windows: int = 5,
        delay: float = 1.5,
        save_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="TransferExtractor", **kwargs)
        self.max_windows = max_windows
        self.delay = delay
        self.save_path = save_path

    def _extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Extract transfer data for the given teams."""
        from src.data_collection.sources.transfermarkt import TEAM_TO_TM_ID
        from src.data_collection.sources.transfers import scrape_transfers

        team_names = kwargs.get("team_names")
        if not team_names:
            # Default to all teams with TM IDs
            seen: set[str] = set()
            team_names = []
            for name in sorted(TEAM_TO_TM_ID):
                if name not in seen:
                    seen.add(name)
                    team_names.append(name)

        df = scrape_transfers(
            team_names=team_names,
            max_windows=kwargs.get("max_windows", self.max_windows),
            delay=kwargs.get("delay", self.delay),
            save_path=kwargs.get("save_path", self.save_path),
        )
        return df.to_dict(orient="records") if not df.empty else []


# ── Weather Extractor ──────────────────────────────────


class WeatherExtractor(BaseExtractor):
    """Extract historical weather data from OpenWeatherMap.

    Parameters
    ----------
    use_cache : bool
        Whether to cache API responses (default True).
    save_path : str, optional
        If provided, saves raw data to this CSV path.
    """

    def __init__(
        self,
        use_cache: bool = True,
        save_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="WeatherExtractor", **kwargs)
        self.use_cache = use_cache
        self.save_path = save_path

    def _extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Extract weather data for the given matches."""
        from src.data_collection.sources.weather_api import collect_weather

        matches_df = kwargs.get("matches_df")
        if matches_df is None:
            raise ValueError("WeatherExtractor requires 'matches_df' (pd.DataFrame)")

        df = collect_weather(
            matches_df=matches_df,
            lat_lon_map=kwargs.get("lat_lon_map"),
            api_key=kwargs.get("api_key"),
            use_cache=kwargs.get("use_cache", self.use_cache),
            output_path=kwargs.get("save_path", self.save_path),
        )
        return df.to_dict(orient="records") if not df.empty else []


# ── Referee Extractor ──────────────────────────────────


class RefereeExtractor(BaseExtractor):
    """Extract referee statistics from FBref.

    Parameters
    ----------
    delay : float
        Seconds between requests (default 2.0).
    save_path : str, optional
        If provided, saves raw data to this CSV path.
    """

    def __init__(
        self,
        delay: float = 2.0,
        save_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="RefereeExtractor", **kwargs)
        self.delay = delay
        self.save_path = save_path

    def _extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Extract referee stats for the given competition/season."""
        from src.data_collection.sources.referee_stats import scrape_referees

        competition_id = kwargs.get("competition_id", "9")
        season = kwargs.get("season", "2024-2025")

        df = scrape_referees(
            competition_id=competition_id,
            season=season,
            delay=kwargs.get("delay", self.delay),
            save_path=kwargs.get("save_path", self.save_path),
        )
        return df.to_dict(orient="records") if not df.empty else []


# ── StatsBomb Extractor ────────────────────────────────


class StatsBombExtractor(BaseExtractor):
    """Extract match/event data from StatsBomb open data.

    Parameters
    ----------
    use_cache : bool
        Whether to cache API responses (default True).
    data_type : str
        Type of data to extract: ``matches``, ``events``, ``lineups``, or ``shots``.
    """

    def __init__(
        self,
        use_cache: bool = True,
        data_type: str = "matches",
        **kwargs: Any,
    ) -> None:
        super().__init__(name="StatsBombExtractor", **kwargs)
        self.use_cache = use_cache
        self.data_type = data_type

    def _extract(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Extract StatsBomb data for the given competition/match."""
        from src.data_collection.sources.statsbomb_open import (
            list_matches,
            get_match_events,
            get_match_lineups,
            shots_to_dataframe,
            matches_to_dataframe,
        )

        competition_name = kwargs.get("competition_name", "")
        match_id = kwargs.get("match_id")

        data_type = kwargs.get("data_type", self.data_type)

        if data_type == "matches":
            if not competition_name:
                raise ValueError("StatsBombExtractor(data_type='matches') requires 'competition_name'")
            matches = list_matches(
                competition_name=competition_name,
                use_cache=kwargs.get("use_cache", self.use_cache),
            )
            df = matches_to_dataframe(matches)

        elif data_type == "events":
            if not match_id:
                raise ValueError("StatsBombExtractor(data_type='events') requires 'match_id'")
            events = get_match_events(match_id, use_cache=kwargs.get("use_cache", self.use_cache))
            df = pd.DataFrame(events)

        elif data_type == "lineups":
            if not match_id:
                raise ValueError("StatsBombExtractor(data_type='lineups') requires 'match_id'")
            from src.data_collection.sources.statsbomb_open import lineups_to_dataframe
            df = lineups_to_dataframe(match_id, use_cache=kwargs.get("use_cache", self.use_cache))

        elif data_type == "shots":
            match_ids = kwargs.get("match_ids", [match_id] if match_id else [])
            if not match_ids:
                raise ValueError("StatsBombExtractor(data_type='shots') requires 'match_ids' or 'match_id'")
            df = shots_to_dataframe(
                match_ids=match_ids,
                use_cache=kwargs.get("use_cache", self.use_cache),
            )

        else:
            raise ValueError(f"Unknown data_type: {data_type}. Choose from: matches, events, lineups, shots")

        return df.to_dict(orient="records") if not df.empty else []


# ── Registry map ────────────────────────────────────────

EXTRACTOR_REGISTRY: dict[str, type[BaseExtractor]] = {
    "transfers": TransferExtractor,
    "weather": WeatherExtractor,
    "referees": RefereeExtractor,
    "statsbomb": StatsBombExtractor,
}


def get_extractor(name: str, **kwargs: Any) -> BaseExtractor:
    """Factory: get an extractor instance by name.

    Parameters
    ----------
    name : str
        Extractor name (``transfers``, ``weather``, ``referees``, ``statsbomb``).
    **kwargs
        Passed to the extractor constructor.

    Returns
    -------
    BaseExtractor
        Configured extractor instance.

    Raises
    ------
    KeyError
        If the name is not in the registry.
    """
    if name not in EXTRACTOR_REGISTRY:
        raise KeyError(
            f"Unknown extractor '{name}'. "
            f"Available: {list(EXTRACTOR_REGISTRY.keys())}"
        )
    cls = EXTRACTOR_REGISTRY[name]
    return cls(**kwargs)


# ═══════════════════════════════════════════════════════════
#  CLI for testing
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Test: list available extractors
    print("\n  Available extractors:")
    for name, cls in EXTRACTOR_REGISTRY.items():
        print(f"    - {name}: {cls.__name__}")
    print()
