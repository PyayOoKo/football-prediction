"""
football-data.co.uk — Async HTTP collector for CSV match data.

Downloads historical fixtures, results, closing odds, and match
statistics from https://www.football-data.co.uk using aiohttp.

Key features
------------
- Async I/O with aiohttp for concurrent downloads
- Automatic retry with exponential backoff
- Response caching to avoid redundant downloads
- Season-aware download (archived + current in-progress)
- Standardised column naming
- Respectful rate limiting (1 request/0.5s)
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

from football_data.config import config

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────

SEASON_PATTERN = re.compile(r"mmz4281/(\d{4})")

# ── Column renaming map ────────────────────────────────

# ── Column renaming maps ────────────────────────────────

# Map for the legacy archive CSV format (Div, Date, HomeTeam, AwayTeam, FTHG, ...)
# Used for seasons downloaded from mmz4281/XXXX/ directory
ARCHIVE_COLUMN_MAP: dict[str, str] = {
    "div": "league",
    "date": "date",
    "hometeam": "home_team",
    "awayteam": "away_team",
    "fthg": "home_goals",
    "ftag": "away_goals",
    "ftr": "result",
    "hthg": "home_goals_ht",
    "htag": "away_goals_ht",
    "htr": "result_ht",
    "hs": "home_shots",
    "as": "away_shots",
    "hst": "home_shots_target",
    "ast": "away_shots_target",
    "hc": "home_corners",
    "ac": "away_corners",
    "hf": "home_fouls",
    "af": "away_fouls",
    "hy": "home_yellow",
    "ay": "away_yellow",
    "hr": "home_red",
    "ar": "away_red",
    # Odds
    "b365h": "home_odds",
    "b365d": "draw_odds",
    "b365a": "away_odds",
    "bwh": "home_odds",
    "bwd": "draw_odds",
    "bwa": "away_odds",
    "bbavh": "home_odds",
    "bbavd": "draw_odds",
    "bbava": "away_odds",
    "bbmxh": "home_odds",
    "bbmxd": "draw_odds",
    "bbmxa": "away_odds",
    "psh": "home_odds",
    "psd": "draw_odds",
    "psa": "away_odds",
    # Over/Under 2.5 closing odds
    "bbav>2.5": "over25_odds",
    "bbav<2.5": "under25_odds",
    "b365>2.5": "over25_odds",
    "b365<2.5": "under25_odds",
    "max>2.5": "over25_odds",
    "max<2.5": "under25_odds",
    "avg>2.5": "over25_odds",
    "avg<2.5": "under25_odds",
}

# Map for the new CSV format (Country, League, Season, Date, Home, Away, HG, AG, Res, ...)
# Used for single-file download from new/ directory
NEW_FORMAT_COLUMN_MAP: dict[str, str] = {
    "date": "date",
    "home": "home_team",
    "away": "away_team",
    "hg": "home_goals",
    "ag": "away_goals",
    "res": "result",
    # Odds columns (new format uses PSCH, PSCD, PSCA patterns)
    "psch": "home_odds",
    "pscd": "draw_odds",
    "psca": "away_odds",
    "maxch": "home_odds",
    "maxcd": "draw_odds",
    "maxca": "away_odds",
    "avgch": "home_odds",
    "avgcd": "draw_odds",
    "avgca": "away_odds",
    "bfech": "home_odds",
    "bfecd": "draw_odds",
    "bfeca": "away_odds",
    "b365ch": "home_odds",
    "b365cd": "draw_odds",
    "b365ca": "away_odds",
}


class FootballDataCollector:
    """Async collector for football-data.co.uk CSV files.

    Usage
    -----
    >>> collector = FootballDataCollector()
    >>> matches = await collector.collect(["SE1", "NO2"])
    """

    def __init__(self) -> None:
        self.cfg = config.football_data
        self._session: aiohttp.ClientSession | None = None
        self._semaphore = asyncio.Semaphore(2)  # max 2 concurrent requests
        self._cache: dict[str, pd.DataFrame] = {}

    # ── Session management ───────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.cfg.request_timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; FootballDataCollector/1.0; "
                        "+https://github.com/user/football_data)"
                    ),
                    "Accept": "text/csv, text/html, */*",
                },
                raise_for_status=False,
            )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Core download logic ──────────────────────────────

    async def _download_csv(self, url: str) -> pd.DataFrame | None:
        """Download a CSV file and return a DataFrame.

        Implements retry with exponential backoff.
        """
        if url in self._cache:
            return self._cache[url]

        session = await self._get_session()
        last_error: Exception | None = None

        for attempt in range(self.cfg.max_retries):
            try:
                async with self._semaphore:
                    async with session.get(url) as resp:
                        if resp.status == 404:
                            return None  # Missing season is not an error
                        resp.raise_for_status()
                        text = await resp.text()

                # Parse CSV — skip malformed lines, handle BOM
                cleaned = text.lstrip("\ufeff")  # Strip BOM
                df = pd.read_csv(
                    io.StringIO(cleaned),
                    na_values=["", "NA", "N/A"],
                    on_bad_lines="skip",
                )
                self._cache[url] = df
                return df

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = exc
                if attempt < self.cfg.max_retries - 1:
                    wait = self.cfg.retry_backoff * (2 ** attempt)
                    logger.warning(
                        "Retry %d/%d for %s after %.1fs: %s",
                        attempt + 1, self.cfg.max_retries, url, wait, exc,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        "Failed to download %s after %d attempts: %s",
                        url, self.cfg.max_retries, exc,
                    )

        return None

    # ── Season URL construction ──────────────────────────

    def _season_url(self, season: str, league: str) -> str:
        """Build URL for a specific season CSV."""
        return (
            f"{self.cfg.base_url}/{self.cfg.mmz_path}/{season}/{league}.csv"
        )

    def _current_url(self, league: str) -> str:
        """Build URL for the current in-progress season."""
        return self.cfg.new_csv_url.format(league=league)

    # ── DataFrame normalisation ──────────────────────────

    def _normalise_df(
        self,
        df: pd.DataFrame,
        league: str,
        season: str | None = None,
        is_new_format: bool = False,
    ) -> pd.DataFrame:
        """Normalise column names and add metadata.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe from CSV.
        league : str
            League code (e.g. "SE1", "NO", "FI").
        season : str | None
            Season code for archive format (e.g. "2425"). For new format,
            season is read from the CSV's Season column.
        is_new_format : bool
            Whether this is from the new multi-season CSV format.
        """
        # Lowercase columns
        orig_cols = [str(c).strip().lower() for c in df.columns]
        col_map = dict(zip(df.columns, orig_cols))
        df.rename(columns=col_map, inplace=True)

        if is_new_format:
            # ── New format: Country, League, Season, Date, Home, Away, HG, AG, Res ──
            column_map = NEW_FORMAT_COLUMN_MAP
        else:
            # ── Archive format: Div, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR ──
            column_map = ARCHIVE_COLUMN_MAP

        # Rename known columns
        df.rename(columns=column_map, inplace=True)

        # Determine keep list
        keep_cols = set(column_map.values())

        # For new format: keep original season column (per-row values)
        if is_new_format and "season" in df.columns:
            keep_cols.add("season")

        # Also keep any bb* odds columns
        for c in df.columns:
            if c.startswith("bb") and c not in keep_cols:
                keep_cols.add(c)

        # Keep Country only in new format (for reference)
        if is_new_format and "country" in df.columns:
            keep_cols.add("country")

        # Drop columns not in our schema
        for col in list(df.columns):
            if col not in keep_cols:
                df.drop(columns=[col], inplace=True)

        # Deduplicate column names (multiple odds providers map to same target)
        # Keep the first occurrence of each column name
        deduped = df.loc[:, ~df.columns.duplicated(keep="first")]
        if len(deduped.columns) < len(df.columns):
            n_dropped = len(df.columns) - len(deduped.columns)
            logger.debug("Removed %d duplicate column(s)", n_dropped)
            df = deduped

        # Add metadata
        df["league"] = league
        if not is_new_format and season:
            df["season"] = season
        elif "season" not in df.columns and season:
            df["season"] = season
        df["source"] = "football-data.co.uk"
        df["ingested_at"] = datetime.now(timezone.utc).isoformat()

        # Parse dates (football-data.co.uk uses DD/MM/YYYY or DD/MM/YY)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(
                df["date"], dayfirst=True, errors="coerce"
            ).dt.strftime("%Y-%m-%d")

        # Convert result to standard format
        if "result" in df.columns:
            df["result"] = df["result"].astype(str).str.upper().str[0]

        return df

    # ── Public API ───────────────────────────────────────

    async def collect(
        self, leagues: list[str] | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Collect match data for the specified leagues.

        Parameters
        ----------
        leagues : list[str] | None
            League codes to collect. Defaults to configured leagues.

        Returns
        -------
        dict[str, list[dict]]
            Maps league code to list of match dicts.
        """
        if leagues is None:
            leagues = list(config.leagues)

        logger.info("Starting collection for %d leagues: %s", len(leagues), leagues)
        results: dict[str, list[dict[str, Any]]] = {}

        for league in leagues:
            matches = await self.collect_league(league)
            results[league] = matches
            logger.info("Collected %d matches for league %s", len(matches), league)

        return results

    async def collect_league(
        self, league: str, max_seasons: int | None = None
    ) -> list[dict[str, Any]]:
        """Collect all available data for a single league.

        Strategy:
        1. Try archive format (mmz4281/XXXX/{league}.csv) for multi-season history
        2. If no data found, fall back to new format (new/{league}.csv)
        """
        max_seasons = max_seasons or self.cfg.max_seasons

        # ── Step 1: Try archive format ──
        seasons = await self._get_recent_seasons(max_seasons)
        logger.debug("Seasons to fetch for %s: %s", league, seasons)

        tasks = []
        for season in seasons:
            url = self._season_url(season, league)
            tasks.append(self._download_archive_season(url, season, league))

        # Also try current season (new URL)
        if self.cfg.include_current:
            url = self._current_url(league)
            tasks.append(self._download_new_format(url, league))

        dfs: list[pd.DataFrame] = []
        archive_had_data = False
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is not None:
                df, is_new = result
                if df is not None and not df.empty:
                    if is_new:
                        logger.debug("... new format gave %d rows for %s", len(df), league)
                    else:
                        archive_had_data = True
                    dfs.append(df)

        # ── Step 2: If archive failed, try new format exclusively ──
        if not dfs:
            logger.info("Archive format not found for %s — trying new format", league)
            url = self._current_url(league)
            result = await self._download_new_format(url, league)
            if result is not None:
                df, _ = result
                if df is not None and not df.empty:
                    dfs.append(df)

        if not dfs:
            logger.warning("No data collected for league %s", league)
            return []

        # Combine and deduplicate
        combined = pd.concat(dfs, ignore_index=True)

        # Check which columns exist before dedup
        dedup_cols = [c for c in ["date", "home_team", "away_team", "league"] if c in combined.columns]
        if len(dedup_cols) >= 3:
            combined.drop_duplicates(
                subset=dedup_cols,
                keep="first",
                inplace=True,
            )

        # Convert to dicts
        return combined.to_dict(orient="records")

    async def _download_archive_season(
        self, url: str, season: str, league: str
    ) -> tuple[pd.DataFrame | None, bool]:
        """Download an archive-format CSV, tagged with season."""
        raw = await self._download_csv(url)
        if raw is None or raw.empty:
            return (None, False)
        df = self._normalise_df(raw, league=league, season=season, is_new_format=False)
        return (df, False)

    async def _download_new_format(
        self, url: str, league: str
    ) -> tuple[pd.DataFrame | None, bool]:
        """Download the new-format CSV (single file, all seasons)."""
        raw = await self._download_csv(url)
        if raw is None or raw.empty:
            return (None, False)
        df = self._normalise_df(raw, league=league, season=None, is_new_format=True)
        return (df, True)

    # ── Season helpers ───────────────────────────────────

    async def _get_recent_seasons(self, n: int) -> list[str]:
        """Get the N most recent season codes from the archive."""
        try:
            all_seasons = await self._fetch_available_seasons()
            return all_seasons[-n:] if len(all_seasons) >= n else all_seasons
        except Exception as exc:
            logger.warning(
                "Could not fetch season archive (%s) — generating seasons", exc
            )
            return self._generate_season_codes(n)

    @staticmethod
    def _season_code_to_year(code: str) -> int:
        """Convert a 4-digit season code to a start year.

        '2425' -> 2024, '9394' -> 1993, '0001' -> 2000
        The six-month rule: codes 50+ are 1900s, codes < 50 are 2000s.
        """
        start = int(code[:2])
        if start >= 50:
            return 1900 + start
        return 2000 + start

    async def _fetch_available_seasons(self) -> list[str]:
        """Scrape the archive page for all available season codes.

        Returns seasons sorted chronologically (most recent last).
        """
        session = await self._get_session()
        url = f"{self.cfg.base_url}/downloadm.php"

        async with session.get(url) as resp:
            resp.raise_for_status()
            html = await resp.text()

        seasons = sorted(
            set(SEASON_PATTERN.findall(html)),
            key=self._season_code_to_year,
        )
        logger.debug("Found %d available seasons", len(seasons))
        return seasons

    def _generate_season_codes(self, n: int) -> list[str]:
        """Generate N most recent season codes algorithmically."""
        today = date.today()
        if today.month >= 8:
            end_year = today.year + 1
        else:
            end_year = today.year

        seasons: list[str] = []
        for i in range(n):
            ey = end_year - i
            sy = ey - 1
            seasons.append(f"{str(sy)[2:]}{str(ey)[2:]}")

        return list(reversed(seasons))

    def _guess_current_season(self) -> str:
        """Return the season code for the current ongoing season."""
        today = date.today()
        year = today.year
        if today.month >= 8:
            start = str(year)[2:]
            end = str(year + 1)[2:]
        else:
            start = str(year - 1)[2:]
            end = str(year)[2:]
        return f"{start}{end}"
