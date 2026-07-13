"""
FBrefScraper — collects football statistics from FBref.

Provides methods for scraping all 7 stat categories plus match-level
and team-level data, with caching, rate limiting, and incremental
update support.

Stat Categories (team/squad level)
----------------------------------
1. Standard stats — goals, assists, shots, cards, etc.
2. Shooting — shots, shots on target, xG, shot distance, etc.
3. Passing — passes completed, pass distance, assists, xAG, etc.
4. Possession — touches, dribbles, carries, dispossessed, etc.
5. Defense — tackles, interceptions, blocks, clearances, etc.
6. Goalkeeping — saves, GA, clean sheets, PSxG, etc.
7. Match stats — per-match summary for a given match URL

Usage
-----
::

    from src.data_collection.sources.fbref import FBrefScraper

    async with FBrefScraper() as scraper:
        # Team-level stats for PL 24-25
        df = await scraper.get_team_stats("9", "2024-2025")

        # Shooting stats for Manchester City
        shooting = await scraper.get_squad_stats(
            "9", "2024-2025", "shooting",
            team_id="b8fd03ef"
        )

        # All stats for a team
        all_stats = await scraper.get_all_squad_stats(
            "b8fd03ef", "2024-2025", "Manchester City"
        )

        # Match-level data
        match = await scraper.get_match_stats(
            "https://fbref.com/en/matches/abc123"
        )

    # Sync usage
    scraper = FBrefScraper()
    df = scraper.get_team_stats_sync("9", "2024-2025")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.data_collection.sources.fbref.client import FBrefClient
from src.data_collection.sources.fbref.models import (
    CATEGORY_URL_MAP,
    COMPETITION_IDS,
    FBrefTable,
    MatchStats,
    PlayerStats,
    SquadStats,
    StatCategory,
)
from src.data_collection.sources.fbref.parser import FBrefTableParser

logger = logging.getLogger(__name__)


@dataclass
class ScrapeJob:
    """Tracking info for a scrape job (for checkpoint/resume).

    Attributes
    ----------
    job_id : str
        Unique job identifier.
    competition_id : str
        FBref competition ID.
    season : str
        Season identifier.
    teams : list[str]
        Team IDs or names to scrape.
    categories : list[str]
        Stat categories to scrape.
    completed_teams : list[str]
        Teams that have been fully scraped.
    started_at : float
        Timestamp when the job started.
    last_updated : float
        Timestamp of last checkpoint save.
    """

    job_id: str = ""
    competition_id: str = ""
    season: str = ""
    teams: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    completed_teams: list[str] = field(default_factory=list)
    started_at: float = 0.0
    last_updated: float = 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> ScrapeJob | None:
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None


class FBrefScraper:
    """Main scraper for FBref statistics.

    Parameters
    ----------
    client : FBrefClient, optional
        Custom FBref client. Defaults to a new instance.
    parser : FBrefTableParser, optional
        Custom table parser. Defaults to a new instance.
    checkpoint_dir : str | Path
        Directory for checkpoint files (default ``data/scrapers/fbref``).
    """

    def __init__(
        self,
        client: FBrefClient | None = None,
        parser: FBrefTableParser | None = None,
        checkpoint_dir: str | Path = "data/scrapers/fbref",
    ) -> None:
        self.client = client or FBrefClient()
        self.parser = parser or FBrefTableParser()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════
    #  Team-level stats (competition page)
    # ══════════════════════════════════════════════════════

    async def get_team_stats(
        self,
        competition_id: str = "9",
        season: str = "2024-2025",
        category: str = "standard",
        force_refresh: bool = False,
    ) -> list[FBrefTable]:
        """Scrape team-level stats for a competition.

        Parameters
        ----------
        competition_id : str
            FBref competition ID (default ``9`` = Premier League).
        season : str
            Season slug (default ``2024-2025``).
        category : str
            Stat category name (default ``standard``).
            See ``StatCategory`` for valid values.
        force_refresh : bool
            Bypass cache.

        Returns
        -------
        list[FBrefTable]
            Parsed tables from the competition stats page.
        """
        category_key = self._resolve_category(category)
        url_path = CATEGORY_URL_MAP.get(category_key, "")
        comp_name = self._competition_name(competition_id)

        # Build URL: /en/comps/{id}/{category}/{name}-Stats
        url = f"/en/comps/{competition_id}/{url_path}/{comp_name}-Stats" if url_path \
            else f"/en/comps/{competition_id}/{comp_name}-Stats"

        logger.info("Fetching %s stats for %s (%s)", category, comp_name, season)
        html = await self.client.get(url, force_refresh=force_refresh)

        tables = self.parser.parse_page(html, url=url)
        for tbl in tables:
            tbl.season = season
            tbl.competition = comp_name

        return tables

    async def get_squad_stats(
        self,
        competition_id: str = "9",
        season: str = "2024-2025",
        category: str = "standard",
        team_id: str | None = None,
        force_refresh: bool = False,
    ) -> list[FBrefTable]:
        """Scrape squad-level stats for a specific category.

        Parameters
        ----------
        competition_id : str
            FBref competition ID.
        season : str
            Season slug.
        category : str
            Stat category name.
        team_id : str, optional
            FBref squad ID (hex). If not provided, fetches the
            competition-level table which includes all teams.
        force_refresh : bool
            Bypass cache.

        Returns
        -------
        list[FBrefTable]
            Parsed tables from the squad stats page.
        """
        if team_id:
            # Squad-specific page: /en/squads/{id}/{season}/{category}/{name}-Stats
            comp_name = self._competition_name(competition_id)
            category_key = self._resolve_category(category)
            url_path = CATEGORY_URL_MAP.get(category_key, "")

            if url_path:
                url = f"/en/squads/{team_id}/{season}/{url_path}/{comp_name}-Stats"
            else:
                url = f"/en/squads/{team_id}/{season}/{comp_name}-Stats"

            logger.info(
                "Fetching %s stats for squad %s", category, team_id,
            )
            html = await self.client.get(url, force_refresh=force_refresh)

            tables = self.parser.parse_page(html, url=url)
            for tbl in tables:
                tbl.season = season
                tbl.team_name = comp_name
            return tables

        # Fall back to competition-level stats
        return await self.get_team_stats(
            competition_id, season, category, force_refresh,
        )

    async def get_all_squad_stats(
        self,
        team_id: str,
        season: str,
        team_name: str = "",
        competition_id: str = "9",
        force_refresh: bool = False,
    ) -> SquadStats:
        """Scrape all stat categories for a single squad.

        Parameters
        ----------
        team_id : str
            FBref squad ID.
        season : str
            Season slug.
        team_name : str
            Team name for metadata.
        competition_id : str
            FBref competition ID (default ``9`` = Premier League).
        force_refresh : bool
            Bypass cache.

        Returns
        -------
        SquadStats
            All scraped stats organised by category.
        """
        squad = SquadStats(
            team_id=team_id,
            team_name=team_name,
            season=season,
        )

        # Fetch all categories concurrently
        categories = list(CATEGORY_URL_MAP.keys())
        tasks = []
        for cat in categories:
            if cat == StatCategory.MATCH_STATS:
                continue
            tasks.append(self.get_squad_stats(
                competition_id=competition_id,
                season=season,
                category=cat.value,
                team_id=team_id,
                force_refresh=force_refresh,
            ))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for cat, result in zip(categories, results):
            if isinstance(result, Exception):
                logger.warning(
                    "Failed to fetch %s for %s: %s",
                    cat.value, team_id, result,
                )
                continue
            if result:
                squad.stat_tables[cat] = result[0]

        return squad

    # ══════════════════════════════════════════════════════
    #  Match-level stats
    # ══════════════════════════════════════════════════════

    async def get_match_stats(
        self,
        match_url: str,
        force_refresh: bool = False,
    ) -> MatchStats:
        """Scrape statistics for a single match.

        Parameters
        ----------
        match_url : str
            Full FBref match URL.
        force_refresh : bool
            Bypass cache.

        Returns
        -------
        MatchStats
            Parsed match statistics.
        """
        html = await self.client.get(match_url, force_refresh=force_refresh)
        return self._parse_match_page(html, match_url)

    async def get_match_stats_batch(
        self,
        match_urls: list[str],
        max_concurrent: int = 3,
    ) -> list[MatchStats]:
        """Scrape statistics for multiple matches concurrently.

        Parameters
        ----------
        match_urls : list[str]
            FBref match URLs.
        max_concurrent : int
            Max concurrent requests (default 3).

        Returns
        -------
        list[MatchStats]
            Parsed match stats in the same order as input URLs.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_one(url: str) -> MatchStats:
            async with semaphore:
                return await self.get_match_stats(url)

        tasks = [fetch_one(url) for url in match_urls]
        return await asyncio.gather(*tasks, return_exceptions=False)

    # ══════════════════════════════════════════════════════
    #  Sync convenience methods
    # ══════════════════════════════════════════════════════

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine synchronously in a fresh event loop.

        Uses a daemon thread to avoid issues when an event loop is
        already running in the current thread.
        """
        import threading
        result: list[Any] = []
        exception: list[Exception] = []

        def run_in_thread() -> None:
            try:
                result.append(asyncio.run(coro))
            except Exception as e:
                exception.append(e)

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        thread.join()

        if exception:
            raise exception[0]
        return result[0]

    def get_team_stats_sync(
        self,
        competition_id: str = "9",
        season: str = "2024-2025",
        category: str = "standard",
    ) -> list[FBrefTable]:
        """Synchronous version of ``get_team_stats()``.

        Uses a fresh event loop in a daemon thread so it works whether
        or not an async context is already running.

        Note: call ``await scraper.close()`` after use to clean up connections.
        """
        return self._run_async(
            self.get_team_stats(competition_id, season, category),
        )

    def get_squad_stats_sync(
        self,
        competition_id: str = "9",
        season: str = "2024-2025",
        category: str = "standard",
        team_id: str | None = None,
    ) -> list[FBrefTable]:
        """Synchronous version of ``get_squad_stats()``.

        Uses a fresh event loop in a daemon thread so it works whether
        or not an async context is already running.

        Note: call ``await scraper.close()`` after use to clean up connections.
        """
        return self._run_async(
            self.get_squad_stats(competition_id, season, category, team_id),
        )

    # ══════════════════════════════════════════════════════
    #  Incremental updates & checkpoint/resume
    # ══════════════════════════════════════════════════════

    async def scrape_competition_incremental(
        self,
        competition_id: str = "9",
        season: str = "2024-2025",
        categories: list[str] | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Scrape a competition incrementally with checkpoint/resume.

        Saves progress after each category so if the job is interrupted,
        it can resume from where it left off.

        Parameters
        ----------
        competition_id : str
            FBref competition ID.
        season : str
            Season slug.
        categories : list[str], optional
            Categories to scrape. Defaults to all except match_stats.
        job_id : str, optional
            Job identifier for resume. Auto-generated if not provided.

        Returns
        -------
        dict[str, Any]
            Results summary with counts per category.
        """
        import hashlib

        if categories is None:
            categories = [
                c.value for c in CATEGORY_URL_MAP.keys()
                if c != StatCategory.MATCH_STATS
            ]

        job_id = job_id or hashlib.md5(
            f"{competition_id}_{season}_{time.time()}".encode()
        ).hexdigest()[:12]

        checkpoint_path = self.checkpoint_dir / f"{job_id}.checkpoint"
        job = ScrapeJob.load(checkpoint_path)

        if job is None:
            job = ScrapeJob(
                job_id=job_id,
                competition_id=competition_id,
                season=season,
                categories=categories,
                started_at=time.time(),
            )

        results: dict[str, Any] = {
            "job_id": job_id,
            "competition_id": competition_id,
            "season": season,
            "categories": {},
            "total_tables": 0,
            "total_rows": 0,
        }

        for category in categories:
            if category in job.completed_teams:
                logger.info("Skipping already-completed category: %s", category)
                continue

            logger.info("Scraping category: %s", category)

            try:
                tables = await self.get_team_stats(
                    competition_id, season, category,
                )

                total_rows = sum(len(t.rows) for t in tables)
                results["categories"][category] = {
                    "tables": len(tables),
                    "rows": total_rows,
                }
                results["total_tables"] += len(tables)
                results["total_rows"] += total_rows

                job.completed_teams.append(category)
                job.last_updated = time.time()
                job.save(checkpoint_path)

                logger.info(
                    "Category %s: %d tables, %d rows",
                    category, len(tables), total_rows,
                )

            except Exception as exc:
                logger.error(
                    "Failed to scrape category %s: %s", category, exc,
                )
                results["categories"][category] = {
                    "error": str(exc),
                }

        # Clean up checkpoint on success
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Checkpoint %s removed (job complete)", job_id)

        return results

    # ══════════════════════════════════════════════════════
    #  DataFrame helpers
    # ══════════════════════════════════════════════════════

    def tables_to_dataframe(self, tables: list[FBrefTable]) -> pd.DataFrame:
        """Convert a list of FBrefTable objects to a single DataFrame.

        Parameters
        ----------
        tables : list[FBrefTable]
            Parsed tables from any scraper method.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with all rows.
        """
        rows: list[dict[str, Any]] = []
        for table in tables:
            for row in table.rows:
                row["_category"] = table.category.value
                row["_competition"] = table.competition
                row["_season"] = table.season
                row["_team"] = table.team_name
                rows.append(row)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    # ══════════════════════════════════════════════════════
    #  Internal helpers
    # ══════════════════════════════════════════════════════

    @staticmethod
    def _resolve_category(category: str) -> StatCategory:
        """Resolve a string to a StatCategory enum."""
        for cat in StatCategory:
            if cat.value == f"stats_{category}" or cat.value == category:
                return cat
            if cat.name.lower() == category.lower():
                return cat
        return StatCategory.STANDARD

    @staticmethod
    def _competition_name(competition_id: str) -> str:
        """Convert competition ID to URL-friendly name."""
        from src.data_collection.sources.fbref.models import COMPETITION_NAMES

        name = COMPETITION_NAMES.get(competition_id, "")
        if name:
            return name.replace(" ", "-")
        return competition_id

    @staticmethod
    def _competition_id(name: str) -> str:
        """Convert competition name to FBref ID."""
        from src.data_collection.sources.fbref.models import COMPETITION_IDS

        return COMPETITION_IDS.get(name, name)

    def _parse_match_page(
        self,
        html: str,
        match_url: str,
    ) -> MatchStats:
        """Parse a match report page into MatchStats.

        Extracts score, teams, date, and match-level stats from the
        scorebox and stats tables on the page.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        match = MatchStats(match_url=match_url)

        # Extract score and teams from scorebox
        scorebox = soup.find("div", class_="scorebox")
        if scorebox:
            teams = scorebox.find_all("strong", itemprop="name")
            if len(teams) >= 2:
                match.home_team = teams[0].get_text(strip=True)
                match.away_team = teams[1].get_text(strip=True)

            # Score
            score_div = scorebox.find("div", class_="score")
            if score_div:
                scores = score_div.get_text(strip=True).split("–")
                if len(scores) >= 2:
                    try:
                        match.home_goals = int(scores[0].strip())
                        match.away_goals = int(scores[1].strip())
                    except (ValueError, IndexError):
                        pass

        # Extract date from page
        date_tag = soup.find("span", attrs={"data-date": True})
        if date_tag:
            match.date = date_tag.get("data-date", "")

        # Extract match stats from the shot summary / match stats table
        # These are usually in the stats tables on the match page
        tables = self.parser.parse_page(html, url=match_url)

        for table in tables:
            if table.category == StatCategory.MATCH_STATS:
                for row in table.rows:
                    match.stats.update(row)

        return match

    # ── Lifecycle ──────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying client."""
        await self.client.close()

    async def __aenter__(self) -> FBrefScraper:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
