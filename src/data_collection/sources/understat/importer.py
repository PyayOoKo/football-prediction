"""
UnderstatImporter — orchestrates data collection from Understat.

Provides high-level methods for:
- League-level xG data (team stats, match results with xG)
- Match-level shot data (locations, xG per shot)
- Batch match processing for entire seasons
- Incremental synchronization with checkpoint/resume
- Duplicate detection via match_id fingerprinting
- Validation at every stage

Optimized for speed:
- Async HTTP with connection pooling
- Response caching with TTL
- Batched match processing with semaphore concurrency
- Only fetches new/updated matches during incremental sync
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_collection.sources.understat.client import UnderstatClient
from src.data_collection.sources.understat.models import (
    LEAGUE_NAMES,
    MatchXG,
    ShotData,
    TeamXG,
)
from src.data_collection.sources.understat.parser import UnderstatParser

logger = logging.getLogger(__name__)

# ── Default batch sizes for concurrent processing ────────
_DEFAULT_MAX_CONCURRENT = 5
_DEFAULT_SYNC_FILE = "data/scrapers/understat/sync_state.json"


@dataclass
class SyncState:
    """Persistent state for incremental synchronization.

    Tracks which matches have been imported per league+season
    to enable incremental updates.
    """

    # { "league_year": { "match_id": "date" } }
    imported_matches: dict[str, dict[str, str]] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"imported_matches": self.imported_matches}, f)

    @classmethod
    def load(cls, path: str | Path) -> SyncState:
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(imported_matches=data.get("imported_matches", {}))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load sync state: %s", exc)
            return cls()

    def is_match_imported(self, league_key: str, match_id: int) -> bool:
        """Check if a match has already been imported."""
        matches = self.imported_matches.get(league_key, {})
        return str(match_id) in matches

    def mark_imported(self, league_key: str, match_id: int, date: str) -> None:
        """Record a match as imported."""
        if league_key not in self.imported_matches:
            self.imported_matches[league_key] = {}
        self.imported_matches[league_key][str(match_id)] = date

    def get_new_match_ids(
        self,
        league_key: str,
        all_matches: list[MatchXG],
    ) -> list[MatchXG]:
        """Return only matches that haven't been imported yet."""
        return [
            m for m in all_matches
            if not self.is_match_imported(league_key, m.match_id)
        ]


@dataclass
class SyncReport:
    """Result of a sync operation."""

    league: str = ""
    season: str = ""
    matches_found: int = 0
    matches_new: int = 0
    matches_imported: int = 0
    matches_skipped: int = 0
    shots_imported: int = 0
    teams_found: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "league": self.league,
            "season": self.season,
            "matches_found": self.matches_found,
            "matches_new": self.matches_new,
            "matches_imported": self.matches_imported,
            "matches_skipped": self.matches_skipped,
            "shots_imported": self.shots_imported,
            "teams_found": self.teams_found,
            "errors": self.errors,
            "duration_seconds": round(self.duration_seconds, 2),
            "success": self.success,
        }


class UnderstatImporter:
    """Main orchestrator for Understat data import.

    Parameters
    ----------
    client : UnderstatClient, optional
        Custom client instance.
    parser : UnderstatParser, optional
        Custom parser instance.
    sync_file : str | Path
        Path to sync state file (default ``data/scrapers/understat/sync_state.json``).
    max_concurrent : int
        Max concurrent match fetches (default 5).
    """

    def __init__(
        self,
        client: UnderstatClient | None = None,
        parser: UnderstatParser | None = None,
        sync_file: str | Path = _DEFAULT_SYNC_FILE,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
    ) -> None:
        self.client = client or UnderstatClient()
        self.parser = parser or UnderstatParser(validate=True, strict=False)
        self.sync_file = Path(sync_file)
        self.max_concurrent = max_concurrent

        # Load sync state
        self._sync = SyncState.load(self.sync_file)

    # ══════════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════════

    async def get_league_xg(
        self,
        league: str = "EPL",
        year: int = 2024,
    ) -> tuple[list[TeamXG], list[MatchXG]]:
        """Fetch league-level xG data (teams + matches).

        Parameters
        ----------
        league : str
            League code (default ``EPL``).
        year : int
            Season starting year (default 2024).

        Returns
        -------
        tuple[list[TeamXG], list[MatchXG]]
            Team xG stats and match-level xG data.
        """
        html = await self.client.get_league_page(league, year)
        teams, matches = self.parser.parse_league_from_html(html, league, year)
        return teams, matches

    async def get_match_shots(self, match_id: int) -> list[ShotData]:
        """Fetch shot data for a single match.

        Parameters
        ----------
        match_id : int
            Understat match identifier.

        Returns
        -------
        list[ShotData]
            Shot-level data with locations and xG.
        """
        shots_data = await self.client.get_match_data(match_id)
        return self.parser.parse_match_shots(shots_data, match_id)

    async def get_match_xg(self, match_id: int) -> MatchXG | None:
        """Extract match-level xG from a match page.

        Computes aggregate xG from shot data for a single match.

        Parameters
        ----------
        match_id : int
            Understat match identifier.

        Returns
        -------
        MatchXG or None
            Match-level xG, or None if parsing fails.
        """
        try:
            shots = await self.get_match_shots(match_id)
            if not shots:
                return None

            home_team = shots[0].home_team
            away_team = shots[0].away_team
            date = shots[0].date

            home_xg = sum(s.xg for s in shots if s.team == home_team)
            away_xg = sum(s.xg for s in shots if s.team == away_team)
            home_goals = sum(1 for s in shots if s.team == home_team and s.is_goal)
            away_goals = sum(1 for s in shots if s.team == away_team and s.is_goal)
            home_sot = sum(1 for s in shots if s.team == home_team and s.result in ("GOAL", "SAVED"))
            away_sot = sum(1 for s in shots if s.team == away_team and s.result in ("GOAL", "SAVED"))

            return MatchXG(
                match_id=match_id,
                date=date,
                home_team=home_team,
                away_team=away_team,
                home_xg=round(home_xg, 2),
                away_xg=round(away_xg, 2),
                home_goals=home_goals,
                away_goals=away_goals,
                home_shots=sum(1 for s in shots if s.team == home_team),
                away_shots=sum(1 for s in shots if s.team == away_team),
                home_shots_on_target=home_sot,
                away_shots_on_target=away_sot,
            )

        except Exception as exc:
            logger.warning("Failed to get match xG for %d: %s", match_id, exc)
            return None

    async def get_batch_match_shots(
        self,
        match_ids: list[int],
    ) -> dict[int, list[ShotData]]:
        """Fetch shot data for multiple matches concurrently.

        Parameters
        ----------
        match_ids : list[int]
            Understat match identifiers.

        Returns
        -------
        dict[int, list[ShotData]]
            Mapping of match_id → shot data.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def fetch_one(mid: int) -> tuple[int, list[ShotData]]:
            async with semaphore:
                shots = await self.get_match_shots(mid)
                return mid, shots

        tasks = [fetch_one(mid) for mid in match_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[int, list[ShotData]] = {}
        for mid, result in zip(match_ids, results):
            if isinstance(result, Exception):
                logger.warning("Failed to fetch match %d: %s", mid, result)
                output[mid] = []
            else:
                output[mid] = result

        return output

    # ══════════════════════════════════════════════════════
    #  Incremental sync
    # ══════════════════════════════════════════════════════

    async def sync_league(
        self,
        league: str = "EPL",
        year: int = 2024,
        include_shots: bool = True,
        force_refresh: bool = False,
    ) -> SyncReport:
        """Synchronize league data incrementally.

        Only fetches matches that haven't been imported yet (based on
        match_id). Saves sync state after completion.

        Parameters
        ----------
        league : str
            League code (default ``EPL``).
        year : int
            Season starting year.
        include_shots : bool
            Whether to also fetch shot-level data (default True).
        force_refresh : bool
            Re-fetch all matches, not just new ones (default False).

        Returns
        -------
        SyncReport
            Report of what was imported.
        """
        report = SyncReport(league=league, season=str(year))
        start = time.perf_counter()
        league_key = f"{league}_{year}"

        try:
            # Step 1: Fetch league page (teams + match list)
            html = await self.client.get_league_page(league, year)
            teams_data_raw = self.client._extract_json(html, "teamsData")
            dates_data_raw = self.client._extract_json(html, "datesData")

            # Step 2: Parse teams
            teams = self.parser.parse_league_teams(teams_data_raw, league, year)
            report.teams_found = len(teams)

            # Step 3: Parse match list
            all_matches = self.parser.parse_league_matches(
                dates_data_raw, teams_data_raw, league, year,
            )
            report.matches_found = len(all_matches)

            # Step 4: Determine which matches to fetch
            if force_refresh:
                new_matches = all_matches
            else:
                new_matches = self._sync.get_new_match_ids(league_key, all_matches)

            report.matches_new = len(new_matches)
            report.matches_skipped = report.matches_found - report.matches_new

            logger.info(
                "%s %s: %d matches total, %d new, %d already imported",
                league, year, report.matches_found,
                report.matches_new, report.matches_skipped,
            )

            # Step 5: Fetch shot data for new matches
            if include_shots and new_matches:
                match_ids = [m.match_id for m in new_matches]
                shots_map = await self.get_batch_match_shots(match_ids)

                total_shots = sum(len(s) for s in shots_map.values())
                report.shots_imported = total_shots

                # Mark matches as imported
                for match in new_matches:
                    self._sync.mark_imported(
                        league_key, match.match_id, match.date,
                    )

            else:
                # Still mark matches as imported (just xG, no shots)
                for match in new_matches:
                    self._sync.mark_imported(
                        league_key, match.match_id, match.date,
                    )

            report.matches_imported = len(new_matches)

            # Step 6: Save sync state
            self._sync.save(self.sync_file)

        except Exception as exc:
            logger.exception("Sync failed for %s %s", league, year)
            report.errors.append(str(exc))

        report.duration_seconds = time.perf_counter() - start
        logger.info(
            "%s %s sync complete: %d matches, %d shots in %.1fs",
            league, year, report.matches_imported,
            report.shots_imported, report.duration_seconds,
        )
        return report

    async def sync_multiple_leagues(
        self,
        leagues: list[tuple[str, int]],
        include_shots: bool = True,
    ) -> list[SyncReport]:
        """Synchronize multiple league seasons.

        Parameters
        ----------
        leagues : list[tuple[str, int]]
            List of ``(league_code, year)`` tuples.
        include_shots : bool
            Whether to fetch shot data.

        Returns
        -------
        list[SyncReport]
            One report per league.
        """
        reports: list[SyncReport] = []
        for league, year in leagues:
            report = await self.sync_league(league, year, include_shots)
            reports.append(report)
        return reports

    # ══════════════════════════════════════════════════════
    #  DataFrame helpers
    # ══════════════════════════════════════════════════════

    def teams_to_dataframe(self, teams: list[TeamXG]) -> pd.DataFrame:
        """Convert TeamXG records to a DataFrame.

        Parameters
        ----------
        teams : list[TeamXG]
            Parsed team xG records.

        Returns
        -------
        pd.DataFrame
        """
        if not teams:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in teams])

    def matches_to_dataframe(self, matches: list[MatchXG]) -> pd.DataFrame:
        """Convert MatchXG records to a DataFrame.

        Parameters
        ----------
        matches : list[MatchXG]
            Parsed match xG records.

        Returns
        -------
        pd.DataFrame
        """
        if not matches:
            return pd.DataFrame()
        return pd.DataFrame([m.to_dict() for m in matches])

    def shots_to_dataframe(self, shots: list[ShotData]) -> pd.DataFrame:
        """Convert ShotData records to a DataFrame.

        Parameters
        ----------
        shots : list[ShotData]
            Parsed shot records.

        Returns
        -------
        pd.DataFrame
        """
        if not shots:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in shots])

    # ══════════════════════════════════════════════════════
    #  Sync state management
    # ══════════════════════════════════════════════════════

    def get_imported_match_ids(self, league: str, year: int) -> set[int]:
        """Get the set of already-imported match IDs for a league.

        Parameters
        ----------
        league : str
            League code.
        year : int
            Season starting year.

        Returns
        -------
        set[int]
            Imported match IDs.
        """
        league_key = f"{league}_{year}"
        matches = self._sync.imported_matches.get(league_key, {})
        return {int(mid) for mid in matches.keys() if mid.isdigit()}

    def reset_sync_state(self, league: str | None = None, year: int | None = None) -> None:
        """Reset sync state for a specific league, or all leagues.

        Parameters
        ----------
        league : str, optional
            League code to reset.
        year : int, optional
            Season year to reset.
        """
        if league and year:
            league_key = f"{league}_{year}"
            self._sync.imported_matches.pop(league_key, None)
            logger.info("Reset sync state for %s", league_key)
        elif league:
            keys = [k for k in self._sync.imported_matches if k.startswith(league)]
            for k in keys:
                self._sync.imported_matches.pop(k, None)
            logger.info("Reset sync state for league %s (%d keys)", league, len(keys))
        else:
            self._sync.imported_matches.clear()
            logger.info("Reset all sync state")

        self._sync.save(self.sync_file)

    # ── Lifecycle ──────────────────────────────────────

    async def close(self) -> None:
        await self.client.close()

    async def __aenter__(self) -> UnderstatImporter:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
