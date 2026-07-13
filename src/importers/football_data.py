"""
FootballDataImporter — production-quality import orchestrator.

Downloads, validates, parses, resolves, and stores historical match
data from football-data.co.uk into the PostgreSQL database.

Usage
-----
::

    from src.importers import FootballDataImporter

    importer = FootballDataImporter()

    # One-shot: download 5 seasons of PL + Championship
    results = importer.import_historical(
        leagues=["E0", "E1"],
        max_seasons=5,
    )

    # Incremental: only new matches since last import
    importer.import_current(["E0", "E1"])

    # Full pipeline report
    for r in results:
        print(r.status, r.season, r.league, r.rows_imported)

Pipeline stages
---------------
1. Download raw CSV (with retry + integrity check)
2. Parse CSV into standardised rows
3. Validate required columns + row-level constraints
4. Resolve team/competition/season FK IDs
5. Store to PostgreSQL (upsert, batch-committed)
6. Log structured report

Features
--------
- **Incremental updates** — only imports matches not already in the DB
- **Resumable** — checkpoints per league+season combination
- **Structured logging** — every stage logs its metrics
- **Config-driven** — league mappings, auto-create flags, batch sizes
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import text as sa_text

from src.database.models.match import Match
from src.database.session import get_session
from src.etl.store import DatabaseStore
from src.importers.downloader import (
    LEAGUE_MAP,
    _guess_season_range,
    _current_season_code,
    DownloadManager,
)
from src.importers.parser import CSVParser
from src.importers.resolver import EntityResolver

logger = logging.getLogger(__name__)


@dataclass
class ImportReport:
    """Result of importing a single league+season combination.

    Attributes
    ----------
    league : str
        League code (e.g. ``\"E0\"``).
    season : str
        Season code (e.g. ``\"2425\"``) or ``\"current\"``.
    season_name : str
        Human-readable season (e.g. ``\"2024/2025\"``).
    status : str
        ``\"success\"``, ``\"skipped\"``, ``\"partial\"``, or ``\"failed\"``.
    rows_downloaded : int
        Rows in the raw CSV.
    rows_parsed : int
        Rows that passed CSV parsing.
    rows_imported : int
        Rows successfully stored in the database.
    rows_skipped : int
        Rows skipped (already exist or validation failure).
    errors : list[str]
        Error messages.
    duration_seconds : float
        Time taken for this import.
    """

    league: str = ""
    season: str = ""
    season_name: str = ""
    status: str = "unknown"
    rows_downloaded: int = 0
    rows_parsed: int = 0
    rows_imported: int = 0
    rows_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def success(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "league": self.league,
            "season": self.season,
            "season_name": self.season_name,
            "status": self.status,
            "rows_downloaded": self.rows_downloaded,
            "rows_parsed": self.rows_parsed,
            "rows_imported": self.rows_imported,
            "rows_skipped": self.rows_skipped,
            "errors": self.errors,
            "duration_seconds": round(self.duration_seconds, 2),
        }


class FootballDataImporter:
    """Orchestrates the full football-data.co.uk import pipeline.

    Parameters
    ----------
    raw_dir : str | Path
        Directory for raw CSV downloads.
    db_batch_size : int
        Rows per database batch commit (default 500).
    auto_create_teams : bool
        Auto-create unknown teams in the DB (default True).
    auto_create_competitions : bool
        Auto-create unknown competitions (default True).
    auto_create_seasons : bool
        Auto-create unknown seasons (default True).
    max_retries : int
        Download retry attempts (default 3).
    download_timeout : int
        HTTP download timeout in seconds (default 30).
    incremental : bool
        If True, skip matches already in the database (default True).
    league_map : dict[str, str]
        League code → display name mapping. Defaults to built-in map.
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw/football-data",
        db_batch_size: int = 500,
        auto_create_teams: bool = True,
        auto_create_competitions: bool = True,
        auto_create_seasons: bool = True,
        max_retries: int = 3,
        download_timeout: int = 30,
        incremental: bool = True,
        league_map: dict[str, str] | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.db_batch_size = db_batch_size
        self.incremental = incremental
        self.league_map = league_map or dict(LEAGUE_MAP)

        # Pipeline components
        self.downloader = DownloadManager(
            raw_dir=self.raw_dir,
            max_retries=max_retries,
            timeout=download_timeout,
        )
        self.parser = CSVParser(strict=True)
        self.resolver = EntityResolver(
            auto_create_teams=auto_create_teams,
            auto_create_competitions=auto_create_competitions,
            auto_create_seasons=auto_create_seasons,
        )
        self.db_store = DatabaseStore(
            model_class=Match,
            unique_columns=["id"],  # Uses generated IDs, not CSV-match_id
            batch_size=db_batch_size,
        )

        # Internal state
        self._known_match_dates: set[str] = set()

    # ── Public API ─────────────────────────────────────

    def import_historical(
        self,
        leagues: list[str] | None = None,
        max_seasons: int = 10,
    ) -> list[ImportReport]:
        """Download and import historical seasons for one or more leagues.

        Parameters
        ----------
        leagues : list[str], optional
            League codes. Defaults to all known leagues.
        max_seasons : int
            Number of most-recent seasons to fetch (default 10).

        Returns
        -------
        list[ImportReport]
            One report per league+season combination.
        """
        leagues = leagues or list(self.league_map.keys())
        seasons = _guess_season_range(max_seasons)

        # Pre-warm entity caches for faster lookup
        self._prewarm_caches(leagues, seasons)

        reports: list[ImportReport] = []

        for league in leagues:
            for season in seasons:
                report = self._import_single(league, season)
                reports.append(report)

            # Also try the current in-progress season
            current_report = self._import_current(league)
            reports.append(current_report)

        # Summary
        success_count = sum(1 for r in reports if r.success)
        total_rows = sum(r.rows_imported for r in reports)
        logger.info(
            "Historical import complete: %d/%d leagues+seasons, %d total rows imported",
            success_count, len(reports), total_rows,
        )

        return reports

    def import_current(self, leagues: list[str] | None = None) -> list[ImportReport]:
        """Import the current in-progress season for the given leagues.

        Parameters
        ----------
        leagues : list[str], optional
            League codes. Defaults to all known leagues.

        Returns
        -------
        list[ImportReport]
            One report per league.
        """
        leagues = leagues or list(self.league_map.keys())
        reports: list[ImportReport] = []

        for league in leagues:
            report = self._import_current(league)
            reports.append(report)

        return reports

    def import_single_file(self, filepath: str, league: str, season: str) -> ImportReport:
        """Import a single CSV file that's already on disk.

        Useful for testing or manual imports.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.
        league : str
            League code (e.g. ``\"E0\"``).
        season : str
            Season code (e.g. ``\"2425\"``) or ``\"current\"``.

        Returns
        -------
        ImportReport
        """
        report = ImportReport(league=league, season=season)
        start = time.perf_counter()

        try:
            path = Path(filepath)
            raw_text = path.read_text(encoding="utf-8")

            report.rows_downloaded = len(raw_text.strip().split("\n")) - 1

            # Parse
            parsed_rows = self.parser.parse_to_dicts(raw_text, source_file=filepath)
            report.rows_parsed = len(parsed_rows)

            if not parsed_rows:
                report.status = "skipped"
                report.duration_seconds = time.perf_counter() - start
                return report

            # Resolve entities
            resolved = self.resolver.resolve_rows(
                parsed_rows, league_map=self.league_map,
            )

            # Filter by incremental
            if self.incremental:
                resolved = self._filter_new_rows(resolved)

            report.rows_skipped = report.rows_parsed - len(resolved)

            # Store
            if resolved:
                store_result = self.db_store.write(resolved, batch_size=self.db_batch_size)
                report.rows_imported = store_result.records_out or 0

            report.status = "success" if report.rows_imported > 0 else "skipped"

        except Exception as exc:
            logger.exception("Import failed for %s: %s", filepath, exc)
            report.status = "failed"
            report.errors.append(str(exc))

        report.duration_seconds = time.perf_counter() - start
        report.season_name = self._season_name(season)
        logger.info(
            "Import %s/%s: %s (%d rows in %.1fs)",
            league, season, report.status, report.rows_imported, report.duration_seconds,
        )
        return report

    # ── Internal ───────────────────────────────────────

    def _import_single(self, league: str, season: str) -> ImportReport:
        """Download, parse, resolve, and store a single league+season."""
        report = ImportReport(league=league, season=season)
        start = time.perf_counter()

        try:
            # Step 1: Download
            download_result = self.downloader.download_season(league, season)

            if not download_result.success:
                report.status = "skipped"
                report.errors.append(download_result.error or "Download failed")
                report.duration_seconds = time.perf_counter() - start
                return report

            report.rows_downloaded = download_result.row_count

            # Verify integrity
            if not self.downloader.verify_integrity(download_result):
                report.status = "failed"
                report.errors.append("Integrity check failed")
                report.duration_seconds = time.perf_counter() - start
                return report

            # Step 2: Parse
            raw_path = download_result.raw_path
            if raw_path is None or not raw_path.exists():
                report.status = "failed"
                report.errors.append("Raw file not found after download")
                report.duration_seconds = time.perf_counter() - start
                return report

            raw_text = raw_path.read_text(encoding="utf-8")
            parsed_rows = self.parser.parse_to_dicts(raw_text, source_file=str(raw_path))
            report.rows_parsed = len(parsed_rows)

            if not parsed_rows:
                report.status = "skipped"
                report.duration_seconds = time.perf_counter() - start
                return report

            # Step 3: Resolve entities
            resolved = self.resolver.resolve_rows(
                parsed_rows, league_map=self.league_map,
            )

            # Step 4: Incremental filtering
            if self.incremental:
                resolved = self._filter_new_rows(resolved)

            report.rows_skipped = report.rows_parsed - len(resolved)

            # Step 5: Store
            if resolved:
                store_result = self.db_store.write(resolved, batch_size=self.db_batch_size)
                report.rows_imported = store_result.records_out or 0

            report.status = "success" if report.rows_imported > 0 else "skipped"

        except Exception as exc:
            logger.exception(
                "Import failed for %s/%s: %s", league, season, exc,
            )
            report.status = "failed"
            report.errors.append(str(exc))

        report.duration_seconds = time.perf_counter() - start
        report.season_name = self._season_name(season)
        logger.info(
            "%s/%s: %s — %d rows imported in %.1fs",
            league, season, report.status,
            report.rows_imported, report.duration_seconds,
        )
        return report

    def _import_current(self, league: str) -> ImportReport:
        """Download and import the current in-progress season for a league."""
        report = ImportReport(league=league, season="current")
        start = time.perf_counter()

        try:
            download_result = self.downloader.download_current(league)

            if not download_result.success:
                report.status = "skipped"
                report.errors.append(download_result.error or "Download failed")
                report.duration_seconds = time.perf_counter() - start
                return report

            report.rows_downloaded = download_result.row_count

            if not self.downloader.verify_integrity(download_result):
                report.status = "failed"
                report.errors.append("Integrity check failed")
                report.duration_seconds = time.perf_counter() - start
                return report

            raw_path = download_result.raw_path
            if raw_path is None or not raw_path.exists():
                report.status = "failed"
                report.errors.append("Raw file not found")
                report.duration_seconds = time.perf_counter() - start
                return report

            raw_text = raw_path.read_text(encoding="utf-8")
            parsed_rows = self.parser.parse_to_dicts(raw_text, source_file=str(raw_path))
            report.rows_parsed = len(parsed_rows)

            if not parsed_rows:
                report.status = "skipped"
                report.duration_seconds = time.perf_counter() - start
                return report

            resolved = self.resolver.resolve_rows(
                parsed_rows, league_map=self.league_map,
            )

            if self.incremental:
                resolved = self._filter_new_rows(resolved)

            report.rows_skipped = report.rows_parsed - len(resolved)

            if resolved:
                store_result = self.db_store.write(resolved, batch_size=self.db_batch_size)
                report.rows_imported = store_result.records_out or 0

            report.status = "success" if report.rows_imported > 0 else "skipped"

        except Exception as exc:
            logger.exception("Import failed for current %s: %s", league, exc)
            report.status = "failed"
            report.errors.append(str(exc))

        report.duration_seconds = time.perf_counter() - start
        report.season_name = self._season_name(_current_season_code())
        logger.info(
            "Current %s: %s — %d rows imported in %.1fs",
            league, report.status, report.rows_imported, report.duration_seconds,
        )
        return report

    def _filter_new_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter out rows that already exist in the database.

        Uses match_date + home/away team as a fingerprint.
        """
        if not self._known_match_dates:
            self._load_known_matches()

        filtered: list[dict[str, Any]] = []
        for row in rows:
            match_date = row.get("match_date")
            home_id = row.get("home_team_id")
            away_id = row.get("away_team_id")
            if match_date and home_id and away_id:
                fingerprint = f"{match_date}:{home_id}:{away_id}"
                if fingerprint in self._known_match_dates:
                    continue
                self._known_match_dates.add(fingerprint)
            filtered.append(row)

        return filtered

    def _setup_existing_matches(self) -> None:
        """Pre-warm known matches from the database."""
        self._load_known_matches()

    def _load_known_matches(self) -> None:
        """Load all existing match fingerprints from the database."""
        try:
            with get_session() as session:
                result = session.execute(
                    sa_text(
                        "SELECT match_date, home_team_id, away_team_id "
                        "FROM matches "
                        "WHERE match_date IS NOT NULL "
                        "AND home_team_id IS NOT NULL "
                        "AND away_team_id IS NOT NULL"
                    )
                )
                for row in result:
                    fingerprint = f"{row.match_date}:{row.home_team_id}:{row.away_team_id}"
                    self._known_match_dates.add(fingerprint)

            logger.info("Loaded %d existing match fingerprints", len(self._known_match_dates))
        except Exception as exc:
            logger.warning("Could not load existing matches: %s", exc)

    def _prewarm_caches(self, leagues: list[str], seasons: list[str]) -> None:
        """Pre-warm entity caches for faster resolution.

        Loads all teams, competitions, and seasons upfront to
        avoid N+1 queries during batch imports.
        """
        try:
            with get_session() as session:
                stats = self.resolver.prewarm_from_db(session)
                logger.info(
                    "Pre-warmed caches: %d teams, %d competitions, %d seasons",
                    stats.get("teams", 0),
                    stats.get("competitions", 0),
                    stats.get("seasons", 0),
                )
        except Exception as exc:
            logger.warning("Cache pre-warm failed: %s", exc)

    @staticmethod
    def _season_name(season_code: str) -> str:
        """Convert 4-digit code to human-readable name, or pass through."""
        if season_code == "current":
            code = _current_season_code()
        else:
            code = season_code
        try:
            start = int(code[:2])
            end = int(code[2:])
            prefix = 1900 if start > 80 else 2000
            return f"{prefix + start}/{prefix + end}"
        except (ValueError, IndexError):
            return code
