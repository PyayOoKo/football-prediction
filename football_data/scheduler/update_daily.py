"""
Daily update orchestrator for the football data pipeline.

Coordinates the full ETL process:
1. Collect data from all sources
2. Clean and normalise
3. Validate
4. Store in SQLite
5. Generate collection report

Can be run as a standalone script or scheduled via cron/Task Scheduler.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_data.collectors import FBrefParser, FootballDataCollector, WeatherCollector
from football_data.collectors.weather import LEAGUE_DEFAULT_COORDS
from football_data.config import config, LEAGUE_NAMES, DEFAULT_LEAGUES
from football_data.database import Database
from football_data.processors import DataCleaner, DataValidator

logger = logging.getLogger(__name__)


class DailyUpdater:
    """Orchestrates the daily data collection pipeline.

    Usage
    -----
    >>> updater = DailyUpdater()
    >>> report = await updater.run()
    >>> print(report["summary"])
    """

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()
        self.cleaner = DataCleaner()
        self.validator = DataValidator()
        self.collectors: dict[str, Any] = {}

    async def run(
        self,
        leagues: list[str] | None = None,
        source: str | None = None,
        skip_weather: bool = False,
    ) -> dict[str, Any]:
        """Run the full collection pipeline.

        Parameters
        ----------
        leagues : list[str] | None
            League codes to collect. Defaults to config.leagues.
        source : str | None
            Only collect from this source ("football-data", "fbref", "weather").
            If None, all sources are collected.
        skip_weather : bool
            Skip weather data collection.

        Returns
        -------
        dict
            Full run report with per-source, per-league stats.
        """
        if leagues is None:
            leagues = list(config.leagues)

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "leagues": leagues,
            "by_source": {},
            "summary": "",
            "errors": [],
        }

        await self.db.connect()

        try:
            # ── Step 1: football-data.co.uk ──
            if source is None or source == "football-data":
                fdc_report = await self._collect_football_data(leagues)
                report["by_source"]["football-data.co.uk"] = fdc_report

            # ── Step 2: FBref (browser-saved pages) ──
            if source is None or source == "fbref":
                fbref_report = await self._collect_fbref()
                report["by_source"]["fbref"] = fbref_report

            # ── Step 3: Weather ──
            if not skip_weather and (source is None or source == "weather"):
                weather_report = await self._collect_weather()
                report["by_source"]["weather"] = weather_report

            # ── Step 4: Validate database ──
            stats = await self.db.get_statistics()
            report["database_stats"] = stats

            # ── Summary ──
            total = stats.get("total_matches", 0)
            by_league = stats.get("by_league", {})

            lines = [
                "═" * 50,
                "  DAILY COLLECTION REPORT",
                f"  Date: {report['timestamp'][:10]}",
                "═" * 50,
            ]
            for source_name, src_report in report["by_source"].items():
                lines.append(f"\n  {source_name}:")
                if isinstance(src_report, dict):
                    for key, val in src_report.items():
                        lines.append(f"    {key}: {val}")

            lines.append(f"\n  Total matches in database: {total}")
            lines.append("\n  By league:")
            for league_code, count in sorted(by_league.items()):
                league_name = LEAGUE_NAMES.get(league_code, league_code)
                lines.append(f"    {league_code} ({league_name}): {count}")

            report["summary"] = "\n".join(lines)
            logger.info("Daily collection complete — %d total matches", total)

        except Exception as exc:
            logger.exception("Daily collection failed: %s", exc)
            report["errors"].append(str(exc))
            report["summary"] = f"Collection FAILED: {exc}"
        finally:
            await self.db.close()

        return report

    # ── Source-specific collection ───────────────────────

    async def _collect_football_data(
        self, leagues: list[str]
    ) -> dict[str, Any]:
        """Collect match data from football-data.co.uk."""
        logger.info("Starting football-data.co.uk collection for %s", leagues)
        collector = FootballDataCollector()

        try:
            all_matches = await collector.collect(leagues)

            total_downloaded = sum(len(m) for m in all_matches.values())
            total_inserted = 0
            total_errors = 0

            for league_code, matches in all_matches.items():
                # Clean
                cleaned = self.cleaner.clean_matches(matches)
                # Validate
                validation = self.validator.validate_matches(cleaned)
                if validation["fatal"] > 0:
                    logger.warning(
                        "%s: %d/%d records had fatal errors",
                        league_code, validation["fatal"], validation["total"],
                    )
                    total_errors += validation["fatal"]

                # Insert into database
                log_id = await self.db.log_collection_start(
                    "football-data.co.uk", league_code, None
                )
                inserted, duplicates = await self.db.insert_matches(cleaned)
                await self.db.log_collection_end(
                    log_id, len(cleaned), duplicates, validation["fatal"]
                )
                total_inserted += inserted

            return {
                "leagues_downloaded": len(all_matches),
                "rows_downloaded": total_downloaded,
                "rows_inserted": total_inserted,
                "rows_duplicates": total_downloaded - total_inserted,
                "errors": total_errors,
            }

        finally:
            await collector.close()

    async def _collect_fbref(self) -> dict[str, Any]:
        """Parse FBref saved HTML pages."""
        parser = FBrefParser()
        saved_dir = config.fbref.saved_pages_dir

        if not saved_dir.exists() or not list(saved_dir.glob("*.html")):
            return {
                "status": "skipped",
                "reason": "No saved HTML files found in fbref_pages/",
                "instructions": FBrefParser.print_instructions(),
            }

        results = parser.parse_all_saved_pages()
        total_inserted = 0
        total_errors = 0

        for league_code, matches in results.items():
            cleaned = self.cleaner.clean_matches(matches)
            validation = self.validator.validate_matches(cleaned)
            if validation["fatal"] > 0:
                total_errors += validation["fatal"]

            log_id = await self.db.log_collection_start("fbref", league_code, None)
            inserted, duplicates = await self.db.insert_matches(cleaned)
            await self.db.log_collection_end(
                log_id, len(cleaned), duplicates, validation["fatal"]
            )
            total_inserted += inserted

        return {
            "pages_parsed": len(results),
            "rows_inserted": total_inserted,
            "errors": total_errors,
        }

    async def _collect_weather(self) -> dict[str, Any]:
        """Collect weather data for historical matches (pre-2025, when data exists)."""
        collector = WeatherCollector()
        try:
            # Query matches with dates before 2025-01-01 so the Open-Meteo archive
            # API can return valid data (it doesn't support future dates).
            import sqlite3
            conn = sqlite3.connect(str(config.database.db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT match_id, date, league FROM matches "
                "WHERE date < '2025-01-01' "
                "AND match_id NOT IN (SELECT match_id FROM weather WHERE match_id IS NOT NULL) "
                "ORDER BY date DESC LIMIT 100"
            )
            rows = cursor.fetchall()
            conn.close()

            matches_with_coords = []
            for match_id, date_str, league_code in rows:
                coords = LEAGUE_DEFAULT_COORDS.get(league_code, {"lat": 55.0, "lon": 10.0})
                matches_with_coords.append({
                    "match_id": match_id,
                    "date": date_str,
                    "lat": coords["lat"],
                    "lon": coords["lon"],
                })

            weather_records = await collector.collect_for_matches(matches_with_coords)
            inserted = await self.db.insert_weather(weather_records)

            return {
                "matches_checked": len(rows),
                "weather_records_inserted": inserted,
            }

        finally:
            await collector.close()

    # ── Static runner for CLI / scheduler ────────────────

    @staticmethod
    async def run_cli() -> None:
        """Run the pipeline from the command line."""
        import argparse

        parser = argparse.ArgumentParser(
            description="Football data daily collection pipeline",
        )
        parser.add_argument(
            "--leagues",
            nargs="+",
            default=list(DEFAULT_LEAGUES),
            help="League codes to collect (default: all)",
        )
        parser.add_argument(
            "--source",
            choices=["football-data", "fbref", "weather"],
            help="Only collect from this source",
        )
        parser.add_argument(
            "--skip-weather",
            action="store_true",
            help="Skip weather data collection",
        )
        parser.add_argument(
            "--db-path",
            type=str,
            help="Path to SQLite database file",
        )

        args = parser.parse_args()

        # Configure logging
        log_path = config.logging.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, config.logging.level),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(str(log_path)),
                logging.StreamHandler(sys.stdout) if config.logging.console_logging else logging.NullHandler(),
            ],
        )

        # Override DB path if specified
        db = None
        if args.db_path:
            from football_data.database.sqlite import Database
            db = Database(Path(args.db_path))

        updater = DailyUpdater(db=db)
        report = await updater.run(
            leagues=args.leagues,
            source=args.source,
            skip_weather=args.skip_weather,
        )

        # Print summary, handling Windows cp1252 encoding
        summary = report["summary"]
        try:
            print("\n" + summary)
        except UnicodeEncodeError:
            # Fall back to ascii-safe output on Windows
            safe = summary.encode("ascii", errors="replace").decode("ascii")
            print("\n" + safe)

        if report["errors"]:
            print(f"\nErrors: {len(report['errors'])}")
            for err in report["errors"]:
                try:
                    print(f"  • {err}")
                except UnicodeEncodeError:
                    safe_err = err.encode("ascii", errors="replace").decode("ascii")
                    print(f"  * {safe_err}")

        sys.exit(1 if report["errors"] else 0)


# ── Entry point ─────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(DailyUpdater.run_cli())
