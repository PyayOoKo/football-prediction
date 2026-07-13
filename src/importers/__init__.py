"""
Production-quality Football-Data importer.

Downloads, validates, normalizes, and stores historical match data from
football-data.co.uk into the PostgreSQL database.

Pipeline
--------
1. **DownloadManager** — download CSV files with retry, integrity checks, raw storage
2. **CSVParser** — parse CSV rows, map columns, validate data types
3. **EntityResolver** — resolve team/competition names to database FK IDs
4. **FootballDataImporter** — orchestrator tying everything together

Usage
-----
::

    from src.importers import FootballDataImporter

    importer = FootballDataImporter()
    importer.download_historical(leagues=["E0", "E1"], max_seasons=5)
    importer.download_current(["E0", "E1"])
    importer.update_incremental()  # Only new matches since last run
"""

from src.importers.downloader import DownloadManager
from src.importers.parser import CSVParser
from src.importers.resolver import EntityResolver
from src.importers.football_data import FootballDataImporter

__all__ = [
    "DownloadManager",
    "CSVParser",
    "EntityResolver",
    "FootballDataImporter",
]
