"""
Transfermarkt data collector.

Transfermarkt (https://www.transfermarkt.com) does not provide an
official API and their Terms of Service prohibit automated scraping.

This module provides:
1. Instructions for manual CSV export from Transfermarkt
2. A CSV import parser for user-downloaded transfer data
3. Documentation of what data is available and how to get it

Data available from Transfermarkt (manually):
- Squad market values
- Player ages
- Injuries and suspensions
- Transfer history

For automated injury/squad data, consider:
- API-Football (paid tier)
- FBref (via browser-save manual mode)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from football_data.config import config

logger = logging.getLogger(__name__)


class TransfermarktImporter:
    """Import Transfermarkt data from manually saved CSV files.

    Transfermarkt doesn't offer CSV export directly. Use a browser
    extension like "Table Capture" or copy-paste from the website
    into a spreadsheet, then save as CSV.

    Usage
    -----
    >>> importer = TransfermarktImporter()
    >>> injuries = importer.import_injuries("data/transfermarkt/injuries.csv")
    """

    def __init__(self) -> None:
        self.import_dir = config.transfermarkt.import_dir
        self.import_dir.mkdir(parents=True, exist_ok=True)

    def import_injuries(
        self, csv_path: str | Path
    ) -> list[dict[str, Any]]:
        """Import injury data from a manually prepared CSV file.

        Expected CSV columns: team, player, injury, expected_return

        Parameters
        ----------
        csv_path : str | Path
            Path to the CSV file.

        Returns
        -------
        list[dict]
            Injury records.
        """
        path = Path(csv_path)
        if not path.exists():
            logger.warning("Transfermarkt CSV not found: %s", path)
            return []

        records: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "team": row.get("team", "").strip(),
                    "player": row.get("player", "").strip(),
                    "injury": row.get("injury", "").strip(),
                    "expected_return": row.get("expected_return", "").strip(),
                })

        logger.info("Imported %d injury records from %s", len(records), path.name)
        return records

    def import_squad_values(
        self, csv_path: str | Path
    ) -> list[dict[str, Any]]:
        """Import squad market value data from a manually prepared CSV.

        Expected CSV columns: team, player, position, age, market_value, season

        Parameters
        ----------
        csv_path : str | Path
            Path to the CSV file.

        Returns
        -------
        list[dict]
            Squad value records.
        """
        path = Path(csv_path)
        if not path.exists():
            logger.warning("Squad CSV not found: %s", path)
            return []

        records: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "team": row.get("team", "").strip(),
                    "player": row.get("player", "").strip(),
                    "position": row.get("position", "").strip(),
                    "age": self._safe_float(row.get("age")),
                    "market_value": self._safe_float(row.get("market_value")),
                    "season": row.get("season", "").strip(),
                })

        logger.info("Imported %d squad records from %s", len(records), path.name)
        return records

    @staticmethod
    def print_instructions() -> str:
        """Print instructions for manually collecting Transfermarkt data."""
        return """
    ── How to collect Transfermarkt data ──

    Transfermarkt prohibits automated scraping. To get the data
    manually:

    INJURIES:
    1. Go to transfermarkt.com → Competition → Injuries
    2. Copy the injury table
    3. Paste into Excel/Google Sheets
    4. Save as CSV: data/transfermarkt_imports/injuries.csv

    Columns: team, player, injury, expected_return

    SQUAD VALUES:
    1. Go to transfermarkt.com → Competition → Squad details
    2. Copy the squad table
    3. Paste into Excel, save as CSV
    4. Save as: data/transfermarkt_imports/{league}_squad.csv

    Columns: team, player, position, age, market_value, season
    """

    @staticmethod
    def _safe_float(value: str | None) -> float | None:
        if value is None:
            return None
        cleaned = value.replace(",", "").replace("€", "").strip()
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None
