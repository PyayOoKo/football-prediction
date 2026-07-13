"""
Unit tests for FBrefScraper — orchestrator and helper methods.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data_collection.sources.fbref.scraper import FBrefScraper, ScrapeJob
from src.data_collection.sources.fbref.models import (
    FBrefTable,
    StatCategory,
)


class TestScrapeJob:
    def test_save_load(self, tmp_path) -> None:
        """Checkpoint save/load round-trip preserves data."""
        job = ScrapeJob(
            job_id="test123",
            competition_id="9",
            season="2024-2025",
            teams=["Team A", "Team B"],
            categories=["standard", "shooting"],
            completed_teams=["standard"],
            started_at=100.0,
        )
        path = tmp_path / "test.checkpoint"
        job.save(path)
        assert path.exists()

        loaded = ScrapeJob.load(path)
        assert loaded is not None
        assert loaded.job_id == "test123"
        assert loaded.competition_id == "9"
        assert loaded.completed_teams == ["standard"]

    def test_load_nonexistent(self, tmp_path) -> None:
        """Loading a non-existent checkpoint returns None."""
        loaded = ScrapeJob.load(tmp_path / "nonexistent.checkpoint")
        assert loaded is None


class TestFBrefScraper:
    def test_init_defaults(self) -> None:
        """Scraper initialises with default components."""
        scraper = FBrefScraper()
        assert scraper.client is not None
        assert scraper.parser is not None
        assert scraper.checkpoint_dir.name == "fbref"

    def test_resolve_category(self) -> None:
        """Category name resolution works for various inputs."""
        assert FBrefScraper._resolve_category("shooting") == StatCategory.SHOOTING
        assert FBrefScraper._resolve_category("passing") == StatCategory.PASSING
        assert FBrefScraper._resolve_category("stats_defense") == StatCategory.DEFENSE
        assert FBrefScraper._resolve_category("possession") == StatCategory.POSSESSION
        assert FBrefScraper._resolve_category("unknown") == StatCategory.STANDARD

    def test_competition_name(self) -> None:
        """Competition ID resolves to URL-friendly name."""
        assert FBrefScraper._competition_name("9") == "Premier-League"
        assert FBrefScraper._competition_name("12") == "La-Liga"
        assert FBrefScraper._competition_name("999") == "999"  # Unknown

    def test_competition_id(self) -> None:
        """Competition name resolves to ID."""
        assert FBrefScraper._competition_id("Premier League") == "9"
        assert FBrefScraper._competition_id("La Liga") == "12"

    def test_tables_to_dataframe_empty(self) -> None:
        """Empty table list returns empty DataFrame."""
        import pandas as pd

        scraper = FBrefScraper()
        df = scraper.tables_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_tables_to_dataframe(self) -> None:
        """Tables are combined into a DataFrame with metadata columns."""
        scraper = FBrefScraper()

        tables = [
            FBrefTable(
                category=StatCategory.STANDARD,
                competition="Premier League",
                season="2024-2025",
                team_name="Manchester City",
                columns=["player_name", "goals"],
                rows=[{"player_name": "P1", "goals": 10}],
            ),
            FBrefTable(
                category=StatCategory.SHOOTING,
                competition="Premier League",
                season="2024-2025",
                team_name="Manchester City",
                columns=["player_name", "xG"],
                rows=[{"player_name": "P1", "xG": 8.5}],
            ),
        ]

        df = scraper.tables_to_dataframe(tables)
        assert len(df) == 2
        assert "_category" in df.columns
        assert df["_competition"].iloc[0] == "Premier League"

    @patch("src.data_collection.sources.fbref.scraper.FBrefClient.get")
    def test_get_team_stats_mocked(self, mock_get) -> None:
        """get_team_stats returns parsed tables from a mocked HTML response."""
        from src.data_collection.sources.fbref.scraper import FBrefScraper

        mock_get.return_value = """\
<html><body>
<!--
<table id="stats_standard">
<thead><tr><th data-stat="player">Player</th><th data-stat="goals">Goals</th></tr></thead>
<tbody><tr><td>Test Player</td><td>5</td></tr></tbody>
</table>
-->
</body></html>"""

        scraper = FBrefScraper()
        tables = scraper.get_team_stats_sync("9", "2024-2025", "standard")

        assert len(tables) == 1
        assert tables[0].category == StatCategory.STANDARD
        assert len(tables[0].rows) == 1
        assert tables[0].rows[0]["player_name"] == "Test Player"
        assert tables[0].rows[0]["goals"] == 5
