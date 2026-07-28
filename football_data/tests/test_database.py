"""Integration tests for the Database layer.

Uses an in-memory SQLite database to avoid side effects.
"""

import pytest

from football_data.database.sqlite import Database


@pytest.mark.asyncio
class TestDatabase:
    """Test the Database class with in-memory SQLite."""

    async def test_connect_and_create_tables(self):
        db = Database(":memory:")
        await db.connect()
        assert db._connected

        # Check tables exist
        stats = await db.get_statistics()
        assert "total_matches" in stats
        assert stats["total_matches"] == 0

        await db.close()
        assert not db._connected

    async def test_insert_and_query_matches(self):
        db = Database(":memory:")
        await db.connect()

        matches = [
            {
                "source": "football-data.co.uk",
                "league": "SE1",
                "season": "2425",
                "date": "2025-04-15",
                "home_team": "IK Brage",
                "away_team": "Degerfors IF",
                "home_goals": 2,
                "away_goals": 1,
                "result": "H",
            },
            {
                "source": "football-data.co.uk",
                "league": "SE1",
                "season": "2425",
                "date": "2025-04-15",
                "home_team": "Gefle IF",
                "away_team": "Örebro SK",
                "home_goals": 0,
                "away_goals": 0,
                "result": "D",
            },
        ]

        inserted, duplicates = await db.insert_matches(matches)
        assert inserted == 2
        assert duplicates == 0

        # Query all
        all_matches = await db.get_matches(limit=10)
        assert len(all_matches) == 2
        assert all_matches[0]["home_team"] == "Gefle IF"  # Most recent first

        # Filter by league
        se1_matches = await db.get_matches(league="SE1")
        assert len(se1_matches) == 2

        # Count
        count = await db.match_count("SE1")
        assert count == 2

        await db.close()

    async def test_duplicate_insertion_skipped(self):
        db = Database(":memory:")
        await db.connect()

        match = [{
            "source": "football-data.co.uk",
            "league": "SE1",
            "season": "2425",
            "date": "2025-04-15",
            "home_team": "IK Brage",
            "away_team": "Degerfors IF",
            "home_goals": 2,
            "away_goals": 1,
            "result": "H",
        }]

        inserted1, _ = await db.insert_matches(match)
        inserted2, duplicates = await db.insert_matches(match)

        assert inserted1 == 1
        assert inserted2 == 0
        assert duplicates == 1

        count = await db.match_count()
        assert count == 1

        await db.close()

    async def test_collection_logging(self):
        db = Database(":memory:")
        await db.connect()

        log_id = await db.log_collection_start("football-data.co.uk", "SE1", "2425")
        assert isinstance(log_id, int)
        assert log_id > 0

        await db.log_collection_end(log_id, 100, 5, 0, "completed")

        # Verify by querying
        from football_data.database.sqlite import sqlite3
        conn = db._conn
        row = conn.execute("SELECT * FROM collection_log WHERE id = ?", (log_id,)).fetchone()
        assert row["status"] == "completed"
        assert row["rows_collected"] == 100

        await db.close()

    async def test_get_statistics(self):
        db = Database(":memory:")
        await db.connect()

        matches = [
            {
                "source": "test",
                "league": "SE1",
                "date": "2025-04-15",
                "home_team": "Team A",
                "away_team": "Team B",
            },
            {
                "source": "test",
                "league": "SE1",
                "date": "2025-04-16",
                "home_team": "Team C",
                "away_team": "Team D",
            },
            {
                "source": "test",
                "league": "NO2",
                "date": "2025-04-15",
                "home_team": "Team E",
                "away_team": "Team F",
            },
        ]

        await db.insert_matches(matches)
        stats = await db.get_statistics()

        assert stats["total_matches"] == 3
        assert stats["by_league"] == {"SE1": 2, "NO2": 1}
        assert stats["unique_teams"] == 6
        assert stats["date_range"]["from"] is not None
        assert stats["date_range"]["to"] is not None

        await db.close()
