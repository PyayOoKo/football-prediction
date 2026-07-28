"""
Async SQLite database layer.

Provides thread-safe, connection-pooled access to the football data
SQLite database. Tables are auto-created on first connection.

Schema
------
matches         — Core match results with odds
team_stats      — Per-match team statistics
player_stats    — Per-match player statistics  
injuries        — Player injury records
weather         — Weather conditions for matches
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_data.config import config

logger = logging.getLogger(__name__)

# ── Schema DDL ──────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS matches (
    match_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL DEFAULT 'football-data.co.uk',
    league          TEXT NOT NULL,
    season          TEXT,
    date            TEXT NOT NULL,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    home_goals      INTEGER,
    away_goals      INTEGER,
    result          TEXT CHECK(result IN ('H', 'D', 'A', NULL)),
    home_odds       REAL,
    draw_odds       REAL,
    away_odds       REAL,
    home_shots      INTEGER,
    away_shots      INTEGER,
    home_shots_target INTEGER,
    away_shots_target INTEGER,
    home_corners    INTEGER,
    away_corners    INTEGER,
    home_fouls      INTEGER,
    away_fouls      INTEGER,
    home_yellow     INTEGER,
    away_yellow     INTEGER,
    home_red        INTEGER,
    away_red        INTEGER,
    home_xg         REAL,
    away_xg         REAL,
    over25_odds     REAL,
    under25_odds    REAL,
    ingested_at     TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source, league, date, home_team, away_team)
);

CREATE TABLE IF NOT EXISTS team_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id        INTEGER NOT NULL REFERENCES matches(match_id),
    team            TEXT NOT NULL,
    is_home         INTEGER NOT NULL CHECK(is_home IN (0, 1)),
    shots           INTEGER,
    shots_on_target INTEGER,
    corners         INTEGER,
    yellow_cards    INTEGER,
    red_cards       INTEGER,
    fouls           INTEGER,
    possession      REAL,
    ingested_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS player_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id        INTEGER NOT NULL REFERENCES matches(match_id),
    player          TEXT NOT NULL,
    team            TEXT NOT NULL,
    minutes         INTEGER,
    rating          REAL,
    goals           INTEGER DEFAULT 0,
    assists         INTEGER DEFAULT 0,
    xg              REAL,
    xa              REAL,
    shots           INTEGER,
    key_passes      INTEGER,
    ingested_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS injuries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    team            TEXT NOT NULL,
    player          TEXT NOT NULL,
    injury          TEXT,
    expected_return TEXT,
    recorded_at     TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(team, player, recorded_at)
);

CREATE TABLE IF NOT EXISTS weather (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id        INTEGER NOT NULL REFERENCES matches(match_id),
    temperature     REAL,
    feels_like      REAL,
    humidity        REAL,
    precipitation   REAL,
    wind_speed      REAL,
    wind_gust       REAL,
    pressure        REAL,
    condition       TEXT,
    ingested_at     TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(match_id)
);

CREATE TABLE IF NOT EXISTS collection_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,
    league          TEXT,
    season          TEXT,
    rows_collected  INTEGER DEFAULT 0,
    rows_duplicates INTEGER DEFAULT 0,
    errors          INTEGER DEFAULT 0,
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    status          TEXT CHECK(status IN ('running', 'completed', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league);
CREATE INDEX IF NOT EXISTS idx_matches_team ON matches(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_team_stats_match ON team_stats(match_id);
CREATE INDEX IF NOT EXISTS idx_player_stats_match ON player_stats(match_id);
CREATE INDEX IF NOT EXISTS idx_injuries_team ON injuries(team);
CREATE INDEX IF NOT EXISTS idx_weather_match ON weather(match_id);
"""


class Database:
    """Async SQLite database manager with connection pooling.

    Uses a single persistent connection with WAL mode for
    concurrent read performance. All write operations are
    serialized through an asyncio lock.

    Usage
    -----
    >>> db = Database()
    >>> await db.connect()
    >>> await db.insert_matches([...])
    >>> await db.close()
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or config.database.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._connected = False

    # ── Connection management ────────────────────────────

    async def connect(self) -> None:
        """Open the database connection and create schema if needed."""
        if self._connected:
            return

        loop = asyncio.get_event_loop()
        self._conn = await loop.run_in_executor(None, self._connect_sync)
        self._connected = True
        logger.info("Connected to database: %s", self.db_path)

    def _connect_sync(self) -> sqlite3.Connection:
        """Synchronous connection setup."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        # Create tables
        conn.executescript(SCHEMA_SQL)
        # Migrate existing DBs: add xG columns if missing
        migs = [
            ("ALTER TABLE matches ADD COLUMN home_xg REAL", "home_xg"),
            ("ALTER TABLE matches ADD COLUMN away_xg REAL", "away_xg"),
            ("ALTER TABLE matches ADD COLUMN over25_odds REAL", "over25_odds"),
            ("ALTER TABLE matches ADD COLUMN under25_odds REAL", "under25_odds"),
        ]
        existing = {r[0] for r in conn.execute("PRAGMA table_info(matches)").fetchall()}
        for sql, col in migs:
            if col not in existing:
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError:
                    pass
        conn.commit()
        return conn

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._conn.close)
            self._conn = None
            self._connected = False
            logger.info("Database connection closed")

    async def __aenter__(self) -> "Database":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ── Connection check ─────────────────────────────────

    async def ensure_connected(self) -> None:
        """Ensure the database is connected, reconnect if needed."""
        if not self._connected or not self._conn:
            await self.connect()

    # ── Insert operations ────────────────────────────────

    async def insert_matches(
        self, matches: list[dict[str, Any]]
    ) -> tuple[int, int]:
        """Insert match records, skipping duplicates.

        Parameters
        ----------
        matches : list[dict]
            List of match dicts with keys matching the matches table columns.

        Returns
        -------
        tuple[int, int]
            (rows_inserted, rows_duplicate)
        """
        if not matches:
            return 0, 0

        await self.ensure_connected()
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._insert_matches_sync, matches
            )

    def _insert_matches_sync(
        self, matches: list[dict[str, Any]]
    ) -> tuple[int, int]:
        """Synchronous batch insert for matches."""
        assert self._conn is not None

        sql = """
        INSERT OR IGNORE INTO matches
            (source, league, season, date, home_team, away_team,
             home_goals, away_goals, result, home_odds, draw_odds, away_odds,
             home_shots, away_shots, home_shots_target, away_shots_target,
             home_corners, away_corners, home_fouls, away_fouls,
             home_yellow, away_yellow, home_red, away_red,
             home_xg, away_xg,
             over25_odds, under25_odds)
        VALUES
             (?, ?, ?, ?, ?, ?,
              ?, ?, ?, ?, ?, ?,
              ?, ?, ?, ?,
              ?, ?, ?, ?,
              ?, ?, ?, ?,
              ?, ?,
              ?, ?)
        """

        before = self._count_matches()
        rows = []
        for m in matches:
            rows.append((
                m.get("source", "football-data.co.uk"),
                m.get("league"),
                m.get("season"),
                m.get("date"),
                m.get("home_team"),
                m.get("away_team"),
                m.get("home_goals"),
                m.get("away_goals"),
                m.get("result"),
                m.get("home_odds"),
                m.get("draw_odds"),
                m.get("away_odds"),
                m.get("home_shots"),
                m.get("away_shots"),
                m.get("home_shots_target"),
                m.get("away_shots_target"),
                m.get("home_corners"),
                m.get("away_corners"),
                m.get("home_fouls"),
                m.get("away_fouls"),
                m.get("home_yellow"),
                m.get("away_yellow"),
                m.get("home_red"),
                m.get("away_red"),
                m.get("home_xg"),
                m.get("away_xg"),
                m.get("over25_odds"),
                m.get("under25_odds"),
            ))

        cursor = self._conn.executemany(sql, rows)
        self._conn.commit()
        after = self._count_matches()
        inserted = after - before
        duplicates = len(matches) - inserted
        return inserted, duplicates

    def _count_matches(self) -> int:
        assert self._conn is not None
        return self._conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]

    async def insert_weather(
        self, weather_records: list[dict[str, Any]]
    ) -> int:
        """Insert weather records, skipping duplicates."""
        if not weather_records:
            return 0

        await self.ensure_connected()
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._insert_weather_sync, weather_records
            )

    def _insert_weather_sync(
        self, records: list[dict[str, Any]]
    ) -> int:
        assert self._conn is not None

        sql = """
        INSERT OR IGNORE INTO weather
            (match_id, temperature, feels_like, humidity,
             precipitation, wind_speed, wind_gust, pressure, condition)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        rows = [
            (
                r.get("match_id"),
                r.get("temperature"),
                r.get("feels_like"),
                r.get("humidity"),
                r.get("precipitation"),
                r.get("wind_speed"),
                r.get("wind_gust"),
                r.get("pressure"),
                r.get("condition"),
            )
            for r in records
        ]

        cursor = self._conn.executemany(sql, rows)
        self._conn.commit()
        return cursor.rowcount

    # ── Query operations ─────────────────────────────────

    async def get_matches(
        self,
        league: str | None = None,
        team: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query matches with optional filters."""
        await self.ensure_connected()

        conditions: list[str] = []
        params: list[Any] = []

        if league:
            conditions.append("league = ?")
            params.append(league)
        if team:
            conditions.append("(home_team = ? OR away_team = ?)")
            params.extend([team, team])

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""
        SELECT * FROM matches
        WHERE {where}
        ORDER BY date DESC
        LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(
            None, self._query_sync, sql, params
        )
        return [dict(row) for row in rows]

    def _query_sync(
        self, sql: str, params: list[Any]
    ) -> list[sqlite3.Row]:
        assert self._conn is not None
        return self._conn.execute(sql, params).fetchall()

    async def match_count(self, league: str | None = None) -> int:
        """Get total match count, optionally filtered by league."""
        await self.ensure_connected()
        if league:
            sql = "SELECT COUNT(*) FROM matches WHERE league = ?"
            params = [league]
        else:
            sql = "SELECT COUNT(*) FROM matches"
            params = []

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self._conn.execute(sql, params).fetchone()[0]  # type: ignore
        )
        return result

    # ── Collection logging ───────────────────────────────

    async def log_collection_start(
        self, source: str, league: str | None, season: str | None
    ) -> int:
        """Log the start of a collection run. Returns log ID."""
        await self.ensure_connected()
        async with self._lock:
            loop = asyncio.get_event_loop()
            log_id = await loop.run_in_executor(
                None, self._log_collection_start_sync, source, league, season
            )
            return log_id

    def _log_collection_start_sync(
        self, source: str, league: str | None, season: str | None
    ) -> int:
        assert self._conn is not None
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            """INSERT INTO collection_log
               (source, league, season, started_at, status)
               VALUES (?, ?, ?, ?, 'running')""",
            (source, league, season, now),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def log_collection_end(
        self,
        log_id: int,
        rows_collected: int = 0,
        rows_duplicates: int = 0,
        errors: int = 0,
        status: str = "completed",
    ) -> None:
        """Mark a collection run as completed."""
        await self.ensure_connected()
        async with self._lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._log_collection_end_sync,
                log_id,
                rows_collected,
                rows_duplicates,
                errors,
                status,
            )

    def _log_collection_end_sync(
        self,
        log_id: int,
        rows_collected: int,
        rows_duplicates: int,
        errors: int,
        status: str,
    ) -> None:
        assert self._conn is not None
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """UPDATE collection_log SET
               completed_at = ?, rows_collected = ?, rows_duplicates = ?,
               errors = ?, status = ?
               WHERE id = ?""",
            (now, rows_collected, rows_duplicates, errors, status, log_id),
        )
        self._conn.commit()

    # ── Dedup and cleanup ────────────────────────────────

    async def remove_duplicates(self) -> int:
        """Remove any duplicate match records, keeping the first.

        Returns
        -------
        int
            Number of duplicates removed.
        """
        await self.ensure_connected()
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._remove_duplicates_sync)

    def _remove_duplicates_sync(self) -> int:
        assert self._conn is not None
        # SQLite: delete duplicates keeping the lowest match_id
        cursor = self._conn.execute("""
            DELETE FROM matches WHERE match_id NOT IN (
                SELECT MIN(match_id) FROM matches
                GROUP BY source, league, date, home_team, away_team
            )
        """)
        self._conn.commit()
        return cursor.rowcount

    async def get_statistics(self) -> dict[str, Any]:
        """Get database statistics."""
        await self.ensure_connected()
        loop = asyncio.get_event_loop()

        def _stats() -> dict[str, Any]:
            assert self._conn is not None
            total = self._conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            leagues = self._conn.execute(
                "SELECT league, COUNT(*) as cnt FROM matches GROUP BY league ORDER BY cnt DESC"
            ).fetchall()
            date_range = self._conn.execute(
                "SELECT MIN(date), MAX(date) FROM matches"
            ).fetchone()
            team_count = self._conn.execute(
                """SELECT COUNT(DISTINCT team) FROM (
                    SELECT home_team AS team FROM matches
                    UNION
                    SELECT away_team AS team FROM matches
                )"""
            ).fetchone()[0]
            return {
                "total_matches": total,
                "by_league": {r["league"]: r["cnt"] for r in leagues},
                "date_range": {
                    "from": date_range[0],
                    "to": date_range[1],
                },
                "unique_teams": team_count,
            }

        return await loop.run_in_executor(None, _stats)
