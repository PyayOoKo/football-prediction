"""
Partition Data Migration — migrate matches table to partitioned structure.

Migrates data from the existing ``matches`` table to the new
partitioned ``matches_new`` table (created by migration 002)
in batches with progress reporting.

Usage
-----
::

    # Dry-run: show what would be migrated without writing
    python scripts/migrate_to_partitions.py --dry-run

    # Full migration with 10,000-row batches
    python scripts/migrate_to_partitions.py --batch-size 10000 --commit-every 50

    # Rollback: move data back to the old table
    python scripts/migrate_to_partitions.py --rollback

Process
-------
1. Validate that both ``matches`` (old) and ``matches_new`` (partitioned) exist
2. Insert data from old table into partitioned table in batches
3. Rebuild indexes and analyze the new table
4. Rename tables: matches → matches_legacy, matches_new → matches
5. Recreate FKs that reference matches.id
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_connection(dsn: str) -> Any:
    """Create a database connection for batch operations."""
    conn = psycopg2.connect(dsn)
    conn.set_session(autocommit=False)
    return conn


def validate_tables(cur: Any) -> dict[str, bool]:
    """Validate that required tables exist."""
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN ('matches', 'matches_new')
    """)
    existing = {row[0] for row in cur.fetchall()}
    return {
        "matches": "matches" in existing,
        "matches_new": "matches_new" in existing,
    }


def count_rows(cur: Any, table: str) -> int:
    """Get approximate row count for a table."""
    cur.execute(f"SELECT reltuples::bigint FROM pg_class WHERE relname = '{table}'")
    row = cur.fetchone()
    return row[0] if row else 0


def migrate_batch(
    cur: Any,
    batch_size: int = 10000,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Migrate data from matches to matches_new in batches.

    Returns
    -------
    tuple[int, int]
        (rows_migrated, total_rows)
    """
    # Get total rows in old table
    cur.execute("SELECT COUNT(*) FROM matches")
    total = cur.fetchone()[0]

    if total == 0:
        logger.info("No rows to migrate.")
        return 0, 0

    logger.info("Total rows to migrate: %s", f"{total:,}")

    if dry_run:
        logger.info("[DRY RUN] Would migrate %s rows in batches of %s",
                     f"{total:,}", f"{batch_size:,}")
        return total, total

    # Migrate in batches
    migrated = 0
    batch_num = 0
    start_time = time.perf_counter()

    cur.execute("""
        SELECT setval(
            'matches_new_id_seq',
            COALESCE((SELECT MAX(id) FROM matches), 1)
        )
    """)

    while migrated < total:
        batch_num += 1
        batch_start = time.perf_counter()

        cur.execute(f"""
            INSERT INTO matches_new
            SELECT * FROM matches
            WHERE id > %s
            ORDER BY id
            LIMIT %s
            ON CONFLICT (id, match_date) DO NOTHING
        """, (migrated, batch_size))

        rows_this_batch = cur.rowcount
        migrated += rows_this_batch

        elapsed = time.perf_counter() - batch_start
        rate = rows_this_batch / elapsed if elapsed > 0 else 0

        logger.info(
            "  Batch %3d: %s rows (%s/s) — %5.1f%% complete",
            batch_num,
            f"{rows_this_batch:,}",
            f"{rate:,.0f}",
            migrated / total * 100,
        )

    total_elapsed = time.perf_counter() - start_time
    logger.info(
        "Migration complete: %s rows in %.1fs (%.0f rows/s)",
        f"{migrated:,}",
        total_elapsed,
        migrated / total_elapsed if total_elapsed > 0 else 0,
    )

    return migrated, total


def rebuild_indexes(cur: Any) -> None:
    """Rebuild indexes and analyze the new partitioned table."""
    logger.info("Rebuilding indexes on matches_new...")

    cur.execute("""
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'matches_new'
          AND indexname NOT LIKE '%_pkey'
    """)
    indexes = [row[0] for row in cur.fetchall()]

    for idx in indexes:
        cur.execute(f"REINDEX INDEX {idx}")
        logger.info("  Reindexed: %s", idx)

    cur.execute("ANALYZE matches_new")
    logger.info("  Analyzed matches_new")


def swap_tables(cur: Any) -> None:
    """Rename old table to matches_legacy, new table to matches."""
    logger.info("Swapping tables...")

    # Drop old table if it was already renamed
    cur.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_class WHERE relname = 'matches_legacy') THEN
                DROP TABLE matches_legacy CASCADE;
            END IF;
        END $$;
    """)

    # Rename: matches → matches_legacy, matches_new → matches
    cur.execute("ALTER TABLE matches RENAME TO matches_legacy")
    cur.execute("ALTER TABLE matches_new RENAME TO matches")

    logger.info("  matches → matches_legacy")
    logger.info("  matches_new → matches")


def recreate_foreign_keys(cur: Any) -> None:
    """Recreate FKs that reference matches.id after the rename."""
    logger.info("Recreating foreign keys...")

    fks = [
        ("odds", "match_id", "fk_odds_match_id"),
        ("match_statistics", "match_id", "fk_match_statistics_match_id"),
        ("weather", "match_id", "fk_weather_match_id"),
        ("lineups", "match_id", "fk_lineups_match_id"),
        ("player_match_stats", "match_id", "fk_pms_match_id"),
        ("predictions", "match_id", "fk_predictions_match_id"),
        ("expected_value_bets", "match_id", "fk_ev_bets_match_id"),
        ("closing_line_values", "match_id", "fk_clv_match_id"),
        ("betting_results", "match_id", "fk_betting_results_match_id"),
        ("team_elo_history", "match_id", "fk_team_elo_match_id"),
        ("team_form", "match_id", "fk_team_form_match_id"),
        ("team_xg_history", "match_id", "fk_team_xg_match_id"),
    ]

    for table, column, fk_name in fks:
        try:
            cur.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {fk_name}")
            cur.execute(f"""
                ALTER TABLE {table} ADD CONSTRAINT {fk_name}
                FOREIGN KEY ({column}) REFERENCES matches(id) ON DELETE CASCADE
            """)
            logger.info("  %s.%s → matches.id", table, column)
        except Exception as exc:
            logger.warning("  Could not recreate FK %s: %s", fk_name, exc)


def run_rollback(cur: Any) -> None:
    """Rollback: restore data from matches_legacy if available."""
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'matches_legacy'
    """)
    if not cur.fetchone():
        logger.error("No matches_legacy table found — cannot rollback.")
        return

    logger.info("Rolling back: restoring matches from matches_legacy...")
    cur.execute("ALTER TABLE matches RENAME TO matches_new")
    cur.execute("ALTER TABLE matches_legacy RENAME TO matches")
    recreate_foreign_keys(cur)
    logger.info("Rollback complete.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate matches data to partitioned table",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Database DSN (default: reads from .env DATABASE_URL)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Rows per batch (default: 10000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback: restore matches from matches_legacy",
    )
    parser.add_argument(
        "--skip-swap",
        action="store_true",
        help="Skip the table rename step (for testing)",
    )
    args = parser.parse_args()

    # Get DSN
    dsn = args.dsn
    if not dsn:
        try:
            from src.config.settings import config
            dsn = config.db.sa_url
        except ImportError:
            import os
            dsn = os.environ.get("DATABASE_URL")

    if not dsn:
        logger.error("No DATABASE_URL found. Set it in .env or pass --dsn.")
        return 1

    try:
        conn = get_connection(dsn)
        cur = conn.cursor()

        # Validate
        tables = validate_tables(cur)
        logger.info("Table status: matches=%s, matches_new=%s",
                     tables["matches"], tables["matches_new"])

        if args.rollback:
            run_rollback(cur)
            conn.commit()
            return 0

        if not tables["matches"]:
            logger.error("Source table 'matches' does not exist.")
            return 1

        if not tables["matches_new"]:
            logger.error(
                "Target table 'matches_new' does not exist.\n"
                "Run migration 002 first: alembic upgrade 002"
            )
            return 1

        # Validate partitioned structure
        cur.execute("""
            SELECT relispartition FROM pg_class WHERE relname = 'matches_new'
        """)
        is_partitioned = cur.fetchone()
        if is_partitioned:
            logger.info("matches_new is a partitioned table ✓")

        # Count rows
        old_count = count_rows(cur, "matches")
        new_count = count_rows(cur, "matches_new")
        logger.info("Rows: matches=%s, matches_new=%s",
                     f"{old_count:,}", f"{new_count:,}")

        if old_count == 0 and new_count > 0:
            logger.info("Data already migrated — skipping.")
            return 0

        if old_count == 0 and new_count == 0:
            logger.info("Both tables empty — nothing to do.")
            return 0

        # Migrate
        migrated, total = migrate_batch(cur, args.batch_size, args.dry_run)
        if args.dry_run:
            return 0

        if migrated == 0:
            logger.warning("No rows migrated.")
            conn.rollback()
            return 1

        # Rebuild indexes
        rebuild_indexes(cur)

        # Swap tables
        if not args.skip_swap:
            swap_tables(cur)
            recreate_foreign_keys(cur)
        else:
            logger.info("Skipping table swap (--skip-swap)")

        # Commit
        conn.commit()

        logger.info("✅ Migration complete!")
        return 0

    except Exception as exc:
        logger.error("Migration failed: %s", exc, exc_info=True)
        try:
            conn.rollback()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
