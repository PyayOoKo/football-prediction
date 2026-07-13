"""
Update connection pool and statement timeout configuration.

Revision ID: 004
Revises: 003
Create Date: 2026-07-13

Changes
-------
1. Set PostgreSQL configuration parameters for the session:
   - statement_timeout: 300s (prevents runaway queries)
   - idle_in_transaction_session_timeout: 10min
   - lock_timeout: 30s
2. Add application_name for pg_stat_activity tracking
3. Update connection pool settings via the application config (Python)
4. Add monitoring views for query performance

Note
----
This migration applies session-level defaults. Actual pool settings
(pool_size, max_overflow) are configured in src/config/settings.py
and src/database/session.py, which are listed in the report.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ══════════════════════════════════════════════════════
    # 1. Set database-level defaults
    # ══════════════════════════════════════════════════════
    # These apply to all new connections. Existing connections
    # can apply them via:
    #   ALTER DATABASE football_prediction SET ...
    op.execute("""
        ALTER DATABASE football_prediction
        SET statement_timeout = '300000'
    """)  # 5 minutes — prevent runaway queries

    op.execute("""
        ALTER DATABASE football_prediction
        SET idle_in_transaction_session_timeout = '600000'
    """)  # 10 minutes — release stuck transactions

    op.execute("""
        ALTER DATABASE football_prediction
        SET lock_timeout = '30000'
    """)  # 30 seconds — fail fast on lock contention

    op.execute("""
        ALTER DATABASE football_prediction
        SET application_name = 'football-prediction'
    """)

    op.execute("""
        ALTER DATABASE football_prediction
        SET default_statistics_target = '500'
    """)  # Better query plans for large tables

    # ══════════════════════════════════════════════════════
    # 2. Create monitoring views
    # ══════════════════════════════════════════════════════
    op.execute("""
        CREATE OR REPLACE VIEW v_slow_queries AS
        SELECT pid,
               now() - pg_stat_activity.query_start AS duration,
               query,
               state,
               wait_event_type,
               wait_event,
               application_name
        FROM pg_stat_activity
        WHERE state != 'idle'
          AND now() - pg_stat_activity.query_start > interval '5 seconds'
        ORDER BY duration DESC
    """)

    op.execute("""
        CREATE OR REPLACE VIEW v_table_bloat AS
        SELECT schemaname,
               tablename,
               pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
               n_dead_tup AS dead_tuples,
               n_live_tup AS live_tuples,
               CASE WHEN n_live_tup > 0
                    THEN round(n_dead_tup::numeric / n_live_tup * 100, 2)
                    ELSE 0
               END AS dead_pct,
               last_autovacuum,
               last_autoanalyze
        FROM pg_stat_user_tables
        WHERE n_dead_tup > 10000
        ORDER BY n_dead_tup DESC
    """)

    op.execute("""
        CREATE OR REPLACE VIEW v_index_usage AS
        SELECT schemaname,
               tablename,
               indexname,
               idx_scan,
               idx_tup_read,
               idx_tup_fetch,
               pg_size_pretty(pg_relation_size(
                   (schemaname||'.'||indexname)::regclass
               )) AS index_size
        FROM pg_stat_user_indexes
        ORDER BY idx_scan ASC
        LIMIT 50
    """)

    op.execute("""
        CREATE OR REPLACE VIEW v_seq_scans AS
        SELECT schemaname,
               tablename,
               seq_scan,
               seq_tup_read,
               idx_scan,
               round(seq_tup_read::numeric /
                     NULLIF(seq_tup_read + idx_tup_fetch, 0) * 100, 2
               ) AS pct_seq_reads
        FROM pg_stat_user_tables
        WHERE seq_tup_read > 1000000
        ORDER BY seq_tup_read DESC
    """)

    op.execute("""
        CREATE OR REPLACE VIEW v_table_sizes AS
        SELECT schemaname,
               tablename,
               pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total,
               pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table,
               pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) -
                              pg_relation_size(schemaname||'.'||tablename)) AS indexes,
               n_live_tup AS estimated_rows
        FROM pg_stat_user_tables
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
    """)

    # ══════════════════════════════════════════════════════
    # 3. Create maintenance function for partitioned tables
    # ══════════════════════════════════════════════════════
    op.execute("""
        CREATE OR REPLACE FUNCTION create_next_match_partition()
        RETURNS void AS $$
        DECLARE
            next_year text;
            partition_name text;
        BEGIN
            next_year := to_char(CURRENT_DATE + interval '1 year', 'YYYY');
            partition_name := 'matches_' || next_year;

            IF NOT EXISTS (
                SELECT 1 FROM pg_class WHERE relname = partition_name
            ) THEN
                EXECUTE format(
                    'CREATE TABLE %I PARTITION OF matches
                     FOR VALUES FROM (%L) TO (%L)',
                    partition_name,
                    next_year || '-01-01',
                    (substr(next_year, 1, 4)::int + 1) || '-01-01'
                );
                RAISE NOTICE 'Created partition: %', partition_name;
            END IF;
        END;
        $$ LANGUAGE plpgsql;
    """)


def downgrade() -> None:
    """Remove monitoring views and reset database defaults."""

    # Drop views
    views = [
        "v_slow_queries",
        "v_table_bloat",
        "v_index_usage",
        "v_seq_scans",
        "v_table_sizes",
    ]
    for view in views:
        op.execute(f"DROP VIEW IF EXISTS {view}")

    # Drop maintenance function
    op.execute("DROP FUNCTION IF EXISTS create_next_match_partition()")

    # Reset database configuration
    for param in [
        "statement_timeout",
        "idle_in_transaction_session_timeout",
        "lock_timeout",
        "application_name",
        "default_statistics_target",
    ]:
        op.execute(f"ALTER DATABASE football_prediction RESET {param}")
