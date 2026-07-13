"""
Tune fillfactor, sequence caches, and autovacuum for write-heavy workloads.

Revision ID: 003
Revises: 002
Create Date: 2026-07-13

Changes
-------
1. Set fillfactor=90 on insert-heavy tables (odds, player_match_stats) to
   leave space for HOT updates and reduce page splits
2. Set fillfactor=70 on update-heavy tables (matches, team_elo_history) to
   accommodate updated_at + result changes without page splits
3. Tune sequence cache to 1000 for all tables expected to exceed 1M rows
4. Enable parallel workers for table creation and index building
5. Add table-level autovacuum tuning for predictable maintenance

Background
----------
fillfactor controls how full a page is when first written:
- 100 (default): Full page → UPDATE causes page split (expensive)
- 90: 10% free space → UPDATE can use same page (HOT update)
- 70: 30% free space → Multiple updates before page split

For football data:
- matches: updated frequently (result, home_goals added post-match) → 70
- odds: rarely updated (append-mostly) → 90
- player_match_stats: rarely updated (append-mostly) → 90
- team_elo_history: rarely updated (append-mostly) → 90
- team_form: rare updates (pre-computed) → 90
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_FILLFACTOR_UPDATE_HEAVY = 70   # matches, team analytics
_FILLFACTOR_INSERT_HEAVY = 90   # odds, pms, weather, lineups


def upgrade() -> None:
    # ══════════════════════════════════════════════════════
    # 1. Fillfactor — Update-heavy tables
    # ══════════════════════════════════════════════════════
    # matches: updated when results come in, status changes
    op.execute(f"ALTER TABLE matches SET (fillfactor = {_FILLFACTOR_UPDATE_HEAVY})")

    # team_elo_history: rare updates, but leave room
    op.execute(f"ALTER TABLE team_elo_history SET (fillfactor = {_FILLFACTOR_UPDATE_HEAVY})")

    # team_form: rare updates
    op.execute(f"ALTER TABLE team_form SET (fillfactor = {_FILLFACTOR_UPDATE_HEAVY})")

    # team_xg_history: rare updates
    op.execute(f"ALTER TABLE team_xg_history SET (fillfactor = {_FILLFACTOR_UPDATE_HEAVY})")

    # predictions: rare updates (backfill corrections)
    op.execute(f"ALTER TABLE predictions SET (fillfactor = {_FILLFACTOR_UPDATE_HEAVY})")

    # expected_value_bets: rare updates
    op.execute(f"ALTER TABLE expected_value_bets SET (fillfactor = {_FILLFACTOR_UPDATE_HEAVY})")

    # betting_results: rare updates
    op.execute(f"ALTER TABLE betting_results SET (fillfactor = {_FILLFACTOR_UPDATE_HEAVY})")

    # ══════════════════════════════════════════════════════
    # 2. Fillfactor — Insert-heavy (append-mostly) tables
    # ══════════════════════════════════════════════════════
    op.execute(f"ALTER TABLE odds SET (fillfactor = {_FILLFACTOR_INSERT_HEAVY})")
    op.execute(f"ALTER TABLE player_match_stats SET (fillfactor = {_FILLFACTOR_INSERT_HEAVY})")
    op.execute(f"ALTER TABLE weather SET (fillfactor = {_FILLFACTOR_INSERT_HEAVY})")
    op.execute(f"ALTER TABLE lineups SET (fillfactor = {_FILLFACTOR_INSERT_HEAVY})")
    op.execute(f"ALTER TABLE injuries SET (fillfactor = {_FILLFACTOR_INSERT_HEAVY})")
    op.execute(f"ALTER TABLE transfers SET (fillfactor = {_FILLFACTOR_INSERT_HEAVY})")
    op.execute(f"ALTER TABLE closing_line_values SET (fillfactor = {_FILLFACTOR_INSERT_HEAVY})")

    # ══════════════════════════════════════════════════════
    # 3. Sequence cache — already set in migration 002 for the
    #    largest tables; ensure remaining sequences are tuned
    # ══════════════════════════════════════════════════════
    # Note: matches_id_seq, odds_id_seq, player_match_stats_id_seq,
    # predictions_id_seq, and expected_value_bets_id_seq were tuned
    # in migration 002.
    op.execute("ALTER SEQUENCE betting_results_id_seq CACHE 1000")

    # ══════════════════════════════════════════════════════
    # 4. Autovacuum — Aggressive settings for big tables
    # ══════════════════════════════════════════════════════
    # These tables have frequent updates → need frequent vacuuming
    big_tables = [
        "matches", "odds", "player_match_stats", "predictions",
        "expected_value_bets", "betting_results", "team_elo_history",
        "team_form", "team_xg_history",
    ]

    for table in big_tables:
        op.execute(f"""
            ALTER TABLE {table} SET (
                autovacuum_vacuum_scale_factor = 0.01,
                autovacuum_analyze_scale_factor = 0.005,
                autovacuum_vacuum_threshold = 100000,
                autovacuum_analyze_threshold = 50000,
                autovacuum_vacuum_cost_delay = 5,
                autovacuum_vacuum_cost_limit = 1000
            )
        """)

    # ══════════════════════════════════════════════════════
    # 5. Toast storage for JSONB columns
    # ══════════════════════════════════════════════════════
    # Lineups stores starting_xi and substitutes as JSONB. For large lineups
    # datasets, use TOAST compression to save storage.
    op.execute("""
        ALTER TABLE lineups ALTER COLUMN starting_xi SET STORAGE EXTENDED
    """)
    op.execute("""
        ALTER TABLE lineups ALTER COLUMN substitutes SET STORAGE EXTENDED
    """)

    # ══════════════════════════════════════════════════════
    # 6. Rebuild indexes to apply fillfactor (requires exclusive lock)
    # ══════════════════════════════════════════════════════
    # Fillfactor only applies to new pages. Existing pages retain their
    # current fill. To apply to all existing data, rebuild affected tables.
    # This is optional and can be done via pg_repack or CLUSTER.
    op.execute("""
        DO $$ BEGIN
            RAISE NOTICE 'To apply fillfactor to existing data, run:';
            RAISE NOTICE '  CLUSTER matches USING ix_matches_match_date;';
            RAISE NOTICE 'Or use pg_repack for zero-downtime:';
            RAISE NOTICE '  pg_repack -t matches -t odds -t player_match_stats';
        END $$;
    """)


def downgrade() -> None:
    """Reset fillfactor, sequence cache, and autovacuum to defaults."""

    # Reset fillfactor to default (100) for all tables
    tables = [
        "matches", "team_elo_history", "team_form", "team_xg_history",
        "predictions", "expected_value_bets", "betting_results",
        "odds", "player_match_stats", "weather", "lineups",
        "injuries", "transfers", "closing_line_values",
    ]
    for table in tables:
        op.execute(f"ALTER TABLE {table} RESET (fillfactor)")

    # Reset autovacuum for big tables — reset each parameter individually
    big_tables = [
        "matches", "odds", "player_match_stats", "predictions",
        "expected_value_bets", "betting_results",
    ]
    autovacuum_params = [
        "autovacuum_vacuum_scale_factor",
        "autovacuum_analyze_scale_factor",
        "autovacuum_vacuum_threshold",
        "autovacuum_analyze_threshold",
        "autovacuum_vacuum_cost_delay",
        "autovacuum_vacuum_cost_limit",
    ]
    for table in big_tables:
        for param in autovacuum_params:
            op.execute(f"ALTER TABLE {table} RESET ({param})")

    # Reset sequence cache to default (1)
    op.execute("ALTER SEQUENCE betting_results_id_seq CACHE 1")
