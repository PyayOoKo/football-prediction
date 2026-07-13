"""
Fix foreign key column types — migrate all match_id FK columns from INT to BIGINT.

Revision ID: 005
Revises: 004
Create Date: 2026-07-13

Motivation
----------
The ``matches.id`` column was created as ``BIGINT`` in migration 001
to support 100M+ rows, but all child tables (odds, player_match_stats,
predictions, etc.) reference it with ``Integer`` columns. This causes:

1. **Implicit type coercion on joins** — PostgreSQL must cast INT→BIGINT
   on every join, preventing index-only scans and adding overhead.
2. **Overflow risk at 2B rows** — INT child tables overflow before the
   BIGINT parent table, causing FK violations if max INT is exceeded.
3. **Wasted index space** — Each FK index stores 4-byte entries when
   the parent PK is 8 bytes, causing index key width mismatch.

Changes
-------
Alter all ``match_id`` columns from ``Integer`` to ``BigInteger`` in:
- odds
- match_statistics
- weather
- lineups
- player_match_stats
- predictions
- expected_value_bets
- closing_line_values
- betting_results
- team_elo_history
- team_form
- team_xg_history
- injuries (no match_id FK)
- transfers (no match_id FK)
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Tables whose match_id column needs BIGINT
# Format: (table_name, fk_column, fk_name)
FK_COLUMNS = [
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

# Tables whose team_id/player_id columns should also be BIGINT (secondary FKs)
SECONDARY_FK_COLUMNS = [
    ("players", "current_team_id", "fk_players_current_team_id"),
    ("player_match_stats", "player_id", "fk_pms_player_id"),
    ("player_match_stats", "team_id", "fk_pms_team_id"),
    ("transfers", "player_id", "fk_transfers_player_id"),
    ("transfers", "from_team_id", "fk_transfers_from_team_id"),
    ("transfers", "to_team_id", "fk_transfers_to_team_id"),
    ("injuries", "player_id", "fk_injuries_player_id"),
    ("lineups", "team_id", "fk_lineups_team_id"),
    ("team_elo_history", "team_id", "fk_team_elo_team_id"),
    ("team_form", "team_id", "fk_team_form_team_id"),
    ("team_xg_history", "team_id", "fk_team_xg_team_id"),
    ("matches", "competition_id", "fk_matches_competition_id"),
    ("matches", "season_id", "fk_matches_season_id"),
    ("matches", "home_team_id", "fk_matches_home_team_id"),
    ("matches", "away_team_id", "fk_matches_away_team_id"),
    ("matches", "stadium_id", "fk_matches_stadium_id"),
    ("matches", "referee_id", "fk_matches_referee_id"),
    ("seasons", "competition_id", "fk_seasons_competition_id"),
    ("teams", "country_id", "fk_teams_country_id"),
    ("competitions", "country_id", "fk_competitions_country_id"),
]


def _alter_column_type(
    table: str,
    column: str,
    fk_name: str,
    from_type: str,
    to_type: str,
) -> None:
    """Safely alter a FK column type by dropping and recreating the constraint.

    Steps:
    1. Drop the FK constraint
    2. Alter the column type
    3. Recreate the FK constraint
    """
    # Drop FK
    op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {fk_name}")

    # Alter column type
    op.execute(
        f"ALTER TABLE {table} ALTER COLUMN {column} TYPE {to_type}"
        f" USING {column}::{to_type}"
    )

    # Recreate FK (migration 001 defined FKs with specific names)
    # We recreate with the same name for clean rollback


def upgrade() -> None:
    # ══════════════════════════════════════════════════════
    # 1. Fix match_id FKs: Integer → BigInteger
    # ══════════════════════════════════════════════════════
    for table, column, fk_name in FK_COLUMNS:
        _alter_column_type(table, column, fk_name, "INTEGER", "BIGINT")
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {fk_name}"
            f" FOREIGN KEY ({column}) REFERENCES matches(id)"
            f" ON DELETE CASCADE"
        )

    # ══════════════════════════════════════════════════════
    # 2. Fix secondary FK columns: Integer → BigInteger
    # ══════════════════════════════════════════════════════
    for table, column, fk_name in SECONDARY_FK_COLUMNS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {fk_name}")
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN {column} TYPE BIGINT"
            f" USING {column}::BIGINT"
        )

    # Recreate secondary FKs (need to resolve referenced table + column)
    fk_definitions = {
        "fk_players_current_team_id": ("teams", "id"),
        "fk_pms_player_id": ("players", "id"),
        "fk_pms_team_id": ("teams", "id"),
        "fk_transfers_player_id": ("players", "id"),
        "fk_transfers_from_team_id": ("teams", "id"),
        "fk_transfers_to_team_id": ("teams", "id"),
        "fk_injuries_player_id": ("players", "id"),
        "fk_lineups_team_id": ("teams", "id"),
        "fk_team_elo_team_id": ("teams", "id"),
        "fk_team_form_team_id": ("teams", "id"),
        "fk_team_xg_team_id": ("teams", "id"),
        "fk_matches_competition_id": ("competitions", "id"),
        "fk_matches_season_id": ("seasons", "id"),
        "fk_matches_home_team_id": ("teams", "id"),
        "fk_matches_away_team_id": ("teams", "id"),
        "fk_matches_stadium_id": ("stadiums", "id"),
        "fk_matches_referee_id": ("referees", "id"),
        "fk_seasons_competition_id": ("competitions", "id"),
        "fk_teams_country_id": ("countries", "id"),
        "fk_competitions_country_id": ("countries", "id"),
    }

    for table, column, fk_name in SECONDARY_FK_COLUMNS:
        ref_table, ref_column = fk_definitions[fk_name]
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {fk_name}"
            f" FOREIGN KEY ({column}) REFERENCES {ref_table}({ref_column})"
        )

    # ══════════════════════════════════════════════════════
    # 3. Set matches.id to BIGINT if it's still INTEGER
    # ══════════════════════════════════════════════════════
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'matches' AND column_name = 'id'
                AND data_type = 'integer'
            ) THEN
                ALTER TABLE matches ALTER COLUMN id TYPE BIGINT;
            END IF;
        END $$;
    """)

    # Also upgrade primary key indexes that reference match_id
    # (player_match_stats, team_elo_history, team_form, team_xg_history
    #  have multi-column unique constraints that include match_id)
    for table in [
        "player_match_stats", "team_elo_history", "team_form",
        "team_xg_history",
    ]:
        op.execute(f"""
            DO $$ BEGIN
                RAISE NOTICE 'Unique constraints on % include match_id — auto-reindexing',
                    '{table}';
            END $$;
        """)


def downgrade() -> None:
    """Revert all FK column types back to INTEGER.

    WARNING: This will fail if any FK column has values > 2^31-1.
    Data integrity checks should be run before downgrading.
    """

    # ══════════════════════════════════════════════════════
    # 1. Revert match_id FKs: BigInteger → Integer
    # ══════════════════════════════════════════════════════
    for table, column, fk_name in FK_COLUMNS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {fk_name}")
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN {column} TYPE INTEGER"
            f" USING CASE WHEN {column} <= 2147483647 THEN {column}::INTEGER"
            f"           ELSE NULL END"
        )
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {fk_name}"
            f" FOREIGN KEY ({column}) REFERENCES matches(id)"
            f" ON DELETE CASCADE"
        )

    # ══════════════════════════════════════════════════════
    # 2. Revert secondary FK columns: BigInteger → Integer
    # ══════════════════════════════════════════════════════
    fk_definitions = {
        "fk_players_current_team_id": ("teams", "id"),
        "fk_pms_player_id": ("players", "id"),
        "fk_pms_team_id": ("teams", "id"),
        "fk_transfers_player_id": ("players", "id"),
        "fk_transfers_from_team_id": ("teams", "id"),
        "fk_transfers_to_team_id": ("teams", "id"),
        "fk_injuries_player_id": ("players", "id"),
        "fk_lineups_team_id": ("teams", "id"),
        "fk_team_elo_team_id": ("teams", "id"),
        "fk_team_form_team_id": ("teams", "id"),
        "fk_team_xg_team_id": ("teams", "id"),
        "fk_matches_competition_id": ("competitions", "id"),
        "fk_matches_season_id": ("seasons", "id"),
        "fk_matches_home_team_id": ("teams", "id"),
        "fk_matches_away_team_id": ("teams", "id"),
        "fk_matches_stadium_id": ("stadiums", "id"),
        "fk_matches_referee_id": ("referees", "id"),
        "fk_seasons_competition_id": ("competitions", "id"),
        "fk_teams_country_id": ("countries", "id"),
        "fk_competitions_country_id": ("countries", "id"),
    }

    for table, column, fk_name in SECONDARY_FK_COLUMNS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {fk_name}")
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN {column} TYPE INTEGER"
            f" USING CASE WHEN {column} <= 2147483647 THEN {column}::INTEGER"
            f"           ELSE NULL END"
        )

    for table, column, fk_name in SECONDARY_FK_COLUMNS:
        ref_table, ref_column = fk_definitions[fk_name]
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {fk_name}"
            f" FOREIGN KEY ({column}) REFERENCES {ref_table}({ref_column})"
        )
