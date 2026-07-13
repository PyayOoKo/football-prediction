"""
Initial schema — fully normalised football analytics database.

Revision ID: 001
Revises: None
Create Date: 2026-07-12
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ═══════════════════════════════════════════════════════════
#  UPGRADE
# ═══════════════════════════════════════════════════════════

def upgrade() -> None:
    # ── 1. Countries ───────────────────────────────────
    op.create_table(
        "countries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("iso_alpha2", sa.String(2), nullable=False),
        sa.Column("iso_alpha3", sa.String(3), nullable=False),
        sa.Column("fifa_code", sa.String(3), nullable=True),
        sa.Column("continent", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
        sa.UniqueConstraint("iso_alpha2"),
        sa.UniqueConstraint("iso_alpha3"),
        sa.UniqueConstraint("fifa_code"),
    )
    op.create_index("ix_countries_iso_alpha2", "countries", ["iso_alpha2"])
    op.create_index("ix_countries_iso_alpha3", "countries", ["iso_alpha3"])

    # ── 2. Competitions ───────────────────────────────
    op.create_table(
        "competitions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("code", sa.String(16), nullable=True),
        sa.Column("type", sa.String(16), nullable=False, server_default="league"),
        sa.Column("country_id", sa.Integer(), nullable=True),
        sa.Column("level", sa.Integer(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("logo_url", sa.String(512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["country_id"], ["countries.id"],
            name="fk_competitions_country_id",
        ),
        sa.CheckConstraint(
            "type IN ('league', 'cup', 'playoff', 'friendly')",
            name="ck_competitions_type",
        ),
        sa.UniqueConstraint("code"),
    )
    op.create_index("ix_competitions_name", "competitions", ["name"])
    op.create_index("ix_competitions_country_id", "competitions", ["country_id"])

    # ── 3. Stadiums ───────────────────────────────────
    op.create_table(
        "stadiums",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("city", sa.String(128), nullable=True),
        sa.Column("country_id", sa.Integer(), nullable=True),
        sa.Column("capacity", sa.Integer(), nullable=True),
        sa.Column("surface", sa.String(32), nullable=True),
        sa.Column("roofed", sa.Boolean(), nullable=True, server_default=sa.false()),
        sa.Column("image_url", sa.String(512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["country_id"], ["countries.id"],
            name="fk_stadiums_country_id",
        ),
    )
    op.create_index("ix_stadiums_name", "stadiums", ["name"])
    op.create_index("ix_stadiums_country_id", "stadiums", ["country_id"])

    # ── 4. Teams ──────────────────────────────────────
    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("short_name", sa.String(8), nullable=True),
        sa.Column("country_id", sa.Integer(), nullable=True),
        sa.Column("stadium_id", sa.Integer(), nullable=True),
        sa.Column("year_founded", sa.Integer(), nullable=True),
        sa.Column("logo_url", sa.String(512), nullable=True),
        sa.Column("website", sa.String(256), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["country_id"], ["countries.id"],
            name="fk_teams_country_id",
        ),
        sa.ForeignKeyConstraint(
            ["stadium_id"], ["stadiums.id"],
            name="fk_teams_stadium_id",
        ),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_teams_name", "teams", ["name"])
    op.create_index("ix_teams_country_id", "teams", ["country_id"])

    # ── 5. Referees ──────────────────────────────────
    op.create_table(
        "referees",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("full_name", sa.String(128), nullable=False),
        sa.Column("country_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["country_id"], ["countries.id"],
            name="fk_referees_country_id",
        ),
    )
    op.create_index("ix_referees_full_name", "referees", ["full_name"])
    op.create_index("ix_referees_country_id", "referees", ["country_id"])

    # ── 6. Seasons ────────────────────────────────────
    op.create_table(
        "seasons",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("competition_id", sa.Integer(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["competition_id"], ["competitions.id"],
            name="fk_seasons_competition_id",
        ),
        sa.UniqueConstraint(
            "competition_id", "name",
            name="uq_seasons_competition_name",
        ),
        sa.CheckConstraint(
            "start_date <= end_date",
            name="ck_seasons_date_range",
        ),
    )
    op.create_index("ix_seasons_competition_id", "seasons", ["competition_id"])

    # ── 7. Players ────────────────────────────────────
    op.create_table(
        "players",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("full_name", sa.String(128), nullable=False),
        sa.Column("date_of_birth", sa.Date(), nullable=True),
        sa.Column("country_id", sa.Integer(), nullable=True),
        sa.Column("position", sa.String(32), nullable=True),
        sa.Column("preferred_foot", sa.String(8), nullable=True),
        sa.Column("height_cm", sa.Integer(), nullable=True),
        sa.Column("weight_kg", sa.Integer(), nullable=True),
        sa.Column("current_team_id", sa.Integer(), nullable=True),
        sa.Column("shirt_number", sa.Integer(), nullable=True),
        sa.Column("market_value_eur", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["country_id"], ["countries.id"],
            name="fk_players_country_id",
        ),
        sa.ForeignKeyConstraint(
            ["current_team_id"], ["teams.id"],
            name="fk_players_current_team_id",
        ),
        sa.CheckConstraint(
            "position IN ('GK','CB','LB','RB','LWB','RWB','CM','CDM','CAM','LM','RM','LW','RW','CF','ST','SS')",
            name="ck_players_position",
        ),
        sa.CheckConstraint(
            "preferred_foot IN ('left','right','both')",
            name="ck_players_foot",
        ),
    )
    op.create_index("ix_players_full_name", "players", ["full_name"])
    op.create_index("ix_players_country_id", "players", ["country_id"])
    op.create_index("ix_players_current_team_id", "players", ["current_team_id"])

    # ── 8. Matches (central fact table) ───────────────
    op.create_table(
        "matches",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("competition_id", sa.Integer(), nullable=True),
        sa.Column("season_id", sa.Integer(), nullable=True),
        sa.Column("home_team_id", sa.Integer(), nullable=False),
        sa.Column("away_team_id", sa.Integer(), nullable=False),
        sa.Column("stadium_id", sa.Integer(), nullable=True),
        sa.Column("referee_id", sa.Integer(), nullable=True),
        sa.Column("match_date", sa.Date(), nullable=False),
        sa.Column("round", sa.String(32), nullable=True),
        sa.Column("is_neutral_venue", sa.Boolean(), nullable=True, server_default=sa.false()),
        sa.Column("attendance", sa.Integer(), nullable=True),
        sa.Column("home_goals", sa.Integer(), nullable=True),
        sa.Column("away_goals", sa.Integer(), nullable=True),
        sa.Column("result", sa.String(4), nullable=True),
        sa.Column("duration", sa.String(8), nullable=True, server_default="regular"),
        sa.Column("status", sa.String(16), nullable=False, server_default="scheduled"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["competition_id"], ["competitions.id"],
            name="fk_matches_competition_id",
        ),
        sa.ForeignKeyConstraint(
            ["season_id"], ["seasons.id"],
            name="fk_matches_season_id",
        ),
        sa.ForeignKeyConstraint(
            ["home_team_id"], ["teams.id"],
            name="fk_matches_home_team_id",
        ),
        sa.ForeignKeyConstraint(
            ["away_team_id"], ["teams.id"],
            name="fk_matches_away_team_id",
        ),
        sa.ForeignKeyConstraint(
            ["stadium_id"], ["stadiums.id"],
            name="fk_matches_stadium_id",
        ),
        sa.ForeignKeyConstraint(
            ["referee_id"], ["referees.id"],
            name="fk_matches_referee_id",
        ),
        sa.CheckConstraint(
            "home_team_id != away_team_id",
            name="ck_matches_different_teams",
        ),
        sa.CheckConstraint(
            "result IN ('H', 'D', 'A')",
            name="ck_matches_result",
        ),
        sa.CheckConstraint(
            "duration IN ('regular', 'extra_time', 'penalties')",
            name="ck_matches_duration",
        ),
        sa.CheckConstraint(
            "status IN ('scheduled', 'live', 'finished', 'postponed', 'cancelled', 'abandoned')",
            name="ck_matches_status",
        ),
    )
    # Indexes for matches (the central table — heavily queried)
    op.create_index("ix_matches_home_team_id", "matches", ["home_team_id"])
    op.create_index("ix_matches_away_team_id", "matches", ["away_team_id"])
    op.create_index("ix_matches_match_date", "matches", ["match_date"])
    op.create_index("ix_matches_competition_id", "matches", ["competition_id"])
    op.create_index("ix_matches_season_id", "matches", ["season_id"])
    op.create_index(
        "ix_matches_comp_season_date",
        "matches", ["competition_id", "season_id", "match_date"],
        postgresql_using="btree",
    )
    op.create_index(
        "ix_matches_home_date",
        "matches", ["home_team_id", "match_date"],
        postgresql_using="btree",
    )
    op.create_index(
        "ix_matches_away_date",
        "matches", ["away_team_id", "match_date"],
        postgresql_using="btree",
    )

    # ── 9. Match Statistics (1:1) ───────────────────────
    op.create_table(
        "match_statistics",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        # Home stats
        sa.Column("home_shots", sa.Integer(), nullable=True),
        sa.Column("home_shots_on_target", sa.Integer(), nullable=True),
        sa.Column("home_possession", sa.Float(), nullable=True),
        sa.Column("home_corners", sa.Integer(), nullable=True),
        sa.Column("home_fouls", sa.Integer(), nullable=True),
        sa.Column("home_yellow_cards", sa.Integer(), nullable=True),
        sa.Column("home_red_cards", sa.Integer(), nullable=True),
        sa.Column("home_offsides", sa.Integer(), nullable=True),
        sa.Column("home_shots_inside_box", sa.Integer(), nullable=True),
        sa.Column("home_shots_outside_box", sa.Integer(), nullable=True),
        # Away stats
        sa.Column("away_shots", sa.Integer(), nullable=True),
        sa.Column("away_shots_on_target", sa.Integer(), nullable=True),
        sa.Column("away_possession", sa.Float(), nullable=True),
        sa.Column("away_corners", sa.Integer(), nullable=True),
        sa.Column("away_fouls", sa.Integer(), nullable=True),
        sa.Column("away_yellow_cards", sa.Integer(), nullable=True),
        sa.Column("away_red_cards", sa.Integer(), nullable=True),
        sa.Column("away_offsides", sa.Integer(), nullable=True),
        sa.Column("away_shots_inside_box", sa.Integer(), nullable=True),
        sa.Column("away_shots_outside_box", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["match_id"], ["matches.id"],
            name="fk_match_statistics_match_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("match_id", name="uq_match_statistics_match_id"),
    )
    op.create_index("ix_match_statistics_match_id", "match_statistics", ["match_id"])

    # ── 10. Odds (1:N) ────────────────────────────────
    op.create_table(
        "odds",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("odds_home", sa.Float(), nullable=True),
        sa.Column("odds_draw", sa.Float(), nullable=True),
        sa.Column("odds_away", sa.Float(), nullable=True),
        sa.Column("implied_prob_home", sa.Float(), nullable=True),
        sa.Column("implied_prob_draw", sa.Float(), nullable=True),
        sa.Column("implied_prob_away", sa.Float(), nullable=True),
        sa.Column("margin", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["match_id"], ["matches.id"],
            name="fk_odds_match_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "match_id", "source", "timestamp",
            name="uq_odds_match_source_time",
        ),
        sa.CheckConstraint(
            "odds_home IS NULL OR odds_home > 1.0",
            name="ck_odds_home_positive",
        ),
        sa.CheckConstraint(
            "odds_draw IS NULL OR odds_draw > 1.0",
            name="ck_odds_draw_positive",
        ),
        sa.CheckConstraint(
            "odds_away IS NULL OR odds_away > 1.0",
            name="ck_odds_away_positive",
        ),
    )
    op.create_index("ix_odds_match_id", "odds", ["match_id"])
    op.create_index("ix_odds_match_source", "odds", ["match_id", "source"])

    # ── 11. Weather (1:1) ─────────────────────────────
    op.create_table(
        "weather",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("temperature_celsius", sa.Float(), nullable=True),
        sa.Column("humidity_pct", sa.Integer(), nullable=True),
        sa.Column("wind_speed_kmh", sa.Float(), nullable=True),
        sa.Column("precipitation_mm", sa.Float(), nullable=True),
        sa.Column("condition", sa.String(32), nullable=True),
        sa.Column("pitch_condition", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["match_id"], ["matches.id"],
            name="fk_weather_match_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("match_id", name="uq_weather_match_id"),
        sa.CheckConstraint(
            "temperature_celsius IS NULL OR (temperature_celsius >= -30 AND temperature_celsius <= 60)",
            name="ck_weather_temperature",
        ),
        sa.CheckConstraint(
            "humidity_pct IS NULL OR (humidity_pct >= 0 AND humidity_pct <= 100)",
            name="ck_weather_humidity",
        ),
    )
    op.create_index("ix_weather_match_id", "weather", ["match_id"])

    # ── 12. Lineups (1:N) ────────────────────────────
    op.create_table(
        "lineups",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("formation", sa.String(8), nullable=True),
        sa.Column("starting_xi", postgresql.JSONB(), nullable=True),
        sa.Column("substitutes", postgresql.JSONB(), nullable=True),
        sa.Column("substitutions_made", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("coach", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["match_id"], ["matches.id"],
            name="fk_lineups_match_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"],
            name="fk_lineups_team_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "match_id", "team_id",
            name="uq_lineups_match_team",
        ),
    )
    op.create_index("ix_lineups_match_id", "lineups", ["match_id"])
    op.create_index("ix_lineups_team_id", "lineups", ["team_id"])

    # ── 13. Transfers (1:N) ──────────────────────────
    op.create_table(
        "transfers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("from_team_id", sa.Integer(), nullable=True),
        sa.Column("to_team_id", sa.Integer(), nullable=False),
        sa.Column("transfer_date", sa.Date(), nullable=False),
        sa.Column("transfer_fee_eur", sa.Float(), nullable=True),
        sa.Column("is_loan", sa.Boolean(), nullable=True, server_default=sa.false()),
        sa.Column("loan_end_date", sa.Date(), nullable=True),
        sa.Column("contract_length_months", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["player_id"], ["players.id"],
            name="fk_transfers_player_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["from_team_id"], ["teams.id"],
            name="fk_transfers_from_team_id",
        ),
        sa.ForeignKeyConstraint(
            ["to_team_id"], ["teams.id"],
            name="fk_transfers_to_team_id",
        ),
        sa.CheckConstraint(
            "from_team_id IS NULL OR from_team_id != to_team_id",
            name="ck_transfers_different_teams",
        ),
        sa.CheckConstraint(
            "transfer_fee_eur IS NULL OR transfer_fee_eur >= 0",
            name="ck_transfers_fee_non_negative",
        ),
    )
    op.create_index("ix_transfers_player_id", "transfers", ["player_id"])
    op.create_index("ix_transfers_from_team_id", "transfers", ["from_team_id"])
    op.create_index("ix_transfers_to_team_id", "transfers", ["to_team_id"])

    # ── 14. Player Match Stats (1:N) ───────────────────
    op.create_table(
        "player_match_stats",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("minutes_played", sa.Integer(), nullable=True),
        sa.Column("is_starter", sa.Boolean(), nullable=True, server_default=sa.false()),
        sa.Column("position", sa.String(8), nullable=True),
        sa.Column("goals", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("assists", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("shots", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("shots_on_target", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("passes", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("pass_accuracy", sa.Float(), nullable=True),
        sa.Column("tackles", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("interceptions", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("fouls_committed", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("fouls_drawn", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("yellow_card", sa.Boolean(), nullable=True, server_default=sa.false()),
        sa.Column("red_card", sa.Boolean(), nullable=True, server_default=sa.false()),
        sa.Column("saves", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("rating", sa.Float(), nullable=True),
        sa.Column("xg", sa.Float(), nullable=True),
        sa.Column("xa", sa.Float(), nullable=True),
        sa.Column("xg_chain", sa.Float(), nullable=True),
        sa.Column("xg_buildup", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["match_id"], ["matches.id"],
            name="fk_pms_match_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["player_id"], ["players.id"],
            name="fk_pms_player_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"],
            name="fk_pms_team_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "match_id", "player_id",
            name="uq_pms_match_player",
        ),
        sa.CheckConstraint(
            "minutes_played IS NULL OR (minutes_played >= 0 AND minutes_played <= 120)",
            name="ck_pms_minutes",
        ),
    )
    op.create_index("ix_pms_match_id", "player_match_stats", ["match_id"])
    op.create_index("ix_pms_player_id", "player_match_stats", ["player_id"])
    op.create_index("ix_pms_team_id", "player_match_stats", ["team_id"])
    op.create_index("ix_pms_player_date", "player_match_stats", ["player_id", "match_id"])

    # ── 15. Injuries (1:N) ────────────────────────────
    op.create_table(
        "injuries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("injury_type", sa.String(64), nullable=True),
        sa.Column("severity", sa.String(16), nullable=True),
        sa.Column("injury_date", sa.Date(), nullable=False),
        sa.Column("expected_return", sa.Date(), nullable=True),
        sa.Column("actual_return", sa.Date(), nullable=True),
        sa.Column("missed_matches", sa.Integer(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["player_id"], ["players.id"],
            name="fk_injuries_player_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_injuries_player_id", "injuries", ["player_id"])

    # ── 16. Team Elo History (1:N) ────────────────────
    op.create_table(
        "team_elo_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("side", sa.String(4), nullable=False),
        sa.Column("elo_before", sa.Float(), nullable=False),
        sa.Column("elo_after", sa.Float(), nullable=False),
        sa.Column("elo_change", sa.Float(), nullable=False),
        sa.Column("k_factor", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"],
            name="fk_team_elo_team_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["match_id"], ["matches.id"],
            name="fk_team_elo_match_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "team_id", "match_id",
            name="uq_team_elo_match",
        ),
        sa.CheckConstraint(
            "side IN ('home', 'away')",
            name="ck_team_elo_side",
        ),
    )
    op.create_index("ix_team_elo_team_id", "team_elo_history", ["team_id"])
    op.create_index("ix_team_elo_match_id", "team_elo_history", ["match_id"])
    op.create_index(
        "ix_team_elo_team_date",
        "team_elo_history", ["team_id", "match_id"],
        postgresql_using="btree",
    )

    # ── 17. Team Form (1:N) ──────────────────────────
    op.create_table(
        "team_form",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("side", sa.String(4), nullable=False),
        sa.Column("last_5_ppg", sa.Float(), nullable=True),
        sa.Column("last_5_goals_scored", sa.Float(), nullable=True),
        sa.Column("last_5_goals_conceded", sa.Float(), nullable=True),
        sa.Column("last_5_wins", sa.Integer(), nullable=True),
        sa.Column("last_5_draws", sa.Integer(), nullable=True),
        sa.Column("last_5_losses", sa.Integer(), nullable=True),
        sa.Column("last_5_clean_sheets", sa.Integer(), nullable=True),
        sa.Column("last_5_btts", sa.Integer(), nullable=True),
        sa.Column("current_streak", sa.String(8), nullable=True),
        sa.Column("last_10_ppg", sa.Float(), nullable=True),
        sa.Column("last_10_goals_scored", sa.Float(), nullable=True),
        sa.Column("last_10_goals_conceded", sa.Float(), nullable=True),
        sa.Column("last_20_ppg", sa.Float(), nullable=True),
        sa.Column("last_20_goals_scored", sa.Float(), nullable=True),
        sa.Column("last_20_goals_conceded", sa.Float(), nullable=True),
        sa.Column("season_ppg", sa.Float(), nullable=True),
        sa.Column("season_matches_played", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"],
            name="fk_team_form_team_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["match_id"], ["matches.id"],
            name="fk_team_form_match_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "team_id", "match_id",
            name="uq_team_form_match",
        ),
    )
    op.create_index("ix_team_form_team_id", "team_form", ["team_id"])
    op.create_index("ix_team_form_match_id", "team_form", ["match_id"])

    # ── 18. Team xG History (1:N) ──────────────────────
    op.create_table(
        "team_xg_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("side", sa.String(4), nullable=False),
        sa.Column("source", sa.String(32), nullable=False, server_default="opta"),
        sa.Column("xg", sa.Float(), nullable=False),
        sa.Column("xg_open_play", sa.Float(), nullable=True),
        sa.Column("xg_set_piece", sa.Float(), nullable=True),
        sa.Column("xg_penalty", sa.Float(), nullable=True),
        sa.Column("xg_first_half", sa.Float(), nullable=True),
        sa.Column("xg_second_half", sa.Float(), nullable=True),
        sa.Column("xa", sa.Float(), nullable=True),
        sa.Column("shots", sa.Integer(), nullable=True),
        sa.Column("shots_on_target", sa.Integer(), nullable=True),
        sa.Column("deep_completions", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"],
            name="fk_team_xg_team_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["match_id"], ["matches.id"],
            name="fk_team_xg_match_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "team_id", "match_id", "source",
            name="uq_team_xg_match_source",
        ),
    )
    op.create_index("ix_team_xg_team_id", "team_xg_history", ["team_id"])
    op.create_index("ix_team_xg_match_id", "team_xg_history", ["match_id"])

    # ── 19. Predictions (1:N) ─────────────────────────
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(64), nullable=False),
        sa.Column("model_version", sa.String(32), nullable=True),
        sa.Column("prob_home", sa.Float(), nullable=True),
        sa.Column("prob_draw", sa.Float(), nullable=True),
        sa.Column("prob_away", sa.Float(), nullable=True),
        sa.Column("predicted_result", sa.String(4), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("expected_value", sa.Float(), nullable=True),
        sa.Column("kelly_stake", sa.Float(), nullable=True),
        sa.Column("features_hash", sa.String(64), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["match_id"], ["matches.id"],
            name="fk_predictions_match_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_predictions_match_id", "predictions", ["match_id"])

    # ── 20. Expected Value Bets (1:N) ─────────────────
    op.create_table(
        "expected_value_bets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("bookmaker", sa.String(32), nullable=False),
        sa.Column("model_prob_home", sa.Float(), nullable=False),
        sa.Column("model_prob_draw", sa.Float(), nullable=False),
        sa.Column("model_prob_away", sa.Float(), nullable=False),
        sa.Column("book_prob_home", sa.Float(), nullable=False),
        sa.Column("book_prob_draw", sa.Float(), nullable=False),
        sa.Column("book_prob_away", sa.Float(), nullable=False),
        sa.Column("ev_home", sa.Float(), nullable=False),
        sa.Column("ev_draw", sa.Float(), nullable=False),
        sa.Column("ev_away", sa.Float(), nullable=False),
        sa.Column("kelly_stake_home", sa.Float(), nullable=True),
        sa.Column("kelly_stake_draw", sa.Float(), nullable=True),
        sa.Column("kelly_stake_away", sa.Float(), nullable=True),
        sa.Column("recommended_bet", sa.String(4), nullable=True),
        sa.Column("recommended_ev", sa.Float(), nullable=True),
        sa.Column("recommended_kelly", sa.Float(), nullable=True),
        sa.Column("model_version", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["match_id"], ["matches.id"],
            name="fk_ev_bets_match_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "match_id", "bookmaker",
            name="uq_ev_bet_match_bookmaker",
        ),
        sa.CheckConstraint(
            "model_prob_home + model_prob_draw + model_prob_away BETWEEN 0.98 AND 1.02",
            name="ck_ev_bet_model_probs_sum",
        ),
    )
    op.create_index("ix_ev_bets_match_id", "expected_value_bets", ["match_id"])

    # ── 21. Closing Line Value (1:N) ──────────────────
    op.create_table(
        "closing_line_values",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("bookmaker", sa.String(32), nullable=False),
        sa.Column("outcome", sa.String(4), nullable=False),
        sa.Column("opening_price", sa.Float(), nullable=False),
        sa.Column("closing_price", sa.Float(), nullable=False),
        sa.Column("clv", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["match_id"], ["matches.id"],
            name="fk_clv_match_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "match_id", "bookmaker", "outcome",
            name="uq_clv_match_bookmaker_outcome",
        ),
    )
    op.create_index("ix_clv_match_id", "closing_line_values", ["match_id"])

    # ── 22. Betting Results (1:N) ─────────────────────
    op.create_table(
        "betting_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=False),
        sa.Column("strategy", sa.String(64), nullable=False),
        sa.Column("bookmaker", sa.String(32), nullable=True),
        sa.Column("bet_outcome", sa.String(4), nullable=False),
        sa.Column("decimal_odds", sa.Float(), nullable=False),
        sa.Column("stake", sa.Float(), nullable=False),
        sa.Column("won", sa.Boolean(), nullable=True),
        sa.Column("profit", sa.Float(), nullable=True),
        sa.Column("roi_pct", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("model_version", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["match_id"], ["matches.id"],
            name="fk_betting_results_match_id",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "bet_outcome IN ('H', 'D', 'A')",
            name="ck_betting_results_outcome",
        ),
        sa.CheckConstraint(
            "stake > 0",
            name="ck_betting_results_stake_positive",
        ),
    )
    op.create_index("ix_betting_results_match_id", "betting_results", ["match_id"])
    op.create_index("ix_betting_results_strategy", "betting_results", ["strategy"])

    # ── Post-creation: PK sequence tuning ──────────────
    # For tables expected to grow to millions of rows,
    # set the sequence cache to 100 for better insert perf.
    for table in ["matches", "match_statistics", "odds", "player_match_stats",
                  "team_elo_history", "team_form", "team_xg_history",
                  "predictions", "expected_value_bets", "betting_results"]:
        op.execute(
            f"ALTER SEQUENCE {table}_id_seq CACHE 100"
        )


# ═══════════════════════════════════════════════════════════
#  DOWNGRADE
# ═══════════════════════════════════════════════════════════

def downgrade() -> None:
    """Drop all tables in reverse dependency order."""
    tables = [
        "betting_results",
        "closing_line_values",
        "expected_value_bets",
        "predictions",
        "team_xg_history",
        "team_form",
        "team_elo_history",
        "injuries",
        "player_match_stats",
        "transfers",
        "lineups",
        "weather",
        "odds",
        "match_statistics",
        "matches",
        "players",
        "seasons",
        "referees",
        "teams",
        "stadiums",
        "competitions",
        "countries",
    ]
    for table in tables:
        op.drop_table(table, cascade=True)
