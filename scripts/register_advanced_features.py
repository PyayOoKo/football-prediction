#!/usr/bin/env python3
"""
Register Advanced Predictive Features in the Feature Store.

This script registers all newly created advanced features (weather, referee,
schedule, extended H2H, extended form) as ``FeatureDefinition`` entries in
the Feature Store. Run once after setting up the database schema.

Usage
-----
::

    python scripts/register_advanced_features.py            # Use DB session
    python scripts/register_advanced_features.py --dry-run   # Print only
    python scripts/register_advanced_features.py --csv       # Save to CSV

Requires a running database with the Feature Store schema migrated.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# ── Feature definitions ─────────────────────────────────────
# Each entry: (name, feature_type, category, entity_type, description,
#              computation_params, dependencies, tags)

FEATURE_DEFINITIONS: list[tuple[str, str, str, str, str, dict[str, Any], list[str], list[str]]] = [
    # ── Weather Features ────────────────────────────────────
    (
        "h_temperature_celsius",
        "composite", "home_advantage", "match",
        "Match-day temperature in Celsius for the home team. "
        "Higher temperatures increase fatigue and reduce scoring rates.",
        {"source": "weather.csv", "fallback": 15.0},
        ["temperature_celsius"],
        ["weather", "temperature", "environment"],
    ),
    (
        "a_temperature_celsius",
        "composite", "away_advantage", "match",
        "Match-day temperature in Celsius for the away team.",
        {"source": "weather.csv", "fallback": 15.0},
        ["temperature_celsius"],
        ["weather", "temperature", "environment"],
    ),
    (
        "h_humidity_pct",
        "composite", "home_advantage", "match",
        "Humidity percentage (0-100) for the home team. "
        "High humidity (>80%) can significantly impact stamina.",
        {"source": "weather.csv", "fallback": 50.0},
        ["humidity_pct"],
        ["weather", "humidity", "environment"],
    ),
    (
        "a_humidity_pct",
        "composite", "away_advantage", "match",
        "Humidity percentage (0-100) for the away team.",
        {"source": "weather.csv", "fallback": 50.0},
        ["humidity_pct"],
        ["weather", "humidity", "environment"],
    ),
    (
        "h_wind_speed_kmh",
        "composite", "home_advantage", "match",
        "Wind speed in km/h. Strong wind (>30 km/h) disrupts long balls "
        "and aerial duels, favouring defensive/ground-based teams.",
        {"source": "weather.csv", "fallback": 10.0},
        ["wind_speed_kmh"],
        ["weather", "wind", "environment"],
    ),
    (
        "a_wind_speed_kmh",
        "composite", "away_advantage", "match",
        "Wind speed in km/h for the away team.",
        {"source": "weather.csv", "fallback": 10.0},
        ["wind_speed_kmh"],
        ["weather", "wind", "environment"],
    ),
    (
        "h_weather_severity",
        "composite", "home_advantage", "match",
        "Composite weather severity score (0-1) combining precipitation, "
        "wind, and extreme temperatures. Higher = more adverse conditions.",
        {"source": "weather.csv", "formula": "0.4*precip_norm + 0.3*wind_norm + 0.3*temp_extreme"},
        ["precipitation_mm", "wind_speed_kmh", "temperature_celsius"],
        ["weather", "composite", "severity"],
    ),
    (
        "a_weather_severity",
        "composite", "away_advantage", "match",
        "Composite weather severity score (0-1) for the away team.",
        {"source": "weather.csv"},
        ["precipitation_mm", "wind_speed_kmh", "temperature_celsius"],
        ["weather", "composite", "severity"],
    ),

    # ── Referee Features ────────────────────────────────────
    (
        "referee_home_yellow_rate",
        "rolling_stat", "home_advantage", "match",
        "Rolling average of home team yellow cards per match under this referee "
        "(last 20 matches). Some referees are stricter than others.",
        {"window": 20, "source": "referees.csv", "leakage_protected": True},
        [],
        ["referee", "yellow_cards", "discipline"],
    ),
    (
        "referee_away_yellow_rate",
        "rolling_stat", "away_advantage", "match",
        "Rolling average of away team yellow cards per match under this referee "
        "(last 20 matches). Captures referee-specific away bias.",
        {"window": 20, "source": "referees.csv", "leakage_protected": True},
        [],
        ["referee", "yellow_cards", "discipline"],
    ),
    (
        "referee_home_win_rate",
        "rolling_stat", "home_advantage", "match",
        "Rolling home win rate under this referee (last 20 matches). "
        "Captures referees who favour aggressive play (benefits counter-attackers).",
        {"window": 20, "source": "referees.csv", "leakage_protected": True},
        [],
        ["referee", "home_advantage", "bias"],
    ),
    (
        "referee_card_total_avg",
        "rolling_stat", "composite", "match",
        "Rolling average of total cards (yellow + red) per match under this "
        "referee. High-card referees cause more stoppages and set-pieces.",
        {"window": 20, "source": "referees.csv", "leakage_protected": True},
        [],
        ["referee", "cards", "discipline"],
    ),

    # ── Schedule / Congestion Features ──────────────────────
    (
        "h_rest_days",
        "rest_days", "fixture_congestion", "team",
        "Days since the home team's last match. "
        "< 3 days = congested schedule, 3-6 = normal, 7+ = well-rested.",
        {"source": "derived", "leakage_protected": True},
        [],
        ["schedule", "rest", "congestion", "fatigue"],
    ),
    (
        "a_rest_days",
        "rest_days", "fixture_congestion", "team",
        "Days since the away team's last match.",
        {"source": "derived", "leakage_protected": True},
        [],
        ["schedule", "rest", "congestion", "fatigue"],
    ),
    (
        "h_matches_last_7_days",
        "fixture_congestion", "fixture_congestion", "team",
        "Number of matches the home team played in the last 7 days. "
        ">= 3 matches in 7 days indicates heavy fixture congestion.",
        {"source": "derived", "leakage_protected": True},
        [],
        ["schedule", "congestion", "fatigue", "frequency"],
    ),
    (
        "a_matches_last_7_days",
        "fixture_congestion", "fixture_congestion", "team",
        "Number of matches the away team played in the last 7 days.",
        {"source": "derived", "leakage_protected": True},
        [],
        ["schedule", "congestion", "fatigue", "frequency"],
    ),
    (
        "h_matches_last_14_days",
        "fixture_congestion", "fixture_congestion", "team",
        "Number of matches the home team played in the last 14 days. "
        ">= 5 matches in 14 days indicates high workload.",
        {"source": "derived", "leakage_protected": True},
        [],
        ["schedule", "congestion", "fatigue", "frequency"],
    ),
    (
        "a_matches_last_14_days",
        "fixture_congestion", "fixture_congestion", "team",
        "Number of matches the away team played in the last 14 days.",
        {"source": "derived", "leakage_protected": True},
        [],
        ["schedule", "congestion", "fatigue", "frequency"],
    ),
    (
        "h_consec_home",
        "rolling_stat", "home_advantage", "team",
        "Consecutive home matches for the home team. "
        "Long home streaks strengthen familiarity and routine.",
        {"source": "derived", "leakage_protected": True},
        [],
        ["schedule", "streak", "home_advantage"],
    ),
    (
        "a_consec_away",
        "rolling_stat", "away_advantage", "team",
        "Consecutive away matches for the away team. "
        "Long away streaks increase travel fatigue.",
        {"source": "derived", "leakage_protected": True},
        [],
        ["schedule", "streak", "away_disadvantage"],
    ),
    (
        "h_travel_distance",
        "composite", "home_advantage", "team",
        "Great-circle distance (km) the home team travelled from their "
        "previous venue. > 5000 km = significant jet lag risk.",
        {"source": "derived", "formula": "haversine", "leakage_protected": True},
        [],
        ["travel", "distance", "fatigue"],
    ),
    (
        "a_travel_distance",
        "composite", "away_advantage", "team",
        "Great-circle distance (km) the away team travelled from their "
        "previous venue. Away teams typically travel more than home teams.",
        {"source": "derived", "formula": "haversine", "leakage_protected": True},
        [],
        ["travel", "distance", "fatigue"],
    ),

    # ── Extended H2H Features (representative samples) ─────
    (
        "h_h2h_overall_points_avg3",
        "h2h_stat", "h2h_stat", "match",
        "Average points per match for the home team against this opponent "
        "over the last 3 meetings (all venues).",
        {"window": 3, "context": "overall", "leakage_protected": True},
        [],
        ["h2h", "rolling", "pairwise"],
    ),
    (
        "a_h2h_overall_points_avg3",
        "h2h_stat", "h2h_stat", "match",
        "Average points per match for the away team against this opponent "
        "over the last 3 meetings.",
        {"window": 3, "context": "overall", "leakage_protected": True},
        [],
        ["h2h", "rolling", "pairwise"],
    ),
    (
        "h_h2h_home_win_rate_last5",
        "h2h_stat", "h2h_stat", "match",
        "Win rate for the home team when playing at home against this "
        "opponent, over the last 5 home meetings.",
        {"window": 5, "context": "home", "leakage_protected": True},
        [],
        ["h2h", "home_advantage", "pairwise"],
    ),
    (
        "a_h2h_away_goals_scored_last5",
        "h2h_stat", "h2h_stat", "match",
        "Average goals scored by the away team when playing away against "
        "this opponent, over the last 5 away meetings.",
        {"window": 5, "context": "away", "leakage_protected": True},
        [],
        ["h2h", "away_goals", "pairwise"],
    ),

    # ── Extended Form Features (representative samples) ─────
    (
        "h_overall_points_avg5",
        "rolling_stat", "team_form", "team",
        "Average points per match for the home team across all venues, "
        "last 5 matches. Standard measure of recent form.",
        {"window": 5, "context": "overall", "leakage_protected": True},
        [],
        ["form", "rolling", "points"],
    ),
    (
        "a_overall_goals_scored_avg5",
        "rolling_stat", "team_form", "team",
        "Average goals scored per match for the away team across all "
        "venues, last 5 matches.",
        {"window": 5, "context": "overall", "leakage_protected": True},
        [],
        ["form", "rolling", "goals"],
    ),
    (
        "h_home_clean_sheets_avg10",
        "rolling_stat", "defense_strength", "team",
        "Clean sheet rate for the home team when playing at home, "
        "last 10 home matches. Measures defensive solidity on home turf.",
        {"window": 10, "context": "home", "leakage_protected": True},
        [],
        ["form", "defence", "clean_sheets", "home"],
    ),
    (
        "a_away_btts_avg10",
        "rolling_stat", "team_momentum", "team",
        "Both teams scored rate for the away team when playing away, "
        "last 10 away matches. Indicates open/vulnerable games.",
        {"window": 10, "context": "away", "leakage_protected": True},
        [],
        ["form", "btts", "away"],
    ),
]


def register_all(
    dry_run: bool = False,
    csv_path: str | None = None,
) -> None:
    """Register all advanced feature definitions."""
    logger = logging.getLogger(__name__)

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "name", "feature_type", "category", "entity_type",
                "description", "computation_params", "dependencies",
            ])
            for name, ftype, cat, etype, desc, params, deps, _ in FEATURE_DEFINITIONS:
                writer.writerow([
                    name, ftype, cat, etype, desc,
                    str(params), ";".join(deps),
                ])
        logger.info("Wrote %d definitions to %s", len(FEATURE_DEFINITIONS), csv_path)
        return

    if dry_run:
        print(f"DRY RUN — would register {len(FEATURE_DEFINITIONS)} features:")
        print()
        print(f"{'Name':45s} {'Type':20s} {'Category':20s}")
        print("-" * 85)
        for name, ftype, cat, *_ in FEATURE_DEFINITIONS:
            print(f"{name:45s} {ftype:20s} {cat:20s}")
        return

    # Try database registration
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from config import config
        from src.feature_store.models import FeatureCategory, FeatureDefinition
        from src.feature_store.registry import FeatureRegistry, FEATURE_TYPES

        # Connect to database
        db_url = config.paths.data / "football.db"
        engine = create_engine(f"sqlite:///{db_url}")
        session = Session(engine)

        registry = FeatureRegistry(session)
        count = 0
        errors = []

        for name, ftype, cat_str, etype, desc, params, deps, tags in FEATURE_DEFINITIONS:
            try:
                # Map category string to enum
                cat_map = {
                    "rolling_stat": FeatureCategory.ROLLING_STAT,
                    "team_form": FeatureCategory.TEAM_FORM,
                    "elo_rating": FeatureCategory.ELO_RATING,
                    "attack_strength": FeatureCategory.ATTACK_STRENGTH,
                    "defense_strength": FeatureCategory.DEFENSE_STRENGTH,
                    "home_advantage": FeatureCategory.HOME_ADVANTAGE,
                    "away_advantage": FeatureCategory.AWAY_ADVANTAGE,
                    "rest_days": FeatureCategory.REST_DAYS,
                    "fixture_congestion": FeatureCategory.FIXTURE_CONGESTION,
                    "league_strength": FeatureCategory.LEAGUE_STRENGTH,
                    "team_momentum": FeatureCategory.TEAM_MOMENTUM,
                    "market_movement": FeatureCategory.MARKET_MOVEMENT,
                    "h2h_stat": FeatureCategory.H2H_STAT,
                    "xg_feature": FeatureCategory.XG_FEATURE,
                    "odds_feature": FeatureCategory.ODDS_FEATURE,
                    "composite": FeatureCategory.COMPOSITE,
                }
                category = cat_map.get(cat_str, FeatureCategory.COMPOSITE)

                # Use a valid feature_type from FEATURE_TYPES
                type_map = {
                    "rolling_stat": "rolling_stat",
                    "team_form": "team_form",
                    "elo": "elo",
                    "attack_strength": "attack_strength",
                    "defense_strength": "defense_strength",
                    "home_advantage": "home_advantage",
                    "away_advantage": "away_advantage",
                    "rest_days": "rest_days",
                    "fixture_congestion": "fixture_congestion",
                    "league_strength": "league_strength",
                    "team_momentum": "team_momentum",
                    "market_movement": "market_movement",
                    "h2h_stat": "h2h_stat",
                    "xg_feature": "xg_feature",
                    "odds_feature": "odds_feature",
                    "composite": "composite",
                }
                feature_type = type_map.get(ftype, ftype)
                if feature_type not in FEATURE_TYPES:
                    feature_type = "composite"  # fallback

                metadata = {"tags": tags, "registered_at": datetime.now(timezone.utc).isoformat()}

                registry.register(
                    name=name,
                    feature_type=feature_type,
                    category=category,
                    entity_type=etype,
                    description=desc,
                    computation_params=params,
                    dependencies=deps or None,
                    metadata=metadata,
                    status="active",
                    changelog="Initial registration of advanced predictive feature",
                )
                count += 1
            except ValueError as ve:
                # Likely duplicate — that's OK
                errors.append(f"  {name}: {ve}")
            except Exception as exc:
                errors.append(f"  {name}: {exc}")

        session.commit()
        session.close()

        print(f"✅ Registered {count}/{len(FEATURE_DEFINITIONS)} advanced features")
        if errors:
            print(f"\n⚠️  {len(errors)} skipped (likely duplicates):")
            for err in errors[:10]:
                print(f"   {err}")
            if len(errors) > 10:
                print(f"   ... and {len(errors) - 10} more")

    except ImportError as ie:
        print(f"❌ Cannot register: missing dependencies ({ie})")
        print("   Install required packages: pip install sqlalchemy")
        sys.exit(1)
    except Exception as exc:
        print(f"❌ Registration failed: {exc}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register advanced features in the Feature Store",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print feature definitions without registering",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Export definitions to CSV file instead of registering",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    register_all(dry_run=args.dry_run, csv_path=args.csv)

    if args.dry_run:
        print(f"\nTotal: {len(FEATURE_DEFINITIONS)} features")
    elif args.csv:
        print(f"Total: {len(FEATURE_DEFINITIONS)} features in {args.csv}")


if __name__ == "__main__":
    main()
