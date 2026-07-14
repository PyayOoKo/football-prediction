"""
Collect World Cup 2026 Data — download, clean, and save.

Downloads the free public-domain 2026 World Cup match dataset from
openfootball/worldcup.json, converts it to the project's standard
schema, runs the validation pipeline, and saves to ``data/raw/worldcup_2026.csv``.

Usage::

    python collect_worldcup.py                   # Full download + validate
    python collect_worldcup.py --no-save          # Download + validate only
    python collect_worldcup.py --list-teams       # List all 48 teams
    python collect_worldcup.py --list-groups      # Show group assignments
    python collect_worldcup.py --upcoming         # Show upcoming fixtures only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and process 2026 World Cup data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Download and validate but don't save to CSV",
    )
    parser.add_argument(
        "--list-teams",
        action="store_true",
        help="List all real (non-placeholder) teams in the tournament",
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="Show group-stage team assignments",
    )
    parser.add_argument(
        "--upcoming",
        action="store_true",
        help="Show upcoming fixtures (matches without results)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a formatted summary of the downloaded data",
    )
    args = parser.parse_args()

    # ── Quick info modes (no full download needed) ─────────
    if args.list_teams:
        _list_teams()
        return
    if args.list_groups:
        _list_groups()
        return

    # ── Full download ──────────────────────────────────────
    _download_and_report(
        save=not args.no_save,
        show_upcoming=args.upcoming,
        show_summary=args.summary,
    )


def _list_teams() -> None:
    """Download and display all 48 World Cup teams."""
    from src.data_collection import list_worldcup_teams

    print("=" * 60)
    print("  2026 World Cup — Teams")
    print("=" * 60)

    teams = list_worldcup_teams()
    print(f"  Total teams: {len(teams)}")
    print()

    # Group by confederation based on known mappings
    asia = {"Australia", "Iran", "Iraq", "Japan", "Jordan", "Saudi Arabia",
            "South Korea", "United Arab Emirates", "Uzbekistan"}
    africa = {"Algeria", "Cameroon", "Côte d'Ivoire", "Egypt", "Ghana",
              "Morocco", "Nigeria", "Senegal", "Tunisia"}
    europe = {"Austria", "Belgium", "Croatia", "Czech Republic", "Denmark",
              "England", "France", "Germany", "Greece", "Hungary", "Italy",
              "Netherlands", "Norway", "Poland", "Portugal", "Romania",
              "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden",
              "Switzerland", "Turkey", "Ukraine", "Wales"}
    n_america = {"Canada", "Costa Rica", "Honduras", "Jamaica", "Mexico",
                 "Panama", "United States"}
    s_america = {"Argentina", "Brazil", "Chile", "Colombia", "Ecuador",
                 "Paraguay", "Peru", "Uruguay", "Venezuela"}
    oceania = {"New Zealand"}

    for label, conf_teams in [
        ("Europe (UEFA)", europe),
        ("Asia (AFC)", asia),
        ("Africa (CAF)", africa),
        ("North & Central America (CONCACAF)", n_america),
        ("South America (CONMEBOL)", s_america),
        ("Oceania (OFC)", oceania),
    ]:
        actual = [t for t in conf_teams if t in teams]
        if actual:
            print(f"  ── {label} ({len(actual)}) ──")
            for t in sorted(actual):
                print(f"    • {t}")
        print()


def _list_groups() -> None:
    """Download and display group-stage assignments."""
    from src.data_collection import list_worldcup_groups

    print("=" * 60)
    print("  2026 World Cup — Group Stage")
    print("=" * 60)

    groups = list_worldcup_groups()
    print(f"  Total groups: {len(groups)}  ({sum(len(v) for v in groups.values())} teams)")
    print()

    for group_name, teams in groups.items():
        print(f"  ── {group_name} ──")
        for t in teams:
            print(f"    • {t}")
        print()


def _download_and_report(
    save: bool = True,
    show_upcoming: bool = False,
    show_summary: bool = False,
) -> None:
    """Download, clean, and optionally display the World Cup data."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.data_collection import collect_worldcup
    from src.data_collection.sources.worldcup import (
        get_group_stage,
        get_knockout_stage,
        get_completed_matches,
        get_upcoming_matches,
    )

    print()
    print("═" * 60)
    print("  2026 FIFA World Cup — Data Collection")
    print("═" * 60)
    print()

    report = collect_worldcup(save=save)

    if "error" in report:
        print(f"  ❌ {report['error']}")
        return

    print()
    print(f"  ✓ Downloaded  {report['total_matches']} matches")
    print(f"  ✓ Completed   {report['completed']}  (with scores)")
    print(f"  ✓ Upcoming    {report['upcoming']}  (fixtures)")
    print(f"  ✓ Teams       {report['n_teams']}")
    if save:
        print(f"  ✓ Saved to    {report['path']}")
    print(f"  ✓ Duration    {report['duration_seconds']:.1f}s")
    print()

    # Validation
    val = report["validation"]
    if val.get("is_valid"):
        print("  ✓ Validation: PASS")
    else:
        print(f"  ⚠ Validation: {len(val.get('warnings', []))} warnings")
        for w in val.get("warnings", []):
            print(f"    ⚠ {w}")

    # Upcoming fixtures
    if show_upcoming:
        from src.data_collection.sources.worldcup import get_upcoming_matches
        import pandas as pd

        print()
        print("  ── Upcoming Fixtures ──")
        print()

        try:
            df = pd.read_csv(report["path"], parse_dates=["date"])
            upcoming = get_upcoming_matches(df)
            if len(upcoming) == 0:
                print("  (no upcoming fixtures — all matches completed)")
            else:
                for _, row in upcoming.iterrows():
                    date_str = row["date"].strftime("%d %b") if pd.notna(row.get("date")) else "TBD"
                    home = row.get("home_team", "?")
                    away = row.get("away_team", "?")
                    round_name = row.get("round", "")
                    print(f"  {date_str:>8}  {home:<20} vs {away:<20}  [{round_name}]")
        except Exception as e:
            print(f"  (error loading fixtures: {e})")
        print()

    # Detailed summary
    if show_summary:
        print()
        print("  ── Match Summary ──")
        print()

        import pandas as pd
        df = pd.read_csv(report["path"], parse_dates=["date"])

    print()
    print("  Run with --list-teams to see all 48 teams,")
    print("  --list-groups to see group assignments, or")
    print("  --upcoming to list fixtures without results.")
    print()


if __name__ == "__main__":
    main()
