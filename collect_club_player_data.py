"""collect_club_player_data.py — Scrape Transfermarkt squad data for top 5 league clubs.

Downloads per-player information (age, market value, position, injury status)
for every club in the top 5 European leagues (E0, SP1, D1, I1, F1), saves to
``data/external/players.csv``, ready for ``src.player_info.add_player_features()``.

Team name normalisation
-----------------------
Transfermarkt uses full club names (e.g. "Manchester City") while the
match database uses abbreviated names (e.g. "Man City"). This script
auto-maps TM names to DB names using fuzzy matching, so player features
correctly connect to match data.

Usage
-----
    python collect_club_player_data.py                           # All top 5 leagues
    python collect_club_player_data.py --leagues E0,SP1          # Specific leagues
    python collect_club_player_data.py --dry-run                 # Preview only
    python collect_club_player_data.py --delay 2.0               # Polite slow mode
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Reuse existing infrastructure ──────────────────────

from src.data_collection.sources.transfermarkt import (
    _parse_market_value,
    _normalise_position,
    _scrape_squad_page,
    PlayerRecord,
)

# ── Logging ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_club_player_data")

# ── Constants ──────────────────────────────────────────

TRANSFERMARKT_BASE = "https://www.transfermarkt.com"
REQUEST_TIMEOUT = 20

# League codes → (Transfermarkt league slug, league ID, display name)
# League IDs are Transfermarkt's competition codes (e.g. GB1, ES1, etc.)
LEAGUE_CONFIG: dict[str, tuple[str, str, str]] = {
    "E0":  ("premier-league",  "GB1", "Premier League"),
    "SP1": ("laliga",          "ES1", "La Liga"),
    "D1":  ("bundesliga",      "L1",  "Bundesliga"),
    "I1":  ("serie-a",         "IT1", "Serie A"),
    "F1":  ("ligue-1",         "FR1", "Ligue 1"),
}

PLAYERS_CSV = Path("data/external/players.csv")


# ═══════════════════════════════════════════════════════════
#  Club discovery from Transfermarkt league overview pages
# ═══════════════════════════════════════════════════════════


def discover_clubs_from_league(
    league_code: str,
    sess: requests.Session,
    season: int | None = None,
) -> list[dict[str, Any]]:
    """Scrape a Transfermarkt league overview page to discover club names & IDs.

    Returns
    -------
    list[dict]
        Each dict: {tm_name, tm_id, slug}
    """
    league_slug, league_id, _ = LEAGUE_CONFIG[league_code]

    url = f"{TRANSFERMARKT_BASE}/{league_slug}/startseite/wettbewerb/{league_id}"
    if season:
        url += f"/plus/?saison_id={season}"

    logger.info("  Discovering clubs from %s ...", url)

    resp = sess.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    clubs: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    # Find all club links pointing to /{slug}/startseite/verein/{id}
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        m = re.search(r"/([^/]+)/startseite/verein/(\d+)", href)
        if m:
            tm_id = int(m.group(2))
            if tm_id in seen_ids:
                continue
            seen_ids.add(tm_id)

            # Get club name from link text or img alt
            name = a_tag.get_text(strip=True)
            if not name or len(name) < 2:
                img = a_tag.find("img")
                if img and img.get("alt"):
                    name = img["alt"].strip()

            if name:
                clubs.append({
                    "tm_name": name.strip(),
                    "tm_id": tm_id,
                    "slug": m.group(1),
                })

    # Deduplicate by name
    unique: dict[str, dict[str, Any]] = {}
    for c in clubs:
        key = c["tm_name"].lower().strip()
        # Keep the entry with the longer name (more descriptive)
        if key not in unique or len(c["tm_name"]) > len(unique[key]["tm_name"]):
            unique[key] = c

    return list(unique.values())


# ═══════════════════════════════════════════════════════════
#  Team name normalisation — TM names → DB names
# ═══════════════════════════════════════════════════════════

# Manual mapping for known name differences
# Format: TM display name → our database name
TM_TO_DB_OVERRIDE: dict[str, str] = {
    # Premier League (E0)
    "Manchester City":           "Man City",
    "Manchester United":         "Man United",
    "Newcastle United":          "Newcastle",
    "Tottenham Hotspur":         "Tottenham",
    "Wolverhampton Wanderers":   "Wolves",
    "Nottingham Forest":         "Nott'm Forest",
    "Ipswich Town":              "Ipswich",
    "Leeds United":              "Leeds",
    "AFC Bournemouth":           "Bournemouth",
    "Brighton & Hove Albion":    "Brighton",
    "West Ham United":           "West Ham",
    "West Bromwich Albion":      "West Brom",
    "Hull City":                 "Hull",
    "Norwich City":              "Norwich",
    "Leicester City":            "Leicester",
    "Sheffield United":          "Sheffield United",
    "Queens Park Rangers":       "Qpr",
    "Coventry City":             "Coventry",
    "Stoke City":                "Stoke",
    "Sunderland AFC":            "Sunderland",
    "Swansea City":              "Swansea",
    "Derby County":              "Derby",
    "Reading FC":                "Reading",
    "Watford FC":                "Watford",
    "Birmingham City":           "Birmingham",
    "Blackburn Rovers":          "Blackburn",
    "Bolton Wanderers":          "Bolton",
    "Burnley FC":                "Burnley",
    "Cardiff City":              "Cardiff",
    "Wigan Athletic":            "Wigan",
    "Middlesbrough FC":          "Middlesbrough",
    "Luton Town":                "Luton",
    "Huddersfield Town":         "Huddersfield",
    # La Liga (SP1)
    "Athletic Bilbao":           "Ath Bilbao",
    "Atl�?©tico Madrid":            "Ath Madrid",
    "Atlético Madrid":           "Ath Madrid",
    "Atletico Madrid":           "Ath Madrid",
    "Alavés":                    "Alaves",
    "Almería":                   "Almeria",
    "Cádiz":                     "Cadiz",
    "Córdoba":                   "Cordoba",
    "Elche CF":                  "Elche",
    "RCD Espanyol Barcelona":    "Espanol",
    "RCD Espanyol":              "Espanol",
    "Girona FC":                 "Girona",
    "Las Palmas":                "Las Palmas",
    "Levante UD":                "Levante",
    "Málaga CF":                 "Malaga",
    "Mallorca":                  "Mallorca",
    "CA Osasuna":                "Osasuna",
    "Burgos CF":                 "Burgos",
    "Real Valladolid":           "Valladolid",
    "Rayo Vallecano":            "Vallecano",
    "Racing Santander":          "Santander",
    "Real Sociedad":             "Sociedad",
    "Sporting Gijón":            "Sp Gijon",
    "Real Oviedo":                "Oviedo",
    "Deportivo Alavés":          "Alaves",
    "Deportivo La Coruña":       "La Coruna",
    "Villarreal CF":             "Villarreal",
    "Real Betis Balompié":       "Betis",
    "Real Betis":                "Betis",
    # Bundesliga (D1)
    "Eintracht Frankfurt":       "Ein Frankfurt",
    "Borussia Mönchengladbach":  "M'gladbach",
    "Borussia M'gladbach":       "M'gladbach",
    "1. FC Köln":                "Fc Koln",
    "1. FC Kaiserslautern":      "Kaiserslautern",
    "1. FC Nürnberg":            "Nurnberg",
    "1. FC Union Berlin":        "Union Berlin",
    "SV Werder Bremen":          "Werder Bremen",
    "FC Schalke 04":             "Schalke 04",
    "FC Augsburg":               "Augsburg",
    "SC Paderborn 07":           "Paderborn",
    "SV Darmstadt 98":           "Darmstadt",
    "SpVgg Greuther Fürth":      "Greuther Furth",
    "VfL Wolfsburg":             "Wolfsburg",
    "TSG 1899 Hoffenheim":       "Hoffenheim",
    "Hertha BSC":                "Hertha",
    "Fortuna Düsseldorf":        "Fortuna Dusseldorf",
    "Arminia Bielefeld":         "Bielefeld",
    "Hannover 96":               "Hannover",
    "FC St. Pauli":              "St Pauli",
    "VfB Stuttgart":             "Stuttgart",
    "Eintracht Braunschweig":    "Braunschweig",
    "Holstein Kiel":             "Holstein Kiel",
    "SV 07 Elversberg":          "Elversberg",
    "FC Ingolstadt 04":          "Ingolstadt",
    "Hamburger SV":              "Hamburg",
    "1. FC Heidenheim 1846":     "Heidenheim",
    "RB Leipzig":                "Rb Leipzig",
    "Bayer 04 Leverkusen":       "Leverkusen",
    "Borussia Dortmund":         "Dortmund",
    # Serie A (I1)
    "AC Milan":                  "Milan",
    "FC Internazionale":         "Inter",
    "Inter Milan":               "Inter",
    "AS Roma":                   "Roma",
    "SSC Napoli":                "Napoli",
    "SS Lazio":                  "Lazio",
    "ACF Fiorentina":            "Fiorentina",
    "Bologna FC 1909":           "Bologna",
    "FC Empoli":                 "Empoli",
    "Cagliari Calcio":           "Cagliari",
    "Genoa CFC":                 "Genoa",
    "Udinese Calcio":            "Udinese",
    "Torino FC":                 "Torino",
    "Hellas Verona":             "Verona",
    "US Salernitana 1919":       "Salernitana",
    "AC Monza":                  "Monza",
    "Como 1907":                 "Como",
    "Frosinone Calcio":          "Frosinone",
    "US Lecce":                  "Lecce",
    "Parma Calcio 1913":         "Parma",
    "Sampdoria":                 "Sampdoria",
    "Spezia Calcio":             "Spezia",
    "SPAL":                      "Spal",
    "Venezia FC":                "Venezia",
    "AC ChievoVerona":           "Chievo",
    "Carpi FC 1909":             "Carpi",
    "FC Crotone":                  "Crotone",
    "Pescara Calcio":            "Pescara",
    "Benevento Calcio":          "Benevento",
    "US Sassuolo":               "Sassuolo",
    "Siena FC":                  "Siena",
    "Novara Calcio":             "Novara",
    "US Cremonese":              "Cremonese",
    "Palermo FC":                "Palermo",
    "Pisa Sporting Club":        "Pisa",
    # Ligue 1 (F1)
    "Paris Saint-Germain":       "Paris Sg",
    "Paris FC":                  "Paris Fc",
    "Olympique Lyonnais":        "Lyon",
    "Olympique Marseille":       "Marseille",
    "Olympique de Marseille":    "Marseille",
    "Stade Rennais FC":          "Rennes",
    "Stade de Reims":            "Reims",
    "Stade Brestois 29":         "Brest",
    "RC Strasbourg Alsace":      "Strasbourg",
    "FC Toulouse":               "Toulouse",
    "LOSC Lille":                "Lille",
    "RC Lens":                   "Lens",
    "AS Monaco":                 "Monaco",
    "OGC Nice":                  "Nice",
    "Montpellier HSC":           "Montpellier",
    "Montpellier Hérault SC":    "Montpellier",
    "Angers SCO":                "Angers",
    "FC Nantes":                 "Nantes",
    "AJ Auxerre":                "Auxerre",
    "Le Havre AC":               "Le Havre",
    "FC Lorient":                "Lorient",
    "FC Metz":                   "Metz",
    "FC Girondins Bordeaux":     "Bordeaux",
    "EA Guingamp":               "Guingamp",
    "ESTAC Troyes":              "Troyes",
    "SC Amiens":                 "Amiens",
    "Clermont Foot 63":          "Clermont",
    "Dijon FCO":                 "Dijon",
    "Nîmes Olympique":           "Nimes",
    "SM Caen":                   "Caen",
    "AC Ajaccio":                "Ajaccio",
    "GFC Ajaccio":               "Ajaccio Gfco",
    "FC Sochaux-Montbéliard":    "Sochaux",
    "AS Nancy Lorraine":         "Nancy",
    "Valenciennes FC":           "Valenciennes",
    "Évian Thonon Gaillard FC":  "Evian Thonon Gaillard",
    "SC Bastia":                 "Bastia",
}

# ── Fuzzy name matcher ─────────────────────────────────


def _normalise_team_name(tm_name: str, db_team_names: set[str]) -> str:
    """Map a Transfermarkt team name to the closest database team name.

    Priority:
    1. Exact match in override dict
    2. Case-insensitive match
    3. Contains match (TM name contains DB name or vice versa)
    4. First-word match
    5. Fallback to original TM name
    """
    # 1. Direct override
    if tm_name in TM_TO_DB_OVERRIDE:
        mapped = TM_TO_DB_OVERRIDE[tm_name]
        if mapped in db_team_names:
            return mapped
        # Fall through — the mapped name might still not be exact

    # 2. Case-insensitive match
    tm_lower = tm_name.lower().strip()
    for db_name in db_team_names:
        if db_name.lower().strip() == tm_lower:
            return db_name

    # 3. Contains match
    for db_name in sorted(db_team_names, key=len, reverse=True):
        db_lower = db_name.lower().strip()
        # TM name contains DB name, or DB name contains TM name
        if db_lower in tm_lower or tm_lower in db_lower:
            return db_name

    # 4. First word match (e.g. "Man" in "Manchester City" → "Man City")
    tm_first = tm_lower.split()[0] if tm_lower.split() else ""
    for db_name in db_team_names:
        if db_name.lower().startswith(tm_first):
            return db_name

    # 5. Fallback
    return tm_name


# ═══════════════════════════════════════════════════════════
#  Main collection
# ═══════════════════════════════════════════════════════════


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Transfermarkt squad data for top 5 league clubs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--leagues", type=str, default="E0,SP1,D1,I1,F1",
        help="Comma-separated league codes (default: all top 5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scrape but don't save to CSV",
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="Seconds between requests (default 1.5)",
    )
    parser.add_argument(
        "--season", type=int, default=None,
        help="Specific season year (default: latest available)",
    )
    parser.add_argument(
        "--max-clubs", type=int, default=999,
        help="Max clubs to scrape per league (default: all)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    league_codes = [c.strip() for c in args.leagues.split(",")]

    # Validate league codes
    for code in league_codes:
        if code not in LEAGUE_CONFIG:
            logger.warning("Unknown league code '%s' — skipping", code)
            league_codes.remove(code)

    if not league_codes:
        logger.error("No valid leagues to collect!")
        return 1

    # ── Load DB team names for normalisation ──
    import sqlite3
    db_path = Path("data/football_data.db")
    db_team_names: set[str] = set()
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            for code in league_codes:
                cur = conn.cursor()
                cur.execute(
                    "SELECT DISTINCT home_team FROM matches WHERE league = ? "
                    "UNION SELECT DISTINCT away_team FROM matches WHERE league = ?",
                    (code, code),
                )
                db_team_names.update(r[0] for r in cur.fetchall())
            conn.close()
            logger.info("Loaded %d DB team names for name normalisation", len(db_team_names))
        except Exception as exc:
            logger.warning("Could not load DB team names: %s", exc)

    # ── Build session ──
    sess = requests.Session()
    retries = Retry(total=3, backoff_factor=2.0, status_forcelist=[502, 503, 504])
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
    })

    # ── Collect ──
    all_records: list[dict[str, Any]] = []
    total_clubs = 0
    missed: list[str] = []
    name_map: dict[str, str] = {}  # TM name → DB name (for report)

    t0 = time.time()

    for league_code in league_codes:
        _, _, league_name = LEAGUE_CONFIG[league_code]
        logger.info("\n─── %s (%s) ───", league_name, league_code)

        # Discover clubs from league page
        clubs = discover_clubs_from_league(league_code, sess, args.season)
        if not clubs:
            logger.warning("  No clubs discovered for %s", league_code)
            continue

        logger.info("  Discovered %d clubs in %s", len(clubs), league_name)

        # Scrape squad for each club
        for idx, club in enumerate(clubs[:args.max_clubs]):
            club_name = club["tm_name"]
            tm_id = club["tm_id"]
            slug = club["slug"]

            logger.info(
                "  [%d/%d] %s (ID %d) ...",
                idx + 1, min(len(clubs), args.max_clubs),
                club_name, tm_id,
            )

            # Use the existing squad scraper
            url = f"{TRANSFERMARKT_BASE}/{slug}/kader/verein/{tm_id}"
            try:
                players = _scrape_squad_page(url, club_name, sess)
            except Exception as exc:
                logger.warning("    [W] Failed: %s", exc)
                missed.append(f"{club_name} ({tm_id})")
                continue

            # Convert PlayerRecord objects to dicts
            if players:
                for p in players:
                    # Normalise team name
                    db_team = _normalise_team_name(club_name, db_team_names)
                    if db_team != club_name:
                        name_map[club_name] = db_team
                    record = p.to_dict()
                    record["team"] = db_team  # Use normalised name
                    all_records.append(record)
                logger.info("    -> %d players → DB team: '%s'", len(players), db_team)
            else:
                missed.append(f"{club_name} ({tm_id})")

            total_clubs += 1
            if idx < len(clubs) - 1:
                time.sleep(args.delay)

    # ── Build DataFrame ──
    if not all_records:
        logger.warning("No player data collected!")
        return 1

    df = pd.DataFrame(all_records)

    # Ensure correct dtypes
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(25.0)
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0.0)
    df["is_starter"] = df["is_starter"].astype(bool)
    df["injured"] = df["injured"].astype(bool)
    df["suspended"] = df["suspended"].astype(bool)
    df["goals_scored"] = pd.to_numeric(df["goals_scored"], errors="coerce").fillna(0).astype(int)

    elapsed = time.time() - t0

    # ── Report ──
    print(f"\n{'=' * 60}")
    print(f"  CLUB PLAYER DATA COLLECTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Leagues:      {', '.join(league_codes)}")
    print(f"  Clubs scraped: {total_clubs}")
    print(f"  Players total: {len(df)}")
    print(f"  Duration:      {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Team name mapping success rate
    matched = sum(1 for tm in name_map.values() if tm.lower() != tm)
    total_tm_names = total_clubs
    match_rate = matched / total_tm_names * 100 if total_tm_names > 0 else 0
    print(f"\n  Team name matching: {matched}/{total_tm_names} ({match_rate:.0f}%)")

    # Per-team stats
    print(f"\n  Top teams by squad value:")
    print(f"  {'Team':<30} {'Players':<8} {'Age':<5} {'Value (€m)':<12} {'Injured':<8}")
    print(f"  {'-' * 63}")
    team_stats = df.groupby("team").agg(
        players=("player_name", "count"),
        avg_age=("age", "mean"),
        squad_value=("market_value", "sum"),
        injuries=("injured", "sum"),
    ).sort_values("squad_value", ascending=False)
    for team_name, row in team_stats.head(20).iterrows():
        print(f"  {team_name:<30} {int(row['players']):<8} "
              f"{row['avg_age']:<5.1f} €{row['squad_value']:<10.1f}m "
              f"{int(row['injuries']):<8}")

    print(f"\n  Total squad value: €{df['market_value'].sum():.0f}m")
    print(f"  Overall avg age: {df['age'].mean():.1f} years")
    print(f"  Injured players: {int(df['injured'].sum())} / {len(df)}")

    if missed:
        print(f"\n  ⚠ Missed clubs ({len(missed)}):")
        for m in missed[:10]:
            print(f"    - {m}")
        if len(missed) > 10:
            print(f"    ... and {len(missed) - 10} more")

    if args.dry_run:
        print(f"\n  [DRY RUN] Not saved. Run without --dry-run to persist.")
        return 0

    # Save
    PLAYERS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PLAYERS_CSV, index=False)
    print(f"\n  [OK] Saved {len(df)} records to {PLAYERS_CSV}")
    print(f"  To enable in pipeline: set config.player_info.enabled = True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
