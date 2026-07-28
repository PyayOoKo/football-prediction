"""
fetch_sofascore_xg.py — Fetch real xG + detailed match stats from SofaScore API.

Supports multiple secondary leagues that FBref blocks but SofaScore exposes
via its public API. Uses curl_cffi to impersonate a real browser.

Leagues
-------
    SE1 (Superettan)     → tournament_id=46  (Swedish second tier)
    NO2 (OBOS-ligaen)    → tournament_id=22  (Norwegian second tier)
    FI2 (Ykkösliiga)     → tournament_id=55  (Finnish second tier)

Usage
-----
    python scripts/fetch_sofascore_xg.py                       # All leagues
    python scripts/fetch_sofascore_xg.py --leagues SE1 NO2     # Specific leagues
    python scripts/fetch_sofascore_xg.py --leagues FI2         # Just Finnish
"""

from __future__ import annotations

import argparse
import io
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Fix Unicode output on Windows (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from curl_cffi import requests

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "football_data.db"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Origin": "https://www.sofascore.com",
    "Referer": "https://www.sofascore.com/",
}

# ── SofaScore tournament config ─────────────────────────
TOURNAMENTS: dict[str, dict[str, Any]] = {
    "SE1": {
        "id": 46,
        "competition_name": "Superettan",
        "team_map": {
            "Falkenbergs FF": "Falkenberg",
            "Norrby IF": "Norrby",
            "IFK Norrkoping": "Norrkoping",
            "IFK Norrköping": "Norrköping",
            "IFK Varnamo": "Varnamo",
            "IFK Värnamo": "Värnamo",
            "Orebro SK": "Orebro",
            "Örebro SK": "Örebro",
            "Osters IF": "Oster",
            "Östers IF": "Öster",
            "Ostersunds FK": "Ostersund",
            "Östersunds FK": "Östersund",
            "Nordic United FC": "Nordic United",
            "Helsingborgs IF": "Helsingborgs IF",
            "GIF Sundsvall": "GIF Sundsvall",
            "IK Brage": "IK Brage",
            "IK Oddevold": "IK Oddevold",
            "Varbergs BoIS": "Varbergs BoIS",
            "Landskrona BoIS": "Landskrona BoIS",
            "Ljungskile SK": "Ljungskile Sk",
            "Sandvikens IF": "Sandvikens If",
            "Utsiktens BK": "Utsikten",
            "Trelleborgs FF": "Trelleborgs FF",
            "Hammarby TFF": "Hammarby Tff",
            "Oskarshamns AIK": "Oskarshamns AIK",
            "Vasteras SK": "Vasteras SK",
            "Västerås SK": "Västerås SK",
            "Jonkopings Sodra": "Jonkopings Sodra",
            "Jönköpings Södra": "Jönköpings Södra",
            "GAIS": "GAIS",
            "AFC Eskilstuna": "AFC Eskilstuna",
            "Akropolis IF": "Akropolis IF",
            "Orgryte IS": "Orgryte IS",
            "Örgryte IS": "Örgryte",
            "Halmstads BK": "Halmstads BK",
        },
    },
    "NO2": {
        "id": 22,
        "competition_name": "1st Division",
        "team_map": {
            # Norwegian team names → DB names (common mappings)
            "Aalesunds FK": "Aalesund",
            "Aalesund": "Aalesund",
            "Bryne FK": "Bryne",
            "FK Bryne": "Bryne",
            "Fk Bryne": "Bryne",
            "Hødd": "Hodd",
            "IL Hødd": "Hodd",
            "Kongsvinger IL": "Kongsvinger",
            "Kongsvinger": "Kongsvinger",
            "Levanger FK": "Levanger",
            "Levanger": "Levanger",
            "Lyn Fotball": "Lyn",
            "Lyn 1896 FK": "Lyn",
            "Lyn": "Lyn",
            "Mjøndalen IF": "Mjoendalen",
            "Mjøndalen": "Mjoendalen",
            "Moss FK": "Moss",
            "Moss": "Moss",
            "Ranheim IL": "Ranheim",
            "Ranheim": "Ranheim",
            "Raufoss IL": "Raufoss",
            "Raufoss": "Raufoss",
            "Sandnes Ulf": "Sandnes Ulf",
            "Sandnes": "Sandnes Ulf",
            "Sogndal IL": "Sogndal",
            "Sogndal": "Sogndal",
            "Stabæk IF": "Stabaek",
            "Stabæk": "Stabaek",
            "Stabaek": "Stabaek",
            "Start IK": "Start",
            "IK Start": "Start",
            "Start": "Start",
            "Tromsdalen UIL": "Tromsdalen",
            "Vålerenga IF": "Valerenga",
            "Vålerenga": "Valerenga",
            "Valerenga": "Valerenga",
            "Åsane Fotball": "Asane",
            "Åsane": "Asane",
            "Asane": "Asane",
            "Egersunds IK": "Egersund",
            "Egersund": "Egersund",
            "Skeid Fotball": "Skeid",
            "Skeid": "Skeid",
            "Grorud IL": "Grorud",
            "Grorud": "Grorud",
            "Ull/Kisa": "Ullensaker Kisa",
            "Ullensaker": "Ullensaker Kisa",
            "KFUM Oslo": "KFUM Oslo",
            "KFUM": "KFUM Oslo",
            "Jerv": "Jerv",
            "FK Jerv": "Jerv",
        },
    },
    "FI2": {
        "id": 55,
        "competition_name": "Ykkösliiga",
        "team_map": {
            # Finnish team names → DB names
            "AC Oulu": "AC Oulu",
            "Ekenas IF": "Ekenas",
            "Ekenäs IF": "Ekenas",
            "Ekenas": "Ekenas",
            "FF Jaro": "Jaro",
            "Jaro": "Jaro",
            "FC Honka": "Honka",
            "Honka": "Honka",
            "IF Gnistan": "Gnistan",
            "Gnistan": "Gnistan",
            "IFK Mariehamn": "Mariehamn",
            "Mariehamn": "Mariehamn",
            "JJK": "JJK",
            "JJK Jyvaskyla": "JJK",
            "Jyvaskyla": "JJK",
            "KTP": "KTP",
            "KTP Kotka": "KTP",
            "KuPs": "KuPS",
            "Kuopion Palloseura": "KuPS",
            "KuPS Akatemia": "KuPS Akatemia",
            "Lahti": "Lahti",
            "FC Lahti": "Lahti",
            "MP": "MP",
            "Mikkelin Palloilijat": "MP",
            "Musan Salama": "MuSa",
            "PK-35 Vantaa": "PK-35",
            "PK-35": "PK-35",
            "PEPO": "PEPO",
            "PEPO Lappeenranta": "PEPO",
            "Pallo-Iirot": "Pallo-Iirot",
            "RoPS": "RoPS",
            "Rovaniemen Palloseura": "RoPS",
            "SJK": "SJK",
            "Seinajoen JK": "SJK",
            "SJK Akatemia": "SJK Akatemia",
            "TPS": "TPS",
            "Turun Palloseura": "TPS",
            "TPS Turku": "TPS",
            "Vaasan Palloseura": "VPS",
            "VPS": "VPS",
            "FC Inter": "Inter Turku",
            "Inter Turku": "Inter Turku",
            "Haka": "Haka",
            "FC Haka": "Haka",
        },
    },
}

# Stat mapping: SofaScore display name → (DB_home_col, DB_away_col, type_caster)
STAT_NAME_MAP: dict[str, tuple[str, str, type]] = {
    "expected goals": ("home_xg", "away_xg", float),
    "total shots": ("home_shots", "away_shots", int),
    "shots on target": ("home_shots_target", "away_shots_target", int),
    "corners": ("home_corners", "away_corners", int),
    "fouls": ("home_fouls", "away_fouls", int),
    "yellow cards": ("home_yellow", "away_yellow", int),
    "red cards": ("home_red", "away_red", int),
}


# ═══════════════════════════════════════════════════════════
#  SofaScore API helpers
# ═══════════════════════════════════════════════════════════


def fetch_seasons(tournament_id: int) -> list[dict[str, Any]]:
    """Fetch available seasons for a tournament from SofaScore API."""
    url = f"https://api.sofascore.com/api/v1/unique-tournament/{tournament_id}/seasons"
    try:
        resp = requests.get(url, headers=HEADERS, impersonate="chrome120", timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
        seasons = data.get("seasons", [])
        return sorted(seasons, key=lambda s: s.get("year", 0), reverse=True)
    except Exception as exc:
        print(f"  [WARN] Failed to fetch seasons for tournament {tournament_id}: {exc}")
        return []


def fetch_season_events(tournament_id: int, season_id: int) -> list[dict]:
    """Fetch all finished events for a tournament season."""
    all_events: dict[int, dict] = {}
    page = 0
    while True:
        url = (
            f"https://api.sofascore.com/api/v1/unique-tournament/{tournament_id}"
            f"/season/{season_id}/events/last/{page}"
        )
        try:
            resp = requests.get(url, headers=HEADERS, impersonate="chrome120", timeout=15)
        except Exception as exc:
            print(f"  [WARN] Page {page} failed: {exc}")
            break
        if resp.status_code != 200:
            break
        data = resp.json()
        events = data.get("events", [])
        if not events:
            break
        for e in events:
            status = e.get("status", {}).get("type", "")
            if status != "finished":
                continue
            # No competition_name filter needed — the API endpoint
            # unique-tournament/{id}/season/{sid}/events already scopes
            # events to the correct tournament.  Removing the filter
            # avoids silent drops from SofaScore using different display
            # names (e.g. "OBOS-ligaen" vs "1st Division").
            eid = e["id"]
            if eid not in all_events:
                all_events[eid] = e
        if not data.get("hasNextPage"):
            break
        page += 1
        time.sleep(0.15)
    return list(all_events.values())


def fetch_match_stats(event_id: int) -> dict[str, dict[str, Any]] | None:
    """Fetch detailed match statistics from SofaScore."""
    url = f"https://api.sofascore.com/api/v1/event/{event_id}/statistics"
    try:
        resp = requests.get(url, headers=HEADERS, impersonate="chrome120", timeout=10)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    stats: dict[str, dict[str, Any]] = {}
    for pg in resp.json().get("statistics", []):
        for g in pg.get("groups", []):
            for item in g.get("statisticsItems", []):
                name = item.get("name", "")
                stats[name.lower()] = {
                    "home": item.get("home"),
                    "away": item.get("away"),
                }
    return stats


def parse_stats(sofa_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Convert SofaScore stats dict to DB column values."""
    row: dict[str, Any] = {}
    for sofa_key, (db_h, db_a, caster) in STAT_NAME_MAP.items():
        if sofa_key in sofa_stats:
            s = sofa_stats[sofa_key]
            try:
                home_val = caster(s["home"]) if s["home"] not in (None, "", "-") else None
            except (ValueError, TypeError):
                home_val = None
            try:
                away_val = caster(s["away"]) if s["away"] not in (None, "", "-") else None
            except (ValueError, TypeError):
                away_val = None
            if db_h:
                row[db_h] = home_val
            if db_a:
                row[db_a] = away_val
    return row


# ═══════════════════════════════════════════════════════════
#  DB Upsert (Update or Insert)
# ═══════════════════════════════════════════════════════════


def compute_result(home_goals: int | None, away_goals: int | None) -> str | None:
    """Compute match result (H/D/A) from goals."""
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "H"
    elif away_goals > home_goals:
        return "A"
    else:
        return "D"


def upsert_match(db: sqlite3.Connection, league: str, date_str: str,
                 home_team: str, away_team: str, season_str: str,
                 home_goals: int | None, away_goals: int | None,
                 stats: dict[str, Any]) -> bool:
    """Update stats on an existing match, or INSERT a new match row if none exists.

    Returns True if a row was updated or inserted.
    """
    # Step 1: Try UPDATE first (match already exists in DB)
    if stats:
        set_clause = ", ".join(f"{c}=?" for c in stats)
        values = list(stats.values()) + [date_str, home_team, away_team, league]
        cur = db.execute(f"""
            UPDATE matches
            SET {set_clause}
            WHERE date=? AND home_team=? AND away_team=? AND league=?
        """, values)
        if cur.rowcount > 0:
            return True

    # Step 2: No existing match — INSERT one with all available data
    result = compute_result(home_goals, away_goals)

    insert_cols = ["date", "home_team", "away_team", "league", "season",
                   "home_goals", "away_goals", "result", "source"]
    insert_vals: list[Any] = [date_str, home_team, away_team, league,
                               season_str, home_goals, away_goals, result, "sofascore"]

    for col in ("home_xg", "away_xg", "home_shots", "away_shots",
                "home_shots_target", "away_shots_target",
                "home_corners", "away_corners", "home_fouls", "away_fouls",
                "home_yellow", "away_yellow", "home_red", "away_red"):
        if col in stats:
            insert_cols.append(col)
            insert_vals.append(stats[col])

    placeholders = ", ".join("?" for _ in insert_cols)
    cols_str = ", ".join(insert_cols)
    try:
        db.execute(f"INSERT INTO matches ({cols_str}) VALUES ({placeholders})", insert_vals)
        return True
    except sqlite3.IntegrityError:
        # Race condition: another process inserted this exact match between our UPDATE and INSERT
        # Try UPDATE again (should find it now)
        if stats:
            set_clause = ", ".join(f"{c}=?" for c in stats)
            values = list(stats.values()) + [date_str, home_team, away_team, league]
            cur = db.execute(f"""
                UPDATE matches
                SET {set_clause}
                WHERE date=? AND home_team=? AND away_team=? AND league=?
            """, values)
            return cur.rowcount > 0
        return False


# ═══════════════════════════════════════════════════════════
#  Per-League Runner
# ═══════════════════════════════════════════════════════════


def collect_league(
    db: sqlite3.Connection,
    league: str,
    max_seasons: int = 5,
) -> dict[str, int]:
    """Collect SofaScore stats for a league. Returns {updated, skipped, errors}."""
    tconf = TOURNAMENTS[league]
    tid = tconf["id"]
    comp_name = tconf["competition_name"]
    team_map = tconf["team_map"]

    print(f"\n{'='*60}")
    print(f"  {league} ({comp_name}) — SofaScore tournament #{tid}")
    print(f"{'='*60}")

    # Fetch available seasons
    seasons = fetch_seasons(tid)
    if not seasons:
        print(f"  [ERR] No seasons found for tournament #{tid}")
        return {"updated": 0, "skipped": 0, "errors": 1}

    # Take the N most recent seasons
    recent = seasons[:max_seasons]
    print(f"  Found {len(seasons)} seasons, processing {len(recent)} most recent")

    total_updated = 0
    total_skipped = 0
    total_errors = 0

    for s in recent:
        season_id = s["id"]
        year_label = s.get("year", str(season_id))
        print(f"\n  ── Season {year_label} (id={season_id}) ──")

        events = fetch_season_events(tid, season_id)
        if not events:
            print(f"  No finished {comp_name} matches found for {year_label}")
            continue

        print(f"  Found {len(events)} finished matches")

        season_updated = 0
        season_skipped = 0

        for e in events:
            eid = e["id"]
            ts = e.get("startTimestamp", 0)
            dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""

            # Map team names
            home_raw = e["homeTeam"]["name"]
            away_raw = e["awayTeam"]["name"]
            home = team_map.get(home_raw, home_raw)
            away = team_map.get(away_raw, away_raw)

            hs = e.get("homeScore", {}).get("current", "?")
            as_ = e.get("awayScore", {}).get("current", "?")

            sofa_stats = fetch_match_stats(eid)
            stats = parse_stats(sofa_stats) if sofa_stats else {}

            # Extract scores from event object
            home_goals = e.get("homeScore", {}).get("current")
            away_goals = e.get("awayScore", {}).get("current")

            if upsert_match(db, league, dt, home, away, year_label,
                            home_goals, away_goals, stats):
                season_updated += 1
                xg = stats.get("home_xg", "?")
                xg_a = stats.get("away_xg", "?")
                print(f"  ✓ {dt} {home:25s} {hs}-{as_} {away:25s}  xG {xg} - {xg_a}")
            else:
                season_skipped += 1

            time.sleep(0.1)

        db.commit()
        print(f"  Season {year_label}: {season_updated} updated, {season_skipped} skipped")
        total_updated += season_updated
        total_skipped += season_skipped

    return {"updated": total_updated, "skipped": total_skipped, "errors": total_errors}


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Fetch xG + match stats from SofaScore for secondary leagues"
    )
    parser.add_argument(
        "--leagues", nargs="+",
        choices=list(TOURNAMENTS.keys()),
        default=list(TOURNAMENTS.keys()),
        help="Leagues to collect (default: all)",
    )
    parser.add_argument(
        "--seasons", type=int, default=5,
        help="Number of recent seasons to collect (default: 5)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  COLLECT xG + STATS FROM SOFASCORE")
    print(f"  Leagues: {', '.join(args.leagues)}")
    print(f"  Seasons: {args.seasons} most recent per league")
    print("=" * 60)

    # Enable WAL mode for concurrent access (prevents 'database is locked' errors)
    db = sqlite3.connect(str(DB_PATH))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=5000")  # Wait up to 5s if another process is writing

    grand_total = {"updated": 0, "skipped": 0, "errors": 0}

    for league in args.leagues:
        result = collect_league(db, league, max_seasons=args.seasons)
        grand_total["updated"] += result["updated"]
        grand_total["skipped"] += result["skipped"]
        grand_total["errors"] += result["errors"]

    db.close()

    print(f"\n{'='*60}")
    print(f"  FINAL: {grand_total['updated']} updated, "
          f"{grand_total['skipped']} skipped, "
          f"{grand_total['errors']} errors")
    print(f"{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
