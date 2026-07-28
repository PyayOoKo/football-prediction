"""
collect_matches.py — Comprehensive match data collection for O/U & BTTS models.

Exports all available matches from the database to structured CSV files.
For leagues with zero data (NO2, FI2), attempts SofaScore events API.

Output files:
    data/matches.csv           — All match results (2016-2024 minimum)
    data/matches_NO2.csv       — NO2 matches from SofaScore (if available)
    data/matches_FI2.csv       — FI2 matches from SofaScore (if available)
"""

from __future__ import annotations

import csv
import logging
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Project root ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from curl_cffi import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("collect_matches")

DB_PATH = PROJECT_ROOT / "data" / "football_data.db"
OUTPUT_DIR = PROJECT_ROOT / "data"

SOFASCORE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Origin": "https://www.sofascore.com",
    "Referer": "https://www.sofascore.com/",
}

SOFASCORE_LEAGUES = {
    "NO2": {"id": 22, "name": "OBOS-ligaen"},
    "FI2": {"id": 55, "name": "Ykkösliiga"},
}


# ═══════════════════════════════════════════════════════════
#  1. Export matches from DB to CSV
# ═══════════════════════════════════════════════════════════


def export_db_matches() -> dict[str, Any]:
    """Export all matches from the database to data/matches.csv.

    Returns stats about what was exported.
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Query: all matches with goals — O/U and BTTS derivable
    rows = conn.execute("""
        SELECT match_id, date, league, season, home_team, away_team,
               home_goals, away_goals, result,
               home_shots, away_shots, home_shots_target, away_shots_target,
               home_xg, away_xg,
               home_odds, draw_odds, away_odds,
               over25_odds, under25_odds,
               home_corners, away_corners, home_yellow, away_yellow, home_red, away_red,
               home_fouls, away_fouls
        FROM matches
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
        ORDER BY date ASC
    """).fetchall()

    conn.close()

    # Derive BTTS and O/U 2.5 targets
    output_path = OUTPUT_DIR / "matches.csv"
    fieldnames = [
        "match_id", "date", "league", "season",
        "home_team", "away_team",
        "home_goals", "away_goals", "result",
        "total_goals", "btts", "over_2_5",
        "home_shots", "away_shots", "home_shots_target", "away_shots_target",
        "home_xg", "away_xg",
        "home_odds", "draw_odds", "away_odds",
        "over25_odds", "under25_odds",
        "home_corners", "away_corners",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for r in rows:
            total = (r["home_goals"] or 0) + (r["away_goals"] or 0)
            row = dict(r)
            row["total_goals"] = total
            row["btts"] = 1 if r["home_goals"] and r["home_goals"] > 0 and r["away_goals"] and r["away_goals"] > 0 else 0
            row["over_2_5"] = 1 if total > 2.5 else 0
            writer.writerow(row)

    # Stats
    leagues: dict[str, int] = {}
    for r in rows:
        leagues[r["league"]] = leagues.get(r["league"], 0) + 1

    stats = {
        "total_matches": len(rows),
        "leagues": leagues,
        "date_range": {"from": rows[0]["date"] if rows else None, "to": rows[-1]["date"] if rows else None},
        "output_file": str(output_path),
    }

    logger.info("Exported %d matches to %s across %d leagues",
                len(rows), output_path, len(leagues))
    return stats


# ═══════════════════════════════════════════════════════════
#  2. Fetch NO2/FI2 matches from SofaScore events API
# ═══════════════════════════════════════════════════════════


def fetch_sofascore_seasons(tournament_id: int) -> list[dict]:
    """Fetch available seasons for a tournament."""
    url = f"https://api.sofascore.com/api/v1/unique-tournament/{tournament_id}/seasons"
    try:
        resp = requests.get(url, headers=SOFASCORE_HEADERS, impersonate="chrome120", timeout=15)
        if resp.status_code != 200:
            return []
        return sorted(resp.json().get("seasons", []), key=lambda s: s.get("year", 0), reverse=True)
    except Exception as exc:
        logger.warning("Failed to fetch seasons for tournament %d: %s", tournament_id, exc)
        return []


def fetch_sofascore_events(tournament_id: int, season_id: int) -> list[dict]:
    """Fetch finished events for a tournament season."""
    all_events: dict[int, dict] = {}
    page = 0
    while True:
        url = (f"https://api.sofascore.com/api/v1/unique-tournament/{tournament_id}"
               f"/season/{season_id}/events/last/{page}")
        try:
            resp = requests.get(url, headers=SOFASCORE_HEADERS, impersonate="chrome120", timeout=15)
        except Exception:
            break
        if resp.status_code != 200:
            break
        data = resp.json()
        events = data.get("events", [])
        if not events:
            break
        for e in events:
            if e.get("status", {}).get("type") != "finished":
                continue
            all_events[e["id"]] = e
        if not data.get("hasNextPage"):
            break
        page += 1
        time.sleep(0.15)
    return list(all_events.values())


def collect_sofascore_league(league: str, max_seasons: int = 5) -> dict[str, Any]:
    """Collect match data for a league from SofaScore events API.

    Returns stats and writes to data/matches_{league}.csv.
    """
    tconf = SOFASCORE_LEAGUES[league]
    tid = tconf["id"]

    logger.info("Fetching %s matches from SofaScore tournament #%d...", league, tid)

    seasons = fetch_sofascore_seasons(tid)
    if not seasons:
        logger.warning("No seasons found for %s", league)
        return {"matches_found": 0, "matches_written": 0}

    recent = seasons[:max_seasons]
    all_events: list[dict] = []

    for s in recent:
        sid = s["id"]
        year = s.get("year", "?")
        logger.info("  Fetching %s season %s (id=%d)...", league, year, sid)
        events = fetch_sofascore_events(tid, sid)
        all_events.extend(events)
        logger.info("    Found %d finished matches", len(events))

    if not all_events:
        logger.warning("No matches found for %s", league)
        return {"matches_found": 0, "matches_written": 0}

    # Compute result from scores
    rows = []
    for e in all_events:
        ts = e.get("startTimestamp", 0)
        date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""
        home = e.get("homeTeam", {}).get("name", "?")
        away = e.get("awayTeam", {}).get("name", "?")
        hg = e.get("homeScore", {}).get("current")
        ag = e.get("awayScore", {}).get("current")
        if hg is None or ag is None:
            continue
        total = hg + ag
        result = "H" if hg > ag else "A" if ag > hg else "D"
        rows.append({
            "date": date_str,
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "total_goals": total,
            "result": result,
            "btts": 1 if hg > 0 and ag > 0 else 0,
            "over_2_5": 1 if total > 2.5 else 0,
        })

    # Write to CSV
    output_path = OUTPUT_DIR / f"matches_{league}.csv"
    fieldnames = ["date", "home_team", "away_team", "home_goals", "away_goals",
                  "total_goals", "result", "btts", "over_2_5"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d %s matches to %s", len(rows), league, output_path)
    return {"matches_found": len(all_events), "matches_written": len(rows), "output_file": str(output_path)}


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("  MATCH DATA COLLECTION")
    print("=" * 60)

    # Step 1: Export existing DB data
    print("\n--- Step 1: Export DB matches to CSV ---")
    stats = export_db_matches()
    print(f"  Total matches: {stats['total_matches']}")
    print(f"  Leagues: {', '.join(f'{k}({v})' for k, v in sorted(stats['leagues'].items()))}")
    print(f"  Date range: {stats['date_range']['from']} to {stats['date_range']['to']}")

    # Step 2: Try SofaScore for NO2 and FI2
    print("\n--- Step 2: Fetch NO2/FI2 from SofaScore events API ---")
    for league in ["NO2", "FI2"]:
        result = collect_sofascore_league(league, max_seasons=3)
        if result.get("matches_written", 0) > 0:
            print(f"  ✅ {league}: {result['matches_written']} matches saved")
        else:
            print(f"  ⚠️  {league}: No matches collected (SofaScore API may be slow or blocked)")

    # Summary
    print("\n" + "=" * 60)
    print("  COLLECTION COMPLETE")
    print("=" * 60)
    print(f"  data/matches.csv          — {stats['total_matches']} matches from DB")
    print(f"  data/matches_NO2.csv      — SofaScore data (if available)")
    print(f"  data/matches_FI2.csv      — SofaScore data (if available)")
    print()

    # Gaps remaining
    print("  REMAINING GAPS:")
    print("  ❌ NO2, FI2 — Need FBref or FootyStats for team stats beyond scores")
    print("  ❌ Secondary leagues — No shots/corners/cards data")
    print("  ❌ Player stats — No player-level data available")
    print("  ❌ Injuries — No injury data available")
    print("  ✅ All other leagues — Full match results exported")


if __name__ == "__main__":
    main()
