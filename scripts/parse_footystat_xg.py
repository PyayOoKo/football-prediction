"""
Parse team-level xG data from saved FootyStats HTML and
use it to calibrate per-match Dixon-Coles xG estimates.

Usage
-----
    python scripts/parse_footystat_xg.py data/scrapers/footystat/Superettan*xG*.html

Extracts team xG/xGA per game from the league table, computes
per-match scaling factors, and updates the DB.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bs4 import BeautifulSoup
from src.dixon_coles import DixonColesModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("footystat_xg")

LEAGUE_KEYWORDS = {
    "superettan": "SE1", "allsvenskan": "SE1",
    "obos": "NO2", "ykkösliiga": "FI2",
}

# Map FootyStats display names → our DB names
# FootyStats uses full club names; we use shorter/common versions
TEAM_NAME_MAP = {
    "IFK Norrköping": "Norrköping",
    "Varbergs BoIS FC": "Varbergs BoIS",
    "Falkenbergs FF": "Falkenberg",
    "GIF Sundsvall": "GIF Sundsvall",
    "Helsingborgs IF": "Helsingborgs IF",
    "IFK Värnamo": "Värnamo",
    "IK Brage": "IK Brage",
    "IK Oddevold": "IK Oddevold",
    "Landskrona BoIS": "Landskrona BoIS",
    "Ljungskile SK": "Ljungskile Sk",
    "Norrby IF": "Norrby",
    "Sandvikens IF": "Sandvikens If",
    "United IK Nordic": "Nordic United",
    "Örebro SK": "Örebro",
    "Östers IF": "Öster",
    "Östersunds FK": "Östersund",
}


def parse_team_xg_table(html_path: str) -> dict[str, dict[str, float]]:
    path = Path(html_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    # Find the xG league table
    table = soup.find("table", class_=re.compile(r"xg-all"))
    if not table:
        raise ValueError("No xG league table found in HTML")

    teams: dict[str, dict[str, float]] = {}
    all_rows = table.find_all("tr")
    for row in all_rows:
        cells = row.find_all(["th", "td"])
        # Main data rows have rank in first cell (a number) and many cells
        rank_text = cells[0].get_text(strip=True) if cells else ""
        if not rank_text.isdigit() or len(cells) < 25:
            continue
        # Team name is in cells[2]; first stripped string is the clean name
        raw_name = next(cells[2].stripped_strings, "")
        if not raw_name:
            continue
        text_cells = [c.get_text(strip=True) for c in cells]
        try:
            mp = int(text_cells[-7])
            xg = float(text_cells[-6])
            xga = float(text_cells[-5])
            teams[raw_name] = {"xG_per_game": xg, "xGA_per_game": xga, "mp": mp}
        except (ValueError, IndexError) as e:
            logger.debug("Skipping row %s: %s", raw_name[:30], e)
            continue

    logger.info("Parsed %d teams from %s", len(teams), path.name)
    return teams


def calibrate_xg(
    db_path: str, league: str, team_xg: dict[str, dict[str, float]],
) -> int:
    conn = sqlite3.connect(db_path)

    query = """
        SELECT * FROM matches
        WHERE league = ?
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(league,))
    logger.info("Loaded %d historical matches for %s", len(df), league)

    dc = DixonColesModel(
        decay_halflife_days=1460.0, use_importance=False, prior_strength=0.01,
    )
    dc.fit(
        df=df,
        home_team_col="home_team",
        away_team_col="away_team",
        home_goals_col="home_goals",
        away_goals_col="away_goals",
        date_col="date",
        verbose=False,
    )

    # Map footystats team names → our internal names
    name_map_path = Path(f"models/per_league/{league}/team_name_map.json")
    team_map = {}
    if name_map_path.exists():
        import json
        team_map = json.loads(name_map_path.read_text())
        logger.info("Loaded %d name mappings from %s", len(team_map), name_map_path)

    fn_name = lambda n: team_map.get(n, n)

    updated = 0
    for _, row in df.iterrows():
        h_team = TEAM_NAME_MAP.get(fn_name(row["home_team"]), fn_name(row["home_team"]))
        a_team = TEAM_NAME_MAP.get(fn_name(row["away_team"]), fn_name(row["away_team"]))

        dc_hxg, dc_axg = dc.expected_goals(row["home_team"], row["away_team"])

        # Scale by FootyStats team-level xG rates
        h_info = team_xg.get(h_team)
        a_info = team_xg.get(a_team)
        if h_info:
            h_factor = h_info["xG_per_game"] / (dc_hxg if dc_hxg > 0 else 0.01)
        else:
            h_factor = 1.0
        if a_info:
            a_factor = a_info["xGA_per_game"] / (dc_axg if dc_axg > 0 else 0.01)
        else:
            a_factor = 1.0

        new_hxg = dc_hxg * h_factor
        new_axg = dc_axg * a_factor

        cur = conn.execute(
            """
            UPDATE matches SET home_xg = ?, away_xg = ?
            WHERE league = ? AND date = ? AND home_team = ? AND away_team = ?
            """,
            (round(new_hxg, 4), round(new_axg, 4),
             league, row["date"], row["home_team"], row["away_team"]),
        )
        updated += cur.rowcount

    conn.commit()
    conn.close()
    logger.info("Updated %d rows with FootyStats-calibrated xG", updated)
    return updated


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_footystat_xg.py <path_to_html>")
        sys.exit(1)

    html_path = sys.argv[1]
    team_data = parse_team_xg_table(html_path)

    fname = Path(html_path).name.lower()
    league = "SE1"
    for kw, code in LEAGUE_KEYWORDS.items():
        if kw in fname:
            league = code
            break

    db_path = "data/football_data.db"
    updated = calibrate_xg(db_path, league, team_data)

    # Summary
    print(f"\nLeague: {league}  |  Teams found: {len(team_data)}")
    for t, d in sorted(team_data.items()):
        internal = TEAM_NAME_MAP.get(t, t)
        print(f"  {t:25s} → {internal:20s}  xG={d['xG_per_game']:.2f}  xGA={d['xGA_per_game']:.2f}  MP={d['mp']}")
    print(f"\nDB matches updated: {updated}")


if __name__ == "__main__":
    main()
