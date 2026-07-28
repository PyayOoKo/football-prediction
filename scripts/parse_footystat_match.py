"""
Parse per-match xG + detailed stats from saved FootyStats match HTML pages.

Usage
-----
    python scripts/parse_footystat_match.py "data/scrapers/footystat/Helsingborg*.html"

Extracts: home_xg, away_xg, shots, corners, cards, fouls, offsides
and updates the DB.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("parse_match")

DB_PATH = Path("data/football_data.db")


def parse_match_html(html_path: str) -> dict | None:
    path = Path(html_path)
    if not path.exists():
        logger.error("File not found: %s", path)
        return None

    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    # --- Match info ---
    match_info = soup.find("div", class_="match-info")
    if not match_info:
        logger.warning("No match-info div found in %s", path.name)
        return None
    info_text = match_info.get_text(strip=True)

    # Extract teams from match-name
    match_name = soup.find("div", class_="match-name")
    if not match_name:
        logger.warning("No match-name div found")
        return None

    # Date: try to find from stats table header or match-info text
    date_str = _extract_date(info_text)

    # Teams: from match-name
    teams = _extract_teams(match_name.get_text(strip=True))
    if not teams:
        logger.warning("Could not extract teams from %s", path.name)

    # Score: from scoreline or from stats section "Final Results"
    score_text = ""
    score_el = soup.find("div", class_="scoreline")
    if score_el:
        score_text = score_el.get_text(strip=True)
    if not score_text or "Stadium" in score_text:
        # Try the stats section
        stats_section = soup.find("section", class_=lambda c: c and "stat-box" in c and "ft-data" in c)
        if stats_section:
            st = stats_section.get_text(strip=True)
            m = re.search(r"Final\s*Results?\s*(\d+)\s*[-–]\s*(\d+)", st, re.IGNORECASE)
            if m:
                score_text = f"{m.group(1)}-{m.group(2)}"

    home_goals, away_goals = _extract_score(score_text)

    # --- Stats table ---
    table = soup.find("table")
    stats = {"home_shots": None, "away_shots": None,
             "home_shots_target": None, "away_shots_target": None,
             "home_corners": None, "away_corners": None,
             "home_fouls": None, "away_fouls": None,
             "home_yellow": None, "away_yellow": None,
             "home_offside": None, "away_offside": None,
             "home_xg": None, "away_xg": None}

    if table:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            label = cells[0].get_text(strip=True).lower()
            vals = [c.get_text(strip=True) for c in cells[1:]]
            if "possession" in label:
                continue
            elif "shots on target" in label:
                stats["home_shots_target"] = _safe_float(vals[0])
                stats["away_shots_target"] = _safe_float(vals[1])
            elif label == "shots" and "target" not in label:
                stats["home_shots"] = _safe_float(vals[0])
                stats["away_shots"] = _safe_float(vals[1])
            elif "corners" in label:
                stats["home_corners"] = _safe_float(vals[0])
                stats["away_corners"] = _safe_float(vals[1])
            elif "fouls" in label:
                stats["home_fouls"] = _safe_float(vals[0])
                stats["away_fouls"] = _safe_float(vals[1])
            elif "cards" in label or label == "cards":
                stats["home_yellow"] = _safe_float(vals[0])
                stats["away_yellow"] = _safe_float(vals[1])
            elif "offside" in label:
                stats["home_offside"] = _safe_float(vals[0])
                stats["away_offside"] = _safe_float(vals[1])
            elif label == "xg":
                stats["home_xg"] = _safe_float(vals[0])
                stats["away_xg"] = _safe_float(vals[1])

    if stats["home_xg"] is None and stats["away_xg"] is None:
        logger.warning("No xG data found in %s", path.name)

    match_data = {
        "date": date_str,
        "home_team": teams[0] if teams else None,
        "away_team": teams[1] if teams else None,
        "home_goals": home_goals,
        "away_goals": away_goals,
        **stats,
    }

    league = _guess_league(path.name)
    match_data["league"] = league
    return match_data


def _extract_date(text: str) -> str | None:
    m = re.search(r"(\w{3,9}\s+\d{1,2},?\s+\d{4})", text)
    if m:
        from datetime import datetime
        for fmt in ["%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"]:
            try:
                return datetime.strptime(m.group(1), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    m2 = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m2:
        return m2.group(1)
    return None


def _extract_teams(text: str) -> tuple[str, str] | None:
    m = re.match(r"(.+?)\s+vs\s+(.+)", text, re.IGNORECASE)
    if m:
        home = m.group(1).strip()
        away_raw = m.group(2).strip()
        # Stop at Stats (may be concatenated like "IFStats")
        away = re.split(r"(?:Stats|H2H|xG)", away_raw)[0].strip()
        return (home, away)
    parts = text.split("vs")
    if len(parts) == 2:
        return (parts[0].strip(), parts[1].strip())
    return None


def _extract_score(text: str) -> tuple[int | None, int | None]:
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)", text)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (None, None)


def _safe_float(v: str) -> float | None:
    try:
        return float(v.replace(",", ".").strip())
    except (ValueError, AttributeError):
        return None


def _guess_league(filename: str) -> str:
    fname = filename.lower()
    if "superettan" in fname:
        return "SE1"
    if "obos" in fname or "ligaen" in fname:
        return "NO2"
    if "ykkös" in fname:
        return "FI2"
    return "SE1"


def update_db(match: dict) -> bool:
    if not match["home_team"] or not match["away_team"]:
        logger.warning("Missing teams, skipping update")
        return False

    from football_data.database.sqlite import SCHEMA_SQL
    set_parts = []
    params = []
    for col in ["home_xg", "away_xg", "home_shots", "away_shots",
                "home_shots_target", "away_shots_target",
                "home_corners", "away_corners", "home_fouls", "away_fouls",
                "home_yellow", "away_yellow"]:
        val = match.get(col)
        if val is not None:
            set_parts.append(f"{col} = ?")
            params.append(val)

    if not set_parts:
        return False

    # Map FootyStats display names to our DB names
    FOOTYSTAT_TO_DB = {
        "Falkenbergs FF": "Falkenberg",
        "Helsingborgs IF": "Helsingborgs IF",
        "GIF Sundsvall": "GIF Sundsvall",
        "IFK Norrköping": "Norrköping",
        "IFK Värnamo": "Värnamo",
        "IK Brage": "IK Brage",
        "IK Oddevold": "IK Oddevold",
        "Landskrona BoIS": "Landskrona BoIS",
        "Ljungskile SK": "Ljungskile Sk",
        "Norrby IF": "Norrby",
        "Sandvikens IF": "Sandvikens If",
        "United IK Nordic": "Nordic United",
        "Varbergs BoIS FC": "Varbergs BoIS",
        "Örebro SK": "Örebro",
        "Östers IF": "Öster",
        "Östersunds FK": "Östersund",
    }
    match["home_team"] = FOOTYSTAT_TO_DB.get(match["home_team"], match["home_team"])
    match["away_team"] = FOOTYSTAT_TO_DB.get(match["away_team"], match["away_team"])

    sql = f"""
        UPDATE matches
        SET {', '.join(set_parts)}
        WHERE league = ?
          AND date = ?
          AND home_team = ?
          AND away_team = ?
    """
    params += [match["league"], match["date"], match["home_team"], match["away_team"]]

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.execute(sql, params)
    conn.commit()
    rows = cur.rowcount
    conn.close()
    return rows > 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_footystat_match.py <html_file_or_glob>")
        sys.exit(1)

    files = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.exists():
            files.append(p)
        else:
            from glob import glob
            files.extend(Path(f) for f in glob(arg))

    if not files:
        print("No files found.")
        sys.exit(1)

    for fpath in sorted(files):
        if not fpath.name.endswith(".html"):
            continue
        logger.info("Parsing %s", fpath.name)
        match = parse_match_html(str(fpath))
        if not match:
            logger.warning("Failed to parse %s", fpath.name)
            continue

        print(f"\n{'='*60}")
        print(f"File: {fpath.name}")
        print(f"  Date:       {match['date']}")
        print(f"  Teams:      {match['home_team']} vs {match['away_team']}")
        print(f"  Score:      {match['home_goals']} - {match['away_goals']}")
        print(f"  xG:         {match['home_xg']} - {match['away_xg']}")
        print(f"  Shots:      {match['home_shots']} - {match['away_shots']}")
        print(f"  ShotsTarg:  {match['home_shots_target']} - {match['away_shots_target']}")
        print(f"  Corners:    {match['home_corners']} - {match['away_corners']}")
        print(f"  Fouls:      {match['home_fouls']} - {match['away_fouls']}")
        print(f"  Cards:      {match['home_yellow']} - {match['away_yellow']}")

        updated = update_db(match)
        print(f"  DB update:  {'OK' if updated else 'FAILED (match not found)'}")


if __name__ == "__main__":
    main()
