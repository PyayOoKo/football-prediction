"""
Import Oddsportal SE1 historical odds into the football database.
"""

import csv
import logging
import sqlite3
import os
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "football_data.db")
CSV_PATH = os.path.join(os.path.dirname(__file__), "oddsportal_se1_historical.csv")

OP_TO_DB = {
    "AFC Eskilstuna": "Afc Eskilstuna",
    "Akropolis": "Akropolis If",
    "Angelholm": "\u00c4ngelholm",
    "Assyriska FF": "Assyriska",
    "Atvidaberg": "\u00c5tvidaberg",
    "Brage": "IK Brage",
    "Brommapojkarna": "Brommapojkarna",
    "Dalkurd": "Dalkurd",
    "Degerfors": "Degerfors IF",
    "Falkenberg": "Falkenberg",
    "Frej": "Frej",
    "GAIS": "Gais",
    "Gefle": "Gefle IF",
    "Halmstad": "Halmstad",
    "Hammarby TFF": "Hammarby Tff",
    "Helsingborg": "Helsingborgs IF",
    "Jonkoping": "J-s\u00f6dra",
    "Kalmar": "Kalmar",
    "Landskrona": "Landskrona BoIS",
    "Ljungskile": "Ljungskile",
    "Lunds": "Lund",
    "Mjallby": "Mj\u00e4llby",
    "Norrby": "Norrby",
    "Norrkoping": "Norrk\u00f6ping",
    "Oddevold": "IK Oddevold",
    "Orebro": "\u00d6rebro",
    "Orgryte": "\u00d6rgryte",
    "Oster": "\u00d6ster",
    "Ostersund": "\u00d6stersund",
    "Sandviken": "Sandvikens If",
    "Sirius": "Sirius",
    "Skovde AIK": "Sk\u00f6vde",
    "Stockholm Internazionale": "Fc Stockholm",
    "Sundsvall": "GIF Sundsvall",
    "Syrianska": "Syrianska",
    "Trelleborg": "Trelleborgs FF",
    "Umea FC": "Ume\u00e5 Fc",
    "Utsikten": "Utsikten",
    "Varberg": "Varbergs BoIS",
    "Varnamo": "V\u00e4rnamo",
    "Vasalund": "Vasalund",
    "Vasteras SK": "V\u00e4ster\u00e5s",
    "Boden": "Boden",
    "Enkoping SK": "Enkoping SK",
    "Forward": "Forward",
    "Friska Viljor": "Friska Viljor",
    "Frolunda": "Frolunda",
    "Husqvarna": "Husqvarna",
    "Oskarshamn": "Oskarshamn",
    "Qviding": "Qviding",
    "Sylvia": "Sylvia",
    "Trollhattan": "Trollhattan",
    "Bunkeflo IF": "Bunkeflo IF",
    "Djursholm": "Djursholm",
    "Hacken": "Hacken",
    "AIK": "AIK",
    "Hammarby": "Hammarby",
}

UNMAPPED_OP_TEAMS = {
    "Boden", "Enkoping SK", "Forward", "Friska Viljor", "Frolunda",
    "Husqvarna", "Oskarshamn", "Qviding", "Sylvia", "Trollhattan",
    "Bunkeflo IF", "Djursholm", "Hacken", "AIK", "Hammarby",
}


def parse_op_date(date_str):
    date_str = date_str.split(" - ")[0].strip()
    try:
        return datetime.strptime(date_str, "%d %b %Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


def result_from_score(hg, ag):
    if hg is None or ag is None:
        return None
    return "H" if hg > ag else ("A" if hg < ag else "D")


def import_odds():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()

    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    logger.info("Total CSV rows: %d", len(rows))

    updated = 0
    inserted = 0
    no_match = 0
    bad_date = 0
    bad_odds = 0

    for r in rows:
        home = r["home_team"].strip()
        away = r["away_team"].strip()
        if not home or not away:
            continue

        home_db = OP_TO_DB.get(home)
        away_db = OP_TO_DB.get(away)

        date_str = parse_op_date(r["date"])
        if not date_str:
            bad_date += 1
            continue

        try:
            ho = float(r["odds_1"])
            do = float(r["odds_x"])
            ao = float(r["odds_2"])
        except (ValueError, TypeError):
            bad_odds += 1
            continue

        try:
            hg = int(r["home_score"])
            ag = int(r["away_score"])
        except ValueError:
            hg = ag = None

        result = result_from_score(hg, ag)
        year = int(date_str[:4])
        season = str(year)

        if home_db and away_db:
            # Try to find existing fbref row
            cursor.execute(
                """SELECT match_id, home_odds, draw_odds, away_odds
                   FROM matches
                   WHERE league='SE1' AND date=? AND home_team=? AND away_team=?
                   ORDER BY source='fbref' DESC LIMIT 1""",
                (date_str, home_db, away_db),
            )
            existing = cursor.fetchone()
        else:
            existing = None

        if existing:
            mid, eho, edo, eao = existing
            new_ho = eho if eho is not None else ho
            new_do = edo if edo is not None else do
            new_ao = eao if eao is not None else ao
            cursor.execute(
                """UPDATE matches SET home_odds=?, draw_odds=?, away_odds=?,
                   home_goals=COALESCE(home_goals,?), away_goals=COALESCE(away_goals,?),
                   result=COALESCE(result,?)
                   WHERE match_id=?""",
                (new_ho, new_do, new_ao, hg, ag, result, mid),
            )
            updated += 1
        else:
            # Insert new row
            cursor.execute(
                """INSERT OR IGNORE INTO matches
                   (source, league, season, date, home_team, away_team,
                    home_goals, away_goals, result,
                    home_odds, draw_odds, away_odds)
                   VALUES ('oddsportal', 'SE1', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (season, date_str,
                 home_db or home, away_db or away,
                 hg, ag, result,
                 ho, do, ao),
            )
            if cursor.rowcount > 0:
                inserted += 1
            else:
                no_match += 1

        if (updated + inserted) % 500 == 0:
            conn.commit()

    conn.commit()
    conn.close()

    logger.info("Done: updated=%d inserted=%d no_match=%d bad_date=%d bad_odds=%d",
                 updated, inserted, no_match, bad_date, bad_odds)


if __name__ == "__main__":
    import_odds()
