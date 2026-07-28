"""
Estimate xG for leagues without detailed match stats using Dixon-Coles.

Usage
-----
    python scripts/estimate_xg.py --league SE1

Estimates expected goals (home_xg, away_xg) from the Dixon-Coles model,
which learns attack/defence parameters from historical goal data.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dixon_coles import DixonColesModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("estimate_xg")

TARGET_LEAGUES = {"SE1", "NO2", "FI2", "FI3", "IRL", "D2", "P1"}


def estimate_xg(
    df: pd.DataFrame, league: str,
) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    dc = DixonColesModel(
        decay_halflife_days=1460.0,
        use_importance=False,
        prior_strength=0.01,
    )
    dc.fit(
        df=df,
        home_team_col="home_team",
        away_team_col="away_team",
        home_goals_col="home_goals",
        away_goals_col="away_goals",
        date_col="date",
        verbose=True,
    )

    home_xg_list = []
    away_xg_list = []
    for _, row in df.iterrows():
        lam, mu = dc.expected_goals(row["home_team"], row["away_team"])
        home_xg_list.append(lam)
        away_xg_list.append(mu)

    df["home_xg"] = home_xg_list
    df["away_xg"] = away_xg_list
    return df


def update_db(conn: sqlite3.Connection, df: pd.DataFrame, league: str) -> int:
    updated = 0
    for _, row in df.iterrows():
        hxg = float(row["home_xg"])
        axg = float(row["away_xg"])
        cur = conn.execute(
            """
            UPDATE matches
            SET home_xg = ?, away_xg = ?
            WHERE league = ?
              AND date = ?
              AND home_team = ?
              AND away_team = ?
              AND (home_xg IS NULL OR away_xg IS NULL)
            """,
            (hxg, axg, row["league"], row["date"], row["home_team"], row["away_team"]),
        )
        updated += cur.rowcount
    return updated


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", "-l", default="SE1", help="League code")
    args = parser.parse_args()

    league = args.league.upper()
    if league not in TARGET_LEAGUES:
        print(f"Unknown league: {league}. Choose from: {sorted(TARGET_LEAGUES)}")
        return

    db_path = Path("data/football_data.db")
    conn = sqlite3.connect(str(db_path))

    query = """
        SELECT * FROM matches
        WHERE league = ?
          AND home_goals IS NOT NULL
          AND away_goals IS NOT NULL
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(league,))

    logger.info("Loaded %d historical matches for %s", len(df), league)

    before = (df["home_xg"].notna() | df["away_xg"].notna()).sum()
    logger.info("Matches with existing xG: %d", before)

    df = estimate_xg(df, league)
    updated = update_db(conn, df, league)
    conn.commit()

    logger.info("Updated %d rows with estimated xG", updated)
    logger.info(
        "xG range: home [%.3f, %.3f], away [%.3f, %.3f]",
        df["home_xg"].min(), df["home_xg"].max(),
        df["away_xg"].min(), df["away_xg"].max(),
    )
    conn.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
