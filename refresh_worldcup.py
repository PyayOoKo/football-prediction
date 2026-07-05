"""
refresh_worldcup.py — Automated World Cup Data Refresh & Prediction.

Downloads the latest World Cup match data from openfootball, resolves knockout
bracket placeholders automatically, retrains the prediction model, and generates
updated predictions for all upcoming matches.

Designed to be run on a schedule (e.g., daily via Windows Task Scheduler) so
predictions stay current as the tournament progresses.

Usage:
    python refresh_worldcup.py                          # Full refresh
    python refresh_worldcup.py --dry-run                # Preview without saving
    python refresh_worldcup.py --log-file refresh.log   # Log to file
    python refresh_worldcup.py --quiet                  # Minimal output (for cron)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# ── Paths ───────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PREDICTIONS_DIR = PROJECT_ROOT / "reports" / "predictions_worldcup"
COMBINED_CSV = DATA_DIR / "worldcup_all.csv"

# ── Sources ─────────────────────────────────────────────

TOURNAMENTS: list[dict[str, Any]] = [
    {"year": 2002, "url": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2002/worldcup.json", "league": "WC"},
    {"year": 2006, "url": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2006/worldcup.json", "league": "WC"},
    {"year": 2010, "url": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2010/worldcup.json", "league": "WC"},
    {"year": 2014, "url": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2014/worldcup.json", "league": "WC"},
    {"year": 2018, "url": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2018/worldcup.json", "league": "WC"},
    {"year": 2022, "url": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2022/worldcup.json", "league": "WC"},
    {"year": 2026, "url": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json", "league": "WC"},
]

# ── Logging ─────────────────────────────────────────────

logger = logging.getLogger("refresh_worldcup")

# ── Placeholder detection ───────────────────────────────

_PLACEHOLDER_RE = re.compile(r"^([WRLP])(\d+)$")
"""Matches placeholder team names like W89, W90, L101, etc."""

# Fallback penalty shootout winners for the 2026 World Cup.
# _convert_match now parses score.p automatically from the openfootball
# JSON, so this dict is only needed in edge cases where the upstream
# data lacks penalty scores.  Format: {match_num: winner_name}
KNOWN_PENALTY_WINNERS: dict[int, str] = {}


def _is_placeholder(name: Any) -> bool:
    """Check if a team name is a bracket placeholder (W89, L101, ...)."""
    return isinstance(name, str) and bool(_PLACEHOLDER_RE.match(name))


# ═══════════════════════════════════════════════════════════
#  Step 1: Download
# ═══════════════════════════════════════════════════════════


def download_tournament(
    url: str,
    season: int,
    league: str,
    timeout: int = 30,
) -> pd.DataFrame:
    """Download a single World Cup tournament from the openfootball JSON API."""
    logger.info("Downloading %d World Cup data ...", season)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    raw_matches: list[dict[str, Any]] = data.get("matches", [])

    rows: list[dict[str, Any]] = []
    for m in raw_matches:
        row = _convert_match(m)
        row["season"] = season
        row["league"] = league
        row["source"] = f"openfootball/worldcup.json/{season}"
        rows.append(row)

    df = pd.DataFrame(rows)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df.sort_values(["date", "home_team"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["downloaded_at"] = datetime.now().isoformat()

    completed = df["result"].notna().sum()
    logger.info(
        "  %d matches (%d completed, %d upcoming)",
        len(df), completed, len(df) - completed,
    )
    return df


def _convert_match(m: dict[str, Any]) -> dict[str, Any]:
    """Convert a single openfootball match dict to the project schema."""
    team1 = m.get("team1", "")
    team2 = m.get("team2", "")
    score = m.get("score") or {}
    ft = score.get("ft") if isinstance(score, dict) else None
    ht = score.get("ht") if isinstance(score, dict) else None
    p  = score.get("p")  if isinstance(score, dict) else None   # penalties

    result: str | None = None
    home_goals: int | None = None
    away_goals: int | None = None
    home_goals_ht: int | None = None
    away_goals_ht: int | None = None

    if isinstance(ft, (list, tuple)) and len(ft) >= 2:
        try:
            home_goals = int(ft[0])
            away_goals = int(ft[1])
            if home_goals > away_goals:
                result = "H"
            elif home_goals < away_goals:
                result = "A"
            else:
                # FT drawn — check penalties for knockout matches
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    try:
                        result = "H" if int(p[0]) > int(p[1]) else "A"
                    except (TypeError, ValueError):
                        result = "D"
                else:
                    result = "D"
        except (TypeError, ValueError):
            pass

    if isinstance(ht, (list, tuple)) and len(ht) >= 2:
        try:
            home_goals_ht = int(ht[0])
            away_goals_ht = int(ht[1])
        except (TypeError, ValueError):
            pass

    return {
        "season": None,  # filled by caller
        "date": m.get("date"),
        "league": None,  # filled by caller
        "round": m.get("round"),
        "group": m.get("group"),
        "ground": m.get("ground"),
        "home_team": team1,
        "away_team": team2,
        "result": result,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_goals_ht": home_goals_ht,
        "away_goals_ht": away_goals_ht,
        "match_num": m.get("num"),
    }


# ═══════════════════════════════════════════════════════════
#  Step 2: Resolve knockout bracket placeholders
# ═══════════════════════════════════════════════════════════


def _get_match_id_winner_map(df: pd.DataFrame) -> dict[int, str | None]:
    """Build a map from match_num -> winning team name for completed matches."""
    id_to_winner: dict[int, str | None] = {}
    for _, match in df.iterrows():
        num = match.get("match_num")
        if pd.isna(num):
            continue
        num = int(num)
        team1 = str(match.get("home_team", ""))
        team2 = str(match.get("away_team", ""))
        result = match.get("result")

        if result and not pd.isna(result):
            if result == "H":
                id_to_winner[num] = team1
            elif result == "A":
                id_to_winner[num] = team2
            elif result == "D":
                if num in KNOWN_PENALTY_WINNERS:
                    id_to_winner[num] = KNOWN_PENALTY_WINNERS[num]
                else:
                    id_to_winner[num] = None  # cannot resolve from D alone

    return id_to_winner


def resolve_bracket(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve placeholder team names using completed match results.

    In the openfootball JSON, knockout matches reference previous match
    winners via placeholders like 'W89' (winner of match 89).  This function
    replaces those placeholders with actual team names once the referenced
    matches have results.
    """
    df = df.copy()
    winner_map = _get_match_id_winner_map(df)

    resolved_count = 0
    for idx, row in df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        changed = False

        # Resolve home team placeholder (W = Winner, R = Runner-up, L = Loser)
        m = _PLACEHOLDER_RE.match(home)
        if m:
            prefix, num_str = m.groups()
            ref_num = int(num_str)
            if prefix == "W" and ref_num in winner_map and winner_map[ref_num] is not None:
                df.at[idx, "home_team"] = winner_map[ref_num]
                changed = True

        # Resolve away team placeholder
        m = _PLACEHOLDER_RE.match(away)
        if m:
            prefix, num_str = m.groups()
            ref_num = int(num_str)
            if prefix == "W" and ref_num in winner_map and winner_map[ref_num] is not None:
                df.at[idx, "away_team"] = winner_map[ref_num]
                changed = True

        if changed:
            resolved_count += 1

    if resolved_count > 0:
        logger.info("  Resolved %d placeholder references from completed results", resolved_count)

    return df


# ═══════════════════════════════════════════════════════════
#  Step 3: Collect data
# ═══════════════════════════════════════════════════════════


def collect_data(dry_run: bool = False) -> pd.DataFrame | None:
    """Download all tournaments, resolve brackets, and save combined CSV."""
    logger.info("-" * 50)
    logger.info("STEP 1: Download data")
    logger.info("-" * 50)

    all_dfs: list[pd.DataFrame] = []
    for t in TOURNAMENTS:
        try:
            df = download_tournament(t["url"], t["year"], t["league"])
            all_dfs.append(df)
        except Exception as e:
            logger.error("  X %d: FAILED - %s", t["year"], e)
            return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.sort_values(["date", "home_team"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    logger.info("-" * 50)
    logger.info("STEP 2: Resolve knockout bracket")
    logger.info("-" * 50)
    combined = resolve_bracket(combined)

    unresolved = combined[
        combined["home_team"].apply(_is_placeholder)
        | combined["away_team"].apply(_is_placeholder)
    ]
    if len(unresolved) > 0:
        logger.info("  Remaining placeholder match-ups: %d", len(unresolved))
        for _, r in unresolved.head(10).iterrows():
            h = r.get("home_team", "?")
            a = r.get("away_team", "?")
            rnd = r.get("round", "?")
            date = str(r.get("date", "?"))[:10]
            logger.info("    %s | %s | %s vs %s", date, rnd, h, a)
    else:
        logger.info("  No remaining placeholder match-ups - all resolved!")

    # Report QF status
    upcoming_qf = combined[
        (combined["round"].str.contains("Quarter", na=False))
        & (combined["result"].isna())
    ]
    if len(upcoming_qf) > 0:
        qf_resolved = upcoming_qf[
            ~upcoming_qf["home_team"].apply(_is_placeholder)
            & ~upcoming_qf["away_team"].apply(_is_placeholder)
        ]
        logger.info(
            "  Quarter-finals: %d/%d with known opponents",
            len(qf_resolved), len(upcoming_qf),
        )

    # Standard output columns
    csv_cols = [
        "season", "date", "league", "round", "group", "ground",
        "home_team", "away_team",
        "result", "home_goals", "away_goals",
        "home_goals_ht", "away_goals_ht",
        "source", "downloaded_at",
    ]
    csv_cols = [c for c in csv_cols if c in combined.columns]
    combined_csv = combined[csv_cols].copy()

    completed = combined_csv["result"].notna().sum()
    upcoming = combined_csv["result"].isna().sum()

    logger.info("")
    logger.info("  Total:     %d matches", len(combined_csv))
    logger.info("  Completed: %d", completed)
    logger.info("  Upcoming:  %d", upcoming)

    if not dry_run:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        combined_csv.to_csv(COMBINED_CSV, index=False)
        logger.info("  Saved to %s", COMBINED_CSV)
    else:
        logger.info("  [DRY RUN] Not saved to disk")

    return combined_csv


# ═══════════════════════════════════════════════════════════
#  Step 4: Train & Predict
# ═══════════════════════════════════════════════════════════


def run_training(dry_run: bool = False) -> bool:
    """Run the training and prediction script.

    Returns True on success, False on failure.
    """
    logger.info("-" * 50)
    logger.info("STEP 3: Train & Predict")
    logger.info("-" * 50)

    train_script = PROJECT_ROOT / "train_worldcup.py"
    if not train_script.exists():
        logger.error("  train_worldcup.py not found at %s", train_script)
        return False

    if dry_run:
        logger.info("  [DRY RUN] Would run: python train_worldcup.py")
        return True

    python_exe = _get_python_exe()

    cmd = [python_exe, "-u", str(train_script)]
    logger.info("  Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(PROJECT_ROOT),
        )

        for line in result.stdout.splitlines():
            logger.info("  %s", line)
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.warning("  [stderr] %s", line)

        if result.returncode != 0:
            logger.error("  Training failed with exit code %d", result.returncode)
            return False

        logger.info("  Training & prediction completed successfully")
        return True

    except subprocess.TimeoutExpired:
        logger.error("  Training timed out (10 min limit)")
        return False
    except Exception as e:
        logger.error("  Training failed: %s", e)
        return False


def _get_python_exe() -> str:
    """Find the correct Python executable."""
    candidates: list[str] = [
        r"C:\Users\dell\AppData\Local\Python\pythoncore-3.14-64\python.exe",
        "python",
        "python3",
    ]
    if sys.executable and os.path.exists(sys.executable):
        candidates.insert(0, sys.executable)

    for exe in candidates:
        try:
            r = subprocess.run(
                [exe, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                logger.debug("Using Python: %s (%s)", exe, r.stdout.strip())
                return exe
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    logger.warning("No Python executable found, falling back to 'python'")
    return "python"


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated World Cup data refresh & prediction",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Download and resolve data, but don't save or train",
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Path to log file (default: stdout)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Minimal output (auto-cron mode)",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Only refresh data, skip training & prediction",
    )
    return parser.parse_args(argv)


def setup_logging(log_file: str | None, quiet: bool) -> None:
    """Configure logging based on CLI flags."""
    handlers: list[logging.Handler] = []
    level = logging.WARNING if quiet else logging.INFO

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%H:%M:%S",
    ))
    handlers.append(stdout_handler)

    if log_file:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        handlers.append(fh)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_file, args.quiet)
    t0 = time.time()

    print("")
    print("/" + "-" * 58 + "\\")
    print("|     WORLD CUP - AUTOMATED DATA REFRESH            |")
    print("\\" + "-" * 58 + "/")
    print("")

    # Step 1-2: Download & resolve bracket
    df = collect_data(dry_run=args.dry_run)
    if df is None:
        logger.error("Data collection failed. Aborting.")
        return 1

    # Step 3: Train & Predict
    if not args.skip_train:
        success = run_training(dry_run=args.dry_run)
        if not success and not args.dry_run:
            logger.error("Training step failed. Check logs above.")
            return 1
    else:
        logger.info("--- Training skipped (--skip-train) ---")

    elapsed = time.time() - t0
    logger.info("-" * 50)
    logger.info("Done in %.1f seconds", elapsed)
    logger.info("-" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
