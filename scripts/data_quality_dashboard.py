"""
Data Quality Dashboard — comprehensive checks + interactive HTML report.

Usage:
    python scripts/data_quality_dashboard.py

Outputs:
    reports/data_quality_{timestamp}.json           — full structured results
    reports/data_quality_summary_{timestamp}.csv    — summary table
    reports/data_quality_dashboard_{timestamp}.html — interactive dashboard
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dq")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

DATA_PATH = ROOT / "data" / "processed" / "results_clean.csv"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

ODDS_COLUMNS = ["B365H", "B365D", "B365A", "BWH", "BWD", "BWA",
                "IWH", "IWD", "IWA", "PSH", "PSD", "PSA",
                "WHH", "WHD", "WHA", "VCH", "VCD", "VCA",
                "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA"]


# ═══════════════════════════════════════════════════════════
#  Check Result Container
# ═══════════════════════════════════════════════════════════

class DQCheck:
    def __init__(self, name: str, category: str, severity: str = "medium"):
        self.name = name
        self.category = category
        self.severity = severity
        self.passed: bool = True
        self.details: dict[str, Any] = {}
        self.issues: list[dict[str, Any]] = []

    def add_issue(self, description: str, count: int = 0, severity: str | None = None,
                  details: str = "") -> None:
        self.passed = False
        self.issues.append({
            "description": description,
            "count": count,
            "severity": severity or self.severity,
            "details": details,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "severity": self.severity,
            "passed": self.passed,
            "details": self.details,
            "issues": self.issues,
        }


# ═══════════════════════════════════════════════════════════
#  A — Completeness Checks
# ═══════════════════════════════════════════════════════════

def check_completeness(df: pd.DataFrame) -> list[DQCheck]:
    results: list[DQCheck] = []

    # A1 — Missing values by column
    ck = DQCheck("missing_by_column", "completeness", "medium")
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / len(df) * 100).round(2)
    cols_with_nulls = null_counts[null_counts > 0]
    ck.details["total_rows"] = len(df)
    ck.details["total_columns"] = len(df.columns)
    ck.details["columns_with_nulls"] = len(cols_with_nulls)
    col_details = {}
    for col in df.columns:
        n = int(null_counts[col])
        p = float(null_pcts[col])
        col_details[col] = {"missing": n, "pct": p}
        if n > 0:
            sev = "high" if p > 10 else "medium" if p > 2 else "low"
            ck.add_issue(f"{n} missing in '{col}' ({p:.1f}%)", n, sev)
    ck.details["column_details"] = col_details
    results.append(ck)

    # A2 — Missing values by league
    if "league" in df.columns:
        ck = DQCheck("missing_by_league", "completeness", "medium")
        league_nulls = df.groupby("league").apply(
            lambda g: int(g.isnull().sum().sum())
        ).sort_values(ascending=False)
        ck.details["by_league"] = {
            str(k): v for k, v in league_nulls.items() if v > 0
        }
        for league, cnt in league_nulls.items():
            if cnt > 0:
                pct = cnt / (len(df[df["league"] == league]) * len(df.columns)) * 100
                ck.add_issue(f"{cnt} missing cells in '{league}'", cnt, "low", f"{pct:.1f}%")
        results.append(ck)

    # A3 — Missing values by season
    if "season" in df.columns:
        ck = DQCheck("missing_by_season", "completeness", "medium")
        season_nulls = df.groupby("season").apply(
            lambda g: int(g.isnull().sum().sum())
        ).sort_values(ascending=False)
        ck.details["by_season"] = {
            str(k): v for k, v in season_nulls.items() if v > 0
        }
        for season, cnt in season_nulls.items():
            if cnt > 0:
                ck.add_issue(f"{cnt} missing cells in season {season}", cnt, "low")
        results.append(ck)

    # A4 — Odds coverage
    ck = DQCheck("odds_coverage", "completeness", "high")
    present_odds = [c for c in ODDS_COLUMNS if c in df.columns]
    ck.details["odds_columns_present"] = len(present_odds)
    ck.details["odds_columns_total"] = len(ODDS_COLUMNS)
    if present_odds:
        odds_null = df[present_odds].isnull().sum()
        for col in present_odds:
            n = int(odds_null.get(col, 0))
            if n > 0:
                pct = n / len(df) * 100
                ck.add_issue(f"{n} missing odds in '{col}' ({pct:.1f}%)", n, "high")
        ck.details["odds_columns"] = list(present_odds)
    else:
        ck.details["odds_columns"] = []
        ck.add_issue("No odds columns found in dataset", 0, "info", "Expected odds columns absent")
    results.append(ck)

    # A5 — xG coverage
    ck = DQCheck("xg_coverage", "completeness", "high")
    xg_cols = [c for c in df.columns if "xg" in c.lower()]
    if xg_cols:
        xg_null = df[xg_cols].isnull().sum()
        for col in xg_cols:
            n = int(xg_null.get(col, 0))
            if n > 0:
                ck.add_issue(f"{n} missing xG values in '{col}'", n, "high")
        ck.details["xg_columns"] = list(xg_cols)
    else:
        ck.details["xg_columns"] = []
        ck.add_issue("No xG columns found in dataset", 0, "info",
                     "Expected xG features absent — xG-dependent checks skipped")
    results.append(ck)

    return results


# ═══════════════════════════════════════════════════════════
#  B — Consistency Checks
# ═══════════════════════════════════════════════════════════

def check_consistency(df: pd.DataFrame) -> list[DQCheck]:
    results: list[DQCheck] = []

    # B1 — Duplicate matches
    ck = DQCheck("duplicate_matches", "consistency", "high")
    match_cols = [c for c in ["date", "home_team", "away_team", "season"] if c in df.columns]
    dupes = df.duplicated(subset=match_cols, keep=False)
    n_dupes = int(dupes.sum())
    ck.details["duplicate_count"] = n_dupes
    ck.details["duplicate_rows"] = (
        df[dupes][match_cols].to_dict("records") if n_dupes > 0 else []
    )
    if n_dupes > 0:
        ck.add_issue(f"{n_dupes} duplicate match rows found", n_dupes, "high")
        # Group duplicates to find exact pairs
        dup_groups = df[dupes].groupby(match_cols).size()
        ck.details["duplicate_groups"] = int(len(dup_groups))
    results.append(ck)

    # B2 — Team name consistency
    ck = DQCheck("team_name_consistency", "consistency", "high")
    all_teams = sorted(set(df["home_team"].dropna().unique()) |
                       set(df["away_team"].dropna().unique()))
    ck.details["total_teams"] = len(all_teams)
    ck.details["teams"] = all_teams
    # Check for suspicious name patterns (digits, very short, etc.)
    suspicious_names = [t for t in all_teams if (isinstance(t, str) and (
        len(t) <= 1 or any(c.isdigit() for c in t) or t != t.strip()
    ))]
    if suspicious_names:
        ck.add_issue(f"Suspicious team names: {suspicious_names}", len(suspicious_names), "high")
    results.append(ck)

    # B3 — Invalid dates
    ck = DQCheck("invalid_dates", "consistency", "high")
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
        n_invalid = int(dates.isna().sum())
        now = pd.Timestamp.now()
        n_future = int((dates > now).sum())
        ck.details["invalid_count"] = n_invalid
        ck.details["future_count"] = n_future
        if n_invalid > 0:
            ck.add_issue(f"{n_invalid} unparseable dates", n_invalid, "high")
        if n_future > 0:
            ck.add_issue(f"{n_future} future dates", n_future, "high")
        ck.details["date_range"] = {
            "min": str(dates.min().date()) if pd.notna(dates.min()) else None,
            "max": str(dates.max().date()) if pd.notna(dates.max()) else None,
        }
    results.append(ck)

    # B4 — Invalid scores
    ck = DQCheck("invalid_scores", "consistency", "high")
    issues_found = False
    if "home_goals" in df.columns and "away_goals" in df.columns:
        neg_home = int((df["home_goals"] < 0).sum())
        neg_away = int((df["away_goals"] < 0).sum())
        unrealistic = int(((df["home_goals"] > 15) | (df["away_goals"] > 15)).sum())
        nan_scores = int(df["home_goals"].isna().sum() + df["away_goals"].isna().sum())
        ck.details["negative_home"] = neg_home
        ck.details["negative_away"] = neg_away
        ck.details["unrealistic_scores"] = unrealistic
        ck.details["nan_scores"] = nan_scores
        if neg_home > 0:
            ck.add_issue(f"{neg_home} negative home goals", neg_home, "high")
        if neg_away > 0:
            ck.add_issue(f"{neg_away} negative away goals", neg_away, "high")
        if unrealistic > 0:
            ck.add_issue(f"{unrealistic} unrealistic scores (>15 goals)", unrealistic, "high")
        if nan_scores > 0:
            ck.add_issue(f"{nan_scores} missing scores", nan_scores, "medium")
    results.append(ck)

    # B5 — Invalid odds
    ck = DQCheck("invalid_odds", "consistency", "high")
    present_odds = [c for c in ODDS_COLUMNS if c in df.columns]
    if present_odds:
        odds_df = df[present_odds]
        neg_odds = int((odds_df < 0).sum().sum())
        zero_odds = int((odds_df == 0).sum().sum())
        extreme_odds = int((odds_df > 100).sum().sum())
        ck.details["negative"] = neg_odds
        ck.details["zero"] = zero_odds
        ck.details["extreme"] = extreme_odds
        if neg_odds > 0:
            ck.add_issue(f"{neg_odds} negative odds values", neg_odds, "high")
        if zero_odds > 0:
            ck.add_issue(f"{zero_odds} zero odds values", zero_odds, "high")
        if extreme_odds > 0:
            ck.add_issue(f"{extreme_odds} extreme odds (>100)", extreme_odds, "medium")
    else:
        ck.details["note"] = "No odds columns present — check skipped"
    results.append(ck)

    return results


# ═══════════════════════════════════════════════════════════
#  C — Validity Checks
# ═══════════════════════════════════════════════════════════

def check_validity(df: pd.DataFrame) -> list[DQCheck]:
    results: list[DQCheck] = []

    # C1 — League name validity
    ck = DQCheck("league_names", "validity", "medium")
    if "league" in df.columns:
        leagues = df["league"].dropna().unique()
        ck.details["leagues"] = sorted(str(l) for l in leagues)
        ck.details["league_count"] = int(len(leagues))
        # Check for suspicious patterns
        suspicious = [str(l) for l in leagues
                      if len(str(l)) <= 1 or any(c.isdigit() for c in str(l)[:3])]
        if suspicious:
            ck.add_issue(f"Suspicious league names: {suspicious}", len(suspicious), "medium")
    results.append(ck)

    # C2 — Season validity
    ck = DQCheck("season_validity", "validity", "medium")
    if "season" in df.columns:
        seasons = sorted(df["season"].dropna().unique())
        ck.details["seasons"] = [int(s) for s in seasons]
        ck.details["season_count"] = int(len(seasons))
        # Check for out-of-range seasons
        current_year = datetime.now().year
        invalid_seasons = [s for s in seasons if s < 1950 or s > current_year + 2]
        if invalid_seasons:
            ck.add_issue(f"Out-of-range seasons: {invalid_seasons}", len(invalid_seasons), "high")
    results.append(ck)

    # C3 — Home/Away team match
    ck = DQCheck("home_away_conflict", "validity", "high")
    if "home_team" in df.columns and "away_team" in df.columns:
        conflicts = df[df["home_team"] == df["away_team"]]
        n_conflicts = len(conflicts)
        ck.details["home_away_conflicts"] = n_conflicts
        if n_conflicts > 0:
            ck.add_issue(f"{n_conflicts} rows where home_team == away_team", n_conflicts, "high")
    results.append(ck)

    # C4 — Result validity
    ck = DQCheck("result_validity", "validity", "high")
    if "result" in df.columns:
        valid_results = {"H", "D", "A"}
        invalid_results = df[~df["result"].isin(valid_results)]
        n_invalid = len(invalid_results)
        ck.details["invalid_results"] = n_invalid
        ck.details["valid_values"] = sorted(df["result"].dropna().unique().tolist())
        if n_invalid > 0:
            ck.add_issue(f"{n_invalid} invalid result values", n_invalid, "high")
    results.append(ck)

    # C5 — Target validity
    ck = DQCheck("target_validity", "validity", "high")
    if "target" in df.columns:
        valid_targets = {0, 1, 2}
        invalid_target = df[~df["target"].isin(valid_targets)]
        n_invalid = len(invalid_target)
        ck.details["invalid_targets"] = n_invalid
        ck.details["target_distribution"] = {
            str(k): int(v) for k, v in df["target"].value_counts().items()
        }
        if n_invalid > 0:
            ck.add_issue(f"{n_invalid} invalid target values (expected 0/1/2)", n_invalid, "high")
    results.append(ck)

    return results


# ═══════════════════════════════════════════════════════════
#  D — Distribution Checks
# ═══════════════════════════════════════════════════════════

def check_distributions(df: pd.DataFrame) -> list[DQCheck]:
    results: list[DQCheck] = []

    # D1 — Goal distribution
    ck = DQCheck("goal_distribution", "distribution", "low")
    if "home_goals" in df.columns:
        home_goals = df["home_goals"].dropna()
        away_goals = df["away_goals"].dropna()
        total_goals = df["total_goals"].dropna() if "total_goals" in df.columns else None
        ck.details["home_goals"] = {
            "mean": round(float(home_goals.mean()), 3),
            "std": round(float(home_goals.std()), 3),
            "median": float(home_goals.median()),
            "min": int(home_goals.min()),
            "max": int(home_goals.max()),
        }
        ck.details["away_goals"] = {
            "mean": round(float(away_goals.mean()), 3),
            "std": round(float(away_goals.std()), 3),
            "median": float(away_goals.median()),
            "min": int(away_goals.min()),
            "max": int(away_goals.max()),
        }
        if total_goals is not None:
            ck.details["total_goals"] = {
                "mean": round(float(total_goals.mean()), 3),
                "std": round(float(total_goals.std()), 3),
                "median": float(total_goals.median()),
                "min": int(total_goals.min()),
                "max": int(total_goals.max()),
            }
        # Goal distribution histogram bins
        hist, edges = np.histogram(total_goals if total_goals is not None else home_goals,
                                    bins=range(0, 12))
        ck.details["histogram"] = {
            "counts": hist.tolist(),
            "edges": edges.tolist(),
        }
    results.append(ck)

    # D2 — Result distribution
    ck = DQCheck("result_distribution", "distribution", "low")
    if "result" in df.columns:
        vc = df["result"].value_counts()
        total = len(df)
        dist = {}
        for label in ["H", "D", "A"]:
            cnt = int(vc.get(label, 0))
            dist[label] = {"count": cnt, "pct": round(cnt / total * 100, 1)}
        ck.details["distribution"] = dist
        ck.details["total"] = total
        home_pct = dist.get("H", {}).get("pct", 0)
        draw_pct = dist.get("D", {}).get("pct", 0)
        away_pct = dist.get("A", {}).get("pct", 0)
        # Expected range: H ~42-48%, D ~24-30%, A ~26-32%
        if not (38 <= home_pct <= 52):
            ck.add_issue(f"Home win rate ({home_pct}%) outside expected range (38-52%)",
                         0, "low")
        if not (20 <= draw_pct <= 34):
            ck.add_issue(f"Draw rate ({draw_pct}%) outside expected range (20-34%)",
                         0, "low")
        if not (22 <= away_pct <= 36):
            ck.add_issue(f"Away win rate ({away_pct}%) outside expected range (22-36%)",
                         0, "low")
    results.append(ck)

    # D3 — Goal distribution by league
    ck = DQCheck("goals_by_league", "distribution", "low")
    if "league" in df.columns and "total_goals" in df.columns:
        league_stats = df.groupby("league")["total_goals"].agg(["mean", "std", "count"])
        league_stats = league_stats.round(3)
        ck.details["by_league"] = {
            str(k): {
                "mean_goals": float(v["mean"]),
                "std_goals": float(v["std"]),
                "matches": int(v["count"]),
            }
            for k, v in league_stats.iterrows()
        }
        # Flag leagues with extreme avg goals
        for league, stats in ck.details["by_league"].items():
            if stats["mean_goals"] > 4 or stats["mean_goals"] < 1:
                ck.add_issue(f"'{league}' avg {stats['mean_goals']:.2f} goals per match "
                             f"({stats['matches']} matches)", stats["matches"], "low")
    results.append(ck)

    # D4 — Goal distribution by season
    ck = DQCheck("goals_by_season", "distribution", "low")
    if "season" in df.columns and "total_goals" in df.columns:
        season_stats = df.groupby("season")["total_goals"].agg(["mean", "std", "count"])
        season_stats = season_stats.round(3)
        ck.details["by_season"] = {
            str(k): {
                "mean_goals": float(v["mean"]),
                "std_goals": float(v["std"]),
                "matches": int(v["count"]),
            }
            for k, v in season_stats.iterrows()
        }
    results.append(ck)

    return results


# ═══════════════════════════════════════════════════════════
#  E — Trend Checks
# ═══════════════════════════════════════════════════════════

def check_trends(df: pd.DataFrame) -> list[DQCheck]:
    results: list[DQCheck] = []

    # E1 — Data volume over time
    ck = DQCheck("volume_over_time", "trends", "info")
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        monthly = dates.dt.to_period("M").value_counts().sort_index()
        yearly = dates.dt.year.value_counts().sort_index()
        ck.details["monthly"] = {str(k): int(v) for k, v in monthly.items()}
        ck.details["yearly"] = {str(k): int(v) for k, v in yearly.items()}
        ck.details["total_matches"] = len(df)
    results.append(ck)

    # E2 — Missing data over time
    ck = DQCheck("missing_over_time", "trends", "info")
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        df_with_date = df.copy()
        df_with_date["_date_parsed"] = dates
        df_with_date["_year"] = dates.dt.year
        missing_by_year = {}
        for year, grp in df_with_date.groupby("_year"):
            null_cells = int(grp.isnull().sum().sum())
            total_cells = int(grp.shape[0] * grp.shape[1])
            missing_by_year[str(int(year))] = {
                "missing_cells": null_cells,
                "total_cells": total_cells,
                "pct": round(null_cells / total_cells * 100, 2) if total_cells > 0 else 0,
            }
        ck.details["by_year"] = missing_by_year
    results.append(ck)

    # E3 — Data quality score over time
    ck = DQCheck("quality_score_trend", "trends", "info")
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        df_with = df.copy()
        df_with["_year"] = dates.dt.year
        scores = {}
        for year, grp in df_with.groupby("_year"):
            total_cells = grp.shape[0] * grp.shape[1]
            filled_cells = int(grp.notna().sum().sum())
            completeness = filled_cells / total_cells if total_cells > 0 else 0
            # Check result validity
            valid_results_pct = (grp["result"].isin({"H", "D", "A"}).sum() /
                                 len(grp)) if "result" in grp.columns else 1
            # Check duplicate rate
            match_cols = [c for c in ["date", "home_team", "away_team"] if c in grp.columns]
            dup_rate = 1 - (grp.duplicated(subset=match_cols).sum() / len(grp)) if len(grp) > 0 else 1
            score = round((completeness * 0.4 + valid_results_pct * 0.3 + dup_rate * 0.3) * 100, 1)
            scores[str(int(year))] = score
        ck.details["scores_by_year"] = scores
        if scores:
            ck.details["overall_score"] = round(
                sum(scores.values()) / len(scores), 1
            )
    results.append(ck)

    return results


# ═══════════════════════════════════════════════════════════
#  Run All Checks
# ═══════════════════════════════════════════════════════════

def run_all_checks(df: pd.DataFrame) -> dict[str, Any]:
    all_checks: list[DQCheck] = []
    all_checks.extend(check_completeness(df))
    all_checks.extend(check_consistency(df))
    all_checks.extend(check_validity(df))
    all_checks.extend(check_distributions(df))
    all_checks.extend(check_trends(df))

    # Aggregate
    total_checks = len(all_checks)
    passed = sum(1 for c in all_checks if c.passed)
    failed = total_checks - passed
    total_issues = sum(len(c.issues) for c in all_checks)

    by_severity: dict[str, int] = {}
    by_category: dict[str, dict] = {}
    for c in all_checks:
        by_severity[c.severity] = by_severity.get(c.severity, 0) + 1
        if c.category not in by_category:
            by_category[c.category] = {"total": 0, "passed": 0, "failed": 0}
        by_category[c.category]["total"] += 1
        if c.passed:
            by_category[c.category]["passed"] += 1
        else:
            by_category[c.category]["failed"] += 1

    # Overall DQ score: weighted by category
    completeness_score = _category_score(all_checks, "completeness")
    consistency_score = _category_score(all_checks, "consistency")
    validity_score = _category_score(all_checks, "validity")
    distribution_score = _category_score(all_checks, "distribution")
    dq_score = round(
        completeness_score * 0.30 +
        consistency_score * 0.25 +
        validity_score * 0.25 +
        distribution_score * 0.20,
        1,
    )

    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data_file": str(DATA_PATH),
            "total_matches": len(df),
            "date_range": {
                "start": str(df["date"].min()) if "date" in df.columns else "N/A",
                "end": str(df["date"].max()) if "date" in df.columns else "N/A",
            },
        },
        "summary": {
            "total_checks": total_checks,
            "passed": passed,
            "failed": failed,
            "total_issues": total_issues,
            "dq_score": dq_score,
            "dq_score_components": {
                "completeness": completeness_score,
                "consistency": consistency_score,
                "validity": validity_score,
                "distribution": distribution_score,
            },
            "by_severity": by_severity,
            "by_category": by_category,
            "issues_by_severity": _count_issues_by_severity(all_checks),
        },
        "checks": [c.to_dict() for c in all_checks],
    }


def _category_score(checks: list[DQCheck], category: str) -> float:
    cat_checks = [c for c in checks if c.category == category]
    if not cat_checks:
        return 100.0
    passed = sum(1 for c in cat_checks if c.passed)
    return round(passed / len(cat_checks) * 100, 1)


def _count_issues_by_severity(checks: list[DQCheck]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in checks:
        for issue in c.issues:
            sev = issue.get("severity", "low")
            counts[sev] = counts.get(sev, 0) + 1
    return counts


# ═══════════════════════════════════════════════════════════
#  Filter Data — Pre-compute per-league & per-year stats
# ═══════════════════════════════════════════════════════════

def _compute_filter_data(df: pd.DataFrame) -> dict[str, Any]:
    """Pre-compute breakdowns by league and year for interactive filtering.

    Returns a dict that gets embedded as JSON in the HTML so the browser
    can switch between "All" leagues and specific leagues without a server.
    """
    fd: dict[str, Any] = {
        "leagues": ["All"],
        "seasons": [],
        "date_min": "",
        "date_max": "",
        "volume": {},        # {league: {year: count}}
        "missing": {},       # {league: {year: pct}}
        "scores": {},        # {league: {year: score}}
        "result_dist": {},   # {league: {H: {count,pct}, D: ..., A: ...}}
        "goal_hist": {},     # {league: {counts: [...], edges: [...]}}
        "goals_avg": {},     # {league: mean_goals}
    }

    if df.empty:
        return fd

    # Parse dates
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
    else:
        dates = pd.Series([pd.NaT] * len(df))

    valid_dates = dates.notna()
    years = dates.dt.year.astype("Int64")
    fd["date_min"] = str(dates.min().date()) if valid_dates.any() and pd.notna(dates.min()) else ""
    fd["date_max"] = str(dates.max().date()) if valid_dates.any() and pd.notna(dates.max()) else ""

    # Collect available leagues
    if "league" in df.columns:
        league_names = sorted(df["league"].dropna().unique())
        fd["leagues"] = ["All"] + [str(l) for l in league_names]
    else:
        league_names = []

    # Years / seasons
    if valid_dates.any():
        fd["seasons"] = sorted(int(y) for y in years.dropna().unique() if pd.notna(y))
    elif "season" in df.columns:
        fd["seasons"] = sorted(int(s) for s in df["season"].dropna().unique())

    def _dq_score_for(sub: pd.DataFrame) -> float:
        """Compute DQ score for a subset (lower bound ~60, upper ~100)."""
        if sub.empty:
            return 100.0
        tc = sub.size
        mc = int(sub.isna().sum().sum())
        completeness = (tc - mc) / tc if tc > 0 else 1.0
        vrp = (sub["result"].isin({"H", "D", "A"}).sum() / len(sub)) if "result" in sub.columns else 1.0
        key = [c for c in ["date", "home_team", "away_team"] if c in sub.columns]
        dr = 1.0
        if key:
            dup = sub.duplicated(subset=key).sum()
            dr = 1.0 - (dup / len(sub)) if len(sub) > 0 else 1.0
        return round((completeness * 0.4 + vrp * 0.3 + dr * 0.3) * 100, 1)

    def _result_dist_for(sub: pd.DataFrame) -> dict[str, dict]:
        if "result" not in sub.columns or sub.empty:
            return {"H": {"count": 0, "pct": 0}, "D": {"count": 0, "pct": 0}, "A": {"count": 0, "pct": 0}}
        vc = sub["result"].value_counts()
        t = len(sub)
        return {
            l: {"count": int(vc.get(l, 0)), "pct": round(int(vc.get(l, 0)) / t * 100, 1)}
            for l in ["H", "D", "A"]
        }

    def _goal_hist_for(sub: pd.DataFrame) -> dict:
        tg = sub["total_goals"].dropna() if "total_goals" in sub.columns else (
            sub["home_goals"].dropna() if "home_goals" in sub.columns else pd.Series(dtype=float)
        )
        if tg.empty:
            return {"counts": [], "edges": []}
        hist, edges = np.histogram(tg, bins=range(0, 12))
        return {"counts": hist.tolist(), "edges": edges.tolist()}

    # ── Pre-compute "All" league data (overall) ───────────
    all_label = "All"
    fd["volume"][all_label] = {}
    fd["missing"][all_label] = {}
    fd["scores"][all_label] = {}
    fd["result_dist"][all_label] = _result_dist_for(df)
    fd["goal_hist"][all_label] = _goal_hist_for(df)
    fd["goals_avg"][all_label] = round(
        float(df["total_goals"].mean()) if "total_goals" in df.columns else 0, 2
    )

    for year in fd["seasons"]:
        mask = (years == year) if valid_dates.any() else (
            df["season"] == year if "season" in df.columns else pd.Series(False, index=df.index)
        )
        sub = df[mask]
        if sub.empty:
            fd["volume"][all_label][str(year)] = 0
            fd["missing"][all_label][str(year)] = 0.0
            fd["scores"][all_label][str(year)] = 100.0
            continue
        fd["volume"][all_label][str(year)] = len(sub)
        tc = sub.size
        mc = int(sub.isna().sum().sum())
        fd["missing"][all_label][str(year)] = round(mc / tc * 100, 2) if tc > 0 else 0.0
        fd["scores"][all_label][str(year)] = _dq_score_for(sub)

    # ── Pre-compute per-league data ───────────────────────
    for league in league_names:
        league_str = str(league)
        ldf = df[df["league"] == league] if "league" in df.columns else pd.DataFrame()
        if ldf.empty:
            continue

        fd["volume"][league_str] = {}
        fd["missing"][league_str] = {}
        fd["scores"][league_str] = {}
        fd["result_dist"][league_str] = _result_dist_for(ldf)
        fd["goal_hist"][league_str] = _goal_hist_for(ldf)
        fd["goals_avg"][league_str] = round(
            float(ldf["total_goals"].mean()) if "total_goals" in ldf.columns else 0, 2
        )

        for year in fd["seasons"]:
            ymask = (years == year) if valid_dates.any() else (
                df["season"] == year if "season" in df.columns else pd.Series(False, index=df.index)
            )
            sub = ldf[ymask.loc[ldf.index]]
            if sub.empty:
                fd["volume"][league_str][str(year)] = 0
                fd["missing"][league_str][str(year)] = 0.0
                fd["scores"][league_str][str(year)] = 100.0
                continue
            fd["volume"][league_str][str(year)] = len(sub)
            tc = sub.size
            mc = int(sub.isna().sum().sum())
            fd["missing"][league_str][str(year)] = round(mc / tc * 100, 2) if tc > 0 else 0.0
            fd["scores"][league_str][str(year)] = _dq_score_for(sub)

    return fd


# ═══════════════════════════════════════════════════════════
#  Save Static Reports
# ═══════════════════════════════════════════════════════════

def save_reports(report: dict[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}

    # JSON
    json_path = REPORTS_DIR / f"data_quality_{TIMESTAMP}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    paths["json"] = str(json_path)
    log.info("JSON report saved to %s", json_path)

    # CSV summary
    rows = []
    for c in report["checks"]:
        for issue in c["issues"]:
            rows.append({
                "check": c["name"],
                "category": c["category"],
                "passed": c["passed"],
                "issue": issue["description"],
                "count": issue["count"],
                "severity": issue["severity"],
            })
    if not rows:
        rows.append({"check": "all", "category": "all", "passed": True,
                      "issue": "No issues found", "count": 0, "severity": "info"})
    summary_df = pd.DataFrame(rows)
    csv_path = REPORTS_DIR / f"data_quality_summary_{TIMESTAMP}.csv"
    summary_df.to_csv(csv_path, index=False)
    paths["csv"] = str(csv_path)
    log.info("CSV summary saved to %s", csv_path)

    return paths


# ═══════════════════════════════════════════════════════════
#  HTML Dashboard Generator
# ═══════════════════════════════════════════════════════════

def _severity_color(sev: str) -> str:
    return {"critical": "#dc3545", "high": "#fd7e14", "medium": "#ffc107",
            "low": "#17a2b8", "info": "#6c757d"}.get(sev, "#6c757d")

def _severity_badge(sev: str) -> str:
    colors = {"critical": "var(--red)", "high": "var(--orange)", "medium": "var(--yellow)",
              "low": "var(--blue)", "info": "var(--gray)"}
    c = colors.get(sev, "var(--gray)")
    return f'<span style="background:{c};color:#fff;padding:2px 8px;border-radius:10px;font-size:0.75rem">{sev}</span>'


def generate_html(report: dict[str, Any], filter_data: dict[str, Any] | None = None) -> str:
    s = report["summary"]
    meta = report["metadata"]
    checks = report["checks"]

    # ── Filter UI options ────────────────────────────────
    if filter_data is None:
        filter_data = _compute_filter_data(pd.DataFrame())
    fd_json = json.dumps(filter_data, default=str)

    league_options = "".join(
        f'<option value="{l}"{" selected" if l == "All" else ""}>{l}</option>'
        for l in filter_data["leagues"]
    )
    season_options = "".join(
        f'<option value="{s}">{s}</option>'
        for s in filter_data["seasons"]
    )

    # Build issues table rows (dynamic — filtered by JS, but we render full set for initial view)
    issue_rows = ""
    all_issue_data: list[dict[str, Any]] = []
    for c in checks:
        for iss in c["issues"]:
            if iss["severity"] in ("info",):
                continue
            all_issue_data.append({
                "check": c["name"],
                "description": iss["description"],
                "count": iss["count"],
                "severity": iss["severity"],
            })
            issue_rows += f"""
            <tr class="issue-row" data-check="{c['name']}">
                <td>{c['name']}</td>
                <td>{iss['description']}</td>
                <td>{iss['count']}</td>
                <td>{_severity_badge(iss['severity'])}</td>
            </tr>"""
    issues_json = json.dumps(all_issue_data, default=str)

    # Build detailed check sections
    check_sections = ""
    for cat in ["completeness", "consistency", "validity", "distribution", "trends"]:
        cat_checks = [c for c in checks if c["category"] == cat]
        if not cat_checks:
            continue
        cat_passed = sum(1 for c in cat_checks if c["passed"])
        cat_total = len(cat_checks)
        cat_issues = sum(len(c["issues"]) for c in cat_checks)
        check_sections += f"""
        <div class="section">
            <h3>{cat.title()}  <span class="badge">{cat_passed}/{cat_total} passed</span>  <span class="badge warn">{cat_issues} issues</span></h3>
        """

        for c in cat_checks:
            status_icon = "+" if c["passed"] else "x"
            status_class = "pass" if c["passed"] else "fail"
            issues_for_check = c.get("issues", [])
            check_sections += f"""
            <div class="check-block {status_class}">
                <div class="check-header">
                    <span class="status">{status_icon}</span>
                    <strong>{c['name']}</strong>
                    <span class="sev-badge">{_severity_badge(c['severity'])}</span>
                </div>
                <div class="check-body">
                    <p>{len(issues_for_check)} issue(s)</p>
                    <table>
                        <tr><th>Issue</th><th>Count</th><th>Severity</th></tr>"""

            for iss in issues_for_check:
                check_sections += f"""
                        <tr>
                            <td>{iss['description']}</td>
                            <td>{iss['count']}</td>
                            <td>{_severity_badge(iss['severity'])}</td>
                        </tr>"""
            check_sections += """
                    </table>
                </div>
            </div>"""

        check_sections += "</div>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Quality Dashboard — Football Prediction</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root {{
    --bg: #f8f9fa;
    --card: #fff;
    --text: #212529;
    --text-secondary: #6c757d;
    --border: #dee2e6;
    --green: #28a745;
    --red: #dc3545;
    --orange: #fd7e14;
    --yellow: #ffc107;
    --blue: #17a2b8;
    --gray: #6c757d;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: var(--bg); color: var(--text); line-height: 1.5; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size: 1.8rem; margin-bottom: 4px; }}
h2 {{ font-size: 1.3rem; margin: 24px 0 12px; padding-bottom: 6px; border-bottom: 2px solid var(--border); }}
h3 {{ font-size: 1.1rem; margin: 16px 0 8px; }}
.subtitle {{ color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 16px; }}

/* ── Filters ───────────────────────────────── */
.filters {{
    display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 12px 16px; margin-bottom: 20px;
}}
.filters label {{ font-size: 0.8rem; font-weight: 600; color: var(--text-secondary);
                  text-transform: uppercase; letter-spacing: 0.03em; }}
.filters select, .filters input {{
    padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px;
    background: var(--bg); color: var(--text); font-size: 0.85rem;
    min-width: 130px; outline: none;
}}
.filters select:focus, .filters input:focus {{
    border-color: var(--blue); box-shadow: 0 0 0 2px rgba(23,162,184,0.15);
}}
.filters .filter-group {{
    display: flex; align-items: center; gap: 6px;
}}
.filter-badge {{
    display: inline-block; font-size: 0.7rem; padding: 2px 8px;
    border-radius: 10px; background: var(--blue); color: #fff;
}}

.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.metric {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; transition: box-shadow 0.2s; }}
.metric:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
.metric h3 {{ font-size: 0.85rem; color: var(--text-secondary); margin: 0 0 4px; text-transform: uppercase; }}
.metric .value {{ font-size: 1.8rem; font-weight: 700; }}
.metric .sub {{ font-size: 0.8rem; color: var(--text-secondary); }}
.metric.good .value {{ color: var(--green); }}
.metric.warn .value {{ color: var(--orange); }}
.metric.bad .value {{ color: var(--red); }}

.charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.chart-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
.chart-card h3 {{ font-size: 0.9rem; margin: 0 0 8px; }}
.chart-card canvas {{ max-height: 250px; }}

.section {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
.section .badge {{ display: inline-block; background: var(--green); color: #fff; padding: 2px 8px;
                  border-radius: 10px; font-size: 0.75rem; }}
.section .badge.warn {{ background: var(--orange); }}

.check-block {{ border: 1px solid var(--border); border-radius: 6px; margin: 8px 0; padding: 12px; }}
.check-block.fail {{ border-left: 4px solid var(--red); }}
.check-block.pass {{ border-left: 4px solid var(--green); }}
.check-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
.check-header .status {{ font-weight: 700; font-size: 1.1rem; }}
.check-block.pass .status {{ color: var(--green); }}
.check-block.fail .status {{ color: var(--red); }}
.check-body {{ padding-left: 24px; }}

table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }}
th {{ font-weight: 600; color: var(--text-secondary); }}
tr:hover td {{ background: #f1f3f5; }}

.sev-badge {{ margin-left: auto; }}

.issues-grid {{ display: grid; grid-template-columns: 1fr; gap: 8px; }}
@media (min-width: 768px) {{ .issues-grid {{ grid-template-columns: 1fr 1fr; }} }}

.active-filter {{ outline: 2px solid var(--blue); outline-offset: 2px; }}

.footer {{ text-align: center; padding: 20px; color: var(--text-secondary); font-size: 0.8rem; }}

@keyframes fadeIn {{ from {{ opacity: 0.5; }} to {{ opacity: 1; }} }}
.chart-card.updating {{ animation: fadeIn 0.3s ease; }}
</style>
</head>
<body>
<div class="container">

<h1>Data Quality Dashboard</h1>
<p class="subtitle">Football Prediction System &mdash; {meta['timestamp']} &mdash; {meta['total_matches']} matches &mdash; {meta['date_range']['start']} to {meta['date_range']['end']}</p>

<!-- ── Interactive Filters ────────────────────────────── -->
<div class="filters" id="filterBar">
    <div class="filter-group">
        <label for="leagueFilter">League</label>
        <select id="leagueFilter" onchange="applyFilters()">
            {league_options}
        </select>
    </div>
    <div class="filter-group">
        <label for="yearFromFilter">Year from</label>
        <select id="yearFromFilter" onchange="applyFilters()">
            {season_options}
        </select>
    </div>
    <div class="filter-group">
        <label for="yearToFilter">to</label>
        <select id="yearToFilter" onchange="applyFilters()">
            {season_options}
        </select>
    </div>
    <div class="filter-group">
        <label for="severityFilter">Min severity</label>
        <select id="severityFilter" onchange="applyFilters()">
            <option value="all">All</option>
            <option value="high">High+</option>
            <option value="critical" selected>Critical</option>
        </select>
    </div>
    <span class="filter-badge" id="filterBadge">All data</span>
    <button onclick="resetFilters()" style="margin-left:auto;padding:4px 12px;border:1px solid var(--border);border-radius:6px;background:var(--bg);color:var(--text);cursor:pointer;font-size:0.8rem;">↺ Reset</button>
</div>

<div class="metrics">
    <div class="metric {'good' if s['dq_score'] >= 90 else 'warn' if s['dq_score'] >= 70 else 'bad'}" id="metricScore">
        <h3>DQ Score</h3>
        <div class="value" id="dqScoreValue">{s['dq_score']}%</div>
        <div class="sub">weighted across categories</div>
    </div>
    <div class="metric {'good' if s['total_issues'] == 0 else 'warn'}" id="metricIssues">
        <h3>Issues</h3>
        <div class="value" id="issuesValue">{s['total_issues']}</div>
        <div class="sub" id="issuesSub">{s['failed']} of {s['total_checks']} checks failed</div>
    </div>
    <div class="metric" id="metricMatches">
        <h3>Matches</h3>
        <div class="value" id="matchesValue">{meta['total_matches']:,}</div>
        <div class="sub" id="matchesSub">{len(filter_data['seasons'])} seasons</div>
    </div>
    <div class="metric {'good' if s['by_severity'].get('critical', 0) == 0 else 'bad'}" id="metricCritical">
        <h3>Critical</h3>
        <div class="value" id="criticalValue">{s['by_severity'].get('critical', 0)}</div>
        <div class="sub" id="criticalSub">{s['by_severity'].get('high', 0)} high, {s['by_severity'].get('medium', 0)} medium</div>
    </div>
</div>

<h2>Trends</h2>
<div class="charts">
    <div class="chart-card" id="ccVolume">
        <h3>Matches per season</h3>
        <canvas id="volumeChart"></canvas>
    </div>
    <div class="chart-card" id="ccMissing">
        <h3>Missing data (%)</h3>
        <canvas id="missingChart"></canvas>
    </div>
    <div class="chart-card" id="ccScore">
        <h3>DQ Score over time</h3>
        <canvas id="scoreChart"></canvas>
    </div>
    <div class="chart-card" id="ccResult">
        <h3>Result distribution</h3>
        <canvas id="resultChart"></canvas>
    </div>
</div>

<h2>Issues</h2>
<div class="issues-grid">
<table id="issuesTable">
    <thead><tr><th>Check</th><th>Issue</th><th>Count</th><th>Severity</th></tr></thead>
    <tbody id="issuesBody">
    {issue_rows if issue_rows else '<tr><td colspan="4" style="text-align:center;color:var(--green);font-weight:600;">No issues found</td></tr>'}
    </tbody>
</table>
</div>

<h2>Detailed Checks</h2>
{check_sections}

<div class="footer">
    Generated by scripts/data_quality_dashboard.py at {meta['timestamp']} &mdash;
    <a href="data_quality_{TIMESTAMP}.json">JSON</a> &middot;
    <a href="data_quality_summary_{TIMESTAMP}.csv">CSV</a>
</div>

</div>

<!-- ── Embed filter data as JSON ──────────────────────── -->
<script id="filterData" type="application/json">{fd_json}</script>
<script id="issuesData" type="application/json">{issues_json}</script>

<script>
// ═══════════════════════════════════════════════════════════
//  Filter Data & Chart Instances
// ═══════════════════════════════════════════════════════════

const FD = JSON.parse(document.getElementById('filterData').textContent);
const ALL_ISSUES = JSON.parse(document.getElementById('issuesData').textContent);

let charts = {{}};

function initCharts() {{
    const years = FD.seasons.map(String);
    const vols = years.map(y => (FD.volume['All'] || {{}})[y] || 0);
    const miss = years.map(y => (FD.missing['All'] || {{}})[y] || 0);
    const scrs = years.map(y => (FD.scores['All'] || {{}})[y] || 100);
    const rd = FD.result_dist['All'] || {{ H:{{pct:0}}, D:{{pct:0}}, A:{{pct:0}} }};

    charts.volume = new Chart(document.getElementById('volumeChart'), {{
        type: 'bar',
        data: {{ labels: years, datasets: [{{ label: 'Matches', data: vols,
                     backgroundColor: '#28a745', borderRadius: 4 }}] }},
        options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }} }}
    }});

    charts.missing = new Chart(document.getElementById('missingChart'), {{
        type: 'line',
        data: {{ labels: years, datasets: [{{ label: 'Missing (%)', data: miss,
                     borderColor: '#fd7e14', backgroundColor: 'rgba(253,126,20,0.1)',
                     fill: true, tension: 0.3 }}] }},
        options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }} }}
    }});

    charts.score = new Chart(document.getElementById('scoreChart'), {{
        type: 'line',
        data: {{ labels: years, datasets: [{{ label: 'DQ Score', data: scrs,
                     borderColor: '#28a745', backgroundColor: 'rgba(40,167,69,0.1)',
                     fill: true, tension: 0.3 }}] }},
        options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }} }}
    }});

    charts.result = new Chart(document.getElementById('resultChart'), {{
        type: 'doughnut',
        data: {{ labels: ['Home Win', 'Draw', 'Away Win'],
                datasets: [{{
                    data: [rd.H.pct, rd.D.pct, rd.A.pct],
                    backgroundColor: ['#28a745', '#ffc107', '#dc3545']
                }}] }},
        options: {{ responsive: true, plugins: {{ legend: {{ position: 'bottom' }} }} }}
    }});
}}

// ═══════════════════════════════════════════════════════════
//  Filter Logic
// ═══════════════════════════════════════════════════════════

function getKey() {{
    const league = document.getElementById('leagueFilter').value;
    return league;
}}

function getYears() {{
    const all = FD.seasons.map(String);
    const from = document.getElementById('yearFromFilter').value;
    const to = document.getElementById('yearToFilter').value;
    if (!from || !to) return all;
    return all.filter(y => y >= from && y <= to);
}}

function applyFilters() {{
    const league = getKey();
    const years = getYears();
    const sevFilter = document.getElementById('severityFilter').value;
    // Update badge
    const badge = document.getElementById('filterBadge');
    badge.textContent = league === 'All' ? 'All leagues' : league;
    if (years.length < FD.seasons.length) {{
        badge.textContent += ` (${{years[0]}}-${{years[years.length-1]}})`;
    }}

    // Get data for selected league (falls back to All)
    const vols = years.map(y => (FD.volume[league] || FD.volume['All'] || {{}})[y] || 0);
    const miss = years.map(y => (FD.missing[league] || FD.missing['All'] || {{}})[y] || 0);
    const scrs = years.map(y => (FD.scores[league] || FD.scores['All'] || {{}})[y] || 100);
    const rd = FD.result_dist[league] || FD.result_dist['All'] || {{ H:{{pct:0}}, D:{{pct:0}}, A:{{pct:0}} }};

    // Update charts
    charts.volume.data.labels = years;
    charts.volume.data.datasets[0].data = vols;
    charts.volume.update();

    charts.missing.data.labels = years;
    charts.missing.data.datasets[0].data = miss;
    charts.missing.update();

    charts.score.data.labels = years;
    charts.score.data.datasets[0].data = scrs;
    charts.score.update();

    charts.result.data.datasets[0].data = [rd.H.pct, rd.D.pct, rd.A.pct];
    charts.result.update();

    // Update KPI cards
    const totalMatches = vols.reduce((a, b) => a + b, 0);
    document.getElementById('matchesValue').textContent = totalMatches.toLocaleString();
    document.getElementById('matchesSub').textContent = years.length + ' seasons';

    // Filter issues table
    const issuesBody = document.getElementById('issuesBody');
    const sevOrder = {{ 'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4 }};
    const minSev = sevFilter === 'all' ? 4 : (sevOrder[sevFilter] !== undefined ? sevOrder[sevFilter] : 0);
    let matched = 0;
    let highMatched = 0;
    let critMatched = 0;
    ALL_ISSUES.forEach(function(iss, i) {{
        const row = issuesBody.children[i];
        if (!row) return;
        const sevNum = sevOrder[iss.severity] || 4;
        const showFilter = sevNum <= minSev;
        row.style.display = showFilter ? '' : 'none';
        if (showFilter) {{
            matched++;
            if (iss.severity === 'high') highMatched++;
            if (iss.severity === 'critical') critMatched++;
        }}
    }});

    document.getElementById('issuesValue').textContent = matched;
    document.getElementById('criticalValue').textContent = critMatched;
    document.getElementById('criticalSub').textContent = highMatched + ' high, ' + (matched - critMatched - highMatched) + ' other';
}}

function resetFilters() {{
    document.getElementById('leagueFilter').value = 'All';
    const seasons = FD.seasons;
    if (seasons.length > 0) {{
        document.getElementById('yearFromFilter').value = String(seasons[0]);
        document.getElementById('yearToFilter').value = String(seasons[seasons.length - 1]);
    }}
    document.getElementById('severityFilter').value = 'all';
    applyFilters();
}}

// ── Init ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {{
    // Set default year range
    const seasons = FD.seasons;
    if (seasons.length > 0) {{
        document.getElementById('yearFromFilter').value = String(seasons[0]);
        document.getElementById('yearToFilter').value = String(seasons[seasons.length - 1]);
    }}
    initCharts();
    applyFilters(); // Apply severity and year filter defaults
}});
</script>
</body>
</html>"""

    return html


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 65)
    print("  DATA QUALITY DASHBOARD")
    print("=" * 65)

    t_start = time.time()

    # Load data
    print(f"\n  Loading data from {DATA_PATH} ...")
    if not DATA_PATH.exists():
        log.error("Data file not found: %s", DATA_PATH)
        return 1

    df = pd.read_csv(DATA_PATH, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} cols")

    # Run checks
    print("\n  Running data quality checks ...")
    t0 = time.time()
    report = run_all_checks(df)
    s = report["summary"]
    print(f"  {s['total_checks']} checks | {s['passed']} passed | "
          f"{s['failed']} failed | {s['total_issues']} issues | "
          f"DQ Score: {s['dq_score']}%  ({time.time()-t0:.1f}s)")

    # Save static reports
    print("\n  Saving static reports ...")
    paths = save_reports(report)

    # Generate HTML dashboard with interactive filters
    print("\n  Computing filter data ...")
    filter_data = _compute_filter_data(df)
    print(f"  Available leagues: {len(filter_data['leagues'])-1} + All")
    print(f"  Season range: {filter_data['seasons'][0]}–{filter_data['seasons'][-1] if filter_data['seasons'] else 'N/A'}")

    print("\n  Generating HTML dashboard ...")
    html = generate_html(report, filter_data)
    html_path = REPORTS_DIR / f"data_quality_dashboard_{TIMESTAMP}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"  Dashboard saved to {html_path}")

    total = time.time() - t_start
    print(f"\n  Total duration: {total:.1f}s")
    print(f"  Reports:")
    for key, path in paths.items():
        print(f"    {key}: {path}")
    print(f"    html: {html_path}")
    print("=" * 65)

    return 0 if s["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
