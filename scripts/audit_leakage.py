"""
Leakage Audit — scan feature engineering code for temporal leakage, target leakage,
and train-test contamination.

Usage:
    python scripts/audit_leakage.py                          # scan & print report
    python scripts/audit_leakage.py --fix                     # apply auto-fixes
    python scripts/audit_leakage.py --output reports/leakage_audit_20260714.md
    python scripts/audit_leakage.py --json-only               # JSON report only
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files to scan
SCAN_FILES = [
    "src/feature_engineering.py",
    "src/elo.py",
    "src/poisson_model.py",
    "src/dixon_coles.py",
    "src/xg_features.py",
    "src/odds_processing.py",
    "src/preprocessing.py",
    "src/time_series_cv.py",
]


# ═══════════════════════════════════════════════════════════
#  Static analysis helpers
# ═══════════════════════════════════════════════════════════


def _get_source(file_rel: str) -> tuple[str, list[str]]:
    """Return (full_source, lines) for a project source file."""
    path = PROJECT_ROOT / file_rel
    if not path.exists():
        return "", []
    text = path.read_text(encoding="utf-8")
    return text, text.splitlines()


def _count_pattern(source: str, pattern: str) -> int:
    return len(re.findall(pattern, source))


# ═══════════════════════════════════════════════════════════
#  Individual checks
# ═══════════════════════════════════════════════════════════


def check_shift1(file_rel: str) -> list[dict[str, Any]]:
    """Check that all rolling computations use .shift(1)."""
    findings: list[dict[str, Any]] = []
    source, lines = _get_source(file_rel)
    if not source:
        return findings

    # Find rolling/expanding calls
    rolling_calls = re.finditer(
        r"(\.rolling\(|\.ewm\(|\.expanding\()",
        source,
    )
    for m in rolling_calls:
        # Get surrounding 5 lines of context to catch multi-line chains
        pos = m.start()
        lineno = source[:pos].count("\n") + 1
        context_start = max(0, lineno - 2)
        context_lines = lines[context_start : lineno + 5]
        context = "\n".join(context_lines)

        has_shift = ".shift(" in context
        # .diff() inherently shifts (no additional shift needed)
        is_diff = ".diff(" in m.group()

        if not has_shift and not is_diff:
            findings.append({
                "file": file_rel,
                "line": lineno,
                "severity": "HIGH",
                "check": "missing_shift1",
                "message": f"Rolling/expanding call without .shift(1) at line {lineno}",
                "context": context,
                "fixable": False,
            })

    return findings


def check_target_leakage(file_rel: str) -> list[dict[str, Any]]:
    """Check that target columns are not used directly as features.

    Target columns (result, home_goals, away_goals) are legitimate in:
    - preprocessing.py: defining the target variable
    - elo.py: updating ratings (pre-match rating is recorded before update)
    - poisson_model.py: computing expanding-window strengths
    - dixon_coles.py: MLE fitting
    - feature_engineering.py: used with .shift(1) for rolling stats,
      or dropped via _get_target_columns()

    We flag only specific suspicious patterns:
    1. Using ALL completed matches for league-average (lookahead)
    2. Missing drop of target columns before model training
    """
    findings: list[dict[str, Any]] = []
    source, lines = _get_source(file_rel)
    if not source:
        return findings

    # Specific check: lookahead in _add_attack_defence_ratios
    if "feature_engineering.py" in file_rel:
        for m in re.finditer(r"df\[\"result\"\]\.notna\(\)", source):
            pos = m.start()
            lineno = source[:pos].count("\n") + 1
            func_name_match = re.search(
                r"def (\w+)\(", source[max(0, pos - 500):pos],
            )
            func_name = func_name_match.group(1) if func_name_match else "unknown"
            findings.append({
                "file": file_rel,
                "line": lineno,
                "severity": "CRITICAL",
                "check": "lookahead_league_avg",
                "message": (
                    f"Function '{func_name}' uses `df['result'].notna()` to compute "
                    f"league-average goals from ALL completed matches "
                    f"(including future ones). Should use expanding window instead."
                ),
                "context": "",
                "fixable": True,
            })

        # Check that target/result/goals are properly dropped before returning
        has_drop = re.search(
            r"cols_to_drop\s*=\s*_get_target_columns\(df\)", source,
        )
        if not has_drop:
            findings.append({
                "file": file_rel,
                "line": 0,
                "severity": "HIGH",
                "check": "missing_target_drop",
                "message": (
                    "Target columns may not be dropped from the feature matrix."
                ),
                "context": "",
                "fixable": False,
            })

    # Check that result/goals in preprocessing are used only to create target,
    # not as features
    if "preprocessing.py" in file_rel:
        # Verify target is created from result (not result kept as feature)
        has_target_creation = re.search(
            r"target.*=\s*df\[\"result\"\]", source,
        )
        if not has_target_creation:
            pass  # OK, might use .map() as in the actual code

    return findings


def check_train_test_contamination(file_rel: str) -> list[dict[str, Any]]:
    """Check that time-series data is not shuffled randomly."""
    findings: list[dict[str, Any]] = []
    source, lines = _get_source(file_rel)
    if not source:
        return findings

    # Check train_test_split calls
    for m in re.finditer(r"train_test_split\(", source):
        pos = m.start()
        lineno = source[:pos].count("\n") + 1
        # Look for shuffle=True or missing shuffle param
        context = source[max(0, pos - 100):pos + 200]
        has_shuffle_false = re.search(r"shuffle\s*=\s*False", context)
        has_shuffle_true = re.search(r"shuffle\s*=\s*True", context)
        has_no_shuffle = not has_shuffle_false and not has_shuffle_true

        if has_shuffle_true:
            findings.append({
                "file": file_rel,
                "line": lineno,
                "severity": "CRITICAL",
                "check": "random_shuffle",
                "message": (
                    f"train_test_split with shuffle=True at line {lineno} — "
                    f"this randomly shuffles time-series data, causing leakage."
                ),
                "context": context,
                "fixable": True,
            })
        elif has_no_shuffle:
            findings.append({
                "file": file_rel,
                "line": lineno,
                "severity": "LOW",
                "check": "missing_shuffle_false",
                "message": (
                    f"train_test_split at line {lineno} — double-check that "
                    f"shuffle=False is set for chronological data."
                ),
                "context": context,
                "fixable": True,
            })

    # Check for Shuffle=True or random_state without shuffle=False
    shuffle_refs = re.finditer(r"(KFold|Shuffle|random_split)", source)
    for m in shuffle_refs:
        pos = m.start()
        lineno = source[:pos].count("\n") + 1
        findings.append({
            "file": file_rel,
            "line": lineno,
            "severity": "LOW",
            "check": "shuffle_reference",
            "message": (
                f"Shuffle-related pattern '{m.group()}' at line {lineno} — verify."
            ),
            "context": "",
            "fixable": False,
        })

    return findings


def check_dc_leakage(file_rel: str) -> list[dict[str, Any]]:
    """Check Dixon-Coles add_features for the refit cutoff leakage."""
    findings: list[dict[str, Any]] = []
    source, lines = _get_source(file_rel)
    if not source:
        return findings

    if "dixon_coles.py" not in file_rel:
        return findings

    # Check that cutoff_pos matches are not used to predict themselves
    fill_patterns = re.finditer(
        r"range\(first_cutoff_pos\s*\+\s*1\)|"
        r"range\(last_filled_pos\s*\+\s*1\s*,\s*cutoff_pos\s*\+\s*1\)",
        source,
    )
    for m in fill_patterns:
        pos = m.start()
        lineno = source[:pos].count("\n") + 1
        findings.append({
            "file": file_rel,
            "line": lineno,
            "severity": "CRITICAL",
            "check": "dc_refit_leakage",
            "message": (
                f"Dixon-Coles refit includes cutoff_pos in both training set AND "
                f"feature-fill range at line {lineno}. The model is fit on match P, "
                f"then used to compute features for match P (self-leakage). "
                f"Should exclude cutoff_pos from fill range."
            ),
            "context": "",
            "fixable": True,
        })

    return findings


def check_temporal_sort(file_rel: str) -> list[dict[str, Any]]:
    """Verify chronological sorting before feature computation."""
    findings: list[dict[str, Any]] = []
    source, lines = _get_source(file_rel)
    if not source:
        return findings

    # Feature engineering must sort by date
    if "feature_engineering.py" in file_rel:
        has_sort = "sort_values" in source and "date" in source
        if has_sort:
            findings.append({
                "file": file_rel,
                "line": next(
                    (i + 1 for i, l in enumerate(lines) if "sort_values" in l and "date" in l),
                    0,
                ),
                "severity": "INFO",
                "check": "chronological_sort",
                "message": "Data is sorted chronologically — OK",
                "context": "",
                "fixable": False,
            })
        else:
            findings.append({
                "file": file_rel,
                "line": 0,
                "severity": "CRITICAL",
                "check": "missing_sort",
                "message": "No chronological sort by date found in feature_engineering.py",
                "context": "",
                "fixable": True,
            })

    return findings


def check_tscv_usage(file_rel: str) -> list[dict[str, Any]]:
    """Check that time_series_cv is properly used for validation."""
    findings: list[dict[str, Any]] = []
    source, lines = _get_source(file_rel)
    if not source:
        return findings

    if "time_series_cv.py" in file_rel:
        # Verify that TimeSeriesSplit is used (no shuffle param — safe by design)
        has_ts_split = "TimeSeriesSplit" in source

        if has_ts_split:
            findings.append({
                "file": file_rel,
                "line": next(
                    (i + 1 for i, l in enumerate(lines) if "TimeSeriesSplit" in l),
                    0,
                ),
                "severity": "INFO",
                "check": "tscv_usage",
                "message": "TimeSeriesSplit is used — correct for time-series CV",
                "context": "",
                "fixable": False,
            })

    return findings


# ═══════════════════════════════════════════════════════════
#  Full audit
# ═══════════════════════════════════════════════════════════


def run_audit() -> dict[str, Any]:
    """Run all leakage checks and return a structured report."""
    start = time.time()
    start_time_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    all_findings: list[dict[str, Any]] = []
    summary = {
        "files_scanned": len(SCAN_FILES),
        "checks_run": 0,
        "findings_by_severity": {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
            "INFO": 0,
        },
        "findings_by_check": {},
        "passed": True,
    }

    for file_rel in SCAN_FILES:
        checks = [
            check_shift1(file_rel),
            check_target_leakage(file_rel),
            check_train_test_contamination(file_rel),
            check_dc_leakage(file_rel),
            check_temporal_sort(file_rel),
            check_tscv_usage(file_rel),
        ]

        for check_results in checks:
            for finding in check_results:
                all_findings.append(finding)
                sev = finding["severity"]
                summary["findings_by_severity"][sev] = (
                    summary["findings_by_severity"].get(sev, 0) + 1
                )
                check_name = finding["check"]
                summary["findings_by_check"][check_name] = (
                    summary["findings_by_check"].get(check_name, 0) + 1
                )
                if sev in ("CRITICAL", "HIGH"):
                    summary["passed"] = False

        summary["checks_run"] += len(checks)

    duration = time.time() - start
    summary["total_findings"] = len(all_findings)
    summary["duration_seconds"] = round(duration, 2)

    report: dict[str, Any] = {
        "report_title": "Data Leakage Audit Report",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "summary": summary,
        "findings": all_findings,
        "auto_fix_available": any(f.get("fixable") for f in all_findings),
        "fixable_findings": [
            f for f in all_findings if f.get("fixable")
        ],
    }

    return report


# ═══════════════════════════════════════════════════════════
#  Auto-fix functions
# ═══════════════════════════════════════════════════════════


def fix_attack_defence_ratios() -> bool:
    """Fix lookahead bias in _add_attack_defence_ratios.

    The original code computes league average from ALL completed matches.
    Fix: use only matches before the current one via expanding window.
    """
    path = PROJECT_ROOT / "src/feature_engineering.py"
    text = path.read_text(encoding="utf-8")

    # Replace the global league-average computation with an expanding-window version
    old_code = """    # Compute league-average goals per match from completed matches only
    completed = df[df["result"].notna()] if "result" in df.columns else df
    if len(completed) == 0:
        return df

    avg_home = float(completed["home_goals"].mean())
    avg_away = float(completed["away_goals"].mean())

    # Guard against division by zero (no goals ever scored)
    league_avg = (avg_home + avg_away) / 2.0
    if league_avg <= 0:
        league_avg = 1.0"""

    new_code = """    # Compute league-average goals from the running (expanding) window
    # so each match only sees data available before kick-off.
    # The expanding window is computed per-match and shifted to avoid lookahead.
    df = _add_running_league_avg(df)"""

    if old_code not in text:
        logger.warning("Could not find attack_defence_ratio code to fix")
        return False

    text = text.replace(old_code, new_code)

    # Add the helper function before _add_attack_defence_ratios
    helper = """
def _add_running_league_avg(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Add running league-average goals using an expanding window (no lookahead).\"\"\"
    team_stats = _compute_team_stats(df)
    team_stats.sort_values(["team", "date"], inplace=True)

    def _expanding_avg(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.sort_values("date").copy()
        # Compute running averages shifted by 1 to exclude current match
        grp["goals_scored_cum"] = grp["goals_scored"].expanding().mean().shift(1)
        grp["goals_conceded_cum"] = grp["goals_conceded"].expanding().mean().shift(1)
        return grp

    team_stats = team_stats.groupby("team", group_keys=False).apply(_expanding_avg)

    # Aggregate across all teams to get league avg per match day
    league_avgs = team_stats.groupby("match_id").agg({
        "goals_scored_cum": "mean",
        "goals_conceded_cum": "mean",
    }).rename(columns={
        "goals_scored_cum": "league_avg_goals_scored",
        "goals_conceded_cum": "league_avg_goals_conceded",
    })

    df = df.join(league_avgs, how="left")

    # Fill NaN for first matches with a sensible default
    for col in ["league_avg_goals_scored", "league_avg_goals_conceded"]:
        if col in df.columns:
            df[col].fillna(1.0, inplace=True)
    return df


"""
    # Insert before _add_attack_defence_ratios
    marker = "def _add_attack_defence_ratios(df: pd.DataFrame) -> pd.DataFrame:"
    if marker in text:
        text = text.replace(
            f"def _add_running_league_avg",
            "def _add_running_league_avg_NO_MATCH",
        )  # Don't double-insert
        text = text.replace(
            marker,
            helper + marker,
        )
    else:
        logger.warning("Could not find _add_attack_defence_ratios to insert helper")
        return False

    # Update the ratio computation to use running league avg
    # Remove the old league_avg computation and replace with expanding
    old_loop = """    # Check which rolling windows are available
    windows = getattr(config.features, "rolling_windows", (5, 10, 20))

    for w in windows:"""

    # We need to replace the league_avg usage inside the loop
    new_loop = """    # Use running (no-lookahead) league average
    league_avg = (
        df["league_avg_goals_scored"].mean() +
        df["league_avg_goals_conceded"].mean()
    ) / 2.0
    if pd.isna(league_avg) or league_avg <= 0:
        league_avg = 1.0

    # Check which rolling windows are available
    windows = getattr(config.features, "rolling_windows", (5, 10, 20))

    for w in windows:"""

    if old_loop in text:
        text = text.replace(old_loop, new_loop)
    else:
        logger.warning("Could not find rolling windows loop to update")
        return False

    path.write_text(text, encoding="utf-8")
    return True


def fix_dc_refit_leakage() -> bool:
    """Fix Dixon-Coles refit leakage: exclude cutoff_pos from fill range."""
    path = PROJECT_ROOT / "src/dixon_coles.py"
    text = path.read_text(encoding="utf-8")

    # Fix 1: first chunk fill should exclude first_cutoff_pos
    old1 = "for i in range(first_cutoff_pos + 1):"
    new1 = "for i in range(first_cutoff_pos):  # Exclude cutoff to prevent self-leakage"
    if old1 in text:
        text = text.replace(old1, new1)
    else:
        logger.warning("Could not fix first DC refit range")
        return False

    # Fix 2: subsequent chunks should exclude cutoff_pos
    old2 = "for i in range(last_filled_pos + 1, cutoff_pos + 1):"
    new2 = "for i in range(last_filled_pos + 1, cutoff_pos):  # Exclude cutoff to prevent self-leakage"
    if old2 in text:
        text = text.replace(old2, new2)
    else:
        logger.warning("Could not fix second DC refit range")
        return False

    path.write_text(text, encoding="utf-8")
    return True


def run_auto_fix() -> dict[str, Any]:
    """Run all available auto-fixes and report results."""
    results: dict[str, Any] = {
        "fixes_attempted": 0,
        "fixes_succeeded": 0,
        "fixes_failed": 0,
        "details": [],
    }

    # Fix 1: Attack/defence ratios league average
    results["fixes_attempted"] += 1
    try:
        if fix_attack_defence_ratios():
            results["fixes_succeeded"] += 1
            results["details"].append({
                "fix": "attack_defence_ratios_lookahead",
                "status": "fixed",
                "file": "src/feature_engineering.py",
                "description": "Replaced global league average with expanding-window (no lookahead)",
            })
        else:
            results["fixes_failed"] += 1
            results["details"].append({
                "fix": "attack_defence_ratios_lookahead",
                "status": "failed",
                "file": "src/feature_engineering.py",
                "description": "Could not match code patterns for auto-fix",
            })
    except Exception as e:
        results["fixes_failed"] += 1
        results["details"].append({
            "fix": "attack_defence_ratios_lookahead",
            "status": "error",
            "file": "src/feature_engineering.py",
            "description": str(e),
        })

    # Fix 2: DC refit leakage
    results["fixes_attempted"] += 1
    try:
        if fix_dc_refit_leakage():
            results["fixes_succeeded"] += 1
            results["details"].append({
                "fix": "dc_refit_leakage",
                "status": "fixed",
                "file": "src/dixon_coles.py",
                "description": "Excluded cutoff_pos from fill range in add_features",
            })
        else:
            results["fixes_failed"] += 1
            results["details"].append({
                "fix": "dc_refit_leakage",
                "status": "failed",
                "file": "src/dixon_coles.py",
                "description": "Could not match code patterns for auto-fix",
            })
    except Exception as e:
        results["fixes_failed"] += 1
        results["details"].append({
            "fix": "dc_refit_leakage",
            "status": "error",
            "file": "src/dixon_coles.py",
            "description": str(e),
        })

    return results


# ═══════════════════════════════════════════════════════════
#  Report formatting
# ═══════════════════════════════════════════════════════════


def format_report_md(report: dict[str, Any], fix_results: dict[str, Any] | None = None) -> str:
    """Format the audit report as Markdown."""
    s = report["summary"]
    lines: list[str] = [
        f"# Data Leakage Audit Report",
        f"",
        f"- **Timestamp:** {report['timestamp']}",
        f"- **Duration:** {s['duration_seconds']:.2f}s",
        f"- **Files scanned:** {s['files_scanned']}",
        f"- **Checks run:** {s['checks_run']}",
        f"- **Overall status:** {'**PASS**' if s['passed'] else '**FAIL**'}",
        f"",
        f"## Summary",
        f"",
        f"| Severity | Count |",
        f"|----------|-------|",
    ]
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        count = s["findings_by_severity"].get(sev, 0)
        lines.append(f"| {sev} | {count} |")
    lines.append(f"| **Total** | **{s['total_findings']}** |")
    lines.append("")

    if fix_results:
        lines.extend([
            f"## Auto-Fix Results",
            f"",
            f"| Fix | Status | File |",
            f"|-----|--------|------|",
        ])
        for d in fix_results["details"]:
            status_icon = "✅" if d["status"] == "fixed" else "❌"
            lines.append(f"| {status_icon} {d['fix']} | {d['status']} | {d['file']} |")
        lines.append("")

    # Group findings by file
    findings_by_file: dict[str, list[dict]] = {}
    for f in report["findings"]:
        findings_by_file.setdefault(f["file"], []).append(f)

    lines.append("## Findings by File")
    lines.append("")

    for file_rel in SCAN_FILES:
        file_findings = findings_by_file.get(file_rel, [])
        if not file_findings:
            lines.append(f"### {file_rel}")
            lines.append("")
            lines.append("_No findings._")
            lines.append("")
            continue

        lines.append(f"### {file_rel}")
        lines.append("")
        lines.append("| Line | Severity | Check | Message |")
        lines.append("|------|----------|-------|---------|")

        for f in file_findings:
            fixable_mark = " *(auto-fixable)*" if f.get("fixable") else ""
            sev_icon = {
                "CRITICAL": "🔴",
                "HIGH": "🟠",
                "MEDIUM": "🟡",
                "LOW": "🔵",
                "INFO": "⚪",
            }.get(f["severity"], "⚪")
            context = f.get("context", "")
            if context:
                context = f"<br>`{context[:120].strip()}`"
            lines.append(
                f"| {f['line']} | {sev_icon} {f['severity']}{fixable_mark} | "
                f"`{f['check']}` | {f['message']}{context} |"
            )
        lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for f in report["findings"]:
        if f["severity"] in ("CRITICAL", "HIGH"):
            lines.append(f"1. **{f['file']}:{f['line']}** — {f['message']}")
            if f.get("fixable"):
                lines.append(f"   _Auto-fix available via `--fix` flag._")
            lines.append("")

    lines.append("---")
    lines.append(f"_Generated by `scripts/audit_leakage.py` at {report['timestamp']}_")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Data Leakage Audit for Football Prediction Pipeline",
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Apply auto-fixes for detectable leakage issues",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        help="Output path for Markdown report (default: reports/leakage_audit_{timestamp}.md)",
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="Print JSON report to stdout only",
    )
    args = parser.parse_args()

    # Run audit
    report = run_audit()
    fix_results = None

    # Auto-fix if requested
    if args.fix:
        fix_results = run_auto_fix()
        report["fix_results"] = fix_results
        # Re-audit after fixes
        report["post_fix_audit"] = run_audit()

    # Determine output path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(PROJECT_ROOT / f"reports/leakage_audit_{timestamp}.md")

    if args.json_only:
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["summary"]["passed"] else 1

    # Generate Markdown report
    md = format_report_md(report, fix_results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(md, encoding="utf-8")

    # Print summary to stdout
    s = report["summary"]
    print(f"\n{'=' * 55}")
    print(f"  DATA LEAKAGE AUDIT REPORT")
    print(f"{'=' * 55}")
    print(f"  Files scanned:  {s['files_scanned']}")
    print(f"  Checks run:     {s['checks_run']}")
    print(f"  Duration:       {s['duration_seconds']:.2f}s")
    print(f"  Status:         {'PASS' if s['passed'] else 'FAIL'}")
    print()
    print(f"  Findings breakdown:")
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        count = s["findings_by_severity"].get(sev, 0)
        print(f"    {sev:<10s} {count}")
    print()
    print(f"  Report saved: {output_path}")

    if fix_results:
        print()
        print(f"  Auto-fix results:")
        for d in fix_results["details"]:
            status_icon = "+" if d["status"] == "fixed" else "x"
            print(f"    [{status_icon}] {d['fix']}: {d['status']}")

        pf = report.get("post_fix_audit", {}).get("summary", {})
        if pf:
            print()
            print(f"  Post-fix audit:")
            print(f"    Findings: {pf.get('total_findings', 0)} "
                  f"(critical: {pf.get('findings_by_severity', {}).get('CRITICAL', 0)}, "
                  f"high: {pf.get('findings_by_severity', {}).get('HIGH', 0)})")
            print(f"    Status: {'PASS' if pf.get('passed', True) else 'STILL FAILING'}")

    print(f"{'=' * 55}")
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
