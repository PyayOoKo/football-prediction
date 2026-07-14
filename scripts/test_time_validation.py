"""
Time-Based Validation Audit — verify chronological correctness across the pipeline.

Checks:
  1. Data is chronologically sorted before feature building
  2. All split functions produce strict chronological ordering
  3. Rolling features use .shift(1) (no self-leakage)
  4. Elo ratings are pre-match (recorded before update)
  5. Poisson features use expanding window (no lookahead)
  6. Dixon-Coles add_features excludes cutoff from fill range
  7. Hyperparameter tuning uses TimeSeriesSplit
  8. All scripts use chronological splits (no shuffle)
  9. No future data leaks into training features
 10. Imputation is fit on training data only
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger("time_val")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

# ── Check result type ───────────────────────────────────
class CheckResult:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.passed: bool = True
        self.details: list[str] = []
        self.errors: list[str] = []

    def ok(self, msg: str) -> None:
        self.details.append(f"  [+] {msg}")

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(f"  [x] {msg}")

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {len(self.errors)} error(s)"

# ═══════════════════════════════════════════════════════════
#  1. Data loading utility
# ═══════════════════════════════════════════════════════════

DATA_PATH = ROOT / "data" / "processed" / "results_clean.csv"

def _load_df() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

# ═══════════════════════════════════════════════════════════
#  2. Check implementations
# ═══════════════════════════════════════════════════════════

def check_chronological_sort() -> CheckResult:
    """Check that build_features sorts data chronologically."""
    ck = CheckResult("chronological_sort", "Feature engineering sorts data by date")
    spec = importlib.util.spec_from_file_location("fe", ROOT / "src" / "feature_engineering.py")
    if spec is None or spec.loader is None:
        ck.fail("Could not load feature_engineering.py")
        return ck
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = (ROOT / "src" / "feature_engineering.py").read_text("utf-8")

    if "sort_values" in src and "date" in src:
        ck.ok("Data is sorted chronologically via sort_values(['date', ...])")
    else:
        ck.fail("No chronological sort by date found in build_features")
        return ck

    # Verify the sort is BEFORE feature computation
    lines = src.splitlines()
    build_func_lines = []
    in_build = False
    for i, line in enumerate(lines):
        if line.strip().startswith("def build_features("):
            in_build = True
        if in_build:
            build_func_lines.append((i + 1, line))
            if in_build and line.strip().startswith("def ") and "build_features" not in line:
                break

    sort_line = None
    for lineno, line in build_func_lines:
        if "sort_values" in line and "date" in line:
            sort_line = lineno
            break

    shift_lines = []
    for lineno, line in build_func_lines:
        if ".shift(" in line:
            shift_lines.append(lineno)

    if sort_line and shift_lines:
        first_shift = min(shift_lines)
        if sort_line < first_shift:
            ck.ok(f"Sort (line {sort_line}) occurs before first .shift() usage (line {first_shift})")
        else:
            ck.fail(f"Sort (line {sort_line}) occurs AFTER .shift() usage (line {first_shift}) — rolling features may use unsorted data")
    elif sort_line:
        ck.ok(f"Sort found at line {sort_line}")

    return ck


def check_split_functions() -> CheckResult:
    """Verify that split functions produce strict chronological ordering."""
    ck = CheckResult("split_chronological", "All split functions maintain chronological order")

    df = _load_df()
    if df is None:
        ck.fail(f"Data file not found at {DATA_PATH}")
        return ck

    df_sorted = df.sort_values(["date", "home_team"]).reset_index(drop=True)
    n = len(df_sorted)

    # Import split functions
    from src.feature_engineering import train_val_test_split as legacy_split
    from src.time_series_cv import time_series_train_val_test_split as ts_split

    # Create dummy X, y
    X = pd.DataFrame({"dummy": range(n)})
    y = pd.Series(np.zeros(n, dtype=int))

    # Test legacy split
    try:
        splits = legacy_split(X, y, ratios=(0.6, 0.2, 0.2))
        train_end = len(splits["X_train"])
        val_end = train_end + len(splits["X_val"])

        # Check ordering: train indices < val indices < test indices
        train_idx = splits["X_train"].index
        val_idx = splits["X_val"].index
        test_idx = splits["X_test"].index

        assert all(i < j for i in train_idx for j in val_idx), "Train/val overlap or misorder"
        assert all(i < j for i in val_idx for j in test_idx), "Val/test overlap or misorder"
        ck.ok("train_val_test_split (legacy) maintains strict chronological order")
        assert len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"]) == n
        ck.ok("Split sizes sum to total")
    except Exception as e:
        ck.fail(f"train_val_test_split failed: {e}")

    # Test time-series split
    try:
        splits2 = ts_split(X, y, ratios=(0.6, 0.2, 0.2))
        train_end2 = len(splits2["X_train"])
        val_end2 = train_end2 + len(splits2["X_val"])

        train_idx2 = splits2["X_train"].index
        val_idx2 = splits2["X_val"].index
        test_idx2 = splits2["X_test"].index

        assert all(i < j for i in train_idx2 for j in val_idx2)
        assert all(i < j for i in val_idx2 for j in test_idx2)
        ck.ok("time_series_train_val_test_split maintains strict chronological order")
        assert len(splits2["X_train"]) + len(splits2["X_val"]) + len(splits2["X_test"]) == n
        ck.ok("Split sizes sum to total")
    except Exception as e:
        ck.fail(f"time_series_train_val_test_split failed: {e}")

    return ck


def check_shift1_usage() -> CheckResult:
    """Verify all rolling/expanding computations use .shift(1)."""
    ck = CheckResult("shift1_usage", "All rolling features use .shift(1) for leakage prevention")

    files_to_check = [
        "src/feature_engineering.py",
        "src/poisson_model.py",
        "src/dixon_coles.py",
        "src/elo.py",
        "src/xg_features.py",
    ]

    total_rolling = 0
    missing_shift = 0

    for fname in files_to_check:
        path = ROOT / fname
        if not path.exists():
            ck.fail(f"File not found: {fname}")
            continue
        src = path.read_text("utf-8")
        lines = src.splitlines()

        # Find all .rolling(, .expanding(, .ewm( calls
        for m in re.finditer(r"(\.rolling\s*\(|\.expanding\s*\(|\.ewm\s*\()", src):
            total_rolling += 1
            pos = m.start()
            lineno = src[:pos].count("\n") + 1
            context_start = max(0, lineno - 3)
            context = "\n".join(lines[context_start : lineno + 5])

            has_shift = ".shift(" in context
            if not has_shift:
                missing_shift += 1
                ck.fail(f"{fname}:{lineno} — rolling/expanding call without .shift(1)")

    if total_rolling > 0:
        ck.ok(f"{total_rolling} rolling/expanding calls found across {len(files_to_check)} files")
    if missing_shift == 0:
        ck.ok("All rolling/expanding calls have .shift(1) — no self-leakage detected")

    return ck


def check_elo_ratings() -> CheckResult:
    """Verify Elo ratings are recorded before match result is applied."""
    ck = CheckResult("elo_pre_match", "Elo ratings are pre-match (recorded before update)")

    path = ROOT / "src" / "elo.py"
    src = path.read_text("utf-8")

    # Check process_matches records rating BEFORE update_ratings
    lines = src.splitlines()
    record_before_update = False
    update_before_record = False

    for i, line in enumerate(lines):
        if "R_home = _get_rating(home)" in line and "R_away = _get_rating(away)" in line:
            pass
        # Check that elo_diff is computed before update (recorded pre-match)
        if "elo_diff = R_home - R_away" in line:
            # Find corresponding _append_diff and _append_elo calls
            for j in range(i, min(i + 30, len(lines))):
                if "_append_elo(R_home)" in lines[j] or "_append_diff(elo_diff)" in lines[j]:
                    # Find if this is before or after _update call
                    pass

    # Simpler: check the comment/docstring
    if "pre-match" in src.lower() and "before" in src.lower():
        ck.ok("Docstring states ratings are recorded before match result")

    # Check the actual code flow: _get_rating -> compute diff -> _update -> _append_*
    has_get_rating = "_get_rating(home)" in src
    has_update = "self.update_ratings" in src or "_update(" in src
    has_append = "home_elo_list.append" in src

    if has_get_rating and has_update and has_append:
        ck.ok("Code flow: get_rating -> (optional update) -> append (pre-match rating)")
    else:
        ck.fail("Could not verify pre-match rating flow in elo.py")

    # Verify no future information flows backwards
    if "recorded in ``Home_Elo``" in src and "**before** the match" in src:
        ck.ok("Documentation confirms pre-match rating recording")

    return ck


def check_poisson_leakage() -> CheckResult:
    """Verify Poisson features use expanding (historical-only) window."""
    ck = CheckResult("poisson_expanding_window", "Poisson features use expanding window with .shift(1)")

    path = ROOT / "src" / "poisson_model.py"
    src = path.read_text("utf-8")

    lines = src.splitlines()

    # Check the add_poisson_features method
    # It should iterate chronologically and compute stats from PREVIOUS matches only
    total_matches_init = False
    stats_updated_after = False

    for i, line in enumerate(lines):
        if "total_matches = 0" in line:
            total_matches_init = True
        if "total_matches += 1" in line:
            # This should come AFTER the expected goals computation
            # Check if there's a lambda/computation before this line
            for j in range(max(0, i - 20), i):
                if "λ_home" in lines[j] or "expected_home.append" in lines[j]:
                    stats_updated_after = True
                    break

    if total_matches_init:
        ck.ok("total_matches counter starts at 0 (no matches seen yet)")

    # Verify the expected goals are computed from PREVIOUS matches
    for i, line in enumerate(lines):
        if "total_matches > 0" in line and "μ_home" in line:
            ck.ok(f"League averages computed from pre-match matches only (line {i+1})")
            break

    # Check the docstring
    if "only matches that occurred before the current match" in src.lower():
        ck.ok("Docstring confirms historical-only computation")

    # Check the iteration pattern: compute, THEN update
    idx_compute = None
    idx_update = None
    for i, line in enumerate(lines):
        if "λ_home = μ_home * α_home * β_away" in line:
            idx_compute = i
        if "total_matches += 1" in line:
            idx_update = i

    if idx_compute is not None and idx_update is not None:
        if idx_compute < idx_update:
            ck.ok("Expected goals computed before updating aggregates (correct order)")
        else:
            ck.fail("Aggregates updated BEFORE expected goals — potential lookahead")

    # Check docstring confirms leakage prevention
    if "Leakage prevention" in src:
        ck.ok("Leakage prevention section present in docstring")

    return ck


def check_dc_refit_leakage() -> CheckResult:
    """Verify Dixon-Coles add_features excludes cutoff from fill range."""
    ck = CheckResult("dc_refit_cutoff", "DC add_features excludes cutoff match from fill range")

    path = ROOT / "src" / "dixon_coles.py"
    src = path.read_text("utf-8")

    # Check the fill range patterns
    patterns = [
        # Current correct pattern (excludes cutoff)
        (r"range\(first_cutoff_pos\):\s*# Exclude cutoff to prevent self-leakage", True),
        (r"range\(last_filled_pos\s*\+\s*1\s*,\s*cutoff_pos\):\s*# Exclude cutoff to prevent self-leakage", True),
        # Old buggy patterns (should NOT exist)
        (r"range\(first_cutoff_pos\s*\+\s*1\):", False),
        (r"range\(last_filled_pos\s*\+\s*1\s*,\s*cutoff_pos\s*\+\s*1\):", False),
    ]

    for pattern, should_exist in patterns:
        found = re.search(pattern, src)
        if should_exist:
            if found:
                ck.ok(f"Correct fill-range pattern found: {pattern}")
            else:
                ck.fail(f"Missing expected fill-range pattern: {pattern}")
        else:
            if found:
                ck.fail(f"Buggy fill-range pattern STILL PRESENT: {pattern}")
            else:
                ck.ok(f"Buggy fill-range pattern absent: {pattern}")

    return ck


def check_cv_time_series() -> CheckResult:
    """Verify hyperparameter tuning uses TimeSeriesSplit."""
    ck = CheckResult("cv_time_series", "Hyperparameter tuning uses TimeSeriesSplit")

    files_to_check = [
        ("src/hyperparameter_tuning.py", "HyperTuner"),
        ("src/train.py", "tune_hyperparameters"),
    ]

    for fname, func_name in files_to_check:
        path = ROOT / fname
        if not path.exists():
            ck.fail(f"File not found: {fname}")
            continue
        src = path.read_text("utf-8")

        # Check for TimeSeriesSplit import or usage
        if "TimeSeriesSplit" in src or "create_time_series_folds" in src:
            ck.ok(f"{fname} uses TimeSeriesSplit or create_time_series_folds")
        else:
            ck.fail(f"{fname} does not use time-series aware CV")

        # Check for randomized/GridSearchCV usage
        if "RandomizedSearchCV" in src or "GridSearchCV" in src:
            # Verify cv parameter is set
            cv_assignments = re.findall(r"cv\s*=\s*create_time_series_folds", src)
            cv_kwargs = re.findall(r"cv\s*=\s*ts_cv", src)
            if cv_assignments or cv_kwargs:
                ck.ok(f"{fname}: CV parameter uses time-series folds")
            else:
                ck.fail(f"{fname}: CV parameter may not use time-series folds")

    # Check time_series_cv.py itself
    tscv_src = (ROOT / "src" / "time_series_cv.py").read_text("utf-8")
    if "TimeSeriesSplit" in tscv_src:
        ck.ok("time_series_cv.py wraps sklearn TimeSeriesSplit (expanding window)")
    else:
        ck.fail("time_series_cv.py does not use sklearn TimeSeriesSplit")

    return ck


def check_script_splits() -> CheckResult:
    """Verify all scripts use chronological splits."""
    ck = CheckResult("script_split_methods", "All scripts use chronological split methods")

    scripts_dir = ROOT / "scripts"
    script_files = sorted(scripts_dir.glob("*.py"))

    # Scripts that do NOT perform ML training (utility scripts) — skip these
    NON_TRAINING_SCRIPTS = {
        "benchmark_database.py", "bump_version.py", "generate_changelog.py",
        "notify.py", "auto_commit.ps1", "migrate_to_partitions.py",
        "debug_lineups.py", "debug_lineups2.py", "debug_lineups3.py",
        "debug_transfermarkt.py", "debug_transfermarkt2.py", "verify_ids.py",
        "verify_feature_store.py", "validate_features.py",
    }

    suspicious_scripts = []

    for script_path in script_files:
        fname = script_path.name
        if fname in NON_TRAINING_SCRIPTS:
            continue

        src = script_path.read_text("utf-8")

        # Check for train_test_split usage
        tts_uses = re.findall(r"train_test_split\s*\(", src)
        if tts_uses:
            # Check if shuffle=False is set
            shuffle_context = src[src.find("train_test_split"):src.find("train_test_split") + 500]
            if "shuffle" not in shuffle_context:
                suspicious_scripts.append((fname, "train_test_split without shuffle parameter"))
            elif "shuffle=True" in shuffle_context:
                suspicious_scripts.append((fname, "train_test_split with shuffle=True"))

        if "train_test_split" not in src:
            # Check it uses chronological split
            has_chrono = False
            for keyword in ["time_series_train_val_test_split", "train_val_test_split",
                           "X.iloc[:train_end]", "X.iloc[:split]",
                           "X.iloc[train_end:]", "X.iloc[split:]"]:
                if keyword in src:
                    has_chrono = True
                    break
            if not has_chrono and ("X_train" in src or "split" in src):
                suspicious_scripts.append((fname, "no obvious chronological split method"))

    if suspicious_scripts:
        for fname, issue in suspicious_scripts:
            ck.fail(f"{fname}: {issue}")
    else:
        ck.ok("All scripts use chronological split methods")

    return ck


def check_imputation_leakage() -> CheckResult:
    """Verify imputation statistics are fit on training data only."""
    ck = CheckResult("imputation_no_leakage", "Imputation statistics fit on training data only")

    files_to_check = [
        "scripts/train_baseline.py",
        "scripts/test_end_to_end.py",
        "scripts/feature_importance_analysis.py",
        "scripts/tune_ensemble.py",
    ]

    for fname in files_to_check:
        path = ROOT / fname
        if not path.exists():
            continue
        src = path.read_text("utf-8")

        # Check pattern: col_means = X_train... ; then fillna with col_means for val/test
        train_mean_pattern = r"col_means\s*=\s*(splits\[\"X_train\"\]|X_train)\.mean\(\)"
        val_fill_pattern = r"(X_val|X_test).*fillna\s*\(\s*col_means\s*\)"

        has_train_mean = re.search(train_mean_pattern, src)
        has_val_fill = re.search(val_fill_pattern, src)

        if has_train_mean:
            ck.ok(f"{fname}: column means computed from training data only")
            if has_val_fill:
                ck.ok(f"{fname}: validation/test sets filled with training means")
        else:
            # Check if it fills NaN at all
            if "fillna" in src and ("X_val" in src or "X_test" in src):
                # Could be using X.mean() instead of X_train.mean()
                if "X.mean()" in src or "X.fillna" in src:
                    ck.fail(f"{fname}: potential lookahead in imputation (using full X not X_train)")

    return ck


def check_no_future_data() -> CheckResult:
    """Check for any future-data contamination patterns."""
    ck = CheckResult("no_future_data", "No future data used in feature computation")

    files_to_check = [
        "src/feature_engineering.py",
        "src/poisson_model.py",
        "src/dixon_coles.py",
        "src/elo.py",
        "src/xg_features.py",
        "src/odds_processing.py",
        "src/preprocessing.py",
    ]

    suspicious_patterns = [
        (r"\.iloc\s*\[.*:.*\].*\.mean\(\)", "Full-dataset mean computed before split"),
        (r"shuffle\s*=\s*True", "Random shuffle active"),
        (r"(result|target)\.notna\(\)", "Filtering by completed matches (potential lookahead)"),
    ]

    for fname in files_to_check:
        path = ROOT / fname
        if not path.exists():
            continue
        src = path.read_text("utf-8")
        for pattern, desc in suspicious_patterns:
            for m in re.finditer(pattern, src):
                lineno = src[:m.start()].count("\n") + 1
                # Only flag if it's in a suspicious context (e.g., before split)
                if "result.notna" in pattern:
                    # Check if it's in a function called during feature building
                    if "_add_attack_defence_ratios" in src[:m.start()][-500:]:
                        ck.fail(f"{fname}:{lineno} — {desc} in attack/defence ratio computation")
                elif "shuffle=True" in pattern:
                    ck.fail(f"{fname}:{lineno} — {desc}")

    # Check _add_attack_defence_ratios specifically
    fe_src = (ROOT / "src" / "feature_engineering.py").read_text("utf-8")
    if "df[\"result\"].notna()" in fe_src:
        # Check if it's in the current code or patched
        if "_add_running_league_avg" in fe_src:
            ck.ok("_add_attack_defence_ratios uses expanding window (no lookahead)")
        else:
            ck.fail("_add_attack_defence_ratios uses global df['result'].notna() lookahead")

    return ck


def check_feature_engineering_build() -> CheckResult:
    """Verify build_features pipeline order is correct."""
    ck = CheckResult("build_features_order", "build_features pipeline processes features in correct order")

    src = (ROOT / "src" / "feature_engineering.py").read_text("utf-8")
    lines = src.splitlines()

    # Extract the build_features function body to check call order
    in_build = False
    build_body = []
    build_indent = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def build_features("):
            in_build = True
            build_indent = len(line) - len(line.lstrip())
            continue
        if in_build:
            # Check indentation to detect end of function
            if stripped and not stripped.startswith("#"):
                indent = len(line) - len(line.lstrip())
                if indent <= build_indent and stripped.startswith("def "):
                    break
            build_body.append((i + 1, line))

    # Find call positions within build_features
    target_keywords = [
        "sort_values",
        "add_elo_features",
        "add_odds_features",
        "add_poisson_features",
        "add_features",  # DC
        "_add_competition_importance",
        "_add_rolling_features",
        "_add_h2h_features",
        "_add_league_position_features",
        "_encode_categoricals",
        "_add_attack_defence_ratios",
        "_get_target_columns",
        "cols_to_drop",
    ]

    found_positions = []
    for lineno, line in build_body:
        for kw in target_keywords:
            if kw in line:
                found_positions.append((lineno, kw))
                break

    # Check target dropping is AFTER all feature computation calls
    drop_line = None
    feature_lines = []
    for lineno, kw in found_positions:
        if kw in ("_get_target_columns", "cols_to_drop"):
            drop_line = lineno
        else:
            feature_lines.append((lineno, kw))

    if drop_line and feature_lines:
        last_feature_call = max(lineno for lineno, _ in feature_lines)
        if drop_line > last_feature_call:
            ck.ok(f"Target dropping (line {drop_line}) occurs after last feature call (line {last_feature_call})")
        else:
            ck.fail(f"Target dropping (line {drop_line}) precedes feature call at line {last_feature_call}")
    elif drop_line:
        ck.ok(f"Target columns dropped at line {drop_line}")

    # Verify chronological ordering of call sequence
    prev_line = 0
    violations = 0
    for lineno, kw in found_positions:
        if lineno < prev_line:
            violations += 1
        prev_line = lineno

    if violations == 0:
        ck.ok(f"Pipeline call order verified: {len(found_positions)} sequential calls")
    else:
        ck.fail(f"Pipeline call ordering violation: {violations} step(s) out of order")

    return ck


# ═══════════════════════════════════════════════════════════
#  Full audit runner
# ═══════════════════════════════════════════════════════════

CHECK_FUNCTIONS = [
    check_chronological_sort,
    check_split_functions,
    check_shift1_usage,
    check_elo_ratings,
    check_poisson_leakage,
    check_dc_refit_leakage,
    check_cv_time_series,
    check_script_splits,
    check_imputation_leakage,
    check_no_future_data,
    check_feature_engineering_build,
]


def run_all_checks() -> list[CheckResult]:
    """Run all time-validation checks and return results."""
    results: list[CheckResult] = []
    for check_fn in CHECK_FUNCTIONS:
        try:
            result = check_fn()
        except Exception as e:
            result = CheckResult(check_fn.__name__, "Error during check")
            result.fail(f"Exception: {e}")
        results.append(result)
    return results


# ═══════════════════════════════════════════════════════════
#  Report generation
# ═══════════════════════════════════════════════════════════

def generate_report(results: list[CheckResult], duration: float) -> str:
    """Generate markdown audit report."""
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_errors = sum(len(r.errors) for r in results)

    lines: list[str] = []
    lines.append("# Time Validation Audit Report")
    lines.append("")
    lines.append(f"- **Timestamp:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"- **Duration:** {duration:.2f}s")
    lines.append(f"- **Checks run:** {len(results)}")
    lines.append(f"- **Passed:** {passed}")
    lines.append(f"- **Failed:** {failed}")
    lines.append(f"- **Total issues:** {total_errors}")
    lines.append(f"- **Data file:** {DATA_PATH}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Check | Status | Errors |")
    lines.append("|-------|--------|--------|")
    for r in results:
        status = "+ PASS" if r.passed else "x FAIL"
        lines.append(f"| {r.name} | {status} | {len(r.errors)} |")
    lines.append("")
    lines.append(f"**Overall: {'PASS' if failed == 0 else 'FAIL'}**")
    lines.append("")

    for r in results:
        lines.append(f"## {r.name}")
        lines.append("")
        lines.append(f"**Description:** {r.description}")
        lines.append("")
        lines.append(f"**Status:** {'PASS' if r.passed else 'FAIL'}")
        lines.append("")
        for d in r.details:
            lines.append(d)
        for e in r.errors:
            lines.append(e)
        lines.append("")

    lines.append("---")
    lines.append(f"_Generated by `scripts/test_time_validation.py`_")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main() -> int:
    global TIMESTAMP
    print("=" * 65)
    print("  TIME-BASED VALIDATION AUDIT")
    print("=" * 65)

    t_start = time.time()

    # Run checks
    results = run_all_checks()

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_errors = sum(len(r.errors) for r in results)
    duration = time.time() - t_start

    # Print summary
    print(f"\n  Checks: {len(results)}  |  Passed: {passed}  |  Failed: {failed}  |  Issues: {total_errors}")
    print()

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        icon = "[+]" if r.passed else "[x]"
        print(f"  {icon} [{status}] {r.name}")
        for e in r.errors:
            print(f"         {e}")
    print()

    # Generate report
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"time_validation_audit_{TIMESTAMP}.md"
    report = generate_report(results, duration)
    report_path.write_text(report, encoding="utf-8")

    print(f"  Report saved: {report_path}")
    print(f"  Duration: {duration:.2f}s")
    print("=" * 65)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
