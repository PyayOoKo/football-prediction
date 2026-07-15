#!/usr/bin/env python3
"""
Phase 4 Completion Checklist — verify Machine Learning Models deliverables.

Checks model artifacts, validation reports, hyperparameter tuning, leaderboard,
Phase 3 vs Phase 4 comparison, and comparison figures. Writes a markdown
checklist report under ``reports/``.

Usage:
    python scripts/phase4_checklist.py
    python scripts/phase4_checklist.py --output reports/phase4_checklist_custom.md
    python scripts/phase4_checklist.py --json-only
    python scripts/phase4_checklist.py --missing

Exit code:
    0 — all required checks pass (COMPLETE)
    1 — one or more required checks fail (PARTIAL / INCOMPLETE)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"
DEFAULT_OUTPUT = REPORTS_DIR / f"phase4_checklist_{TIMESTAMP}.md"

# Metric gates (1X2)
BRIER_MAX = 0.50
LOG_LOSS_MAX = 1.20
ACCURACY_MIN = 0.55
LEADERBOARD_BEST_BRIER_MAX = 0.45

REQUIRED_LEADERBOARD_MODELS = [
    "Poisson",
    "Dixon-Coles",
    "Elo",
    "XGBoost",
    "LightGBM",
    "RandomForest",
    "NeuralNetwork",
]

ML_MODELS = [
    ("XGBoost", "xgboost_model.joblib", "xgboost_validation_*.json", "xgboost"),
    ("LightGBM", "lightgbm_model.joblib", "lightgbm_validation_*.json", "lightgbm"),
    ("Random Forest", "random_forest_model.joblib", "random_forest_validation_*.json", "random_forest"),
    ("Neural Network", "neural_network_model.joblib", "neural_network_validation_*.json", "neural_network"),
]

TUNING_ML_KEYS = ("xgboost", "lightgbm", "random_forest", "neural_network")


@dataclass
class CheckResult:
    """Single checklist item."""

    name: str
    passed: bool
    evidence: str = ""
    severity: str = "required"  # required | warning


@dataclass
class ModelMetrics:
    model: str
    found: bool = False
    report_path: str | None = None
    brier_score: float | None = None
    log_loss: float | None = None
    accuracy: float | None = None
    btts_accuracy: float | None = None
    over25_accuracy: float | None = None
    metrics_ok: bool = False
    notes: list[str] = field(default_factory=list)


def _latest_glob(pattern: str, directory: Path = REPORTS_DIR) -> Path | None:
    hits = sorted(directory.glob(pattern))
    return hits[-1] if hits else None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"__load_error__": str(exc)}


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float | None]:
    """Pull metric fields from nested or flat validation JSON."""
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload
    if not isinstance(metrics, dict):
        metrics = {}

    def _get(*keys: str) -> float | None:
        for key in keys:
            if key in metrics and metrics[key] is not None:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    continue
            if key in payload and payload[key] is not None:
                try:
                    return float(payload[key])
                except (TypeError, ValueError):
                    continue
        return None

    return {
        "brier_score": _get("brier_score", "brier", "test_brier_score"),
        "log_loss": _get("log_loss", "logloss", "test_log_loss"),
        "accuracy": _get("accuracy", "acc", "test_accuracy"),
        "btts_accuracy": _get("btts_accuracy", "BTTS_accuracy"),
        "over25_accuracy": _get(
            "over25_accuracy",
            "over_under_2_5_accuracy",
            "ou_accuracy",
            "OU_accuracy",
        ),
    }


def _model_file_exists(filename: str) -> tuple[bool, str]:
    """Check exact .joblib path; fall back to extensionless equivalent."""
    exact = MODELS_DIR / filename
    if exact.exists() and exact.stat().st_size > 0:
        return True, str(exact.relative_to(ROOT))

    stem = filename.replace(".joblib", "")
    alt = MODELS_DIR / stem
    if alt.exists() and alt.stat().st_size > 0:
        return False, f"MISSING exact `{filename}`; found equivalent `{stem}`"

    # Tuned variant
    tuned = MODELS_DIR / f"{stem.replace('_model', '')}_tuned_model"
    tuned_joblib = MODELS_DIR / f"{stem}_tuned.joblib"
    for candidate in (tuned, tuned_joblib, MODELS_DIR / f"{stem}.joblib"):
        if candidate.exists() and candidate.stat().st_size > 0:
            return False, (
                f"MISSING exact `{filename}`; found related `{candidate.name}`"
            )

    return False, f"MISSING `{filename}` (no equivalent found)"


def _figure_exists(basename: str) -> tuple[bool, str]:
    """Accept exact name or timestamped variants under reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    exact = FIGURES_DIR / basename
    if exact.exists() and exact.stat().st_size > 0:
        return True, str(exact.relative_to(ROOT))

    stem = Path(basename).stem
    suffix = Path(basename).suffix
    hits = sorted(FIGURES_DIR.glob(f"{stem}*{suffix}"))
    if hits:
        latest = hits[-1]
        return True, f"found timestamped `{latest.relative_to(ROOT)}` (exact `{basename}` missing)"

    # Also accept non-figures location used historically
    root_hits = sorted(REPORTS_DIR.glob(f"{stem}*{suffix}"))
    if root_hits:
        return False, (
            f"MISSING under figures/; found `{root_hits[-1].relative_to(ROOT)}`"
        )

    return False, f"MISSING `reports/figures/{basename}`"


def _normalize_model_name(name: str) -> str:
    s = re.sub(r"[\s_\-]+", "", str(name)).lower()
    aliases = {
        "poisson": "poisson",
        "dixoncoles": "dixon-coles",
        "dixoncole": "dixon-coles",
        "elo": "elo",
        "xgboost": "xgboost",
        "xgb": "xgboost",
        "lightgbm": "lightgbm",
        "lgbm": "lightgbm",
        "randomforest": "randomforest",
        "rf": "randomforest",
        "neuralnetwork": "neuralnetwork",
        "neuralnet": "neuralnetwork",
        "nn": "neuralnetwork",
        "mlp": "neuralnetwork",
    }
    return aliases.get(s, s)


def _required_normalized() -> dict[str, str]:
    return {_normalize_model_name(m): m for m in REQUIRED_LEADERBOARD_MODELS}


def check_model_artifacts() -> list[CheckResult]:
    results: list[CheckResult] = []
    for display, filename, _glob, _key in ML_MODELS:
        ok_exact, evidence = _model_file_exists(filename)
        # Pass only when exact required path exists
        exact_path = MODELS_DIR / filename
        passed = exact_path.exists() and exact_path.stat().st_size > 0
        results.append(
            CheckResult(
                name=f"{display} model trained and saved",
                passed=passed,
                evidence=evidence if not passed else str(exact_path.relative_to(ROOT)),
            )
        )
    return results


def check_model_reports() -> tuple[list[CheckResult], list[ModelMetrics]]:
    checks: list[CheckResult] = []
    metrics_list: list[ModelMetrics] = []

    for display, _filename, pattern, key in ML_MODELS:
        path = _latest_glob(pattern)
        mm = ModelMetrics(model=display)
        if path is None:
            checks.append(
                CheckResult(
                    name=f"{display} validation report generated",
                    passed=False,
                    evidence=f"No files matching reports/{pattern}",
                )
            )
            metrics_list.append(mm)
            continue

        mm.found = True
        mm.report_path = str(path.relative_to(ROOT))
        payload = _load_json(path) or {}
        if "__load_error__" in payload:
            checks.append(
                CheckResult(
                    name=f"{display} validation report generated",
                    passed=False,
                    evidence=f"{mm.report_path} unreadable: {payload['__load_error__']}",
                )
            )
            metrics_list.append(mm)
            continue

        extracted = _extract_metrics(payload)
        mm.brier_score = extracted["brier_score"]
        mm.log_loss = extracted["log_loss"]
        mm.accuracy = extracted["accuracy"]
        mm.btts_accuracy = extracted["btts_accuracy"]
        mm.over25_accuracy = extracted["over25_accuracy"]

        metric_failures: list[str] = []
        if mm.brier_score is None:
            metric_failures.append("brier_score missing")
        elif mm.brier_score >= BRIER_MAX:
            metric_failures.append(f"brier_score {mm.brier_score:.4f} >= {BRIER_MAX}")

        if mm.log_loss is None:
            metric_failures.append("log_loss missing")
        elif mm.log_loss >= LOG_LOSS_MAX:
            metric_failures.append(f"log_loss {mm.log_loss:.4f} >= {LOG_LOSS_MAX}")

        if mm.accuracy is None:
            metric_failures.append("accuracy missing")
        elif mm.accuracy <= ACCURACY_MIN:
            metric_failures.append(f"accuracy {mm.accuracy:.4f} <= {ACCURACY_MIN}")

        if mm.btts_accuracy is None:
            mm.notes.append("btts_accuracy not present (optional)")
        if mm.over25_accuracy is None:
            mm.notes.append("over25_accuracy not present (optional)")

        mm.metrics_ok = len(metric_failures) == 0
        checks.append(
            CheckResult(
                name=f"{display} validation report generated",
                passed=True,
                evidence=mm.report_path or "",
            )
        )
        checks.append(
            CheckResult(
                name=f"{display} metrics meet gates",
                passed=mm.metrics_ok,
                evidence=(
                    f"Brier={mm.brier_score}, LogLoss={mm.log_loss}, Acc={mm.accuracy}"
                    if mm.metrics_ok
                    else "; ".join(metric_failures)
                ),
            )
        )
        metrics_list.append(mm)

    return checks, metrics_list


def check_hyperparameter_tuning() -> list[CheckResult]:
    checks: list[CheckResult] = []
    path = _latest_glob("hyperparameter_tuning_*.json")
    if path is None:
        checks.append(
            CheckResult(
                name="Hyperparameter tuning report generated",
                passed=False,
                evidence="No reports/hyperparameter_tuning_*.json",
            )
        )
        return checks

    checks.append(
        CheckResult(
            name="Hyperparameter tuning report generated",
            passed=True,
            evidence=str(path.relative_to(ROOT)),
        )
    )

    payload = _load_json(path) or {}
    if "__load_error__" in payload:
        checks.append(
            CheckResult(
                name="Hyperparameter tuning report readable",
                passed=False,
                evidence=payload["__load_error__"],
            )
        )
        return checks

    best_params = payload.get("best_params") or {}
    if not isinstance(best_params, dict):
        best_params = {}

    # Also accept nested per-model structures
    if not best_params:
        for key in TUNING_ML_KEYS:
            block = payload.get(key)
            if isinstance(block, dict) and block.get("best_params"):
                best_params[key] = block["best_params"]

    missing_params = [k for k in TUNING_ML_KEYS if k not in best_params or not best_params[k]]
    # Neural network may intentionally be untuned — treat as warning if others present
    hard_missing = [k for k in missing_params if k != "neural_network"]
    soft_missing = [k for k in missing_params if k == "neural_network"]

    checks.append(
        CheckResult(
            name="Best parameters exist for each ML model",
            passed=len(hard_missing) == 0,
            evidence=(
                f"best_params keys: {sorted(best_params.keys())}"
                + (f"; missing: {hard_missing}" if hard_missing else "")
                + (
                    f"; warning missing neural_network params: {soft_missing}"
                    if soft_missing
                    else ""
                )
            ),
        )
    )
    if soft_missing and not hard_missing:
        checks.append(
            CheckResult(
                name="Neural Network tuning params (optional)",
                passed=True,
                evidence="neural_network often skipped by tune_all_models.py — noted",
                severity="warning",
            )
        )

    # Time-series CV verification
    blob = json.dumps(payload).lower()
    ts_markers = (
        "timeseries",
        "time_series",
        "time-series",
        "timeseriessplit",
        "create_time_series_folds",
        "chronological",
    )
    has_ts_field = any(m in blob for m in ts_markers)
    has_n_folds = "n_folds" in payload or any(
        isinstance(v, dict) and "n_folds" in v for v in payload.values()
    )

    # Code-level confirmation that tuner uses time-series folds
    tuner_src = (ROOT / "src" / "hyperparameter_tuning.py").read_text(encoding="utf-8", errors="ignore")
    tune_script = (ROOT / "scripts" / "tune_all_models.py").read_text(encoding="utf-8", errors="ignore")
    code_uses_ts = (
        "TimeSeriesSplit" in tuner_src
        or "create_time_series_folds" in tuner_src
        or "time_series" in tune_script.lower()
    )

    ts_ok = has_ts_field or (has_n_folds and code_uses_ts)
    checks.append(
        CheckResult(
            name="Tuning used time-series cross-validation",
            passed=ts_ok,
            evidence=(
                "report/code confirms time-series CV"
                if ts_ok
                else "No time-series CV markers in report; tuner code check failed"
            ),
        )
    )
    return checks


def _brier_column(fieldnames: list[str]) -> str | None:
    preferred = [
        "1X2_brier_score",
        "brier_score",
        "Brier",
        "brier",
        "Brier_Score",
    ]
    for name in preferred:
        if name in fieldnames:
            return name
    for name in fieldnames:
        if "brier" in name.lower():
            return name
    return None


def _model_column(fieldnames: list[str]) -> str | None:
    for name in ("Model", "model", "model_name", "name"):
        if name in fieldnames:
            return name
    return fieldnames[0] if fieldnames else None


def check_leaderboard() -> list[CheckResult]:
    checks: list[CheckResult] = []
    path = _latest_glob("phase4_leaderboard_*.csv")
    if path is None:
        checks.append(
            CheckResult(
                name="Leaderboard CSV generated",
                passed=False,
                evidence="No reports/phase4_leaderboard_*.csv",
            )
        )
        return checks

    checks.append(
        CheckResult(
            name="Leaderboard CSV generated",
            passed=True,
            evidence=str(path.relative_to(ROOT)),
        )
    )

    try:
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
    except Exception as exc:  # noqa: BLE001
        checks.append(
            CheckResult(
                name="Leaderboard CSV readable",
                passed=False,
                evidence=str(exc),
            )
        )
        return checks

    model_col = _model_column(fieldnames)
    brier_col = _brier_column(fieldnames)
    if not model_col or not rows:
        checks.append(
            CheckResult(
                name="Leaderboard has model rows",
                passed=False,
                evidence=f"columns={fieldnames}",
            )
        )
        return checks

    present_norm = {_normalize_model_name(r.get(model_col, "")) for r in rows}
    required = _required_normalized()
    missing = [label for key, label in required.items() if key not in present_norm]
    checks.append(
        CheckResult(
            name="All 7 models listed on leaderboard",
            passed=len(missing) == 0,
            evidence=(
                f"models={ [r.get(model_col) for r in rows] }"
                if not missing
                else f"missing: {missing}"
            ),
        )
    )

    if brier_col is None:
        checks.append(
            CheckResult(
                name="Brier scores sorted ascending",
                passed=False,
                evidence="No brier column found in leaderboard CSV",
            )
        )
        return checks

    briers: list[float] = []
    parse_ok = True
    for r in rows:
        raw = r.get(brier_col, "")
        try:
            briers.append(float(raw))
        except (TypeError, ValueError):
            parse_ok = False
            break

    if not parse_ok or not briers:
        checks.append(
            CheckResult(
                name="Brier scores sorted ascending",
                passed=False,
                evidence="Could not parse brier values",
            )
        )
        return checks

    # Allow sorted by brier ascending, or verify we can sort equivalently
    is_sorted = all(briers[i] <= briers[i + 1] + 1e-12 for i in range(len(briers) - 1))
    checks.append(
        CheckResult(
            name="Brier scores sorted ascending",
            passed=is_sorted,
            evidence=(
                f"briers={briers}"
                if is_sorted
                else f"not ascending: {briers} (column `{brier_col}`)"
            ),
        )
    )

    best_brier = min(briers)
    checks.append(
        CheckResult(
            name=f"Best model Brier Score < {LEADERBOARD_BEST_BRIER_MAX}",
            passed=best_brier < LEADERBOARD_BEST_BRIER_MAX,
            evidence=f"best_brier={best_brier:.4f}",
        )
    )
    return checks


def check_comparison_artifacts() -> list[CheckResult]:
    checks: list[CheckResult] = []

    cmp_path = _latest_glob("phase3_vs_phase4_*.json")
    checks.append(
        CheckResult(
            name="Phase 3 vs Phase 4 comparison JSON generated",
            passed=cmp_path is not None,
            evidence=(
                str(cmp_path.relative_to(ROOT))
                if cmp_path
                else "No reports/phase3_vs_phase4_*.json"
            ),
        )
    )

    brier_ok, brier_ev = _figure_exists("brier_comparison.png")
    acc_ok, acc_ev = _figure_exists("accuracy_comparison.png")
    # Exact basename preferred; timestamped OK for "visualizations generated"
    # but require both present somehow
    checks.append(
        CheckResult(
            name="brier_comparison.png generated",
            passed=brier_ok and "MISSING under figures" not in brier_ev,
            evidence=brier_ev,
        )
    )
    checks.append(
        CheckResult(
            name="accuracy_comparison.png generated",
            passed=acc_ok and "MISSING under figures" not in acc_ev,
            evidence=acc_ev,
        )
    )
    checks.append(
        CheckResult(
            name="Visualizations generated (brier_comparison.png, accuracy_comparison.png)",
            passed=checks[-2].passed and checks[-1].passed,
            evidence=f"brier: {brier_ev}; accuracy: {acc_ev}",
        )
    )
    return checks


def run_checklist() -> dict[str, Any]:
    model_training = check_model_artifacts()
    model_reports, metrics_list = check_model_reports()
    tuning = check_hyperparameter_tuning()
    leaderboard = check_leaderboard()
    comparison = check_comparison_artifacts()

    sections = {
        "Model Training": model_training,
        "Model Reports": [c for c in model_reports if "metrics meet" not in c.name],
        "Metric Gates": [c for c in model_reports if "metrics meet" in c.name],
        "Hyperparameter Tuning": tuning,
        "Model Comparison": [
            c
            for c in leaderboard + comparison
            if c.name
            in {
                "Leaderboard CSV generated",
                "Phase 3 vs Phase 4 comparison JSON generated",
                "Visualizations generated (brier_comparison.png, accuracy_comparison.png)",
            }
            or c.name.startswith("All 7")
            or c.name.startswith("Brier scores")
            or c.name.startswith("Best model")
        ],
    }

    all_checks = model_training + model_reports + tuning + leaderboard + comparison
    required = [c for c in all_checks if c.severity == "required"]
    passed_required = sum(1 for c in required if c.passed)
    blockers = [
        {"name": c.name, "evidence": c.evidence}
        for c in required
        if not c.passed
    ]

    # Section completion for summary header
    def _ratio(items: list[CheckResult]) -> tuple[int, int]:
        return sum(1 for i in items if i.passed), len(items)

    training_pass, training_total = _ratio(model_training)
    # Report existence only (4 items) — exclude metric gate checks from 4/4 count
    report_existence = [
        c for c in model_reports if c.name.endswith("validation report generated")
    ]
    reports_pass, reports_total = _ratio(report_existence)
    tuning_exist = [c for c in tuning if c.name == "Hyperparameter tuning report generated"]
    tuning_pass, tuning_total = _ratio(tuning_exist)
    comparison_core = [
        c
        for c in comparison + leaderboard
        if c.name
        in {
            "Leaderboard CSV generated",
            "Phase 3 vs Phase 4 comparison JSON generated",
            "Visualizations generated (brier_comparison.png, accuracy_comparison.png)",
        }
    ]
    cmp_pass, cmp_total = _ratio(comparison_core)

    if not blockers:
        status = "COMPLETE"
    elif training_pass + reports_pass + tuning_pass + cmp_pass > 0:
        status = "PARTIAL"
    else:
        status = "INCOMPLETE"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "sections": {k: [c.__dict__ for c in v] for k, v in sections.items()},
        "model_metrics": [m.__dict__ for m in metrics_list],
        "counts": {
            "model_training": {"passed": training_pass, "total": training_total},
            "model_reports": {"passed": reports_pass, "total": reports_total},
            "hyperparameter_tuning": {"passed": tuning_pass, "total": max(tuning_total, 1)},
            "model_comparison": {"passed": cmp_pass, "total": cmp_total},
            "required_passed": passed_required,
            "required_total": len(required),
        },
        "blockers": blockers,
        "all_checks": [c.__dict__ for c in all_checks],
    }


def _checkbox(passed: bool) -> str:
    return "[x]" if passed else "[ ]"


def _status_emoji(status: str) -> str:
    return {
        "COMPLETE": "COMPLETE",
        "PARTIAL": "PARTIAL",
        "INCOMPLETE": "INCOMPLETE",
    }.get(status, status)


def _fmt_metric(value: float | None, fmt: str = ".4f") -> str:
    if value is None:
        return "N/A"
    return format(value, fmt)


def generate_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    metrics = report["model_metrics"]

    def find_check(section: str, substring: str) -> bool:
        for item in report["sections"].get(section, []):
            if substring in item["name"]:
                return bool(item["passed"])
        # search all
        for item in report["all_checks"]:
            if substring in item["name"]:
                return bool(item["passed"])
        return False

    mt = counts["model_training"]
    mr = counts["model_reports"]
    ht = counts["hyperparameter_tuning"]
    mc = counts["model_comparison"]

    lines: list[str] = []
    lines.append("# Phase 4 Completion Checklist")
    lines.append("")
    lines.append(f"**Generated:** {report['timestamp']}")
    lines.append(f"**Status:** {_status_emoji(report['status'])}")
    lines.append("")
    lines.append(
        f"## Model Training ({mt['passed']}/{mt['total']})"
    )
    lines.append(
        f"- {_checkbox(find_check('Model Training', 'XGBoost model'))} "
        "XGBoost model trained and saved"
    )
    lines.append(
        f"- {_checkbox(find_check('Model Training', 'LightGBM model'))} "
        "LightGBM model trained and saved"
    )
    lines.append(
        f"- {_checkbox(find_check('Model Training', 'Random Forest model'))} "
        "Random Forest model trained and saved"
    )
    lines.append(
        f"- {_checkbox(find_check('Model Training', 'Neural Network model'))} "
        "Neural Network model trained and saved"
    )
    lines.append("")
    lines.append(f"## Model Reports ({mr['passed']}/{mr['total']})")
    lines.append(
        f"- {_checkbox(find_check('Model Reports', 'XGBoost validation'))} "
        "XGBoost validation report generated"
    )
    lines.append(
        f"- {_checkbox(find_check('Model Reports', 'LightGBM validation'))} "
        "LightGBM validation report generated"
    )
    lines.append(
        f"- {_checkbox(find_check('Model Reports', 'Random Forest validation'))} "
        "Random Forest validation report generated"
    )
    lines.append(
        f"- {_checkbox(find_check('Model Reports', 'Neural Network validation'))} "
        "Neural Network validation report generated"
    )
    lines.append("")
    lines.append(f"## Hyperparameter Tuning ({ht['passed']}/{ht['total']})")
    lines.append(
        f"- {_checkbox(find_check('Hyperparameter Tuning', 'Hyperparameter tuning report'))} "
        "Hyperparameter tuning report generated"
    )
    lines.append("")
    lines.append(f"## Model Comparison ({mc['passed']}/{mc['total']})")
    lines.append(
        f"- {_checkbox(find_check('Model Comparison', 'Leaderboard CSV'))} "
        "Leaderboard CSV generated"
    )
    lines.append(
        f"- {_checkbox(find_check('Model Comparison', 'Phase 3 vs Phase 4'))} "
        "Phase 3 vs Phase 4 comparison JSON generated"
    )
    lines.append(
        f"- {_checkbox(find_check('Model Comparison', 'Visualizations generated'))} "
        "Visualizations generated (brier_comparison.png, accuracy_comparison.png)"
    )
    lines.append("")
    lines.append("## Performance Metrics")
    lines.append("| Model | Brier Score | Log Loss | Accuracy | Status |")
    lines.append("|-------|-------------|----------|----------|--------|")
    for m in metrics:
        gate_ok = bool(m.get("metrics_ok"))
        status = "PASS" if gate_ok else "FAIL"
        if not m.get("found"):
            status = "MISSING"
        lines.append(
            f"| {m['model']} | {_fmt_metric(m.get('brier_score'))} | "
            f"{_fmt_metric(m.get('log_loss'))} | {_fmt_metric(m.get('accuracy'))} | "
            f"{status} |"
        )
        extras = []
        if m.get("btts_accuracy") is not None:
            extras.append(f"btts_accuracy={m['btts_accuracy']:.4f}")
        if m.get("over25_accuracy") is not None:
            extras.append(f"over25_accuracy={m['over25_accuracy']:.4f}")
        if extras:
            lines.append(f"|  ↳ extras | {' · '.join(extras)} | | | |")

    lines.append("")
    lines.append("## Detailed Checks")
    lines.append("")
    for section, items in report["sections"].items():
        lines.append(f"### {section}")
        for item in items:
            mark = "PASS" if item["passed"] else "FAIL"
            lines.append(f"- [{mark}] {item['name']} — {item.get('evidence', '')}")
        lines.append("")

    lines.append("## Blockers")
    blockers = report.get("blockers") or []
    if not blockers:
        lines.append("None — all required checks passed.")
    else:
        for i, b in enumerate(blockers, 1):
            lines.append(f"{i}. **{b['name']}** — {b['evidence']}")
    lines.append("")
    lines.append("## Next Steps")
    if report["status"] == "COMPLETE":
        lines.append("1. Archive this checklist as Phase 4 sign-off.")
        lines.append("2. Proceed to Phase 5 (if applicable).")
    else:
        steps: list[str] = []
        blob = " ".join(b["name"] + " " + b["evidence"] for b in blockers).lower()
        if "model trained" in blob or "joblib" in blob:
            steps.append(
                "Train/save models with exact names: "
                "`python scripts/train_xgboost.py`, `train_lightgbm.py`, "
                "`train_random_forest.py`, `train_neural_network.py` "
                "(ensure `.joblib` suffix)."
            )
        if "validation report" in blob:
            steps.append(
                "Re-run training scripts so `reports/*_validation_*.json` are written."
            )
        if "hyperparameter" in blob:
            steps.append("Run `python scripts/tune_all_models.py` to produce tuning JSON.")
        if "leaderboard" in blob or "phase 3 vs" in blob or "visualizations" in blob:
            steps.append(
                "Run `python scripts/compare_phase4_vs_phase3.py` to generate "
                "leaderboard, comparison JSON, and figures."
            )
        if "brier" in blob or "metrics meet" in blob or "accuracy" in blob:
            steps.append(
                "Improve ML models until Brier < 0.50, Log Loss < 1.20, Accuracy > 0.55 "
                "(and best leaderboard Brier < 0.45)."
            )
        if not steps:
            steps.append("Address each blocker listed above, then re-run this checklist.")
        steps.append("Re-run: `python scripts/phase4_checklist.py`")
        for i, s in enumerate(steps, 1):
            lines.append(f"{i}. {s}")

    lines.append("")
    lines.append("---")
    lines.append(
        f"_Generated by `scripts/phase4_checklist.py` at {report['timestamp']}_"
    )
    lines.append("")
    return "\n".join(lines)


def generate_console(report: dict[str, Any]) -> str:
    lines = [
        "",
        "=" * 60,
        "  PHASE 4 COMPLETION CHECKLIST",
        f"  Status: {report['status']}",
        f"  Required: {report['counts']['required_passed']}/"
        f"{report['counts']['required_total']} passed",
        "=" * 60,
    ]
    for section, items in report["sections"].items():
        lines.append(f"\n  [{section}]")
        for item in items:
            mark = "OK" if item["passed"] else "X "
            lines.append(f"    [{mark}] {item['name']}")
    if report["blockers"]:
        lines.append("\n  Blockers:")
        for b in report["blockers"][:15]:
            lines.append(f"    - {b['name']}: {b['evidence'][:100]}")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 Completion Checklist — verify ML model deliverables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Markdown report path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print JSON report to stdout (still writes markdown unless --no-write)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write report files to disk",
    )
    parser.add_argument(
        "--missing",
        action="store_true",
        help="Print only failed/missing checks",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Alias: run checks against existing artifacts only (default behaviour)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_checklist()

    if args.missing:
        print("Missing / failed checks:")
        for b in report["blockers"]:
            print(f"  - {b['name']}: {b['evidence']}")
        if not report["blockers"]:
            print("  (none)")
    else:
        print(generate_console(report))

    if not args.no_write:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        md = generate_markdown(report)
        output_path.write_text(md, encoding="utf-8")
        print(f"  Report saved: {output_path}")

        json_path = output_path.with_suffix(".json")
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"  JSON saved:   {json_path}")

    if args.json_only:
        print(json.dumps(report, indent=2, default=str))

    print(f"  Overall: {report['status']}")
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
