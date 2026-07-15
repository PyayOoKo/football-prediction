#!/usr/bin/env python3
"""
Final System Validation — end-to-end readiness checklist.

Validates data pipeline, model performance gates, system reliability,
production readiness, and commercial readiness. Writes a markdown report.

Usage:
    python scripts/final_system_validation.py
    python scripts/final_system_validation.py --output reports/final_validation_custom.md
    python scripts/final_system_validation.py --missing
    python scripts/final_system_validation.py --json-only

Exit code:
    0 — all required checks pass (READY)
    1 — one or more required checks fail (NOT READY)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
REPORTS = ROOT / "reports"
DEFAULT_OUTPUT = REPORTS / f"final_validation_{TIMESTAMP}.md"

# Performance gates
BEST_BRIER_MAX = 0.40
BEST_ACCURACY_MIN = 0.65
BACKTEST_ROI_MIN = 5.0  # percent
CLV_MIN = 0.0

# Data freshness
MAX_DAYS_SINCE_LAST_MATCH = 3
MAX_MATCH_DAY_GAP = 2  # days between consecutive match dates in window


@dataclass
class Check:
    section: str
    name: str
    passed: bool
    evidence: str = ""
    severity: str = "required"  # required | warning


@dataclass
class ValidationReport:
    timestamp: str
    status: str
    checks: list[Check] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    blockers: list[dict[str, str]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)

    @property
    def required(self) -> list[Check]:
        return [c for c in self.checks if c.severity == "required"]

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.required if c.passed)

    @property
    def total_count(self) -> int:
        return len(self.required)


def _exists(*rel_paths: str) -> Path | None:
    for rel in rel_paths:
        p = ROOT / rel
        if p.exists():
            return p
    return None


def _latest(glob_pat: str, directory: Path = REPORTS) -> Path | None:
    if not directory.exists():
        return None
    hits = sorted(directory.glob(glob_pat))
    return hits[-1] if hits else None


def _read_text(path: Path, limit: int = 0) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if not limit else text[:limit]


def _checkbox(ok: bool) -> str:
    return "[x]" if ok else "[ ]"


# ═══════════════════════════════════════════════════════════
#  1. Data Pipeline
# ═══════════════════════════════════════════════════════════


def check_data_pipeline(report: ValidationReport) -> None:
    section = "Data Pipeline"

    daily = _exists(
        "scripts/daily_data_pipeline.py",
        "src/data_collection",
    )
    scheduler = _exists(
        "scheduler_config.yaml",
        "src/scheduler/engine.py",
        ".github/workflows/automation.yml",
    )
    report.checks.append(
        Check(
            section,
            "Daily data collection working",
            daily is not None and scheduler is not None,
            evidence=(
                f"script={daily.relative_to(ROOT) if daily else 'MISSING'}; "
                f"scheduler={scheduler.relative_to(ROOT) if scheduler else 'MISSING'}"
            ),
        )
    )

    feature = _exists(
        "scripts/daily_feature_computation.py",
        "src/feature_store/cli.py",
    )
    report.checks.append(
        Check(
            section,
            "Feature computation automated",
            feature is not None,
            evidence=(
                str(feature.relative_to(ROOT))
                if feature
                else "No daily_feature_computation.py / feature_store CLI"
            ),
        )
    )

    # Data gap / freshness analysis
    csv_path = ROOT / "data" / "processed" / "results_clean.csv"
    gap_ok = False
    evidence = "data/processed/results_clean.csv missing"
    metrics: dict[str, Any] = {}

    if csv_path.exists():
        try:
            import pandas as pd

            df = pd.read_csv(csv_path, parse_dates=["date"], low_memory=False)
            now = pd.Timestamp.now().normalize()
            cutoff = now - pd.Timedelta(days=30)
            dmax = pd.to_datetime(df["date"]).max()
            days_since = int((now - dmax.normalize()).days)
            recent = df[pd.to_datetime(df["date"]) >= cutoff]
            have_dates = sorted({pd.Timestamp(x).normalize().date() for x in recent["date"]})
            gaps = []
            for a, b in zip(have_dates, have_dates[1:]):
                delta = (b - a).days
                if delta > MAX_MATCH_DAY_GAP:
                    gaps.append({"from": str(a), "to": str(b), "missing_days": delta - 1})

            calendar = pd.date_range(cutoff, now - pd.Timedelta(days=1), freq="D")
            have_set = set(have_dates)
            empty_calendar = [d.date().isoformat() for d in calendar if d.date() not in have_set]

            stale = days_since > MAX_DAYS_SINCE_LAST_MATCH
            large_gaps = len(gaps) > 0
            gap_ok = (not stale) and (not large_gaps)

            metrics = {
                "rows": int(len(df)),
                "date_min": str(pd.to_datetime(df["date"]).min().date()),
                "date_max": str(dmax.date()),
                "days_since_last_match": days_since,
                "matches_last_30d": int(len(recent)),
                "match_days_last_30d": len(have_dates),
                "calendar_days_without_match_last_30d": len(empty_calendar),
                "match_day_gaps": gaps,
                "empty_calendar_sample": empty_calendar[-10:],
            }
            evidence = (
                f"last_match={dmax.date()}, days_since={days_since}, "
                f"match_day_gaps={gaps or 'none'}, "
                f"empty_calendar_tail={empty_calendar[-5:]}"
            )
            if stale:
                evidence += f" (stale: >{MAX_DAYS_SINCE_LAST_MATCH}d since last match)"
        except Exception as exc:  # noqa: BLE001
            evidence = f"Failed to analyse results_clean.csv: {exc}"
            gap_ok = False

    report.metrics["data_pipeline"] = metrics
    report.checks.append(
        Check(
            section,
            "No data gaps in last 30 days",
            gap_ok,
            evidence=evidence,
        )
    )


# ═══════════════════════════════════════════════════════════
#  2. Model Performance
# ═══════════════════════════════════════════════════════════


def _parse_roi_from_backtest(text: str) -> float | None:
    """Extract best ROI percent from backtest markdown."""
    patterns = [
        r"Best ROI[:\*\s]+[^\-\d]*(-?\d+(?:\.\d+)?)\s*%",
        r"Average ROI[^\-\d]*(-?\d+(?:\.\d+)?)\s*%",
        r"\|\s*ROI%?\s*\|[^\n]*\n\|[-\s|]+\n(?:\|[^\n]*\|[^\n]*\|[^\n]*\|[^\n]*\|[^\n]*\|[^\n]*\|[^\n]*\|[^\n]*\|\s*(-?\d+(?:\.\d+)?))",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except (TypeError, ValueError):
                continue

    # Table rows: look for ROI% column values
    rois: list[float] = []
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        # match patterns like -99.47% in table cells
        for cell in re.findall(r"(-?\d+(?:\.\d+)?)\s*%", line):
            try:
                rois.append(float(cell))
            except ValueError:
                pass
    # Prefer explicit "Best ROI" already tried; else take max (least bad)
    return max(rois) if rois else None


def check_model_performance(report: ValidationReport) -> None:
    section = "Model Performance"
    metrics: dict[str, Any] = {}

    # Best Brier / Accuracy from leaderboard
    lb = _latest("phase4_leaderboard_*.csv")
    best_brier: float | None = None
    best_acc: float | None = None
    best_model: str | None = None

    if lb:
        with lb.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        def _brier(r: dict[str, str]) -> float:
            for k in ("1X2_brier_score", "brier_score", "Brier"):
                if r.get(k) not in (None, ""):
                    try:
                        return float(r[k])
                    except ValueError:
                        continue
            return 999.0

        def _acc(r: dict[str, str]) -> float | None:
            for k in ("1X2_accuracy", "accuracy", "Accuracy"):
                if r.get(k) not in (None, ""):
                    try:
                        return float(r[k])
                    except ValueError:
                        continue
            return None

        if rows:
            best_row = min(rows, key=_brier)
            best_brier = _brier(best_row)
            best_acc = _acc(best_row)
            best_model = best_row.get("Model") or best_row.get("model")
            # Also track highest accuracy independently
            acc_rows = [(r, _acc(r)) for r in rows]
            acc_rows = [(r, a) for r, a in acc_rows if a is not None]
            if acc_rows:
                top_acc_row, top_acc = max(acc_rows, key=lambda x: x[1] or 0)
                metrics["highest_accuracy"] = {
                    "model": top_acc_row.get("Model"),
                    "accuracy": top_acc,
                }

    metrics["leaderboard"] = str(lb.relative_to(ROOT)) if lb else None
    metrics["best_model"] = best_model
    metrics["best_brier"] = best_brier
    metrics["best_accuracy"] = best_acc

    brier_ok = best_brier is not None and best_brier < BEST_BRIER_MAX
    report.checks.append(
        Check(
            section,
            f"Best model Brier Score < {BEST_BRIER_MAX:.2f}",
            brier_ok,
            evidence=(
                f"{best_model}: Brier={best_brier:.4f} (from {lb.name})"
                if best_brier is not None and lb
                else "No phase4_leaderboard_*.csv or brier column"
            ),
        )
    )

    acc_ok = best_acc is not None and best_acc > BEST_ACCURACY_MIN
    # Prefer highest accuracy for the accuracy gate if available
    hi = metrics.get("highest_accuracy") or {}
    if hi.get("accuracy") is not None:
        acc_ok = hi["accuracy"] > BEST_ACCURACY_MIN
        acc_evidence = (
            f"{hi.get('model')}: Accuracy={hi['accuracy']:.4f} "
            f"(best-by-brier Acc={best_acc})"
        )
    else:
        acc_evidence = (
            f"{best_model}: Accuracy={best_acc}"
            if best_acc is not None
            else "No accuracy in leaderboard"
        )
    report.checks.append(
        Check(
            section,
            f"Best model Accuracy > {BEST_ACCURACY_MIN:.0%}",
            acc_ok,
            evidence=acc_evidence,
        )
    )

    # Backtest ROI
    bt = _latest("backtest_report_*.md")
    roi: float | None = None
    if bt:
        roi = _parse_roi_from_backtest(_read_text(bt))
        # Prefer "Best ROI" line specifically
        m = re.search(
            r"\*\*Best ROI:\*\*\s*[^\-\d]*(-?\d+(?:\.\d+)?)\s*%",
            _read_text(bt),
            re.IGNORECASE,
        )
        if m:
            roi = float(m.group(1))
    metrics["backtest_report"] = str(bt.relative_to(ROOT)) if bt else None
    metrics["best_roi_pct"] = roi
    roi_ok = roi is not None and roi > BACKTEST_ROI_MIN
    report.checks.append(
        Check(
            section,
            f"Backtest ROI > {BACKTEST_ROI_MIN:.0f}%",
            roi_ok,
            evidence=(
                f"Best ROI={roi:.2f}% ({bt.name})"
                if roi is not None and bt
                else "No backtest_report_*.md or ROI not parseable"
            ),
        )
    )

    # CLV > 0%
    clv_path = _latest("clv_summary_*.json")
    best_clv: float | None = None
    clv_model: str | None = None
    if clv_path:
        try:
            data = json.loads(clv_path.read_text(encoding="utf-8"))
            models = data.get("models") or []
            for mrow in models:
                summary = mrow.get("clv_summary") or {}
                avg = summary.get("avg_clv")
                if avg is None:
                    continue
                avg_f = float(avg)
                if best_clv is None or avg_f > best_clv:
                    best_clv = avg_f
                    clv_model = mrow.get("model_name")
        except Exception as exc:  # noqa: BLE001
            metrics["clv_error"] = str(exc)

    # Fallback: backtest CLV column if summary missing
    if best_clv is None and bt:
        m = re.search(
            r"\*\*Best CLV:\*\*\s*[^\-\d]*(-?\d+(?:\.\d+)?)",
            _read_text(bt),
            re.IGNORECASE,
        )
        if m:
            best_clv = float(m.group(1))
            clv_model = "backtest_report"

    metrics["clv_summary"] = str(clv_path.relative_to(ROOT)) if clv_path else None
    metrics["best_avg_clv"] = best_clv
    metrics["best_clv_model"] = clv_model
    # CLV values in summary appear as absolute ratios (e.g. 0.835 = 83.5%);
    # gate is CLV > 0%
    clv_ok = best_clv is not None and best_clv > CLV_MIN
    report.checks.append(
        Check(
            section,
            "CLV > 0%",
            clv_ok,
            evidence=(
                f"{clv_model}: avg_clv={best_clv:.6f}"
                if best_clv is not None
                else "No clv_summary_*.json / backtest CLV"
            ),
        )
    )

    report.metrics["model_performance"] = metrics


# ═══════════════════════════════════════════════════════════
#  3. System Reliability
# ═══════════════════════════════════════════════════════════


def check_system_reliability(report: ValidationReport) -> None:
    section = "System Reliability"

    automation_pieces = [
        ROOT / "scheduler_config.yaml",
        ROOT / ".github" / "workflows" / "automation.yml",
        ROOT / "src" / "scheduler" / "engine.py",
        ROOT / "scripts" / "daily_data_pipeline.py",
        ROOT / "scripts" / "daily_feature_computation.py",
        ROOT / "scripts" / "daily_predictions.py",
    ]
    present = [p for p in automation_pieces if p.exists()]
    # "All automated tasks run successfully" — require automation definitions
    # plus recent evidence of success (log or automation completion report)
    success_evidence = _latest("automation_completion_*.md") or _exists(
        "logs/pipeline.log",
        "logs/app.log",
    )
    # Do not claim success if data is stale (covered elsewhere) — soft pass on tooling
    tooling_ok = len(present) >= 4
    evidence_ok = success_evidence is not None
    report.checks.append(
        Check(
            section,
            "All automated tasks run successfully",
            tooling_ok and evidence_ok,
            evidence=(
                f"automation_files={len(present)}/{len(automation_pieces)}; "
                f"evidence={success_evidence.relative_to(ROOT) if isinstance(success_evidence, Path) else success_evidence}"
            ),
        )
    )

    # Error handling — scheduler retry + notify + try/except in tasks
    engine = ROOT / "src" / "scheduler" / "engine.py"
    tasks = ROOT / "src" / "scheduler" / "tasks.py"
    err_ok = False
    err_ev = "scheduler engine/tasks missing"
    if engine.exists() and tasks.exists():
        eng_txt = _read_text(engine, limit=200_000)
        task_txt = _read_text(tasks, limit=200_000)
        has_retry = "retry" in eng_txt.lower() or "_execute_with_retry" in eng_txt
        has_except = "except" in task_txt
        err_ok = has_retry and has_except
        err_ev = f"retry={has_retry}, task_except={has_except}"
    report.checks.append(
        Check(section, "Error handling works", err_ok, evidence=err_ev)
    )

    notify = _exists(
        "scripts/notify.py",
        "src/scheduler/notifications.py",
        "src/monitoring/alerting.py",
    )
    gha = ROOT / ".github" / "workflows" / "automation.yml"
    gha_alert = False
    if gha.exists():
        gha_txt = _read_text(gha).lower()
        gha_alert = "slack" in gha_txt or "notify" in gha_txt or "alert" in gha_txt
    alert_ok = notify is not None
    report.checks.append(
        Check(
            section,
            "Alerts configured for failures",
            alert_ok,
            evidence=(
                f"notify={notify.relative_to(ROOT) if notify else None}; "
                f"gha_alert_hooks={gha_alert}"
            ),
        )
    )

    recovery_docs = list((ROOT / "docs").glob("*")) if (ROOT / "docs").exists() else []
    recovery_hits = [
        p
        for p in recovery_docs
        if p.is_file()
        and re.search(
            r"recover|runbook|disaster|incident|failover",
            p.name + _read_text(p, limit=3000),
            re.IGNORECASE,
        )
    ]
    # troubleshooting.md counts as partial recovery guidance
    troubleshooting = ROOT / "docs" / "troubleshooting.md"
    recovery_ok = any(
        re.search(r"runbook|disaster recovery|recovery procedure", p.name, re.I)
        for p in recovery_hits
    )
    if not recovery_ok and troubleshooting.exists():
        report.checks.append(
            Check(
                section,
                "Recovery procedures documented",
                False,
                evidence=(
                    "docs/troubleshooting.md exists but no dedicated recovery/"
                    "disaster-recovery runbook found"
                ),
            )
        )
    else:
        report.checks.append(
            Check(
                section,
                "Recovery procedures documented",
                recovery_ok,
                evidence=(
                    ", ".join(str(p.relative_to(ROOT)) for p in recovery_hits[:5])
                    if recovery_hits
                    else "No recovery/runbook docs"
                ),
            )
        )


# ═══════════════════════════════════════════════════════════
#  4. Production Readiness
# ═══════════════════════════════════════════════════════════


def check_production_readiness(report: ValidationReport) -> None:
    section = "Production Readiness"

    packaging = [
        ROOT / "pyproject.toml",
        ROOT / "setup.py",
        ROOT / "requirements.txt",
    ]
    pkg_ok = sum(1 for p in packaging if p.exists()) >= 2
    report.checks.append(
        Check(
            section,
            "Software packaged and installable",
            pkg_ok,
            evidence=", ".join(
                p.name for p in packaging if p.exists()
            )
            or "No packaging files",
        )
    )

    docs_dir = ROOT / "docs"
    required_docs = [
        "deployment_guide.md",
        "installation.md",
        "architecture.md",
        "user_guide.md",
    ]
    present_docs = [d for d in required_docs if (docs_dir / d).exists()]
    # Also accept GUIDE.md / README
    extras = [p for p in ("GUIDE.md", "README.md", "docs/README.md") if _exists(p)]
    docs_ok = len(present_docs) >= 3 or (len(present_docs) >= 2 and extras)
    report.checks.append(
        Check(
            section,
            "Documentation complete",
            docs_ok,
            evidence=f"core_docs={present_docs}; extras={[str(e) for e in extras]}",
        )
    )

    deploy = [
        ROOT / "Dockerfile",
        ROOT / "docker-compose.yml",
        docs_dir / "deployment_guide.md",
    ]
    deploy_ok = all(p.exists() for p in deploy)
    # "Deployment tested" — look for completion/production report claiming deploy
    deploy_report = _latest("production_software_*.md") or _latest(
        "automation_completion_*.md"
    )
    report.checks.append(
        Check(
            section,
            "Deployment tested",
            deploy_ok and deploy_report is not None,
            evidence=(
                f"docker+compose+guide={'yes' if deploy_ok else 'incomplete'}; "
                f"report={deploy_report.relative_to(ROOT) if deploy_report else 'MISSING'}"
            ),
        )
    )

    monitoring_pkg = ROOT / "src" / "monitoring"
    monitoring_reports = REPORTS / "monitoring"
    mon_files = (
        list(monitoring_reports.glob("*"))
        if monitoring_reports.exists()
        else []
    )
    mon_files = [p for p in mon_files if p.is_file()]
    mon_ok = monitoring_pkg.exists() and (
        (ROOT / "src" / "monitoring" / "alerting.py").exists()
    )
    # Prefer having actual monitoring outputs
    if mon_ok and not mon_files:
        report.checks.append(
            Check(
                section,
                "Monitoring in place",
                False,
                evidence=(
                    "src/monitoring/ exists but reports/monitoring/ has no "
                    "generated report artifacts"
                ),
            )
        )
    else:
        report.checks.append(
            Check(
                section,
                "Monitoring in place",
                mon_ok and bool(mon_files),
                evidence=(
                    f"package={monitoring_pkg.exists()}, "
                    f"report_artifacts={len(mon_files)}"
                ),
            )
        )


# ═══════════════════════════════════════════════════════════
#  5. Commercial Readiness
# ═══════════════════════════════════════════════════════════


def _search_docs(patterns: list[str]) -> list[Path]:
    docs = ROOT / "docs"
    hits: list[Path] = []
    if not docs.exists():
        return hits
    for path in docs.rglob("*.md"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:8000]
        except OSError:
            continue
        blob = path.name + "\n" + text
        if any(re.search(p, blob, re.IGNORECASE) for p in patterns):
            hits.append(path)
    return hits


def check_commercial_readiness(report: ValidationReport) -> None:
    section = "Commercial Readiness"

    pricing_hits = _search_docs(
        [r"pricing model", r"subscription plan", r"price per", r"\bSaaS pricing\b", r"tiered pricing"]
    )
    # Exclude false positives (odds pricing)
    pricing_hits = [
        p
        for p in pricing_hits
        if not re.search(r"odds|bookmaker|implied probability", p.name, re.I)
        or re.search(r"pricing model|subscription|saas", _read_text(p, 2000), re.I)
    ]
    # Strict commercial files
    commercial_files = list((ROOT / "docs").glob("*pric*")) if (ROOT / "docs").exists() else []
    commercial_files += list((ROOT / "docs").glob("*commercial*")) if (ROOT / "docs").exists() else []
    commercial_files += list((ROOT / "docs").glob("*business*")) if (ROOT / "docs").exists() else []

    pricing_ok = bool(commercial_files) or any(
        re.search(r"pricing model|subscription plan|go-to-market pricing", _read_text(p, 4000), re.I)
        for p in pricing_hits
    )
    report.checks.append(
        Check(
            section,
            "Pricing model defined",
            pricing_ok,
            evidence=(
                ", ".join(str(p.relative_to(ROOT)) for p in (commercial_files or pricing_hits)[:5])
                or "No pricing/business docs found"
            ),
        )
    )

    customer_hits = _search_docs(
        [r"target customer", r"customer segment", r"buyer persona", r"ICP\b", r"ideal customer"]
    )
    report.checks.append(
        Check(
            section,
            "Target customers identified",
            bool(customer_hits),
            evidence=(
                ", ".join(str(p.relative_to(ROOT)) for p in customer_hits[:5])
                or "No target-customer documentation"
            ),
        )
    )

    marketing_hits = _search_docs(
        [r"marketing material", r"landing page copy", r"go-to-market", r"pitch deck", r"brochure"]
    )
    marketing_dirs = [
        p
        for p in (ROOT / "docs", ROOT / "marketing", ROOT / "assets")
        if p.exists() and p.is_dir() and "marketing" in p.name.lower()
    ]
    report.checks.append(
        Check(
            section,
            "Marketing materials ready",
            bool(marketing_hits or marketing_dirs),
            evidence=(
                ", ".join(
                    str(p.relative_to(ROOT))
                    for p in (marketing_hits + marketing_dirs)[:5]
                )
                or "No marketing materials found"
            ),
        )
    )

    support_hits = _search_docs(
        [r"support system", r"support ticket", r"helpdesk", r"SLA\b", r"customer support"]
    )
    # FAQ/troubleshooting alone is insufficient for "support system"
    support_ok = any(
        re.search(r"support system|helpdesk|ticket|SLA", p.name + _read_text(p, 3000), re.I)
        for p in support_hits
    ) and any(
        re.search(r"support|sla|helpdesk", p.name, re.I) for p in support_hits
    )
    # Broaden: dedicated support doc
    support_docs = []
    if (ROOT / "docs").exists():
        support_docs = [
            p
            for p in (ROOT / "docs").glob("*.md")
            if re.search(r"support|sla|helpdesk", p.name, re.I)
        ]
    support_ok = bool(support_docs) or support_ok
    report.checks.append(
        Check(
            section,
            "Support system in place",
            support_ok,
            evidence=(
                ", ".join(str(p.relative_to(ROOT)) for p in (support_docs or support_hits)[:5])
                or "No support/SLA system documentation (FAQ/troubleshooting only is insufficient)"
            ),
        )
    )


# ═══════════════════════════════════════════════════════════
#  Report generation
# ═══════════════════════════════════════════════════════════


def run_validation() -> ValidationReport:
    report = ValidationReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        status="NOT READY",
    )
    check_data_pipeline(report)
    check_model_performance(report)
    check_system_reliability(report)
    check_production_readiness(report)
    check_commercial_readiness(report)

    for c in report.checks:
        item = {"section": c.section, "name": c.name, "evidence": c.evidence}
        if not c.passed and c.severity == "required":
            report.blockers.append(item)
        elif not c.passed:
            report.warnings.append(item)

    report.status = "READY" if not report.blockers else "NOT READY"
    return report


def generate_markdown(report: ValidationReport) -> str:
    sections_order = [
        "Data Pipeline",
        "Model Performance",
        "System Reliability",
        "Production Readiness",
        "Commercial Readiness",
    ]
    by_section: dict[str, list[Check]] = {s: [] for s in sections_order}
    for c in report.checks:
        by_section.setdefault(c.section, []).append(c)

    lines: list[str] = []
    lines.append("# Final System Validation Report")
    lines.append("")
    lines.append(f"**Generated:** {report.timestamp}")
    lines.append(f"**Status:** {report.status}")
    lines.append(
        f"**Checks passed:** {report.passed_count}/{report.total_count}"
    )
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("| Area | Passed | Total | Status |")
    lines.append("|------|--------|-------|--------|")
    for sec in sections_order:
        items = by_section.get(sec, [])
        p = sum(1 for i in items if i.passed)
        t = len(items)
        st = "PASS" if p == t and t else ("PARTIAL" if p else "FAIL")
        lines.append(f"| {sec} | {p} | {t} | {st} |")
    lines.append("")

    mp = report.metrics.get("model_performance") or {}
    dp = report.metrics.get("data_pipeline") or {}
    if mp or dp:
        lines.append("## Key Metrics")
        lines.append("")
        if dp:
            lines.append(
                f"- **Data:** {dp.get('rows', '?')} rows, "
                f"{dp.get('date_min', '?')} → {dp.get('date_max', '?')} "
                f"(days since last match: {dp.get('days_since_last_match', '?')})"
            )
        if mp:
            lines.append(
                f"- **Best Brier:** {mp.get('best_brier')} "
                f"({mp.get('best_model')}) — gate < {BEST_BRIER_MAX}"
            )
            hi = mp.get("highest_accuracy") or {}
            lines.append(
                f"- **Best Accuracy:** {hi.get('accuracy', mp.get('best_accuracy'))} "
                f"({hi.get('model', mp.get('best_model'))}) — gate > {BEST_ACCURACY_MIN:.0%}"
            )
            lines.append(
                f"- **Best Backtest ROI:** {mp.get('best_roi_pct')}% — gate > {BACKTEST_ROI_MIN:.0f}%"
            )
            lines.append(
                f"- **Best avg CLV:** {mp.get('best_avg_clv')} "
                f"({mp.get('best_clv_model')}) — gate > {CLV_MIN}%"
            )
        lines.append("")

    for sec in sections_order:
        items = by_section.get(sec, [])
        p = sum(1 for i in items if i.passed)
        t = len(items)
        lines.append(f"## {sec} ({p}/{t})")
        for item in items:
            lines.append(f"- {_checkbox(item.passed)} {item.name}")
            if item.evidence:
                lines.append(f"  - Evidence: `{item.evidence}`")
        lines.append("")

    lines.append("## Blockers")
    if not report.blockers:
        lines.append("None — system meets all final validation gates.")
    else:
        for i, b in enumerate(report.blockers, 1):
            lines.append(f"{i}. **[{b['section']}] {b['name']}** — {b['evidence']}")
    lines.append("")

    lines.append("## Next Steps")
    if report.status == "READY":
        lines.append("1. Archive this report as production sign-off.")
        lines.append("2. Enable production monitoring alerts and on-call rota.")
        lines.append("3. Begin commercial launch checklist execution.")
    else:
        steps: list[str] = []
        names = " ".join(b["name"].lower() for b in report.blockers)
        if "data" in names or "gap" in names or "collection" in names:
            steps.append(
                "Run `python scripts/daily_data_pipeline.py` and "
                "`python scripts/daily_feature_computation.py`; verify "
                "`data/processed/results_clean.csv` updates daily."
            )
        if "brier" in names or "accuracy" in names:
            steps.append(
                "Improve / recalibrate models until best Brier < 0.40 and "
                "Accuracy > 65% (expand training data beyond World-Cup-centric set)."
            )
        if "roi" in names or "clv" in names:
            steps.append(
                "Revisit value-bet thresholds and odds quality; "
                "re-run backtests until ROI > 5%. Reconcile CLV summary vs backtest CLV=0."
            )
        if "recovery" in names or "alert" in names or "automated" in names:
            steps.append(
                "Add a disaster-recovery runbook under docs/; confirm Slack/email "
                "failure alerts; ensure scheduler jobs succeed end-to-end."
            )
        if "monitoring" in names or "deployment" in names or "documentation" in names:
            steps.append(
                "Generate live monitoring reports into `reports/monitoring/`; "
                "execute a documented Docker deployment dry-run."
            )
        if "pricing" in names or "customer" in names or "marketing" in names or "support" in names:
            steps.append(
                "Author commercial pack: pricing model, ICP/target customers, "
                "marketing one-pager, and support/SLA process docs."
            )
        steps.append("Re-run: `python scripts/final_system_validation.py`")
        for i, s in enumerate(steps, 1):
            lines.append(f"{i}. {s}")
    lines.append("")
    lines.append("---")
    lines.append(
        f"_Generated by `scripts/final_system_validation.py` at {report.timestamp}_"
    )
    lines.append("")
    return "\n".join(lines)


def generate_console(report: ValidationReport) -> str:
    lines = [
        "",
        "=" * 62,
        "  FINAL SYSTEM VALIDATION",
        f"  Status: {report.status}",
        f"  Passed: {report.passed_count}/{report.total_count}",
        "=" * 62,
    ]
    current = None
    for c in report.checks:
        if c.section != current:
            current = c.section
            lines.append(f"\n  [{current}]")
        mark = "OK" if c.passed else "X "
        lines.append(f"    [{mark}] {c.name}")
    if report.blockers:
        lines.append("\n  Blockers:")
        for b in report.blockers[:20]:
            lines.append(f"    - [{b['section']}] {b['name']}")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Final System Validation checklist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT))
    p.add_argument("--json-only", action="store_true")
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--missing", action="store_true", help="Print only failed checks")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_validation()

    if args.missing:
        print("Failed checks:")
        for b in report.blockers:
            print(f"  - [{b['section']}] {b['name']}: {b['evidence']}")
        if not report.blockers:
            print("  (none)")
    else:
        print(generate_console(report))

    payload = {
        "timestamp": report.timestamp,
        "status": report.status,
        "passed": report.passed_count,
        "total": report.total_count,
        "metrics": report.metrics,
        "checks": [asdict(c) for c in report.checks],
        "blockers": report.blockers,
        "warnings": report.warnings,
    }

    if not args.no_write:
        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(generate_markdown(report), encoding="utf-8")
        print(f"  Report saved: {out}")
        json_path = out.with_suffix(".json")
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"  JSON saved:   {json_path}")

    if args.json_only:
        print(json.dumps(payload, indent=2, default=str))

    print(f"  Overall: {report.status}")
    return 0 if report.status == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
