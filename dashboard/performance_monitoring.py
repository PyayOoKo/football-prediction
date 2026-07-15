"""
Live Performance Monitoring Dashboard — real-time tracking of model accuracy,
Brier Score, ROI, CLV, bet frequency, bankroll, and alerting.

Integrates with:
  - src.live_predictions — live prediction engine
  - src.monitoring — monitoring store + alert engine
  - src.evaluate — model evaluation metrics
  - reports/ — stored backtest, CLV, and bankroll reports

Usage
-----
    streamlit run dashboard/performance_monitoring.py
    python -m streamlit run dashboard/performance_monitoring.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Live Performance Monitoring",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark theme consistent with existing dashboard) ──
st.markdown("""
<style>
    .stApp { background: #0e1117; }
    .stApp header { background: #1a1d27; }
    .metric-card {
        background: linear-gradient(135deg, #1a1d27 0%, #222639 100%);
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .metric-positive { color: #4caf50; }
    .metric-negative { color: #f44336; }
    .metric-neutral { color: #ffc107; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #fff; margin: 0; }
    .metric-label { font-size: 0.7rem; color: #8b8fa3; text-transform: uppercase; letter-spacing: 0.05em; }
    .alert-critical {
        background: linear-gradient(135deg, #2a0a0a 0%, #3a1414 100%);
        border: 1px solid #f44336;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .alert-warning {
        background: linear-gradient(135deg, #2a1f0a 0%, #3a2a14 100%);
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .alert-info {
        background: linear-gradient(135deg, #0a1a2a 0%, #14243a 100%);
        border: 1px solid #2196f3;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .hero {
        background: linear-gradient(135deg, #1a1d27 0%, #16213e 50%, #1a1d27 100%);
        border: 1px solid #2a2d3a;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
        background: linear-gradient(90deg, #4fc3f7, #81c784);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .badge-green { background: #1b5e20; color: #81c784; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
    .badge-red { background: #b71c1c; color: #ef9a9a; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
    .badge-yellow { background: #7c5a00; color: #ffd54f; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
    .status-dot {
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .stAlert { border-radius: 10px; }
    div[data-testid="stExpander"] { border: 1px solid #2a2d3a; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  Session state initialisation
# ═══════════════════════════════════════════════════════════

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True
if "refresh_interval" not in st.session_state:
    st.session_state.refresh_interval = 60  # seconds
if "model_performance_history" not in st.session_state:
    st.session_state.model_performance_history = []
if "alert_history" not in st.session_state:
    st.session_state.alert_history = []


# ═══════════════════════════════════════════════════════════
#  Data loading helpers (cached with TTL)
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def load_live_predictions() -> list[dict]:
    """Load the latest live predictions from disk."""
    latest_path = Path("reports/live/latest_predictions.json")
    if not latest_path.exists():
        return []
    try:
        with open(latest_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


@st.cache_data(ttl=60)
def load_monitoring_metrics(days: int = 30) -> dict[str, pd.DataFrame]:
    """Load monitoring store metrics for the last N days.

    Returns a dict with keys: ``etl``, ``system``, ``data_quality``, ``cache``.
    """
    result: dict[str, pd.DataFrame] = {}
    try:
        from src.monitoring.store import MonitoringStore

        store = MonitoringStore(db_path="data/monitoring/monitor.db")

        etl = store.get_etl_history(days=days)
        sys_data = store.get_system_history(days=days)
        dq = store.get_data_quality_history(days=days)
        cache = store.get_cache_history(days=days)

        if etl:
            result["etl"] = pd.DataFrame(etl)
        if sys_data:
            result["system"] = pd.DataFrame(sys_data)
        if dq:
            result["data_quality"] = pd.DataFrame(dq)
        if cache:
            result["cache"] = pd.DataFrame(cache)
    except Exception as exc:
        st.warning(f"Monitoring store not available: {exc}")

    return result


@st.cache_data(ttl=120)
def load_clv_reports() -> pd.DataFrame:
    """Load and aggregate all CLV reports from the reports directory."""
    rows: list[dict] = []
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return pd.DataFrame()

    for pattern in ["clv_*.json", "clv_summary_*.json", "clv_comparison_*.csv",
                     "clv_tracking_*.json", "clv_report*.json"]:
        for f in sorted(reports_dir.glob(pattern), reverse=True)[:20]:
            try:
                if f.suffix == ".csv":
                    df = pd.read_csv(f)
                    for _, r in df.iterrows():
                        rows.append({
                            "file": f.name,
                            "model": r.get("model", r.get("model_name", r.get("name", "Unknown"))),
                            "avg_clv": float(r.get("avg_clv", r.get("clv", 0))),
                            "positive_clv_pct": float(r.get("positive_clv_pct", r.get("clv_gt_0_pct", 0))),
                            "clv_gt_5_pct": float(r.get("clv_gt_5_pct", 0)),
                            "bets": int(r.get("bets", r.get("n_bets", r.get("total_bets", 0)))),
                        })
                else:
                    with open(f) as fh:
                        data = json.load(fh)
                    _extract_clv_from_json(data, f.name, rows)
            except Exception:
                pass

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _extract_clv_from_json(data: Any, filename: str, rows: list[dict]) -> None:
    """Recursively extract CLV metrics from JSON reports of various formats."""
    if isinstance(data, dict):
        # Per-model CLV results
        for key in ["clv_values", "clv_per_bet", "results", "models"]:
            items = data.get(key, [])
            if isinstance(items, list) and items:
                for item in items:
                    if isinstance(item, dict):
                        rows.append({
                            "file": filename,
                            "model": item.get("model", item.get("model_name", item.get("name", "Unknown"))),
                            "avg_clv": float(item.get("clv", item.get("avg_clv", item.get("average_clv", 0)))),
                            "positive_clv_pct": float(item.get("positive_clv_pct", item.get("clv_gt_0_pct", 0))),
                            "clv_gt_5_pct": float(item.get("clv_gt_5_pct", 0)),
                            "bets": int(item.get("bets", item.get("n_bets", item.get("total_bets", 0)))),
                        })
                return
        # Single result
        clv_val = data.get("avg_clv", data.get("clv", data.get("average_clv", None)))
        if clv_val is not None:
            rows.append({
                "file": filename,
                "model": data.get("model", data.get("model_name", "Aggregate")),
                "avg_clv": float(clv_val),
                "positive_clv_pct": float(data.get("positive_clv_pct", data.get("clv_gt_0_pct", 0))),
                "clv_gt_5_pct": float(data.get("clv_gt_5_pct", 0)),
                "bets": int(data.get("bets", data.get("n_bets", data.get("total_bets", 0)))),
            })


@st.cache_data(ttl=120)
def load_backtest_reports() -> pd.DataFrame:
    """Load and aggregate backtest/bankroll reports for ROI and strategy metrics."""
    rows: list[dict] = []
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return pd.DataFrame()

    for pattern in ["backtest_*.json", "backtest_summary_*.json",
                     "bankroll_management_*.json", "bankroll_optimization_*.json",
                     "bankroll_report*.json"]:
        for f in sorted(reports_dir.glob(pattern), reverse=True)[:30]:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                _extract_backtest_metrics(data, f.name, rows)
            except Exception:
                pass

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _extract_backtest_metrics(data: Any, filename: str, rows: list[dict]) -> None:
    """Extract strategy metrics from backtest JSON reports."""
    if isinstance(data, dict):
        # Best strategy format
        if "best_strategy" in data:
            bs = data["best_strategy"]
            rows.append({
                "file": filename,
                "strategy": bs.get("strategy", bs.get("name", "Best")),
                "roi_pct": float(bs.get("roi_pct", bs.get("roi", 0))),
                "sharpe_ratio": float(bs.get("sharpe_ratio", 0)),
                "max_drawdown_pct": float(bs.get("max_drawdown_pct", 0)),
                "total_profit": float(bs.get("total_profit", 0)),
                "total_bets": int(bs.get("total_bets", 0)),
                "win_rate_pct": float(bs.get("win_rate_pct", bs.get("win_rate", 0))),
                "profit_factor": float(bs.get("profit_factor", 0)),
            })

        # Strategy comparison format
        for section in ["stake_strategies", "risk_scenarios", "strategies", "results"]:
            items = data.get(section, [])
            if isinstance(items, dict):
                items = items.get("results", [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and item.get("total_bets", 0) > 0:
                        rows.append({
                            "file": filename,
                            "strategy": item.get("strategy", item.get("name", item.get("model_name", "Unknown"))),
                            "roi_pct": float(item.get("roi_pct", item.get("roi", 0))),
                            "sharpe_ratio": float(item.get("sharpe_ratio", 0)),
                            "max_drawdown_pct": float(item.get("max_drawdown_pct", 0)),
                            "total_profit": float(item.get("total_profit", item.get("profit", 0))),
                            "total_bets": int(item.get("total_bets", item.get("n_bets", 0))),
                            "win_rate_pct": float(item.get("win_rate_pct", item.get("win_rate", 0))),
                            "profit_factor": float(item.get("profit_factor", 0)),
                        })

        # Direct metrics format
        if "metrics" in data and isinstance(data["metrics"], dict):
            m = data["metrics"]
            rows.append({
                "file": filename,
                "strategy": "Default",
                "roi_pct": float(m.get("roi_pct", m.get("roi", 0))),
                "sharpe_ratio": float(m.get("sharpe_ratio", 0)),
                "max_drawdown_pct": float(m.get("max_drawdown_pct", 0)),
                "total_profit": float(m.get("total_profit", 0)),
                "total_bets": int(m.get("total_bets", m.get("n_bets", 0))),
                "win_rate_pct": float(m.get("win_rate_pct", m.get("win_rate", 0))),
                "profit_factor": float(m.get("profit_factor", 0)),
            })


@st.cache_data(ttl=30)
def detect_model_files() -> dict[str, bool]:
    """Detect which model files exist on disk."""
    model_paths = [
        "models/ensemble.pkl",
        "models/ensemble_model.joblib",
        "models/stacking_ensemble.joblib",
        "models/weighted_ensemble.joblib",
        "models/xgboost_model.pkl",
        "models/xgboost_tuned.joblib",
        "models/lightgbm_tuned.joblib",
        "models/model.pkl",
    ]
    return {p: Path(p).exists() for p in model_paths}


@st.cache_data(ttl=30)
def detect_data_files() -> dict[str, int]:
    """Count data files in raw and processed directories."""
    counts: dict[str, int] = {}
    for d in ["data/raw", "data/processed", "data/external"]:
        path = Path(d)
        if path.exists():
            counts[d] = len(list(path.glob("*.csv")))
        else:
            counts[d] = 0
    return counts


@st.cache_data(ttl=60)
def load_evaluation_history() -> list[dict]:
    """Load historical evaluation results from report files."""
    rows: list[dict] = []
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return rows

    for pattern in ["*validation*.json", "*performance*.json",
                     "calibration_results*.json", "phase4_*.json",
                     "phase3_vs_phase4*.json"]:
        for f in sorted(reports_dir.glob(pattern), reverse=True)[:20]:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                # Try to extract metrics from various formats
                metrics = data.get("metrics", data.get("overall", data))
                if isinstance(metrics, dict):
                    accuracy = metrics.get("accuracy", metrics.get("test_accuracy", None))
                    logloss = metrics.get("log_loss", metrics.get("test_log_loss", None))
                    brier = metrics.get("brier_score", metrics.get("brier", None))
                    f1 = metrics.get("f1", None)
                    roc_auc = metrics.get("roc_auc", None)
                    if accuracy is not None:
                        rows.append({
                            "file": f.name,
                            "timestamp": f.stat().st_mtime,
                            "datetime": datetime.fromtimestamp(f.stat().st_mtime),
                            "accuracy": float(accuracy),
                            "log_loss": float(logloss) if logloss is not None else None,
                            "brier_score": float(brier) if brier is not None else None,
                            "f1": float(f1) if f1 is not None else None,
                            "roc_auc": float(roc_auc) if roc_auc is not None else None,
                        })
            except Exception:
                pass

    return sorted(rows, key=lambda r: r["timestamp"])


# ═══════════════════════════════════════════════════════════
#  Alert engine integration
# ═══════════════════════════════════════════════════════════

def build_performance_snapshot(
    accuracy: float | None,
    brier_score: float | None,
    log_loss: float | None,
    roi_pct: float | None,
    avg_clv: float | None,
    win_rate_pct: float | None,
    sharpe_ratio: float | None,
    max_drawdown_pct: float | None,
    bankroll_change_pct: float | None,
    bets_per_day: float | None,
    avg_ev: float | None,
    avg_confidence: float | None,
) -> dict[str, Any]:
    """Build a metric snapshot dict suitable for the AlertEngine."""
    snapshot: dict[str, Any] = {
        "performance": {},
    }
    if accuracy is not None:
        snapshot["performance"]["accuracy"] = accuracy
    if brier_score is not None:
        snapshot["performance"]["brier_score"] = brier_score
    if log_loss is not None:
        snapshot["performance"]["log_loss"] = log_loss
    if roi_pct is not None:
        snapshot["performance"]["roi_pct"] = roi_pct
    if avg_clv is not None:
        snapshot["performance"]["avg_clv"] = avg_clv
    if win_rate_pct is not None:
        snapshot["performance"]["win_rate_pct"] = win_rate_pct
    if sharpe_ratio is not None:
        snapshot["performance"]["sharpe_ratio"] = sharpe_ratio
    if max_drawdown_pct is not None:
        snapshot["performance"]["max_drawdown_pct"] = max_drawdown_pct
    if bankroll_change_pct is not None:
        snapshot["performance"]["bankroll_change_pct"] = bankroll_change_pct
    if bets_per_day is not None:
        snapshot["performance"]["bets_per_day"] = bets_per_day
    if avg_ev is not None:
        snapshot["performance"]["avg_ev"] = avg_ev
    if avg_confidence is not None:
        snapshot["performance"]["avg_confidence"] = avg_confidence

    return snapshot


def evaluate_performance_alerts(
    snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate performance snapshot against alert rules.

    Returns a list of triggered alert dicts with keys:
    ``rule_name``, ``severity``, ``message``, ``metric``, ``actual_value``, ``threshold``.
    """
    try:
        from src.monitoring.alerting import AlertEngine
        engine = AlertEngine()
        events = engine.evaluate(snapshot)
        return [e.to_dict() for e in events]
    except Exception as exc:
        return [{
            "rule_name": "engine_error",
            "severity": "info",
            "message": f"Alert engine unavailable: {exc}",
            "metric": "system",
            "actual_value": 0,
            "threshold": 0,
        }]


# ═══════════════════════════════════════════════════════════
#  Chart rendering helpers
# ═══════════════════════════════════════════════════════════

def render_metric_card(value_html: str, label: str, col) -> None:
    """Render a styled metric card in a Streamlit column."""
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{value_html}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_alert_card(event: dict) -> None:
    """Render a styled alert event card."""
    severity = event.get("severity", "info")
    cls = {
        "critical": "alert-critical",
        "warning": "alert-warning",
        "info": "alert-info",
    }.get(severity, "alert-info")

    icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(severity, "🔵")

    st.markdown(
        f'<div class="{cls}">'
        f'<strong>{icon} [{severity.upper()}]</strong> '
        f'{event.get("message", "")}'
        f'</div>',
        unsafe_allow_html=True,
    )


def plot_accuracy_trend(eval_history: list[dict]) -> go.Figure:
    """Plot accuracy over time from evaluation history."""
    if not eval_history:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        return fig

    df = pd.DataFrame(eval_history)
    fig = go.Figure()

    if "accuracy" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df["accuracy"],
            mode="lines+markers", name="Accuracy",
            line=dict(color="#4fc3f7", width=2),
            marker=dict(size=6, color="#4fc3f7"),
            hovertemplate="%{y:.1%}<extra></extra>",
        ))
    if "f1" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df["f1"],
            mode="lines+markers", name="F1 Score",
            line=dict(color="#81c784", width=2, dash="dot"),
            marker=dict(size=6, color="#81c784"),
            hovertemplate="%{y:.3f}<extra></extra>",
        ))
    if "roc_auc" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df["roc_auc"],
            mode="lines+markers", name="ROC-AUC",
            line=dict(color="#ba68c8", width=2, dash="dash"),
            marker=dict(size=6, color="#ba68c8"),
            hovertemplate="%{y:.3f}<extra></extra>",
        ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="#ffc107", line_width=1,
                  annotation_text="Random (50%)")
    fig.add_hline(y=0.333, line_dash="dot", line_color="#f44336", line_width=1,
                  annotation_text="Uniform (33%)")

    fig.update_layout(
        title="Model Accuracy Over Time",
        xaxis_title="Date",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(color="#ccc", size=11),
        ),
    )
    return fig


def plot_brier_trend(eval_history: list[dict]) -> go.Figure:
    """Plot Brier Score trend over time."""
    fig = go.Figure()

    df = pd.DataFrame(eval_history)
    if "brier_score" not in df.columns or df["brier_score"].isna().all():
        fig.add_annotation(text="No Brier Score data", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        return fig

    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["brier_score"],
        mode="lines+markers",
        name="Brier Score",
        line=dict(color="#ff7043", width=2),
        marker=dict(size=7, color="#ff7043", symbol="diamond"),
        hovertemplate="%{y:.4f}<extra></extra>",
    ))

    # Reference zones
    fig.add_hrect(y0=0, y1=0.15, fillcolor="green", opacity=0.05, layer="below",
                  line_width=0, annotation_text="Excellent", annotation_position="top left")
    fig.add_hrect(y0=0.15, y1=0.25, fillcolor="yellow", opacity=0.05, layer="below",
                  line_width=0, annotation_text="Good", annotation_position="top left")
    fig.add_hrect(y0=0.25, y1=0.40, fillcolor="orange", opacity=0.05, layer="below",
                  line_width=0, annotation_text="Poor", annotation_position="top left")
    fig.add_hline(y=0.40, line_dash="dash", line_color="#f44336", line_width=1,
                  annotation_text="Critical")

    fig.update_layout(
        title="Brier Score Trend (lower is better)",
        xaxis_title="Date",
        yaxis_title="Brier Score",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        hovermode="x unified",
    )
    return fig


def plot_log_loss_trend(eval_history: list[dict]) -> go.Figure:
    """Plot Log Loss trend over time."""
    fig = go.Figure()

    df = pd.DataFrame(eval_history)
    if "log_loss" not in df.columns or df["log_loss"].isna().all():
        fig.add_annotation(text="No Log Loss data", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        return fig

    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["log_loss"],
        mode="lines+markers",
        name="Log Loss",
        line=dict(color="#ab47bc", width=2),
        marker=dict(size=7, color="#ab47bc", symbol="triangle-up"),
        hovertemplate="%{y:.4f}<extra></extra>",
    ))

    fig.add_hline(y=1.1, line_dash="dash", line_color="#ffc107", line_width=1,
                  annotation_text="Threshold (1.1)")
    fig.add_hline(y=1.5, line_dash="dash", line_color="#f44336", line_width=1,
                  annotation_text="Random (1.5)")

    fig.update_layout(
        title="Log Loss Trend (lower is better)",
        xaxis_title="Date",
        yaxis_title="Log Loss",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        hovermode="x unified",
    )
    return fig


def plot_roi_by_model(backtest_df: pd.DataFrame) -> go.Figure:
    """Plot ROI comparison across strategies/models."""
    fig = go.Figure()

    if backtest_df.empty:
        fig.add_annotation(text="No backtest data", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        return fig

    # Aggregate by strategy
    agg = backtest_df.groupby("strategy").agg({
        "roi_pct": "mean",
        "total_bets": "sum",
        "sharpe_ratio": "mean",
        "max_drawdown_pct": "mean",
        "profit_factor": "mean",
    }).reset_index().sort_values("roi_pct", ascending=True)

    colors = ["#f44336" if v < 0 else "#4caf50" for v in agg["roi_pct"]]

    fig.add_trace(go.Bar(
        y=agg["strategy"],
        x=agg["roi_pct"],
        orientation="h",
        marker=dict(color=colors),
        text=agg["roi_pct"].apply(lambda x: f"{x:+.1f}%"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>ROI: %{x:+.1f}%<br>"
                      "Bets: %{customdata[0]:,}<br>Sharpe: %{customdata[1]:.2f}",
        customdata=agg[["total_bets", "sharpe_ratio"]],
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="#666", line_width=1)

    fig.update_layout(
        title="ROI by Strategy / Model",
        xaxis_title="ROI (%)",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        margin=dict(l=10, r=30, t=40, b=10),
        height=max(300, len(agg) * 35),
    )
    return fig


def plot_clv_comparison(clv_df: pd.DataFrame) -> go.Figure:
    """Plot CLV comparison across models."""
    fig = go.Figure()

    if clv_df.empty:
        fig.add_annotation(text="No CLV data", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        return fig

    agg = clv_df.groupby("model").agg({
        "avg_clv": "mean",
        "positive_clv_pct": "mean",
        "bets": "sum",
    }).reset_index().sort_values("avg_clv", ascending=True)

    colors = ["#f44336" if v < -0.005 else "#4caf50" if v > 0.005 else "#ffc107"
              for v in agg["avg_clv"]]

    fig.add_trace(go.Bar(
        y=agg["model"],
        x=agg["avg_clv"] * 100,  # Convert to percentage
        orientation="h",
        marker=dict(color=colors),
        text=agg["avg_clv"].apply(lambda x: f"{x:+.2%}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Avg CLV: %{x:+.2f}%<br>"
                      "CLV > 0%: %{customdata[0]:.1f}%<br>Bets: %{customdata[1]:,}",
        customdata=agg[["positive_clv_pct", "bets"]],
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="#666", line_width=1)

    fig.update_layout(
        title="Average CLV by Model",
        xaxis_title="CLV (%)",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        margin=dict(l=10, r=30, t=40, b=10),
        height=max(300, len(agg) * 35),
    )
    return fig


def plot_bet_frequency(live_predictions: list[dict]) -> go.Figure:
    """Plot bet frequency / match volume over time from live predictions."""
    fig = go.Figure()

    if not live_predictions:
        fig.add_annotation(text="No live prediction data", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        return fig

    df = pd.DataFrame(live_predictions)
    df["timestamp"] = pd.to_datetime(df.get("timestamp", df.get("match_date")), errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if df.empty:
        fig.add_annotation(text="No timestamp data", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        return fig

    # Count value bets per time bucket
    df["value_bets"] = df.get("n_value_bets", 0)
    df["hour_bucket"] = df["timestamp"].dt.floor("H")

    hourly = df.groupby("hour_bucket").agg(
        total_matches=("home_team", "count"),
        total_value_bets=("value_bets", "sum"),
    ).reset_index().sort_values("hour_bucket")

    fig.add_trace(go.Bar(
        x=hourly["hour_bucket"],
        y=hourly["total_matches"],
        name="Matches",
        marker=dict(color="#4fc3f7", opacity=0.7),
        hovertemplate="%{x}<br>Matches: %{y}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=hourly["hour_bucket"],
        y=hourly["total_value_bets"],
        name="Value Bets",
        marker=dict(color="#81c784", opacity=0.7),
        hovertemplate="%{x}<br>Value Bets: %{y}<extra></extra>",
    ))

    fig.update_layout(
        title="Match & Value Bet Volume",
        xaxis_title="Time",
        yaxis_title="Count",
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_bankroll_growth(backtest_df: pd.DataFrame) -> go.Figure:
    """Plot bankroll vs drawdown for the best strategy."""
    fig = go.Figure()

    if backtest_df.empty:
        fig.add_annotation(text="No bankroll data", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        return fig

    # Try to find bankroll history from report files
    reports_dir = Path("reports")
    bankroll_data = None
    for pattern in ["bankroll_management_*.json", "bankroll_optimization_*.json",
                     "bankroll_report*.json"]:
        for f in sorted(reports_dir.glob(pattern), reverse=True)[:5]:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                # Check for best_strategy with bankroll_history
                if "best_strategy" in data:
                    bs = data["best_strategy"]
                    bh = bs.get("bankroll_history", None)
                    if bh:
                        bankroll_data = bh
                        break
                # Check for strategies with bankroll_history
                for section in ["stake_strategies", "risk_scenarios"]:
                    items = data.get(section, {})
                    results = items.get("results", []) if isinstance(items, dict) else []
                    for item in results:
                        bh = item.get("bankroll_history", [])
                        if bh:
                            bankroll_data = bh
                            break
            except Exception:
                pass
        if bankroll_data:
            break

    if bankroll_data:
        br_df = pd.DataFrame({
            "Bet": list(range(len(bankroll_data))),
            "Bankroll": bankroll_data,
        })

        fig.add_trace(go.Scatter(
            x=br_df["Bet"], y=br_df["Bankroll"],
            mode="lines",
            name="Bankroll",
            line=dict(color="#7c3aed", width=2),
            fill="tozeroy",
            fillcolor="rgba(124, 58, 237, 0.1)",
            hovertemplate="Bet %{x}<br>£%{y:,.2f}<extra></extra>",
        ))

        # Initial bankroll line
        initial = bankroll_data[0]
        fig.add_hline(y=initial, line_dash="dash", line_color="#666", line_width=1,
                      annotation_text=f"Initial: £{initial:,.0f}")

        # Peak line
        peak = max(bankroll_data)
        fig.add_hline(y=peak, line_dash="dot", line_color="#4caf50", line_width=1,
                      annotation_text=f"Peak: £{peak:,.0f}")
    else:
        # Show ROI by strategy as an alternative
        return plot_roi_by_model(backtest_df)

    fig.update_layout(
        title="Bankroll Growth (Best Strategy)",
        xaxis_title="Bet Number",
        yaxis_title="Bankroll (£)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
        hovermode="x unified",
    )
    return fig


# ═══════════════════════════════════════════════════════════
#  Dashboard sections
# ═══════════════════════════════════════════════════════════

def render_header() -> None:
    """Render the hero header with system status info."""
    st.markdown('<div class="hero">', unsafe_allow_html=True)

    now = datetime.now()
    last_refresh = st.session_state.last_refresh
    time_since = (now - last_refresh).total_seconds()

    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("<h1>📈 Live Performance Monitoring</h1>", unsafe_allow_html=True)
        st.markdown(
            f"<p>Real-time tracking of model accuracy, calibration, betting ROI, "
            f"CLV, and bankroll status • "
            f"<span style='color:#8b8fa3'>Last refresh: {last_refresh.strftime('%H:%M:%S')} "
            f"({time_since:.0f}s ago)</span></p>",
            unsafe_allow_html=True,
        )
    with col_h2:
        status = "🟢 Active" if st.session_state.auto_refresh else "🔴 Paused"
        st.markdown(
            f"<div style='text-align:right;padding-top:0.5rem'>"
            f"<span style='font-size:0.9rem'>{status}</span><br>"
            f"<span style='font-size:0.75rem;color:#8b8fa3'>"
            f"Every {st.session_state.refresh_interval}s</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Controls row
    cc1, cc2, cc3, cc4, cc5 = st.columns([1, 1, 1, 2, 1])
    with cc1:
        if st.button("🔄 Refresh Now", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    with cc2:
        new_val = st.number_input(
            "Refresh interval (s)", min_value=5, max_value=600,
            value=st.session_state.refresh_interval, step=5,
            label_visibility="collapsed",
        )
        st.session_state.refresh_interval = new_val
    with cc3:
        if st.button(
            "⏸ Pause" if st.session_state.auto_refresh else "▶ Resume",
            use_container_width=True,
        ):
            st.session_state.auto_refresh = not st.session_state.auto_refresh
            st.rerun()


def render_system_status() -> None:
    """Render system status row (models, data, reports, config)."""
    st.markdown("### 🔌 System Status")
    c1, c2, c3, c4 = st.columns(4)

    # Model status
    with c1:
        model_files = detect_model_files()
        n_models = sum(1 for v in model_files.values() if v)
        status_color = "#4caf50" if n_models > 0 else "#f44336"
        render_metric_card(
            f'<span style="color:{status_color}">{"✅" if n_models > 0 else "❌"} {n_models}</span>',
            "Models Available",
            c1,
        )

    # Data status
    with c2:
        data_counts = detect_data_files()
        total_files = sum(data_counts.values())
        status_color = "#4caf50" if total_files > 0 else "#f44336"
        render_metric_card(
            f'<span style="color:{status_color}">{total_files}</span>',
            "Data Files",
            c2,
        )

    # Live predictions
    with c3:
        live_preds = load_live_predictions()
        n_live = len(live_preds)
        n_value = sum(1 for p in live_preds if p.get("n_value_bets", 0) > 0)
        color = "#4caf50" if n_live > 0 else "#ffc107"
        render_metric_card(
            f'<span style="color:{color}">{n_live}</span>',
            f"Live Matches ({n_value} value bets)",
            c3,
        )

    # Config
    with c4:
        try:
            from config import config as cfg
            model_type = cfg.train.model_type
        except Exception:
            model_type = "default"
        render_metric_card(
            f'<span style="color:#4fc3f7">{model_type.upper()}</span>',
            "Active Model Config",
            c4,
        )


def render_live_metrics_row() -> None:
    """Render top-level live metrics: accuracy, brier, ROI, CLV, bet freq, bankroll."""
    st.markdown("### 🎯 Live Performance Metrics")

    eval_history = load_evaluation_history()
    clv_df = load_clv_reports()
    backtest_df = load_backtest_reports()
    live_preds = load_live_predictions()

    # Compute current metrics
    latest_eval = eval_history[-1] if eval_history else {}
    current_accuracy = latest_eval.get("accuracy")
    current_brier = latest_eval.get("brier_score")
    current_logloss = latest_eval.get("log_loss")

    # Best ROI from backtest
    best_roi = 0
    if not backtest_df.empty and "roi_pct" in backtest_df.columns:
        best_roi = backtest_df["roi_pct"].max()

    # Avg CLV
    avg_clv = 0
    if not clv_df.empty and "avg_clv" in clv_df.columns:
        avg_clv = clv_df["avg_clv"].mean()

    # Bet frequency (from live predictions)
    n_live = len(live_preds)
    n_value = sum(1 for p in live_preds if p.get("n_value_bets", 0) > 0)

    # Best sharpe / drawdown
    best_sharpe = 0
    best_drawdown = 0
    if not backtest_df.empty:
        if "sharpe_ratio" in backtest_df.columns:
            best_sharpe = backtest_df["sharpe_ratio"].max()
        if "max_drawdown_pct" in backtest_df.columns:
            best_drawdown = backtest_df["max_drawdown_pct"].min()

    # Build and evaluate snapshot against alert rules
    snapshot = build_performance_snapshot(
        accuracy=current_accuracy,
        brier_score=current_brier,
        log_loss=current_logloss,
        roi_pct=best_roi,
        avg_clv=avg_clv,
        win_rate_pct=None,
        sharpe_ratio=best_sharpe if best_sharpe > 0 else None,
        max_drawdown_pct=best_drawdown if best_drawdown > 0 else None,
        bankroll_change_pct=None,
        bets_per_day=n_live if n_live > 0 else None,
        avg_ev=None,
        avg_confidence=None,
    )
    alert_events = evaluate_performance_alerts(snapshot)
    st.session_state.alert_history = alert_events

    # ── Metrics row ──
    mcol1, mcol2, mcol3, mcol4, mcol5, mcol6 = st.columns(6)

    # 1. Accuracy
    with mcol1:
        if current_accuracy is not None:
            color = "#4caf50" if current_accuracy >= 0.6 else "#ffc107" if current_accuracy >= 0.5 else "#f44336"
            render_metric_card(
                f'<span style="color:{color}">{current_accuracy:.1%}</span>',
                "Model Accuracy",
                mcol1,
            )
        else:
            render_metric_card('<span style="color:#8b8fa3">—</span>', "Model Accuracy", mcol1)

    # 2. Brier Score
    with mcol2:
        if current_brier is not None:
            color = "#4caf50" if current_brier <= 0.15 else "#ffc107" if current_brier <= 0.25 else "#f44336"
            render_metric_card(
                f'<span style="color:{color}">{current_brier:.4f}</span>',
                "Brier Score",
                mcol2,
            )
        else:
            render_metric_card('<span style="color:#8b8fa3">—</span>', "Brier Score", mcol2)

    # 3. ROI
    with mcol3:
        roi_color = "#4caf50" if best_roi > 0 else "#f44336"
        render_metric_card(
            f'<span style="color:{roi_color}">{best_roi:+.1f}%</span>',
            "Best ROI",
            mcol3,
        )

    # 4. CLV
    with mcol4:
        clv_color = "#4caf50" if avg_clv > 0 else "#f44336" if avg_clv < -0.005 else "#ffc107"
        render_metric_card(
            f'<span style="color:{clv_color}">{avg_clv:+.2%}</span>',
            "Avg CLV",
            mcol4,
        )

    # 5. Bet Frequency
    with mcol5:
        render_metric_card(
            f'<span style="color:#4fc3f7">{n_live}</span>',
            f"Live Matches ({n_value} value)",
            mcol5,
        )

    # 6. Sharpe Ratio
    with mcol6:
        sharpe_color = "#4caf50" if best_sharpe >= 1 else "#ffc107" if best_sharpe >= 0 else "#f44336"
        render_metric_card(
            f'<span style="color:{sharpe_color}">{best_sharpe:.2f}</span>',
            "Best Sharpe Ratio",
            mcol6,
        )

    # ── Alert panel ──
    if alert_events:
        st.markdown("### 🚨 Active Performance Alerts")
        for event in alert_events:
            render_alert_card(event)
    else:
        st.markdown(
            '<p style="color:#4caf50;font-size:0.9rem">✅ All performance metrics within normal ranges</p>',
            unsafe_allow_html=True,
        )


def render_model_performance_section() -> None:
    """Render model accuracy, brier score, and log loss trends."""
    eval_history = load_evaluation_history()

    if not eval_history:
        st.info("No model evaluation history found. Run training scripts to generate evaluation data.")
        return

    st.markdown("### 📊 Model Performance Trends")

    tab1, tab2, tab3 = st.tabs(["📈 Accuracy & Metrics", "🎯 Brier Score", "📉 Log Loss"])

    with tab1:
        fig = plot_accuracy_trend(eval_history)
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        df = pd.DataFrame(eval_history)
        if not df.empty and "accuracy" in df.columns:
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Latest Accuracy", f"{df['accuracy'].iloc[-1]:.1%}" if pd.notna(df['accuracy'].iloc[-1]) else "N/A")
            sc2.metric("Best Accuracy", f"{df['accuracy'].max():.1%}")
            sc3.metric("Worst Accuracy", f"{df['accuracy'].min():.1%}")
            sc4.metric("Avg Accuracy", f"{df['accuracy'].mean():.1%}" if pd.notna(df['accuracy'].mean()) else "N/A")

    with tab2:
        fig = plot_brier_trend(eval_history)
        st.plotly_chart(fig, use_container_width=True)

        df = pd.DataFrame(eval_history)
        if not df.empty and "brier_score" in df.columns and df["brier_score"].notna().any():
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Latest Brier", f"{df['brier_score'].iloc[-1]:.4f}" if pd.notna(df['brier_score'].iloc[-1]) else "N/A")
            sc2.metric("Best Brier", f"{df['brier_score'].min():.4f}")
            sc3.metric("Avg Brier", f"{df['brier_score'].mean():.4f}" if pd.notna(df['brier_score'].mean()) else "N/A")

    with tab3:
        fig = plot_log_loss_trend(eval_history)
        st.plotly_chart(fig, use_container_width=True)

        df = pd.DataFrame(eval_history)
        if not df.empty and "log_loss" in df.columns and df["log_loss"].notna().any():
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Latest Log Loss", f"{df['log_loss'].iloc[-1]:.4f}" if pd.notna(df['log_loss'].iloc[-1]) else "N/A")
            sc2.metric("Best Log Loss", f"{df['log_loss'].min():.4f}")
            sc3.metric("Avg Log Loss", f"{df['log_loss'].mean():.4f}" if pd.notna(df['log_loss'].mean()) else "N/A")


def render_roi_clv_section() -> None:
    """Render ROI by model and CLV comparison side by side."""
    backtest_df = load_backtest_reports()
    clv_df = load_clv_reports()

    st.markdown("### 💰 ROI & CLV by Model")

    col1, col2 = st.columns(2)

    with col1:
        fig_roi = plot_roi_by_model(backtest_df)
        st.plotly_chart(fig_roi, use_container_width=True)

    with col2:
        fig_clv = plot_clv_comparison(clv_df)
        st.plotly_chart(fig_clv, use_container_width=True)

    # Detail tables
    dexp = st.expander("📋 Detailed Metrics Table", expanded=False)
    with dexp:
        if not backtest_df.empty:
            st.markdown("**Backtest / Strategy Metrics**")
            cols = [c for c in ["strategy", "roi_pct", "sharpe_ratio", "max_drawdown_pct",
                                 "total_profit", "total_bets", "win_rate_pct", "profit_factor"]
                     if c in backtest_df.columns]
            display = backtest_df[cols].sort_values("roi_pct", ascending=False) if "roi_pct" in cols else backtest_df
            st.dataframe(
                display.style.format({
                    "roi_pct": "{:+.2f}%",
                    "sharpe_ratio": "{:.2f}",
                    "max_drawdown_pct": "{:.1f}%",
                    "total_profit": "£{:+,.2f}",
                    "win_rate_pct": "{:.1f}%",
                    "profit_factor": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        if not clv_df.empty:
            st.markdown("**CLV Metrics**")
            cols = [c for c in ["model", "avg_clv", "positive_clv_pct", "clv_gt_5_pct", "bets"]
                     if c in clv_df.columns]
            display = clv_df[cols].sort_values("avg_clv", ascending=False) if "avg_clv" in cols else clv_df
            st.dataframe(
                display.style.format({
                    "avg_clv": "{:+.4%}",
                    "positive_clv_pct": "{:.1f}%",
                    "clv_gt_5_pct": "{:.1f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )


def render_bankroll_section() -> None:
    """Render bankroll growth and drawdown monitoring."""
    backtest_df = load_backtest_reports()

    st.markdown("### 🏦 Bankroll & Volume Monitoring")

    col1, col2 = st.columns(2)

    with col1:
        fig_bank = plot_bankroll_growth(backtest_df)
        st.plotly_chart(fig_bank, use_container_width=True)

    with col2:
        live_preds = load_live_predictions()
        fig_vol = plot_bet_frequency(live_preds)
        st.plotly_chart(fig_vol, use_container_width=True)

    # Risk metrics summary
    if not backtest_df.empty:
        st.markdown("### ⚠️ Risk Metrics Summary")
        rc1, rc2, rc3, rc4 = st.columns(4)

        max_dd = backtest_df["max_drawdown_pct"].max() if "max_drawdown_pct" in backtest_df.columns else 0
        avg_dd = backtest_df["max_drawdown_pct"].mean() if "max_drawdown_pct" in backtest_df.columns else 0
        best_sharpe = backtest_df["sharpe_ratio"].max() if "sharpe_ratio" in backtest_df.columns else 0
        total_bets = backtest_df["total_bets"].sum() if "total_bets" in backtest_df.columns else 0

        dd_color = "#4caf50" if max_dd < 10 else "#ffc107" if max_dd < 20 else "#f44336"
        rc1.markdown(
            f'<div class="metric-card"><div class="metric-value" style="color:{dd_color}">'
            f'{max_dd:.1f}%</div><div class="metric-label">Max Drawdown</div></div>',
            unsafe_allow_html=True,
        )
        rc2.markdown(
            f'<div class="metric-card"><div class="metric-value" style="color:{"#4caf50" if avg_dd < 10 else "#ffc107"}">'
            f'{avg_dd:.1f}%</div><div class="metric-label">Avg Drawdown</div></div>',
            unsafe_allow_html=True,
        )
        rc3.markdown(
            f'<div class="metric-card"><div class="metric-value" style="color:{"#4caf50" if best_sharpe >= 1 else "#ffc107"}">'
            f'{best_sharpe:.2f}</div><div class="metric-label">Best Sharpe</div></div>',
            unsafe_allow_html=True,
        )
        rc4.markdown(
            f'<div class="metric-card"><div class="metric-value">'
            f'{total_bets:,}</div><div class="metric-label">Total Bets Tracked</div></div>',
            unsafe_allow_html=True,
        )


def render_live_predictions_section() -> None:
    """Render current live predictions and value bets."""
    live_preds = load_live_predictions()

    st.markdown("### 🔮 Current Live Predictions")

    if not live_preds:
        st.info("No live predictions available. Run the live prediction engine first:\n"
                "`python -c 'from src.live_predictions import live_predictions; print(live_predictions())'`")
        return

    df = pd.DataFrame(live_preds)

    # Sort by best_value_ev descending
    if "best_value_ev" in df.columns:
        df = df.sort_values("best_value_ev", ascending=False)

    # Select display columns
    display_cols = [
        "home_team", "away_team", "match_date",
        "home_prob", "draw_prob", "away_prob",
        "predicted_outcome", "confidence_score",
        "best_value_ev", "n_value_bets", "value_outcomes",
        "home_ev", "draw_ev", "away_ev",
        "home_clv", "draw_clv", "away_clv",
    ]
    available_cols = [c for c in display_cols if c in df.columns]
    display_df = df[available_cols] if available_cols else df

    # Color EV values
    def color_ev(val):
        if isinstance(val, (int, float)):
            return "color: #4caf50" if val > 0 else "color: #f44336" if val < 0 else ""
        return ""

    ev_cols = [c for c in ["best_value_ev", "home_ev", "draw_ev", "away_ev"]
               if c in display_df.columns]
    styled = display_df.style.map(color_ev, subset=ev_cols) if ev_cols else display_df
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
    )

    # Summary row
    n_value = sum(1 for p in live_preds if p.get("n_value_bets", 0) > 0)
    n_high_conf = sum(1 for p in live_preds if p.get("confidence_score", 0) >= 70)
    st.markdown(
        f"<p style='color:#8b8fa3;font-size:0.85rem'>"
        f"{len(live_preds)} matches • {n_value} with value bets • {n_high_conf} high confidence (≥70)</p>",
        unsafe_allow_html=True,
    )


def render_alert_history_section() -> None:
    """Render alert history from the monitoring store."""
    st.markdown("### 📋 Alert History")

    try:
        from src.monitoring.alerting import AlertEngine
        engine = AlertEngine()
        history = engine.get_alert_history(days=7)

        if history:
            for alert in history[:50]:  # Last 50 alerts
                severity = alert.get("severity", "info")
                cls_map = {"critical": "alert-critical", "warning": "alert-warning", "info": "alert-info"}
                cls = cls_map.get(severity, "alert-info")
                st.markdown(
                    f'<div class="{cls}">'
                    f'<strong>{alert.get("message", "")}</strong>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No recent alerts. This is good — all systems operating normally.")
    except Exception as exc:
        st.info(f"Alert history not available: {exc}")


# ═══════════════════════════════════════════════════════════
#  Main render
# ═══════════════════════════════════════════════════════════

def main() -> None:
    """Main dashboard render function."""
    render_header()
    st.markdown("---")
    render_system_status()
    st.markdown("---")
    render_live_metrics_row()
    st.markdown("---")
    render_model_performance_section()
    st.markdown("---")
    render_roi_clv_section()
    st.markdown("---")
    render_bankroll_section()
    st.markdown("---")
    render_live_predictions_section()
    st.markdown("---")
    render_alert_history_section()

    # ── Auto-refresh ────────────────────────────────────
    if st.session_state.auto_refresh:
        interval = st.session_state.refresh_interval
        time_since = (datetime.now() - st.session_state.last_refresh).total_seconds()
        if time_since >= interval:
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#555;font-size:0.8rem'>"
        "Live Performance Monitoring Dashboard | Auto-refresh every "
        f"{st.session_state.refresh_interval}s | "
        f"Built with Streamlit</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
