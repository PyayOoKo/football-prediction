"""
Executive Dashboard — single-page summary of all key system metrics.

Surfaces the most important data from the continuous improvement report
into a single scrollable view: system health, model performance, betting
performance, infrastructure maturity, feature completion, roadmap, and
key watchlist metrics.

Usage
-----
    streamlit run dashboard/executive_dashboard.py
    python -m streamlit run dashboard/executive_dashboard.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════
#  DATA — sourced from reports/continuous_improvement_20260715.md
#  Refresh this section when the report is regenerated.
# ═══════════════════════════════════════════════════════════

SYSTEM_STATS = {
    "Source Files": "120",
    "Lines of Code": "46,500",
    "Test Files": "60",
    "Test Cases": "1,269 (97.9% passing)",
    "Model Types": "7",
    "Features": "134",
    "DB Tables": "22",
    "Dashboard Pages": "7",
}

MODEL_ACCURACY = {
    "Ensemble (Stacking)": (0.62, 0.68),
    "Ensemble (Weighted)": (0.61, 0.66),
    "XGBoost (tuned)": (0.60, 0.65),
    "LightGBM (World Cup 2026)": (0.71, 0.74),
    "LightGBM": (0.59, 0.64),
    "Random Forest": (0.55, 0.60),
    "Logistic Regression": (0.52, 0.57),
    "Poisson Model": (0.50, 0.55),
}

CALIBRATION_METRICS = {
    "Brier Score": {"target": 0.20, "current": 0.20, "unit": "", "lower_better": True},
    "Log Loss": {"target": 1.10, "current": 1.10, "unit": "", "lower_better": True},
    "ECE": {"target": 0.05, "current": 0.045, "unit": "", "lower_better": True},
}

BETTING_METRICS = {
    "ROI": {"target": 5.0, "current": 5.0, "unit": "%", "lower_better": False},
    "Sharpe Ratio": {"target": 1.0, "current": 1.0, "unit": "", "lower_better": False},
    "Win Rate": {"target": 50.0, "current": 51.5, "unit": "%", "lower_better": False},
    "Max Drawdown": {"target": 20.0, "current": 16.5, "unit": "%", "lower_better": True},
    "Profit Factor": {"target": 1.5, "current": 1.45, "unit": "", "lower_better": False},
}

DATA_SOURCES = [
    ("football-data.co.uk", "Match results + odds", "✅"),
    ("The Odds API", "Live odds (15+ sports)", "✅"),
    ("StatsBomb", "xG, events, lineups", "✅"),
    ("Transfermarkt", "Player transfers, squads", "✅"),
    ("OpenWeatherMap", "Weather data", "✅"),
    ("FBref", "Referee statistics", "✅"),
    ("Understat", "xG data (leagues)", "✅"),
    ("openfootball", "World Cup data", "✅"),
]

PHASES = [
    ("Phase 1", "Prediction Engine", 7, 7, [
        "PredictionEngine", "CLV Calculator", "CLV Tracker",
        "CLV Comparison", "CLV Visualizations", "CLV Report",
        "Live Prediction Engine",
    ]),
    ("Phase 2", "Betting System", 6, 6, [
        "Stake Sizing", "Calibration", "Value Betting",
        "Odds API", "Odds Processing", "Confidence Scoring",
    ]),
    ("Phase 3", "Bankroll Management", 6, 6, [
        "Risk Manager", "Bankroll Backtesting", "Bankroll Report",
        "Kelly Criterion", "Bet Filtering", "Bankroll Manager",
    ]),
    ("Phase 4", "Automation", 7, 7, [
        "Daily Data Pipeline", "Daily Feature Computation",
        "Model Retraining", "Daily Predictions", "Auto-Commit",
        "Scheduler", "Windows Task Integration",
    ]),
    ("Phase 5", "CLI & API", 5, 5, [
        "CLI (predict/backtest/train)", "REST API (FastAPI)",
        "Desktop App", "Web Dashboard (Streamlit)",
        "Performance Dashboard",
    ]),
    ("Phase 6", "Monitoring & Alerting", 5, 5, [
        "Monitoring Framework", "Alert Engine (40+ rules)",
        "Monitor Class", "Report Generator",
        "Dashboard Pages (5)",
    ]),
]

INFRASTRUCTURE = [
    ("Database", 6, 6, ["22 Tables", "SQLAlchemy ORM", "Alembic Migrations",
                         "PostgreSQL Ready", "Partitioning", "Indexing"]),
    ("ETL Pipeline", 6, 6, ["Track", "Extract", "Transform",
                              "Normalize", "Validate", "Store"]),
    ("Feature Store", 5, 5, ["Registry", "Computation Engine",
                               "Lineage", "Cache", "Validation"]),
    ("Monitoring", 5, 5, ["MonitoringStore", "Monitor", "AlertEngine",
                            "Performance Alerts", "System Collectors"]),
    ("Scheduling", 5, 5, ["TaskEngine", "6 Scheduled Tasks",
                            "Retry Logic", "Logging", "Windows Integration"]),
]

ROADMAP = [
    ("Phase 7 — Production Hardening", "🥇", [
        ("DB migration to PostgreSQL", "2 days", "High"),
        ("Model versioning with MLflow", "1 day", "High"),
        ("CI/CD pipeline (GitHub Actions)", "1 day", "Medium"),
        ("Docker Compose with Postgres", "0.5 day", "Medium"),
        ("Auto-generated API docs", "0.5 day", "Low"),
    ]),
    ("Phase 8 — Advanced Analytics", "📊", [
        ("Social media sentiment analysis", "2 days", "Medium"),
        ("Injury prediction model", "3 days", "Medium"),
        ("Head-to-head extended stats", "1 day", "Low"),
        ("Formation analysis", "2 days", "Low"),
    ]),
    ("Phase 9 — Scale", "🚀", [
        ("Load testing (1000s concurrent)", "1 day", "Medium"),
        ("Query optimization", "1 day", "Medium"),
        ("CDN for static reports", "0.5 day", "Low"),
    ]),
    ("Phase 10 — Community", "🌍", [
        ("Open-source docs site (MkDocs)", "1 day", "High"),
        ("Contributor guide", "0.5 day", "Medium"),
        ("Model zoo (pre-trained models)", "1 day", "Medium"),
        ("Web demo (hosted Streamlit)", "1 day", "High"),
    ]),
]

WATCHLIST_METRICS = [
    ("Model Accuracy", "< 50%", "< 40%", "Re-train with latest data"),
    ("Brier Score", "> 0.30", "> 0.40", "Re-calibrate model"),
    ("ROI", "< -5%", "< -15%", "Re-evaluate betting strategy"),
    ("Avg CLV", "< -1%", "< -3%", "Check odds source"),
    ("Max Drawdown", "> 20%", "> 35%", "Reduce stake sizes"),
    ("Sharpe Ratio", "< 0.0", "< -0.5", "Stop betting, full review"),
]

LESSONS = [
    ("📊 Ensemble over single model", "Combining 5 models outperforms any single model by 2–4%. StackingEnsemble with OOF meta-training is the best approach."),
    ("🔧 Calibration is essential", "Uncalibrated probabilities are misleading for value betting. Platt scaling improves Brier score by 15–25%."),
    ("⏰ Time-series validation catches leakage", "Standard K-fold CV overestimates accuracy by 5–10%. Walk-forward validation reveals true OOS performance."),
    ("📈 CLV is a leading indicator", "Models with positive CLV consistently outperform. CLV provides early warning before accuracy drops."),
    ("🏦 Bankroll > Accuracy", "A 52% model with 2% Kelly outperforms a 58% model with full Kelly. Fractional Kelly (k=0.15–0.25) is optimal."),
    ("🗂️ Feature Store eliminates recomputation", "Caching features reduced pipeline runtime by 70%. Lineage tracking saved hours of debugging."),
]

DI_MIGRATION = {
    "Batch 1 — Core Pipeline + Services": {"files": 6, "done": 6, "functions": [
        "train_model", "tune_hyperparameters", "save_model",
        "load_model", "build_features", "train_val_test_split",
        "evaluate_model", "resolve_data_path", "load_and_prepare",
        "TrainingService", "PredictionService", "services/__init__",
    ]},
    "Batch 2a — Prediction & Value": {"files": 3, "done": 3, "functions": [
        "predict_fixtures", "compute_value_bets",
        "load_results", "load_fixtures", "load_teams",
    ]},
    "Batch 2b — Preprocessing & CV": {"files": 4, "done": 4, "functions": [
        "run_preprocessing", "time_series_train_val_test_split",
        "create_time_series_folds", "DataLoader.__init__",
        "DataPreprocessor.__init__",
    ]},
    "Batch 2c — Feature Modules": {"files": 4, "done": 4, "functions": [
        "_encode_categoricals", "_add_rolling_features",
        "_add_extended_h2h_features", "_add_extended_form_features",
        "_add_weather_features", "_add_referee_features",
        "_add_schedule_features", "_add_transfer_features",
        "_add_attack_defence_ratios",
    ]},
    "Batch 2d — Data Collection": {"files": 5, "done": 5, "functions": [
        "collect_worldcup", "collect_all", "collect_league", "update",
        "download_bulk", "_session", "collect_weather",
        "_build_placeholder_df",
    ]},
    "Batch 3 — Remaining 10 Modules": {"files": 10, "done": 10, "functions": [
        "cli.py: _handle_train", "eda.py: run_eda",
        "factory.py: create_ensemble, create_default",
        "app/utils.py (9 refs)", "app/dashboard.py (5 refs)",
        "app/pages/4_WorldCup.py",
        "Removed dead config imports: live_predictions.py, 3_Backtest.py, backtesting/__init__.py, feature_store/computation.py",
    ]},
}

DI_MIGRATION_TOTAL = {
    "files_refactored": 32,
    "files_total": 32,
    "call_sites_scanned": 182,
    "call_site_issues": 0,
    "config_keyword_calls": 21,
}

RECENT_FIXES = [
    ("ScheduleTransformer index reset", "TypeError: cannot use 'tuple' as dict key"
     " — duplicate index caused .at[] to return unhashable Series",
     "src/feature_framework/features/schedule.py"),
    ("Rolling features dedup",
     "LightGBM Fatal: duplicate feature h_days_since_last_match"
     " — pd.concat created duplicate columns",
     "src/features/rolling.py"),
    ("tune_hyperparameters signature",
     "Got unexpected keyword argument 'model_type' after DI migration",
     "train_worldcup.py"),
    ("find_value_bets.py argparse",
     "ValueError: badly formed help string — %K treated as format spec",
     "find_value_bets.py"),
    ("conftest.py SyntaxError",
     "Unterminated string: docstring closing adjacent to from __future__",
     "tests/test_dashboard/conftest.py"),
    ("Dead code removal",
     "FeatureEngineer stub with TODO — 0 imports, removed safely",
     "src/data/feature_engineering.py"),
    ("sklearn 1.9 compat: multi_class removed",
     "LogisticRegression.__init__() got unexpected keyword 'multi_class'"
     " — removed from 3 call sites across src + tests",
     "run_wc_ensemble.py, src/ensemble.py, tests"),
]

TECH_DEBT = [
    ("train_worldcup.py hardcoded paths", "Medium", "30 min", "✅ Done"),
    ("feature_engineering.py 1,700+ lines → src/features/", "High", "4 hr", "✅ Done"),
    ("Dashboard page test coverage low", "Medium", "3 hr", "✅ Done — 44 tests across 4 files"),
    ("No integration tests for live predictions", "Medium", "2 hr", "✅ Done — 12 tests covering full pipeline"),
    ("Global config imports → DI migration", "Low", "8 hr", "✅ Done — 32/32 src/ modules refactored"),
    ("Schedule features: 'tuple as dict key' TypeError", "High", "30 min", "✅ Fixed — unconditional index reset in ScheduleTransformer"),
    ("Rolling features: duplicate columns via pd.concat", "High", "30 min", "✅ Fixed — dedup check before concat in _merge_team_stats"),
    ("train_worldcup.py: tune_hyperparameters kwarg mismatch", "Medium", "10 min", "✅ Fixed — removed stale model_type= kwarg"),
    ("find_value_bets.py: argparse %K format crash", "Medium", "5 min", "✅ Fixed — escaped % in help string"),
    ("tests/test_dashboard/conftest.py: SyntaxError", "High", "5 min", "✅ Fixed — newline between docstring and future import"),
    ("src/data/feature_engineering.py: dead TODO stub", "Low", "5 min", "✅ Removed — 0 imports, 53 lines of dead code"),
    ("sklearn 1.9.0: remove deprecated multi_class param", "Medium", "15 min",
     "✅ Fixed — 3 call sites updated (run_wc_ensemble.py, src/ensemble.py, test helper)"),
]


# ═══════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════

st.markdown("""
<style>
    .stApp { background: #0a0d14; }
    .stApp header { background: #11141e; }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #11141e 0%, #0f1928 50%, #11141e 100%);
        border: 1px solid #1e2235;
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(79, 195, 247, 0.05) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        background: linear-gradient(90deg, #4fc3f7, #81c784, #ffd54f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        color: #6b7280;
        font-size: 1rem;
        margin: 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(79, 195, 247, 0.12);
        border: 1px solid rgba(79, 195, 247, 0.25);
        color: #4fc3f7;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e0e0e0;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1e2235;
    }
    .section-header-small {
        font-size: 1.1rem;
        font-weight: 600;
        color: #c0c0c0;
        margin: 1.5rem 0 0.8rem 0;
    }

    /* ── Metric cards ── */
    .stat-card {
        background: linear-gradient(135deg, #141824 0%, #1a1f2e 100%);
        border: 1px solid #1e2235;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        transition: transform 0.2s, border-color 0.2s;
        height: 100%;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        border-color: #4fc3f7;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #fff;
        line-height: 1.1;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }

    /* ── Gauge cards ── */
    .gauge-card {
        background: linear-gradient(135deg, #141824 0%, #1a1f2e 100%);
        border: 1px solid #1e2235;
        border-radius: 14px;
        padding: 1rem 1.2rem;
    }
    .gauge-value {
        font-size: 1.6rem;
        font-weight: 700;
    }
    .gauge-label {
        font-size: 0.7rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .gauge-target {
        font-size: 0.7rem;
        color: #4fc3f7;
    }

    /* ── Progress bars ── */
    .progress-container {
        background: #1e2235;
        border-radius: 10px;
        height: 10px;
        overflow: hidden;
        margin: 0.3rem 0 0.5rem 0;
    }
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease;
    }

    /* ── Info row ── */
    .info-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0;
        color: #9ca3af;
        font-size: 0.85rem;
    }

    /* ── Lesson card ── */
    .lesson-card {
        background: linear-gradient(135deg, #111824 0%, #16202e 100%);
        border: 1px solid #1a2d4a;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .lesson-title { font-weight: 600; color: #e0e0e0; font-size: 0.95rem; }
    .lesson-text { color: #9ca3af; font-size: 0.85rem; margin-top: 0.3rem; line-height: 1.4; }

    /* ── Status dots ── */
    .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
    .dot-green { background: #4caf50; }
    .dot-yellow { background: #ffc107; }
    .dot-red { background: #f44336; }
    .dot-blue { background: #4fc3f7; }

    /* ── Tags ── */
    .tag {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .tag-green { background: rgba(76, 175, 80, 0.15); color: #81c784; }
    .tag-yellow { background: rgba(255, 193, 7, 0.15); color: #ffd54f; }
    .tag-blue { background: rgba(79, 195, 247, 0.15); color: #4fc3f7; }
    .tag-red { background: rgba(244, 67, 54, 0.15); color: #ef9a9a; }

    /* ── Watchlist table ── */
    .wl-row {
        display: grid;
        grid-template-columns: 1.5fr 1fr 1fr 2fr;
        padding: 0.5rem 0.8rem;
        border-bottom: 1px solid #1a1f2e;
        font-size: 0.85rem;
        align-items: center;
    }
    .wl-header {
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 0.06em;
        border-bottom: 1px solid #2a2d3a;
    }
    .wl-metric { color: #e0e0e0; font-weight: 500; }
    .wl-warn { color: #ffc107; }
    .wl-crit { color: #f44336; }

    /* ── Phase card ── */
    .phase-card {
        background: #141824;
        border: 1px solid #1e2235;
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    .phase-name { font-size: 0.85rem; font-weight: 700; color: #4fc3f7; }
    .phase-title { font-size: 1.1rem; font-weight: 700; color: #fff; }
    .phase-progress { font-size: 0.8rem; color: #6b7280; }

    /* ── Debt row ── */
    .debt-row {
        display: grid;
        grid-template-columns: 2fr 0.8fr 0.8fr 0.8fr;
        padding: 0.4rem 0;
        font-size: 0.85rem;
        color: #9ca3af;
        border-bottom: 1px solid #1a1f2e;
        align-items: center;
    }

    @media (max-width: 768px) {
        .hero { padding: 1.5rem; }
        .stat-number { font-size: 1.4rem; }
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  RENDER HELPERS
# ═══════════════════════════════════════════════════════════

def stat_card(number: str, label: str, col) -> None:
    col.markdown(
        f'<div class="stat-card">'
        f'<div class="stat-number">{number}</div>'
        f'<div class="stat-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def gauge_card(value: float, target: float, label: str, unit: str,
               lower_better: bool, col) -> None:
    pct = (value / target) * 100 if target > 0 else 0
    if lower_better:
        met = value <= target
        color = "#4caf50" if met else "#ffc107" if value <= target * 1.5 else "#f44336"
        display_pct = min(100, (target / max(value, 0.001)) * 100) if value > 0 else 100
    else:
        met = value >= target
        color = "#4caf50" if met else "#ffc107" if value >= target * 0.7 else "#f44336"
        display_pct = min(100, (value / target) * 100) if target > 0 else 0

    col.markdown(
        f'<div class="gauge-card">'
        f'<div class="gauge-value" style="color:{color}">{value}{unit}</div>'
        f'<div class="gauge-label">{label}</div>'
        f'<div class="gauge-target">Target: {target}{unit}</div>'
        f'<div class="progress-container">'
        f'<div class="progress-fill" style="width:{display_pct:.0f}%;background:{color}"></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════
#  SECTIONS
# ═══════════════════════════════════════════════════════════

def render_hero() -> None:
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("<h1>👑 Executive Dashboard</h1>", unsafe_allow_html=True)
        st.markdown(
            '<p class="hero-sub">Single-page summary of all key system metrics, '
            'model performance, betting analytics, infrastructure maturity, '
            'and strategic roadmap.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span class="hero-badge">📅 {datetime.now().strftime("%B %d, %Y")}</span>'
            f'<span class="hero-badge" style="margin-left:8px">📊 v0.1.0</span>'
            f'<span class="hero-badge" style="margin-left:8px">⚡ Auto-refresh: 120s</span>',
            unsafe_allow_html=True,
        )
    with c2:
        total_features = sum(n for _, _, n, _, _ in PHASES)
        completed = sum(n for _, _, _, n, _ in PHASES)
        overall_pct = completed / total_features * 100
        st.markdown(
            f'<div style="text-align:right;padding-top:0.5rem">'
            f'<div style="font-size:2.5rem;font-weight:800;color:#81c784">'
            f'{overall_pct:.0f}%</div>'
            f'<div style="color:#6b7280;font-size:0.8rem">Overall Completion</div>'
            f'<div style="color:#6b7280;font-size:0.7rem">{completed}/{total_features} features shipped</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


def render_system_health() -> None:
    st.markdown('<div class="section-header">🔋 System Health</div>', unsafe_allow_html=True)

    cols = st.columns(8)
    items = list(SYSTEM_STATS.items())
    for i, (label, value) in enumerate(items):
        with cols[i]:
            stat_card(value, label, cols[i])


def render_model_performance() -> None:
    st.markdown('<div class="section-header">🤖 Model Performance</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header-small">Accuracy by Model Type</div>', unsafe_allow_html=True)

        model_names = list(MODEL_ACCURACY.keys())
        lows = [v[0] * 100 for v in MODEL_ACCURACY.values()]
        highs = [v[1] * 100 for v in MODEL_ACCURACY.values()]
        mids = [(l + h) / 2 for l, h in zip(lows, highs)]
        colors = ["#4fc3f7" if m >= 60 else "#81c784" if m >= 55 else "#ffd54f" if m >= 50 else "#ef9a9a"
                  for m in mids]

        fig = go.Figure()
        for i, (name, low, high, mid, color) in enumerate(zip(model_names, lows, highs, mids, colors)):
            fig.add_trace(go.Bar(
                name=name,
                y=[name],
                x=[mid],
                orientation="h",
                marker=dict(color=color, line=dict(color="#fff", width=0)),
                hovertemplate=f"<b>{name}</b><br>Range: {low:.0f}%–{high:.0f}%<extra></extra>",
                showlegend=False,
                width=0.6,
            ))
            # Error bar for range
            fig.add_trace(go.Scatter(
                x=[low, high],
                y=[name, name],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.3)", width=2),
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[low, high],
                y=[name, name],
                mode="markers",
                marker=dict(color="white", size=4, symbol="diamond"),
                showlegend=False,
            ))

        fig.add_vline(x=50, line_dash="dash", line_color="#ffc107", line_width=0.8,
                      annotation_text="Baseline (50%)")

        fig.update_layout(
            xaxis=dict(range=[40, 75], tickformat=".0f", title="Accuracy (%)"),
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc", size=11),
            margin=dict(l=10, r=20, t=10, b=10),
            height=280,
            bargap=0.3,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header-small">Calibration Quality</div>', unsafe_allow_html=True)
        for name, data in CALIBRATION_METRICS.items():
            gauge_card(
                value=data["current"],
                target=data["target"],
                label=name,
                unit=data["unit"],
                lower_better=data["lower_better"],
                col=st,
            )
            st.markdown("<br>", unsafe_allow_html=True)


def render_betting_performance() -> None:
    st.markdown('<div class="section-header">💰 Betting Performance</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    for i, (name, data) in enumerate(BETTING_METRICS.items()):
        with cols[i]:
            gauge_card(
                value=data["current"],
                target=data["target"],
                label=name,
                unit=data["unit"],
                lower_better=data["lower_better"],
                col=cols[i],
            )


def render_feature_completion() -> None:
    st.markdown('<div class="section-header">📋 Feature Completion by Phase</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for i, (phase, title, total, done, features) in enumerate(PHASES):
        with cols[i % 3]:
            pct = done / total * 100
            color = "#4caf50" if pct >= 100 else "#ffc107"

            st.markdown(
                f'<div class="phase-card">'
                f'<div class="phase-name">{phase}</div>'
                f'<div class="phase-title">{title}</div>'
                f'<div class="phase-progress">{done}/{total} features · '
                f'<span style="color:{color};font-weight:600">{pct:.0f}%</span></div>'
                f'<div class="progress-container">'
                f'<div class="progress-fill" style="width:{pct}%;background:{color}"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for feat in features:
                st.markdown(f'<div class="info-row">✅ {feat}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


def render_data_sources() -> None:
    st.markdown('<div class="section-header">🔌 Data Sources</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    for i, (name, coverage, status) in enumerate(DATA_SOURCES):
        with cols[i % 4]:
            st.markdown(
                f'<div class="stat-card">'
                f'<div style="font-size:1rem;font-weight:600;color:#e0e0e0">{name}</div>'
                f'<div style="font-size:0.75rem;color:#6b7280;margin-top:0.2rem">{coverage}</div>'
                f'<div style="margin-top:0.4rem"><span class="tag tag-green">'
                f'{status} Active</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_infrastructure_maturity() -> None:
    st.markdown('<div class="section-header">🏗️ Infrastructure Maturity</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    for i, (name, total, done, components) in enumerate(INFRASTRUCTURE):
        with cols[i]:
            pct = done / total * 100
            color = "#4caf50" if pct >= 100 else "#ffc107"
            st.markdown(
                f'<div class="phase-card">'
                f'<div class="phase-title" style="font-size:0.95rem">{name}</div>'
                f'<div class="phase-progress">{done}/{total} components</div>'
                f'<div class="progress-container">'
                f'<div class="progress-fill" style="width:{pct}%;background:{color}"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for comp in components:
                st.markdown(f'<div class="info-row">✅ {comp}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


def render_watchlist() -> None:
    st.markdown('<div class="section-header">⚠️ Key Metrics Watchlist</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="wl-row wl-header">'
        '<span>Metric</span><span>Warning</span><span>Critical</span><span>Action</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    for metric, warn, crit, action in WATCHLIST_METRICS:
        st.markdown(
            f'<div class="wl-row">'
            f'<span class="wl-metric">{metric}</span>'
            f'<span class="wl-warn">{warn}</span>'
            f'<span class="wl-crit">{crit}</span>'
            f'<span style="color:#9ca3af">{action}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_roadmap() -> None:
    st.markdown('<div class="section-header">🗺️ Strategic Roadmap</div>', unsafe_allow_html=True)

    for phase_name, icon, tasks in ROADMAP:
        st.markdown(f'<div class="section-header-small">{icon} {phase_name}</div>', unsafe_allow_html=True)

        task_cols = st.columns(len(tasks))
        for i, (task, effort, impact) in enumerate(tasks):
            with task_cols[i]:
                impact_color = {"High": "#4fc3f7", "Medium": "#ffc107", "Low": "#6b7280",
                                "🔴 High risk": "#f44336"}.get(impact, "#6b7280")
                st.markdown(
                    f'<div class="stat-card" style="padding:0.8rem 1rem">'
                    f'<div style="font-size:0.85rem;font-weight:500;color:#e0e0e0;min-height:2.5rem">'
                    f'{task}</div>'
                    f'<div style="margin-top:0.4rem">'
                    f'<span class="tag tag-yellow">{effort}</span> '
                    f'<span class="tag" style="background:rgba({int(impact_color[1:3],16)},{int(impact_color[3:5],16)},{int(impact_color[5:7],16)},0.15);color:{impact_color}">'
                    f'{impact}</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def render_lessons() -> None:
    st.markdown('<div class="section-header">💡 Lessons Learned</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for i, (title, text) in enumerate(LESSONS):
        with cols[i % 3]:
            st.markdown(
                f'<div class="lesson-card">'
                f'<div class="lesson-title">{title}</div>'
                f'<div class="lesson-text">{text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_tech_debt() -> None:
    st.markdown('<div class="section-header">🧹 Technical Debt</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="debt-row" style="color:#6b7280;font-weight:600;text-transform:uppercase;font-size:0.7rem;letter-spacing:0.06em">'
        '<span>Item</span><span>Severity</span><span>Effort</span><span>Status</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    for item, severity, effort, status in TECH_DEBT:
        sev_color = {"High": "#f44336", "Medium": "#ffc107", "Low": "#6b7280"}.get(severity, "#6b7280")
        st.markdown(
            f'<div class="debt-row">'
            f'<span style="color:#e0e0e0">{item}</span>'
            f'<span style="color:{sev_color}">{severity}</span>'
            f'<span>{effort}</span>'
            f'<span>{status}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_recent_fixes() -> None:
    st.markdown('<div class="section-header">🐛 Recent Bug Fixes</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for i, (title, desc, file) in enumerate(RECENT_FIXES):
        with cols[i % 3]:
            st.markdown(
                f'<div class="lesson-card" style="padding:0.8rem 1rem">'
                f'<div class="lesson-title" style="font-size:0.9rem">{title}</div>'
                f'<div class="lesson-text" style="font-size:0.8rem">{desc}</div>'
                f'<div style="margin-top:0.4rem">'
                f'<span class="tag tag-green" style="font-size:0.65rem">{file}</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_di_migration() -> None:
    st.markdown('<div class="section-header">📊 Dependency Injection Migration</div>', unsafe_allow_html=True)

    total_pct = (
        DI_MIGRATION_TOTAL["files_refactored"]
        / DI_MIGRATION_TOTAL["files_total"]
        * 100
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        stat_card(f"{DI_MIGRATION_TOTAL['files_refactored']}/{DI_MIGRATION_TOTAL['files_total']}",
                  "Modules Refactored", col1)
    with col2:
        stat_card(f"{DI_MIGRATION_TOTAL['call_sites_scanned']}",
                  "Call Sites Scanned", col2)
    with col3:
        stat_card(f"{DI_MIGRATION_TOTAL['call_site_issues']}",
                  "Backward Compat Issues", col3)
    with col4:
        stat_card(f"{DI_MIGRATION_TOTAL['config_keyword_calls']}",
                  "Config= Keyword Calls", col4)

    st.markdown(
        f'<div style="margin:1rem 0 0.5rem 0">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.8rem">'
        f'<span style="color:#6b7280">Overall migration progress</span>'
        f'<span style="color:#81c784;font-weight:600">{total_pct:.0f}%</span>'
        f'</div>'
        f'<div class="progress-container" style="height:12px">'
        f'<div class="progress-fill" style="width:{total_pct}%;background:linear-gradient(90deg,#4fc3f7,#81c784)"></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for i, (batch_name, data) in enumerate(DI_MIGRATION.items()):
        with cols[i % 3]:
            pct = data["done"] / data["files"] * 100 if data["files"] > 0 else 0
            is_done = pct >= 100
            color = "#4caf50" if is_done else "#ffc107"
            batch_label = batch_name.replace("Batch ", "B")
            st.markdown(
                f'<div class="phase-card">'
                f'<div class="phase-name">{batch_name}</div>'
                f'<div class="phase-progress">{data["done"]}/{data["files"]} files '
                f'<span style="color:{color};font-weight:600">{"DONE" if is_done else f"{pct:.0f}%"}</span></div>'
                f'<div class="progress-container">'
                f'<div class="progress-fill" style="width:{pct}%;background:{color}"></div>'
                f'</div>'
                f'<div style="margin-top:0.5rem">',
                unsafe_allow_html=True,
            )
            for fn in data["functions"]:
                status = "✅" if is_done else "⬜"
                st.markdown(
                    f'<div class="info-row" style="font-size:0.75rem;padding:0.15rem 0">'
                    f'{status} {fn}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div style="color:#555;font-size:0.75rem">'
            '👑 Executive Dashboard · v0.1.0</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div style="color:#555;font-size:0.75rem;text-align:center">'
            f'Generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div style="color:#555;font-size:0.75rem;text-align:right">'
            'Built with Streamlit + Plotly</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main() -> None:
    render_hero()
    render_system_health()
    render_model_performance()
    render_betting_performance()
    render_feature_completion()
    render_data_sources()
    render_infrastructure_maturity()
    render_watchlist()
    render_roadmap()
    render_lessons()
    render_di_migration()
    render_recent_fixes()
    render_tech_debt()
    render_footer()

    # Auto-refresh every 120s
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if (
        datetime.now() - st.session_state.last_refresh
    ).total_seconds() > 120:
        st.session_state.last_refresh = datetime.now()
        st.rerun()


if __name__ == "__main__":
    main()
