"""
Shared fixtures for dashboard tests.

Provides sample JSON report data that matches the various report
formats consumed by the dashboard pages: CLV reports, backtest
reports, validation reports, and bankroll optimisation reports.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


# ═══════════════════════════════════════════════════════════
#  Streamlit mocking — prevents StreamlitAPIException when
#  importing dashboard modules in a test context.
#  Applied automatically to every test in this package.
# ═══════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _mock_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock all Streamlit functions to prevent crashes in pytest.

    Dashboard modules call ``st.set_page_config()``, ``st.markdown()``,
    etc. at module-import time, which raises ``StreamlitAPIException``
    outside a Streamlit runtime. This fixture patches every relevant
    ``streamlit`` attribute before any dashboard import occurs.
    """
    mock_st = MagicMock()
    monkeypatch.setattr("streamlit.set_page_config", mock_st)
    monkeypatch.setattr("streamlit.markdown", mock_st)
    monkeypatch.setattr("streamlit.dataframe", mock_st)
    monkeypatch.setattr("streamlit.plotly_chart", mock_st)
    monkeypatch.setattr("streamlit.selectbox", mock_st)
    monkeypatch.setattr("streamlit.multiselect", mock_st)
    monkeypatch.setattr("streamlit.slider", mock_st)
    monkeypatch.setattr("streamlit.checkbox", mock_st)
    monkeypatch.setattr("streamlit.tabs", mock_st)
    monkeypatch.setattr("streamlit.columns", lambda *args, **kwargs: [MagicMock() for _ in range(args[0] if isinstance(args[0], int) else len(args[0]) if args else 2)])
    monkeypatch.setattr("streamlit.button", mock_st)
    monkeypatch.setattr("streamlit.metric", mock_st)
    monkeypatch.setattr("streamlit.info", mock_st)
    monkeypatch.setattr("streamlit.warning", mock_st)
    monkeypatch.setattr("streamlit.error", mock_st)
    monkeypatch.setattr("streamlit.success", mock_st)
    monkeypatch.setattr("streamlit.stop", mock_st)
    monkeypatch.setattr("streamlit.spinner", lambda *a, **kw: MagicMock().__enter__())
    monkeypatch.setattr("streamlit.download_button", mock_st)
    monkeypatch.setattr("streamlit.expander", lambda *a, **kw: MagicMock().__enter__())
    monkeypatch.setattr("streamlit.json", mock_st)
    monkeypatch.setattr("streamlit.cache_data", lambda **kw: lambda f: f)
    monkeypatch.setattr("streamlit.session_state", MagicMock())
    monkeypatch.setattr("streamlit.rerun", mock_st)
    monkeypatch.setattr("streamlit.date_input", mock_st)
    monkeypatch.setattr("streamlit.number_input", mock_st)


@pytest.fixture
def sample_clv_report_data() -> dict[str, Any]:
    """CLV report with per-model results (most common format)."""
    return {
        "clv_values": [
            {"model": "XGBoost", "clv": 0.012, "positive_clv_pct": 55.0, "clv_gt_5_pct": 12.0, "bets": 200},
            {"model": "LightGBM", "clv": 0.008, "positive_clv_pct": 52.0, "clv_gt_5_pct": 8.0, "bets": 150},
            {"model": "Ensemble", "clv": 0.015, "positive_clv_pct": 58.0, "clv_gt_5_pct": 15.0, "bets": 300},
        ],
    }


@pytest.fixture
def sample_clv_single_result() -> dict[str, Any]:
    """CLV report with a single aggregate result."""
    return {
        "model": "Aggregate",
        "avg_clv": 0.0105,
        "positive_clv_pct": 53.2,
        "clv_gt_5_pct": 10.0,
        "bets": 650,
    }


@pytest.fixture
def sample_backtest_best_strategy() -> dict[str, Any]:
    """Backtest report with best_strategy format."""
    return {
        "best_strategy": {
            "strategy": "Kelly 25%",
            "roi_pct": 8.5,
            "sharpe_ratio": 1.2,
            "max_drawdown_pct": 12.0,
            "total_profit": 425.0,
            "total_bets": 250,
            "win_rate_pct": 53.5,
            "profit_factor": 1.45,
        },
    }


@pytest.fixture
def sample_backtest_strategy_comparison() -> dict[str, Any]:
    """Backtest report with strategy comparison format."""
    return {
        "stake_strategies": {
            "results": [
                {"strategy": "Full Kelly", "roi_pct": 15.0, "sharpe_ratio": 0.8,
                 "max_drawdown_pct": 35.0, "total_profit": 750.0, "total_bets": 250,
                 "win_rate_pct": 53.5, "profit_factor": 1.2},
                {"strategy": "Half Kelly", "roi_pct": 10.0, "sharpe_ratio": 1.1,
                 "max_drawdown_pct": 18.0, "total_profit": 500.0, "total_bets": 250,
                 "win_rate_pct": 53.5, "profit_factor": 1.35},
                {"strategy": "Quarter Kelly", "roi_pct": 8.5, "sharpe_ratio": 1.2,
                 "max_drawdown_pct": 12.0, "total_profit": 425.0, "total_bets": 250,
                 "win_rate_pct": 53.5, "profit_factor": 1.45},
            ],
        },
    }


@pytest.fixture
def sample_validation_report() -> dict[str, Any]:
    """Validation/performance report with metrics and confusion matrix."""
    return {
        "file": "phase4_validation_20260715.json",
        "type": "json",
        "metrics": {
            "accuracy": 0.726,
            "test_accuracy": 0.726,
            "log_loss": 0.887,
            "brier_score": 0.195,
            "confusion_matrix": [[18, 5, 2], [4, 10, 6], [1, 2, 25]],
            "class_labels": ["Away Win", "Draw", "Home Win"],
        },
        "feature_importance": {
            "elo_rating_diff": 0.152,
            "rolling_goals_scored": 0.098,
            "h2h_win_rate": 0.075,
        },
    }


@pytest.fixture
def sample_bankroll_report() -> dict[str, Any]:
    """Bankroll optimisation report."""
    return {
        "best_strategy": {
            "strategy": "Optimal Kelly 20%",
            "sharpe_ratio": 1.35,
            "total_profit": 1230.50,
            "max_drawdown_pct": 8.5,
            "win_rate": 54.0,
            "total_bets": 500,
            "roi": 12.3,
            "profit_factor": 1.55,
            "bankroll_history": [1000, 1025, 1050, 1030, 1080, 1060, 1100, 1090, 1120, 1150],
        },
    }


@pytest.fixture
def sample_backtest_metrics_direct() -> dict[str, Any]:
    """Backtest report with direct metrics format."""
    return {
        "metrics": {
            "roi_pct": 5.2,
            "sharpe_ratio": 0.95,
            "max_drawdown_pct": 15.0,
            "total_profit": 260.0,
            "total_bets": 200,
            "win_rate_pct": 51.0,
            "profit_factor": 1.3,
        },
    }


@pytest.fixture
def sample_predictions_csv_data() -> list[dict[str, Any]]:
    """Sample prediction data as loaded from CSV."""
    return [
        {"date": "2026-07-14", "home_team": "Brazil", "away_team": "Norway",
         "home_win_prob": 0.55, "draw_prob": 0.25, "away_win_prob": 0.20,
         "prediction": "Home Win", "confidence": 0.55},
        {"date": "2026-07-14", "home_team": "France", "away_team": "Morocco",
         "home_win_prob": 0.45, "draw_prob": 0.30, "away_win_prob": 0.25,
         "prediction": "Home Win", "confidence": 0.45},
        {"date": "2026-07-15", "home_team": "Argentina", "away_team": "Egypt",
         "home_win_prob": 0.70, "draw_prob": 0.18, "away_win_prob": 0.12,
         "prediction": "Home Win", "confidence": 0.70},
    ]
