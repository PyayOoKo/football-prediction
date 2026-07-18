"""
Tests for ``dashboard/performance_monitoring.py`` — data extraction helpers,
snapshot builder, alert evaluation, and chart generators.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════
#  _extract_clv_from_json
# ═══════════════════════════════════════════════════════════

class TestExtractClvFromJson:
    """Tests for the recursive CLV extraction helper."""

    def test_per_model_list(self, sample_clv_report_data: dict) -> None:
        """Should extract per-model CLV results from 'clv_values' key."""
        from dashboard.performance_monitoring import _extract_clv_from_json

        rows: list[dict] = []
        _extract_clv_from_json(sample_clv_report_data, "test.json", rows)

        assert len(rows) == 3
        models = {r["model"] for r in rows}
        assert models == {"XGBoost", "LightGBM", "Ensemble"}
        ensemble = next(r for r in rows if r["model"] == "Ensemble")
        assert ensemble["avg_clv"] == 0.015
        assert ensemble["bets"] == 300

    def test_single_result(self, sample_clv_single_result: dict) -> None:
        """Should extract a single aggregate CLV result."""
        from dashboard.performance_monitoring import _extract_clv_from_json

        rows: list[dict] = []
        _extract_clv_from_json(sample_clv_single_result, "clv_report.json", rows)

        assert len(rows) == 1
        assert rows[0]["model"] == "Aggregate"
        assert rows[0]["avg_clv"] == 0.0105
        assert rows[0]["bets"] == 650

    def test_empty_dict(self) -> None:
        """Should produce no rows for an empty dict."""
        from dashboard.performance_monitoring import _extract_clv_from_json

        rows: list[dict] = []
        _extract_clv_from_json({}, "empty.json", rows)
        assert len(rows) == 0

    def test_list_data(self) -> None:
        """Should handle list data gracefully (bails early)."""
        from dashboard.performance_monitoring import _extract_clv_from_json

        rows: list[dict] = []
        _extract_clv_from_json([], "list.json", rows)
        assert len(rows) == 0

    def test_fallback_to_single_result(self) -> None:
        """Should try single result extraction as last resort."""
        from dashboard.performance_monitoring import _extract_clv_from_json

        data = {"avg_clv": 0.02, "model_name": "TestModel", "bets": 10}
        rows: list[dict] = []
        _extract_clv_from_json(data, "test.json", rows)

        assert len(rows) == 1
        assert rows[0]["avg_clv"] == 0.02

    def test_partial_data_no_bets(self) -> None:
        """Should handle data where 'bets' key uses alternative names."""
        from dashboard.performance_monitoring import _extract_clv_from_json

        data = {"model": "Test", "clv": 0.01, "n_bets": 50}
        rows: list[dict] = []
        _extract_clv_from_json(data, "test.json", rows)

        assert len(rows) >= 1
        assert rows[0]["model"] == "Test"


# ═══════════════════════════════════════════════════════════
#  _extract_backtest_metrics
# ═══════════════════════════════════════════════════════════

class TestExtractBacktestMetrics:
    """Tests for the recursive backtest metric extraction helper."""

    def test_best_strategy_format(self, sample_backtest_best_strategy: dict) -> None:
        """Should extract from 'best_strategy' format."""
        from dashboard.performance_monitoring import _extract_backtest_metrics

        rows: list[dict] = []
        _extract_backtest_metrics(sample_backtest_best_strategy, "backtest.json", rows)

        assert len(rows) == 1
        assert rows[0]["strategy"] == "Kelly 25%"
        assert rows[0]["roi_pct"] == 8.5
        assert rows[0]["sharpe_ratio"] == 1.2

    def test_strategy_comparison_format(
        self, sample_backtest_strategy_comparison: dict,
    ) -> None:
        """Should extract from stake_strategies.results format."""
        from dashboard.performance_monitoring import _extract_backtest_metrics

        rows: list[dict] = []
        _extract_backtest_metrics(
            sample_backtest_strategy_comparison, "comparison.json", rows,
        )

        assert len(rows) == 3
        strategies = {r["strategy"] for r in rows}
        assert strategies == {"Full Kelly", "Half Kelly", "Quarter Kelly"}
        quarter = next(r for r in rows if r["strategy"] == "Quarter Kelly")
        assert quarter["sharpe_ratio"] == 1.2

    def test_direct_metrics_format(self, sample_backtest_metrics_direct: dict) -> None:
        """Should extract from direct 'metrics' key format."""
        from dashboard.performance_monitoring import _extract_backtest_metrics

        rows: list[dict] = []
        _extract_backtest_metrics(sample_backtest_metrics_direct, "metrics.json", rows)

        assert len(rows) == 1
        assert rows[0]["roi_pct"] == 5.2
        assert rows[0]["total_bets"] == 200

    def test_empty_data(self) -> None:
        """Should produce no rows for empty dict."""
        from dashboard.performance_monitoring import _extract_backtest_metrics

        rows: list[dict] = []
        _extract_backtest_metrics({}, "empty.json", rows)
        assert len(rows) == 0


# ═══════════════════════════════════════════════════════════
#  build_performance_snapshot
# ═══════════════════════════════════════════════════════════

class TestBuildPerformanceSnapshot:
    """Tests for building the metric snapshot dict."""

    def test_all_metrics_provided(self) -> None:
        """Should include all metrics in the snapshot."""
        from dashboard.performance_monitoring import build_performance_snapshot

        snapshot = build_performance_snapshot(
            accuracy=0.72,
            brier_score=0.20,
            log_loss=0.89,
            roi_pct=5.5,
            avg_clv=0.01,
            win_rate_pct=53.0,
            sharpe_ratio=1.1,
            max_drawdown_pct=12.5,
            bankroll_change_pct=2.0,
            bets_per_day=3.5,
            avg_ev=0.05,
            avg_confidence=0.65,
        )

        perf = snapshot["performance"]
        assert perf["accuracy"] == 0.72
        assert perf["brier_score"] == 0.20
        assert perf["log_loss"] == 0.89
        assert perf["roi_pct"] == 5.5
        assert perf["avg_clv"] == 0.01
        assert perf["sharpe_ratio"] == 1.1
        assert perf["avg_confidence"] == 0.65

    def test_partial_metrics(self) -> None:
        """Should only include provided metrics, skip None."""
        from dashboard.performance_monitoring import build_performance_snapshot

        snapshot = build_performance_snapshot(
            accuracy=0.70,
            brier_score=None, log_loss=None,
            roi_pct=3.0, avg_clv=None,
            win_rate_pct=52.0, sharpe_ratio=None,
            max_drawdown_pct=10.0,
            bankroll_change_pct=None, bets_per_day=None,
            avg_ev=None, avg_confidence=None,
        )

        perf = snapshot["performance"]
        assert "accuracy" in perf
        assert "brier_score" not in perf
        assert "log_loss" not in perf
        assert "avg_clv" not in perf
        assert "sharpe_ratio" not in perf

    def test_empty_snapshot(self) -> None:
        """Should return a dict with empty performance when all None."""
        from dashboard.performance_monitoring import build_performance_snapshot

        snapshot = build_performance_snapshot(
            accuracy=None, brier_score=None, log_loss=None,
            roi_pct=None, avg_clv=None, win_rate_pct=None,
            sharpe_ratio=None, max_drawdown_pct=None,
            bankroll_change_pct=None, bets_per_day=None,
            avg_ev=None, avg_confidence=None,
        )

        assert snapshot["performance"] == {}


# ═══════════════════════════════════════════════════════════
#  evaluate_performance_alerts
# ═══════════════════════════════════════════════════════════

class TestEvaluatePerformanceAlerts:
    """Tests for alert evaluation against performance snapshot."""

    def test_returns_list_of_dicts(self) -> None:
        """Should always return a list of dicts (never None)."""
        from dashboard.performance_monitoring import evaluate_performance_alerts

        snapshot = {"performance": {"accuracy": 0.50}}
        alerts = evaluate_performance_alerts(snapshot)
        assert isinstance(alerts, list)

    def test_handles_engine_unavailable(self) -> None:
        """Should gracefully return info alert when AlertEngine can't be imported."""
        from dashboard.performance_monitoring import evaluate_performance_alerts

        # Patch sys.modules to make src.monitoring.alerting unavailable
        with patch.dict("sys.modules", {"src.monitoring.alerting": None}):
            snapshot = {"performance": {"accuracy": 0.50}}
            alerts = evaluate_performance_alerts(snapshot)

        assert isinstance(alerts, list)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "info"
        assert "engine_error" in alerts[0]["rule_name"]


# ═══════════════════════════════════════════════════════════
#  Chart generator functions (return Plotly Figure objects)
# ═══════════════════════════════════════════════════════════

class TestPlotFunctions:
    """Tests that plot functions return valid Plotly Figure objects."""

    def test_plot_accuracy_trend_empty(self) -> None:
        """Should return a figure even with empty data."""
        from dashboard.performance_monitoring import plot_accuracy_trend

        fig = plot_accuracy_trend([])
        assert fig is not None
        # Should have "No data" annotation when no traces exist
        if len(fig.data) == 0:
            assert any("No data" in str(a.text) for a in fig.layout.annotations)

    def test_plot_accuracy_trend_with_data(self) -> None:
        """Should plot accuracy over time from evaluation history."""
        from dashboard.performance_monitoring import plot_accuracy_trend

        eval_history = [
            {"accuracy": 0.65, "datetime": "2026-07-01", "timestamp": 1000},
            {"accuracy": 0.70, "datetime": "2026-07-02", "timestamp": 2000},
        ]
        fig = plot_accuracy_trend(eval_history)
        assert fig is not None
        assert len(fig.data) >= 1

    def test_plot_brier_trend_empty(self) -> None:
        """Should return figure with 'No data' when no Brier scores exist."""
        from dashboard.performance_monitoring import plot_brier_trend

        fig = plot_brier_trend([{"accuracy": 0.7, "datetime": "2026-07-01"}])
        assert fig is not None

    def test_plot_brier_trend_with_data(self) -> None:
        """Should plot Brier Score trend."""
        from dashboard.performance_monitoring import plot_brier_trend

        eval_history = [
            {"brier_score": 0.22, "datetime": "2026-07-01"},
            {"brier_score": 0.20, "datetime": "2026-07-02"},
        ]
        fig = plot_brier_trend(eval_history)
        assert fig is not None
        assert len(fig.data) >= 1

    def test_plot_log_loss_with_data(self) -> None:
        """Should plot Log Loss trend."""
        from dashboard.performance_monitoring import plot_log_loss_trend

        eval_history = [
            {"log_loss": 1.05, "datetime": "2026-07-01"},
            {"log_loss": 0.95, "datetime": "2026-07-02"},
        ]
        fig = plot_log_loss_trend(eval_history)
        assert fig is not None

    def test_plot_roi_by_model_empty(self) -> None:
        """Should handle empty DataFrame gracefully."""
        from dashboard.performance_monitoring import plot_roi_by_model

        fig = plot_roi_by_model(pd.DataFrame())
        assert fig is not None
        # Should have "No backtest data" annotation
        if len(fig.data) == 0:
            assert any("No backtest data" in str(a.text) for a in fig.layout.annotations)

    def test_plot_clv_comparison_empty(self) -> None:
        """Should handle empty DataFrame gracefully."""
        from dashboard.performance_monitoring import plot_clv_comparison

        fig = plot_clv_comparison(pd.DataFrame())
        assert fig is not None

    def test_plot_clv_comparison_with_data(self) -> None:
        """Should plot CLV comparison with model data."""
        from dashboard.performance_monitoring import plot_clv_comparison

        clv_df = pd.DataFrame({
            "model": ["XGBoost", "LightGBM"],
            "avg_clv": [0.012, 0.008],
            "positive_clv_pct": [55.0, 52.0],
            "bets": [200, 150],
        })
        fig = plot_clv_comparison(clv_df)
        assert fig is not None
        assert len(fig.data) >= 1

    def test_plot_bet_frequency_empty(self) -> None:
        """Should handle empty list gracefully."""
        from dashboard.performance_monitoring import plot_bet_frequency

        fig = plot_bet_frequency([])
        assert fig is not None

    def test_plot_figure_layout_properties(self) -> None:
        """Figure should have expected paper_bgcolor."""
        from dashboard.performance_monitoring import plot_accuracy_trend

        fig = plot_accuracy_trend([
            {"accuracy": 0.65, "datetime": "2026-07-01", "timestamp": 1000},
        ])
        assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
