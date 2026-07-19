"""
Tests for dashboard sub-page helpers — metric extraction, data loading
patterns, and report parsing used across the 5 Streamlit sub-pages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════
#  Betting Results page helpers
# ═══════════════════════════════════════════════════════════

class TestBettingResultsExtractMetrics:
    """Tests for the ``extract_metrics`` function in 3_Betting_Results.py.

    Uses ``importlib.import_module()`` because the file name starts with
    a digit (``3_Betting_Results.py``), which is not a valid Python
    identifier and cannot be imported via a standard ``import`` statement.
    """

    @pytest.fixture(autouse=True)
    def _import_extract(self) -> None:
        """Import extract_metrics from the betting results page.

        Uses AST to extract just the function without running the
        module-level code (which has side effects like loading report files).
        """
        import ast
        from pathlib import Path

        source_path = (
            Path(__file__).resolve().parent.parent.parent
            / "dashboard" / "pages" / "3_Betting_Results.py"
        )
        source = source_path.read_text(encoding="utf-8")

        # Parse AST and find extract_metrics function
        tree = ast.parse(source, str(source_path))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "extract_metrics":
                # Compile and execute just this function
                func_code = compile(
                    ast.Module(body=[node], type_ignores=[]),
                    str(source_path), "exec",
                )
                ns: dict[str, Any] = {"__builtins__": __builtins__}
                exec(func_code, ns)
                self.extract_metrics = ns["extract_metrics"]
                return

        pytest.skip("extract_metrics function not found in source")

    def test_metrics_format(self, sample_backtest_best_strategy: dict) -> None:
        """Should extract from 'metrics' key directly."""
        m = self.extract_metrics(sample_backtest_best_strategy)
        assert isinstance(m, dict)

    def test_stake_strategies_format(
        self, sample_backtest_strategy_comparison: dict,
    ) -> None:
        """Should extract best strategy from stake_strategies format."""
        m = self.extract_metrics(sample_backtest_strategy_comparison)
        assert "sharpe_ratio" in m
        # Quarter Kelly has the highest Sharpe, so it should be selected
        assert m.get("sharpe_ratio") == 1.2

    def test_direct_results_list(self) -> None:
        """Should extract from 'results' list."""
        data = {"results": [{"roi": 5.0, "total_bets": 100}]}
        m = self.extract_metrics(data)
        assert m.get("roi") == 5.0

    def test_fallback_to_full_dict(self) -> None:
        """Should fall back to the full dict if no known format matches."""
        data = {"custom_metric": 42, "label": "test"}
        m = self.extract_metrics(data)
        assert m.get("custom_metric") == 42

    def test_empty_dict(self) -> None:
        """Should handle empty dict gracefully."""
        m = self.extract_metrics({})
        assert isinstance(m, dict)
        assert len(m) == 0


# ═══════════════════════════════════════════════════════════
#  CLV data loading helpers
# ═══════════════════════════════════════════════════════════

class TestCLVReportParsing:
    """Tests for CLV data extraction patterns used in 4_CLV_Tracking.py."""

    def test_json_per_model_extraction(self, sample_clv_report_data: dict) -> None:
        """Should parse per-model CLV from 'clv_values' list in JSON."""
        data = sample_clv_report_data
        all_clvs: list[dict] = []

        # Replicate the logic from the dashboard page
        if isinstance(data, dict):
            for key in ["clv_values", "clv_per_bet", "results", "models"]:
                items = data.get(key, [])
                if items:
                    for item in items:
                        if isinstance(item, dict):
                            all_clvs.append({
                                "model": item.get("model", item.get("model_name", "Unknown")),
                                "clv": item.get("clv", item.get("avg_clv", 0)),
                                "positive_clv_pct": item.get("positive_clv_pct", 0),
                                "bets": item.get("bets", item.get("n_bets", 0)),
                            })
                    break

        assert len(all_clvs) == 3
        assert all_clvs[0]["model"] == "XGBoost"
        assert all_clvs[0]["clv"] == 0.012

    def test_csv_clv_parsing(self) -> None:
        """Should extract CLV metrics from CSV-style dict rows."""
        csv_rows = [
            {"model": "Ensemble", "avg_clv": "0.015", "positive_clv_pct": "58.0", "bets": "300"},
            {"model": "XGBoost", "avg_clv": "0.012", "positive_clv_pct": "55.0", "bets": "200"},
        ]

        all_clvs = []
        for row in csv_rows:
            all_clvs.append({
                "model": row.get("model", "Unknown"),
                "clv": float(row.get("avg_clv", row.get("clv", 0))),
                "positive_clv_pct": float(row.get("positive_clv_pct", 0)),
                "bets": int(row.get("bets", row.get("n_bets", 0))),
            })

        assert len(all_clvs) == 2
        assert all_clvs[0]["clv"] == 0.015


# ═══════════════════════════════════════════════════════════
#  Bankroll Monitoring helpers
# ═══════════════════════════════════════════════════════════

class TestBankrollReportParsing:
    """Tests for bankroll report parsing patterns from 5_Bankroll_Monitoring.py."""

    def test_best_strategy_extraction(self, sample_bankroll_report: dict) -> None:
        """Should extract best_strategy from bankroll optimisation report."""
        data = sample_bankroll_report
        assert "best_strategy" in data
        bs = data["best_strategy"]
        assert bs["strategy"] == "Optimal Kelly 20%"
        assert bs["sharpe_ratio"] == 1.35
        assert len(bs["bankroll_history"]) == 10

    def test_bankroll_history_length(self, sample_bankroll_report: dict) -> None:
        """Bankroll history should be a list of floats/ints."""
        history = sample_bankroll_report["best_strategy"]["bankroll_history"]
        assert isinstance(history, list)
        assert all(isinstance(v, (int, float)) for v in history)
        assert history[0] == 1000
        assert history[-1] == 1150

    def test_strategy_format_parsing(self, sample_backtest_strategy_comparison: dict) -> None:
        """Should extract strategies from stake_strategies/risk_scenarios sections."""
        data = sample_backtest_strategy_comparison
        strategies = []

        for section in ["stake_strategies", "risk_scenarios"]:
            if section in data:
                strategies = data[section].get("results", [])

        assert len(strategies) == 3
        assert all(s.get("total_bets", 0) > 0 for s in strategies)


# ═══════════════════════════════════════════════════════════
#  Prediction History helpers
# ═══════════════════════════════════════════════════════════

class TestPredictionHistoryHelpers:
    """Tests for prediction history data handling (2_Prediction_History.py)."""

    def test_team_column_detection(self) -> None:
        """Should detect team columns by keyword matching."""
        cols = ["date", "home_team", "away_team", "home_win_prob", "draw_prob"]
        team_cols = [c for c in cols if any(kw in c.lower() for kw in ["team", "opponent"])]
        assert team_cols == ["home_team", "away_team"]

    def test_probability_column_detection(self) -> None:
        """Should detect probability columns by keyword matching."""
        cols = ["date", "home_team", "away_team", "home_win_prob", "confidence"]
        prob_cols = [c for c in cols if any(
            kw in c.lower() for kw in ["prob", "confidence", "prediction"]
        )]
        assert prob_cols == ["home_win_prob", "confidence"]

    def test_date_column_detection(self) -> None:
        """Should detect date columns by keyword matching."""
        cols = ["date", "home_team", "away_team", "result"]
        date_cols = [c for c in cols if any(kw in c.lower() for kw in ["date", "time", "day"])]
        assert date_cols == ["date"]

    def test_prediction_file_loading_pattern(self, tmp_path: Path) -> None:
        """Should load CSV prediction files from configured directory."""
        # Create a sample predictions CSV
        csv_content = (
            "date,home_team,away_team,home_win_prob,prediction\n"
            "2026-07-14,Brazil,Norway,0.55,Home Win\n"
            "2026-07-14,France,Morocco,0.45,Draw\n"
        )
        pred_file = tmp_path / "worldcup_predictions.csv"
        pred_file.write_text(csv_content)

        # Test the loading pattern from the dashboard
        import pandas as pd
        df = pd.read_csv(pred_file)
        assert len(df) == 2
        assert list(df.columns) == ["date", "home_team", "away_team", "home_win_prob", "prediction"]
        assert df.iloc[0]["home_team"] == "Brazil"

    def test_team_filter_logic(self, sample_predictions_csv_data: list[dict]) -> None:
        """Should correctly filter predictions by team."""
        df = pd.DataFrame(sample_predictions_csv_data)

        # Replicate the filtering logic from the page
        team_cols = ["home_team", "away_team"]
        selected_team = "Brazil"
        mask = pd.Series(False, index=df.index)
        for col in team_cols:
            mask |= df[col].astype(str).str.contains(selected_team, case=False, na=False)
        filtered = df[mask]

        assert len(filtered) == 1
        assert filtered.iloc[0]["home_team"] == "Brazil"


# ═══════════════════════════════════════════════════════════
#  Model Performance helpers
# ═══════════════════════════════════════════════════════════

class TestModelPerformanceHelpers:
    """Tests for model performance report parsing (1_Model_Performance.py)."""

    def test_validation_report_metric_extraction(
        self, sample_validation_report: dict,
    ) -> None:
        """Should extract metrics from validation report at various nesting levels."""
        data = sample_validation_report

        # Replicate the extraction logic from the page
        metrics = {}
        if "metrics" in data:
            metrics = data["metrics"]
        if "overall" in data:
            metrics.update(data["overall"])
        for key in ["accuracy", "log_loss", "brier_score", "roc_auc", "f1"]:
            if key in data:
                metrics[key] = data[key]

        assert metrics["accuracy"] == 0.726
        assert metrics["log_loss"] == 0.887
        assert metrics["brier_score"] == 0.195

    def test_confusion_matrix_extraction(self, sample_validation_report: dict) -> None:
        """Should extract confusion matrix from nested or flat structure."""
        data = sample_validation_report
        cm = data.get("confusion_matrix") or data.get("metrics", {}).get("confusion_matrix")
        assert cm is not None
        assert len(cm) == 3
        assert len(cm[0]) == 3
        # Total should equal sum of all cells
        assert sum(sum(row) for row in cm) == 73

    def test_feature_importance_parsing(self, sample_validation_report: dict) -> None:
        """Should parse feature importance dict and sort by value."""
        data = sample_validation_report
        fi = data.get("feature_importance") or data.get("metrics", {}).get("feature_importance", {})
        assert isinstance(fi, dict)
        sorted_fi = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)
        assert sorted_fi[0][0] == "elo_rating_diff"
        assert sorted_fi[0][1] == 0.152

    def test_json_report_loading(self, tmp_path: Path) -> None:
        """Should load JSON validation reports."""
        import json

        report = tmp_path / "validation_report.json"
        report.write_text(json.dumps({
            "metrics": {"accuracy": 0.72, "log_loss": 0.89},
        }))

        with open(report) as f:
            data = json.load(f)
        assert data["metrics"]["accuracy"] == 0.72
