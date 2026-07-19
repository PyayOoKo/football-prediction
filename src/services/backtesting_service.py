"""
Backtesting Service — orchestrates historical betting simulation.

Handles loading models and data, running backtests, calculating metrics,
and generating reports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.di_container import ConfigProvider, get_container

logger = logging.getLogger(__name__)


class BacktestingService:
    """Service for running betting backtests on historical data.

    Parameters
    ----------
    model_dir : Path, optional
        Directory where trained models are stored. Defaults to
        ``config.paths.models``.
    config : ConfigProvider, optional
        Config provider for dependency injection. Defaults to the
        global container's ConfigProvider.
    """

    def __init__(self, model_dir: Path | None = None, config: ConfigProvider | None = None) -> None:
        self._config = config or get_container().resolve(ConfigProvider)
        self._model_dir = model_dir or self._config.paths.models
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = self._config.paths.reports / "backtest"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────

    def run_backtest(
        self,
        model_name: str | None = None,
        data_path: str | Path | None = None,
        initial_bankroll: float = 1000.0,
        kelly_fraction: float = 0.25,
        min_ev: float = 0.0,
    ) -> dict:
        """Run a complete backtest simulation.

        Loads a trained model and historical data, runs the backtest engine,
        calculates performance metrics, and saves results.

        Parameters
        ----------
        model_name : str, optional
            Model file name within ``models/``. If omitted, uses the
            most recent ``.joblib`` file.
        data_path : str | Path, optional
            Path to preprocessed data CSV. Auto-resolved if omitted.
        initial_bankroll : float
            Starting bankroll in currency units (default 1000).
        kelly_fraction : float
            Fraction of full Kelly to use (default 0.25 = conservative).
        min_ev : float
            Minimum EV threshold for placing bets (default 0.0).

        Returns
        -------
        dict
            Backtest results with metrics, bet records, and output paths.
        """
        from src.backtesting import BacktestEngine
        import joblib

        logger.info("Running backtest (bankroll=%.0f, kelly=%.0f%%)", initial_bankroll, kelly_fraction * 100)

        # ── 1. Load model ────────────────────────────────────
        model = self._load_model(model_name)

        # ── 2. Load & prepare data ───────────────────────────
        data_path = resolve_data_path(data_path, config=self._config)
        
        # For backtesting, we need preprocessed data
        if not data_path.exists():
            # Try processed directory
            processed_path = self._config.paths.processed / "results_clean.csv"
            if processed_path.exists():
                data_path = processed_path
            else:
                raise FileNotFoundError(
                    f"Data not found at {data_path} or {processed_path}. "
                    "Run data collection and preprocessing first."
                )

        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, low_memory=False)

        # ── 3. Prepare features and splits ───────────────────
        from src.feature_engineering import build_features, train_val_test_split
        
        # Add target column if missing
        if "target" not in df.columns:
            result_to_target = {"H": 2, "D": 1, "A": 0}
            df["target"] = df["result"].map(result_to_target).fillna(-1).astype("int8")

        X, y = build_features(df, is_training=True)
        splits = train_val_test_split(X, y)

        X_test = splits["X_test"]
        y_test = splits["y_test"]

        logger.info(f"Backtesting on {len(X_test)} test matches")

        # ── 4. Run backtest engine ───────────────────────────
        engine = BacktestEngine(
            model=model,
            initial_bankroll=initial_bankroll,
            kelly_fraction=kelly_fraction,
            min_ev=min_ev,
        )

        # Get odds from the original dataframe (need to align indices)
        # This requires careful index matching
        odds_cols = ("BbAvA", "BbAvD", "BbAvH")
        team_cols = ("home_team", "away_team")
        
        # Check if we have odds in the original df
        has_odds = all(col in df.columns for col in odds_cols)
        
        if has_odds:
            # Need to align odds with test set - this is tricky
            # For now, pass the full df and let the engine handle it
            odds_df = df.copy()
        else:
            odds_df = None

        metrics = engine.run(
            X_test=X_test,
            y_test=y_test,
            odds_df=odds_df,
            odds_cols=odds_cols,
            team_cols=team_cols,
        )

        # ── 5. Generate report ───────────────────────────────
        report = {
            "metrics": {
                "roi": metrics.roi_pct,
                "yield_pct": metrics.yield_pct,
                "profit": metrics.total_profit,
                "total_bets": metrics.total_bets,
                "winning_bets": metrics.winning_bets,
                "win_rate": metrics.win_rate_pct,
                "avg_stake": metrics.total_staked / max(metrics.total_bets, 1),
                "max_drawdown": metrics.max_drawdown_pct,
                "final_bankroll": metrics.final_bankroll,
            },
            "parameters": {
                "model_name": model_name or "latest",
                "initial_bankroll": initial_bankroll,
                "kelly_fraction": kelly_fraction,
                "min_ev": min_ev,
                "test_matches": len(X_test),
            },
        }

        # ── 6. Save results ──────────────────────────────────
        output_path = self._output_dir / "backtest_results.csv"
        if engine._bets:
            bets_df = pd.DataFrame([
                {
                    "match_index": b.match_index,
                    "match_label": b.match_label,
                    "outcome_bet": b.outcome_bet,
                    "outcome_actual": b.outcome_actual,
                    "decimal_odds": b.decimal_odds,
                    "model_prob": b.model_prob,
                    "fair_prob": b.fair_prob,
                    "ev": b.ev,
                    "stake_pct": b.stake_pct,
                    "stake_amount": b.stake_amount,
                    "profit": b.profit,
                    "won": b.won,
                    "bankroll_after": b.bankroll_after,
                }
                for b in engine._bets
            ])
            bets_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(bets_df)} bet records to {output_path}")
            report["bet_records_path"] = str(output_path)

        # Generate plots
        try:
            plot_dir = self._output_dir / "plots"
            plot_dir.mkdir(exist_ok=True)
            engine.plot_results(output_dir=str(plot_dir))
            report["plots_path"] = str(plot_dir)
        except Exception as exc:
            logger.warning(f"Failed to generate plots: {exc}")

        # Print summary
        self._print_summary(report)

        return report

    def _load_model(self, model_name: str | None = None) -> Any:
        """Load a trained model from disk."""
        import joblib

        if model_name:
            model_path = self._model_dir / model_name
            if not model_path.exists():
                model_path = self._model_dir / f"{model_name}.joblib"
        else:
            # Find most recent model
            model_files = list(self._model_dir.glob("*.joblib"))
            if not model_files:
                raise FileNotFoundError(
                    f"No trained models found in {self._model_dir}. "
                    "Train a model first."
                )
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)

        logger.info(f"Loading model from {model_path}")
        return joblib.load(model_path)

    def _print_summary(self, report: dict) -> None:
        """Print a formatted backtest summary."""
        m = report["metrics"]
        p = report["parameters"]

        print("\n" + "=" * 70)
        print("  BACKTEST RESULTS".center(68))
        print("=" * 70)
        print(f"\n  Model: {p['model_name']}")
        print(f"  Test matches: {p['test_matches']}")
        print(f"  Total bets: {m['total_bets']}")
        print(f"\n  Performance:")
        print(f"    ROI:          {m['roi']:+.2f}%")
        print(f"    Yield:        {m['yield']:+.2f}%")
        print(f"    Profit:       {m['profit']:+.2f}")
        print(f"    Win rate:     {m['win_rate']:.1f}%")
        print(f"    Max drawdown: {m['max_drawdown']:.2f}%")
        print(f"\n  Bankroll: {p['initial_bankroll']:.0f} → {m['final_bankroll']:.2f}")
        print("=" * 70 + "\n")
