"""
Command-line interface for the Football Prediction System.

Provides subcommands for training, prediction, data collection,
betting analysis, evaluation, and system management.

Usage:
    football-predict train           Train a model
    football-predict predict         Generate predictions
    football-predict evaluate        Evaluate model performance
    football-predict collect         Download match data
    football-predict backtest        Run backtesting
    football-predict dashboard       Launch the dashboard
    football-predict api             Start the REST API
    football-predict desktop         Launch the desktop app
    football-predict pipeline        Run the full pipeline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import NoReturn

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="football-predict",
        description="Football Prediction System — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Show version and exit",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── train ────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Train a prediction model")
    train_parser.add_argument(
        "--model-type", default=None,
        choices=["xgboost", "lightgbm", "logistic_regression", "random_forest", "neural_network"],
        help="Model type to train (default: config setting)",
    )
    train_parser.add_argument(
        "--tune", action="store_true",
        help="Run hyper-parameter tuning before training",
    )
    train_parser.add_argument(
        "--save", action="store_true", default=True,
        help="Save the trained model (default: True)",
    )

    # ── predict ──────────────────────────────────────────
    pred_parser = subparsers.add_parser("predict", help="Generate match predictions")
    pred_parser.add_argument(
        "--fixtures", type=str, default=None,
        help="Path to fixtures CSV (default: config setting)",
    )
    pred_parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for predictions CSV",
    )
    pred_parser.add_argument(
        "--model", type=str, default=None,
        help="Model file to use (default: latest from config)",
    )

    # ── collect ──────────────────────────────────────────
    collect_parser = subparsers.add_parser("collect", help="Download match data")
    collect_parser.add_argument(
        "--source", type=str, default="all",
        choices=["all", "worldcup", "leagues", "odds", "players"],
        help="Data source to collect",
    )

    # ── backtest ─────────────────────────────────────────
    bt_parser = subparsers.add_parser("backtest", help="Run betting backtest")
    bt_parser.add_argument(
        "--model", type=str, default=None,
        help="Model to backtest",
    )
    bt_parser.add_argument(
        "--output", type=str, default="reports/backtest",
        help="Output directory for results",
    )

    # ── dashboard ────────────────────────────────────────
    subparsers.add_parser("dashboard", help="Launch the Streamlit dashboard")
    subparsers.add_parser("dashboard-monitor", help="Launch the monitoring dashboard")

    # ── api ──────────────────────────────────────────────
    api_parser = subparsers.add_parser("api", help="Start the REST API server")
    api_parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to bind (default: 8000)",
    )
    api_parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)",
    )

    # ── desktop ──────────────────────────────────────────
    subparsers.add_parser("desktop", help="Launch the desktop application")

    # ── evaluate ────────────────────────────────────────
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    eval_parser.add_argument(
        "--model", type=str, default=None,
        help="Path or name of the model to evaluate (default: latest)",
    )
    eval_parser.add_argument(
        "--test-data", type=str, default=None,
        help="Path to test data CSV (default: config setting)",
    )
    eval_parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for evaluation report",
    )
    eval_parser.add_argument(
        "--plot", action="store_true", default=True,
        help="Generate performance plots (default: True)",
    )

    # ── pipeline ─────────────────────────────────────────
    # ── run-all ──────────────────────────────────────────
    run_all_parser = subparsers.add_parser("run-all", help="Run the complete pipeline: collect → train → predict → value bets → dashboard")
    run_all_parser.add_argument(
        "--skip-collect", action="store_true",
        help="Skip data collection",
    )
    run_all_parser.add_argument(
        "--skip-value-bets", action="store_true",
        help="Skip value bets",
    )
    run_all_parser.add_argument(
        "--skip-dashboard", action="store_true",
        help="Don't open dashboard",
    )
    run_all_parser.add_argument(
        "--predict-only", action="store_true",
        help="Quick: predict only",
    )
    run_all_parser.add_argument(
        "--model", default="lgb", choices=["lgb", "xgb", "lr", "rf"],
        help="Model type (default: lgb — LightGBM)",
    )

    # ── pipeline ────────────────────────────────────────────
    pipe_parser = subparsers.add_parser("pipeline", help="Run the full pipeline")
    pipe_parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip data download",
    )
    pipe_parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip model retraining",
    )
    pipe_parser.add_argument(
        "--lightweight", action="store_true",
        help="Skip download + train (predict only)",
    )

    args = parser.parse_args(argv)

    if args.version:
        from src import __version__
        print(f"football-predict v{__version__}")
        return 0

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if not args.command:
        parser.print_help()
        return 0

    # Route to handler
    command_map = {
        "train": _handle_train,
        "predict": _handle_predict,
        "collect": _handle_collect,
        "backtest": _handle_backtest,
        "dashboard": _handle_dashboard,
        "dashboard-monitor": _handle_dashboard_monitor,
        "evaluate": _handle_evaluate,
        "api": _handle_api,
        "desktop": _handle_desktop,
        "pipeline": _handle_pipeline,
        "run-all": _handle_run_all,
    }

    handler = command_map.get(args.command)
    if handler:
        return handler(args)
    return 1


# ── Command handlers ─────────────────────────────────────


def _handle_train(args: argparse.Namespace) -> int:
    """Handle the ``train`` subcommand."""
    logger.info("Training model...")
    try:
        if args.model_type:
            from config import config
            config.train.model_type = args.model_type

        if args.tune:
            from src.preprocessing import load_preprocessed
            from src.feature_engineering import build_features
            df = load_preprocessed()
            X, y = build_features(df, is_training=True)
            from src.train import tune_hyperparameters
            best_params = tune_hyperparameters(X, y)
            print(f"Best params: {best_params}")

        from run_pipeline import main as pipeline_main
        return pipeline_main(["--skip-download"])

    except ImportError as exc:
        logger.error("Training failed: %s", exc)
        print(f"  Error: {exc}")
        print("  Ensure required packages are installed: pip install xgboost lightgbm torch")
        return 1
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        return 1


def _handle_predict(args: argparse.Namespace) -> int:
    """Handle the ``predict`` subcommand."""
    logger.info("Generating predictions...")
    try:
        from run_pipeline import main as pipeline_main
        return pipeline_main(["--skip-download", "--skip-train"])
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        return 1


def _handle_collect(args: argparse.Namespace) -> int:
    """Handle the ``collect`` subcommand."""
    source = args.source
    logger.info("Collecting data: %s", source)
    try:
        if source == "worldcup" or source == "all":
            print("  Collecting World Cup data...")
            from collect_all_worldcups import main as wc_main
            wc_main()
        if source == "leagues" or source == "all":
            print("  Collecting league data...")
            from collect_leagues import main as league_main
            league_main()
        if source == "players" or source == "all":
            print("  Collecting player data...")
            from collect_player_data import main as player_main
            player_main()
        return 0
    except Exception as exc:
        logger.error("Data collection failed: %s", exc)
        return 1


def _handle_backtest(args: argparse.Namespace) -> int:
    """Handle the ``backtest`` subcommand."""
    logger.info("Running backtest...")
    try:
        from run_backtest import main as bt_main
        bt_main()
        return 0
    except Exception as exc:
        logger.error("Backtest failed: %s", exc)
        return 1


def _handle_evaluate(args: argparse.Namespace) -> int:
    """Handle the ``evaluate`` subcommand.

    Loads a trained model and evaluates it on test data, reporting
    accuracy, log-loss, Brier score, confusion matrix, and per-class
    metrics. Optionally generates performance plots.
    """
    logger.info("Evaluating model performance...")
    try:
        # Load model
        model_path = args.model
        model = None
        model_name = "unknown"

        import joblib
        from pathlib import Path

        if model_path:
            p = Path(model_path)
            if p.exists():
                model = joblib.load(p)
                model_name = p.stem
                logger.info("Loaded model from: %s", p)

        # If no explicit path, use the PredictionEngine's auto-detect
        if model is None:
            from src.prediction_engine import ModelLoader
            model, meta = ModelLoader.load()
            model_name = meta.get("name", "auto")
            if model is None:
                print("  No trained model found. Train a model first:")
                print("    football-predict train")
                return 1

        # Load test data
        test_path = args.test_data
        import pandas as pd

        df = None
        if test_path:
            t = Path(test_path)
            if t.exists():
                df = pd.read_csv(t, low_memory=False)
                logger.info("Loaded test data: %s (%d rows)", t, len(df))

        if df is None:
            from src.data_loader import load_clean_data
            df = load_clean_data()
            if df is None or df.empty:
                processed = Path("data/processed/results_clean.csv")
                if processed.exists():
                    df = pd.read_csv(processed, low_memory=False)

        if df is None or df.empty:
            print("  No test data available. Collect data first:")
            print("    football-predict collect")
            return 1

        # Build features if the model expects them
        has_proba = hasattr(model, "predict_proba")
        has_predict = hasattr(model, "predict")

        if not has_proba and not has_predict:
            print(f"  Model '{model_name}' has no predict or predict_proba method")
            return 1

        metrics = {}

        if has_proba:
            # Try feature-based evaluation
            try:
                from src.feature_engineering import build_features, train_val_test_split

                X, y = build_features(df, is_training=True)
                splits = train_val_test_split(X, y)

                X_test = splits["X_test"]
                y_test = splits["y_test"]

                probs = model.predict_proba(X_test)
                preds = model.predict(X_test) if has_predict else probs.argmax(axis=1)

                from sklearn.metrics import (
                    accuracy_score, log_loss, brier_score_loss,
                    classification_report, confusion_matrix,
                )

                metrics["accuracy"] = float(accuracy_score(y_test, preds))
                metrics["log_loss"] = float(log_loss(y_test, probs))
                metrics["brier_score"] = float(
                    sum(brier_score_loss(y_test == c, probs[:, i])
                        for i, c in enumerate(range(probs.shape[1])))
                ) / probs.shape[1]

                # Classification report
                report = classification_report(y_test, preds, output_dict=True, zero_division=0)
                metrics["classification_report"] = report

                # Confusion matrix
                cm = confusion_matrix(y_test, preds)
                metrics["confusion_matrix"] = cm.tolist()

                logger.info(
                    "Evaluation complete: acc=%.3f, log_loss=%.4f",
                    metrics["accuracy"], metrics["log_loss"],
                )

            except Exception as exc:
                logger.warning("Feature-based evaluation failed: %s", exc)
                # Fallback: direct prediction
                if hasattr(model, "predict_matches"):
                    result = model.predict_matches(df)
                    if result is not None:
                        metrics["note"] = "Phase 3 model — no feature-based metrics available"
                        metrics["predictions_generated"] = len(result)

        # Print results
        print(f"\n{'=' * 55}")
        print(f"  MODEL EVALUATION: {model_name}")
        print(f"{'=' * 55}")

        if "accuracy" in metrics:
            print(f"  Accuracy:      {metrics['accuracy']:.4f}")
        if "log_loss" in metrics:
            print(f"  Log loss:      {metrics['log_loss']:.4f}")
        if "brier_score" in metrics:
            print(f"  Brier score:   {metrics['brier_score']:.4f}")

        if "classification_report" in metrics:
            report = metrics["classification_report"]
            print(f"\n  Per-class performance:")
            for cls, cls_metrics in report.items():
                if isinstance(cls_metrics, dict) and "precision" in cls_metrics:
                    print(f"    Class {cls}: prec={cls_metrics['precision']:.3f} "
                          f"rec={cls_metrics['recall']:.3f} "
                          f"f1={cls_metrics['f1-score']:.3f} "
                          f"support={int(cls_metrics['support'])}")

        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            print(f"\n  Confusion matrix:")
            for row in cm:
                print(f"    {row}")

        if "predictions_generated" in metrics:
            print(f"  Predictions generated: {metrics['predictions_generated']}")

        print(f"{'=' * 55}\n")

        # Save report
        output_path = args.output
        if output_path:
            import json
            from datetime import datetime

            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            report_data = {
                "model": model_name,
                "evaluated_at": datetime.now().isoformat(),
                "metrics": {k: v for k, v in metrics.items()
                           if k not in ("confusion_matrix", "classification_report")},
            }
            with open(out, "w") as f:
                json.dump(report_data, f, indent=2, default=str)
            logger.info("Evaluation report saved: %s", out)

        # Generate plots
        if args.plot and "confusion_matrix" in metrics:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import seaborn as sns

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d",
                           cmap="Blues", ax=ax)
                ax.set_title(f"Confusion Matrix — {model_name}")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")

                plot_path = Path("reports/figures")
                plot_path.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fig.savefig(plot_path / f"confusion_matrix_{model_name}_{ts}.png",
                           bbox_inches="tight", dpi=150)
                plt.close(fig)
                logger.info("Confusion matrix plot saved")
            except Exception as exc:
                logger.warning("Plot generation failed: %s", exc)

        return 0

    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        import traceback
        traceback.print_exc()
        return 1


def _handle_dashboard(args: argparse.Namespace) -> int:
    """Handle the ``dashboard`` subcommand."""
    logger.info("Launching Streamlit dashboard...")
    try:
        import subprocess
        import sys
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/app/dashboard.py",
            "--browser.serverAddress", "localhost",
        ], check=True)
        return 0
    except Exception as exc:
        logger.error("Dashboard failed to start: %s", exc)
        return 1


def _handle_dashboard_monitor(args: argparse.Namespace) -> int:
    """Handle the ``dashboard-monitor`` subcommand."""
    logger.info("Launching monitoring dashboard...")
    try:
        import subprocess
        import sys
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "dashboard/app.py",
            "--browser.serverAddress", "localhost",
        ], check=True)
        return 0
    except Exception as exc:
        logger.error("Monitoring dashboard failed to start: %s", exc)
        return 1


def _handle_api(args: argparse.Namespace) -> int:
    """Handle the ``api`` subcommand."""
    logger.info("Starting API server on %s:%d...", args.host, args.port)
    try:
        import uvicorn
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            log_level="info",
        )
        return 0
    except ImportError:
        logger.error("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")
        return 1
    except Exception as exc:
        logger.error("API server failed: %s", exc)
        return 1


def _handle_desktop(args: argparse.Namespace) -> int:
    """Handle the ``desktop`` subcommand."""
    logger.info("Launching desktop application...")
    try:
        from app.main import main as desktop_main
        desktop_main()
        return 0
    except ImportError:
        logger.error(
            "CustomTkinter not installed. Run: pip install customtkinter"
        )
        return 1
    except Exception as exc:
        logger.error("Desktop app failed: %s", exc)
        return 1


def _handle_run_all(args: argparse.Namespace) -> int:
    """Handle the ``run-all`` subcommand."""
    logger.info("Running complete pipeline...")
    try:
        from run_all import main as run_all_main
        cmd_args = []
        if args.skip_collect:
            cmd_args.append("--skip-collect")
        if args.skip_value_bets:
            cmd_args.append("--skip-value-bets")
        if args.skip_dashboard:
            cmd_args.append("--skip-dashboard")
        if args.predict_only:
            cmd_args.append("--predict-only")
        if args.model:
            cmd_args.extend(["--model", args.model])
        return run_all_main(cmd_args)
    except Exception as exc:
        logger.error("Run-all failed: %s", exc)
        return 1


def _handle_pipeline(args: argparse.Namespace) -> int:
    """Handle the ``pipeline`` subcommand."""
    logger.info("Running prediction pipeline...")
    cmd_args = []
    if args.skip_download:
        cmd_args.append("--skip-download")
    if args.skip_train:
        cmd_args.append("--skip-train")
    if args.lightweight:
        cmd_args.append("--lightweight")

    try:
        from run_pipeline import main as pipeline_main
        return pipeline_main(cmd_args)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1
