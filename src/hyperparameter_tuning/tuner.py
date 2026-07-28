"""HyperTuner — orchestrate hyper-parameter tuning across all model types."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd

from src.config import HyperTuneConfig, config
from src.hyperparameter_tuning.models import (
    build_baseline,
    build_with_params,
    get_params,
    impute,
    needs_impute,
)
from src.hyperparameter_tuning.optimisers import (
    evaluate,
    optimise_lgbm,
    optimise_lr,
    optimise_rf,
    optimise_xgb,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    model_type: str
    baseline_model: Any
    tuned_model: Any
    baseline_params: dict[str, Any]
    tuned_params: dict[str, Any]
    baseline_val_log_loss: float
    tuned_val_log_loss: float
    baseline_val_accuracy: float
    tuned_val_accuracy: float
    baseline_train_time: float
    tuned_train_time: float
    cv_log_loss: float | None = None
    improvement_log_loss: float = 0.0
    improvement_accuracy: float = 0.0

    def __post_init__(self) -> None:
        self.improvement_log_loss = self.baseline_val_log_loss - self.tuned_val_log_loss
        self.improvement_accuracy = self.tuned_val_accuracy - self.baseline_val_accuracy


class HyperTuner:
    def __init__(self, config_override: HyperTuneConfig | None = None) -> None:
        self.cfg = config_override or config.hyper_tune
        self.results: list[ModelResult] = []
        self.summary_df: pd.DataFrame | None = None
        self.report_text: str = ""

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
    ) -> dict[str, Any]:
        self.results = []
        self._print_header()
        for model_type in self.cfg.model_types:
            result = self._tune_one(model_type, X_train, y_train, X_val, y_val)
            self.results.append(result)
            self._print_result(result)
        self.summary_df = self._build_summary_df()
        report = self._build_report()
        self.report_text = report["text"]
        if self.cfg.verbose:
            logger.info(report["text"])
        best_result = min(self.results, key=lambda r: r.tuned_val_log_loss)
        best_model = best_result.tuned_model
        best_model_type = best_result.model_type
        test_results: dict[str, Any] = {}
        if X_test is not None and y_test is not None:
            test_results = self._evaluate_on_test(
                best_model, best_model_type, X_test, y_test
            )
        report_path: str | None = None
        if self.cfg.save_report:
            report_path = self._save_report(report["text"])
            if self.cfg.verbose:
                logger.info("  Report saved to: %s", report_path)
        self._print_footer(best_result)
        return {
            "results": self.results,
            "summary_df": self.summary_df,
            "report_text": report["text"],
            "best_model": best_model,
            "best_model_type": best_model_type,
            "best_val_log_loss": best_result.tuned_val_log_loss,
            "report_path": report_path,
            "test_results": test_results,
        }

    def _tune_one(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> ModelResult:
        cfg = self.cfg
        if cfg.verbose:
            logger.info("  ┌─ %s", "=" * 60)
            logger.info("  │  MODEL: %s", model_type)
            logger.info("  └─ %s", "=" * 60)
        if cfg.verbose:
            logger.info("  │  Training baseline ...")
        baseline = build_baseline(model_type)
        t0 = time.time()
        if needs_impute(model_type):
            baseline.fit(impute(X_train), y_train)
        else:
            baseline.fit(X_train, y_train)
        baseline_time = time.time() - t0
        baseline_ll, baseline_acc = evaluate(baseline, X_val, y_val, model_type)
        if cfg.verbose:
            logger.info(
                "  │    ✓ Baseline  |  log-loss: %.4f  |  accuracy: %.2f%%",
                baseline_ll,
                baseline_acc * 100,
            )
        if model_type == "logistic_regression":
            best_params, cv_loss = optimise_lr(
                X_train, y_train, cfg.cv_folds, cfg.verbose
            )
        elif model_type == "random_forest":
            best_params, cv_loss = optimise_rf(
                X_train, y_train, cfg.cv_folds, cfg.n_iter_random, cfg.verbose
            )
        elif model_type == "xgboost":
            best_params, cv_loss = optimise_xgb(
                X_train, y_train, cfg.cv_folds, cfg.n_iter_random, cfg.verbose
            )
        elif model_type == "lightgbm":
            best_params, cv_loss = optimise_lgbm(
                X_train, y_train, cfg.cv_folds, cfg.n_iter_random, cfg.verbose
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        if cfg.verbose:
            logger.info("  │    ✓ Best CV log-loss: %.4f", cv_loss)
        if cfg.verbose:
            logger.info("  │  Training optimised ...")
        tuned = build_with_params(model_type, best_params)
        t0 = time.time()
        if needs_impute(model_type):
            tuned.fit(impute(X_train), y_train)
        else:
            tuned.fit(X_train, y_train)
        tuned_time = time.time() - t0
        tuned_ll, tuned_acc = evaluate(tuned, X_val, y_val, model_type)
        if cfg.verbose:
            logger.info(
                "  │    ✓ Tuned     |  log-loss: %.4f  |  accuracy: %.2f%%",
                tuned_ll,
                tuned_acc * 100,
            )
            imp_ll = baseline_ll - tuned_ll
            imp_acc = tuned_acc - baseline_acc
            logger.info(
                "  │    Δ log-loss: %+.4f  |  Δ accuracy: %+.4f", imp_ll, imp_acc
            )
        if cfg.save_models:
            self._save_models(model_type, baseline, tuned)
        return ModelResult(
            model_type=model_type,
            baseline_model=baseline,
            tuned_model=tuned,
            baseline_params=get_params(baseline),
            tuned_params=best_params,
            baseline_val_log_loss=baseline_ll,
            tuned_val_log_loss=tuned_ll,
            baseline_val_accuracy=baseline_acc,
            tuned_val_accuracy=tuned_acc,
            baseline_train_time=baseline_time,
            tuned_train_time=tuned_time,
            cv_log_loss=cv_loss,
        )

    @staticmethod
    def _save_models(model_type: str, baseline: Any, tuned: Any) -> None:
        base_path = config.paths.models / f"{model_type}_baseline.joblib"
        tuned_path = config.paths.models / f"{model_type}_tuned.joblib"
        config.paths.models.mkdir(parents=True, exist_ok=True)
        joblib.dump(baseline, base_path)
        joblib.dump(tuned, tuned_path)
        logger.info("Saved baseline -> %s", base_path)
        logger.info("Saved tuned   -> %s", tuned_path)

    def _save_report(self, text: str) -> str:
        report_dir = config.paths.data.parent / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "hyperparameter_tuning_report.txt"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def _build_summary_df(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            rows.append(
                {
                    "Model": r.model_type,
                    "Baseline LogLoss": round(r.baseline_val_log_loss, 4),
                    "Tuned LogLoss": round(r.tuned_val_log_loss, 4),
                    "LogLoss Δ": round(r.improvement_log_loss, 4),
                    "Baseline Accuracy": round(r.baseline_val_accuracy, 4),
                    "Tuned Accuracy": round(r.tuned_val_accuracy, 4),
                    "Accuracy Δ": round(r.improvement_accuracy, 4),
                    "CV LogLoss": round(r.cv_log_loss, 4)
                    if r.cv_log_loss is not None
                    else None,
                    "Baseline Time (s)": round(r.baseline_train_time, 2),
                    "Tuned Time (s)": round(r.tuned_train_time, 2),
                }
            )
        return pd.DataFrame(rows)

    def _build_report(self) -> dict[str, str]:
        df = self.summary_df
        lines: list[str] = []
        sep = "=" * 90
        lines.append("")
        lines.append(sep)
        lines.append("  HYPER-PARAMETER TUNING — COMPARISON REPORT".center(88))
        lines.append(sep)
        lines.append("")
        lines.append(f"  Date:        {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"  CV folds:    {self.cfg.cv_folds}")
        lines.append(f"  Random iters:{self.cfg.n_iter_random}")
        if df is not None:
            lines.append(f"  Train size:  {df['Baseline LogLoss'].count()} models")
        else:
            lines.append("  Train size:  (no results)")
        lines.append("")
        lines.append(f"  {'─' * 88}")
        lines.append(
            f"  {'Model':<22s} {'Baseline':>12s} {'Tuned':>12s} {'Δ LogLoss':>12s}  "
            f"{'Baseline':>10s} {'Tuned':>10s} {'Δ Acc':>10s}"
        )
        lines.append(
            f"  {'':<22s} {'LogLoss':>12s} {'LogLoss':>12s} {'':>12s}  "
            f"{'Accuracy':>10s} {'Accuracy':>10s} {'':>10s}"
        )
        if df is None:
            lines.append("  No results to display.")
            lines.append(sep)
            return {"text": "\n".join(lines)}
        best_row = df.loc[df["Tuned LogLoss"].idxmin()]
        for _, row in df.iterrows():
            is_best = row["Model"] == best_row["Model"]
            marker = " ★" if is_best else "  "
            ll_delta_str = (
                f"{row['LogLoss Δ']:+.4f}"
                if pd.notna(row.get("LogLoss Δ"))
                else "  N/A"
            )
            acc_delta_str = (
                f"{row['Accuracy Δ']:+.4f}"
                if pd.notna(row.get("Accuracy Δ"))
                else "  N/A"
            )
            lines.append(
                f"  {row['Model']:<20s}{marker} "
                f"{row['Baseline LogLoss']:>12.4f} "
                f"{row['Tuned LogLoss']:>12.4f} "
                f"{ll_delta_str:>12s}  "
                f"{row['Baseline Accuracy']:>10.4f} "
                f"{row['Tuned Accuracy']:>10.4f} "
                f"{acc_delta_str:>10s}"
            )
        lines.append(f"  {'─' * 88}")
        lines.append("")
        lines.append(f"  {'★' * 30}  BEST MODEL  {'★' * 30}")
        lines.append("")
        lines.append(f"    {best_row['Model']}")
        lines.append(
            f"      Validation log-loss: {best_row['Baseline LogLoss']:.4f} -> "
            f"{best_row['Tuned LogLoss']:.4f} "
            f"(Δ = {best_row['LogLoss Δ']:+.4f})"
        )
        lines.append(
            f"      Validation accuracy: {best_row['Baseline Accuracy']:.2%} -> "
            f"{best_row['Tuned Accuracy']:.2%} "
            f"(Δ = {best_row['Accuracy Δ']:+.4f})"
        )
        lines.append("")
        lines.append(f"  {'★' * 76}")
        lines.append("")
        lines.append(f"  {'=' * 90}")
        lines.append("  PARAMETER DETAILS")
        lines.append(f"  {'=' * 90}")
        lines.append("")
        for r in self.results:
            lines.append(f"  ── {r.model_type} ──")
            lines.append(f"    Default params:  {r.baseline_params}")
            lines.append(f"    Tuned params:    {r.tuned_params}")
            lines.append(
                f"    CV log-loss:     {r.cv_log_loss:.4f}" if r.cv_log_loss else ""
            )
            lines.append(
                f"    Train time:      {r.baseline_train_time:.2f}s -> {r.tuned_train_time:.2f}s"
            )
            lines.append("")
        lines.append(sep)
        lines.append("")
        return {"text": "\n".join(lines)}

    def _evaluate_on_test(
        self,
        model: Any,
        model_type: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        test_ll, test_acc = evaluate(model, X_test, y_test, model_type)
        all_test: dict[str, dict[str, float]] = {}
        for r in self.results:
            ll, acc = evaluate(r.tuned_model, X_test, y_test, r.model_type)
            all_test[r.model_type] = {"log_loss": ll, "accuracy": acc}
        if self.cfg.verbose:
            logger.info("")
            logger.info("  %s", "=" * 90)
            logger.info("  TEST SET EVALUATION")
            logger.info("  %s", "=" * 90)
            logger.info("    Best model (%s):", model_type)
            logger.info("      Test log-loss: %.4f", test_ll)
            logger.info("      Test accuracy: %.2f%%", test_acc * 100)
            logger.info("    All tuned models on test set:")
            for mt, m in all_test.items():
                marker = " ★" if mt == model_type else "  "
                logger.info(
                    "      %-22s%s  log-loss: %.4f  |  accuracy: %.2f%%",
                    mt,
                    marker,
                    m["log_loss"],
                    m["accuracy"] * 100,
                )
        return {
            "best_model_log_loss": test_ll,
            "best_model_accuracy": test_acc,
            "all_tuned_test_metrics": all_test,
        }

    def _print_header(self) -> None:
        if not self.cfg.verbose:
            return
        logger.info("")
        logger.info("=" * 90)
        logger.info("  HYPER-PARAMETER TUNING".center(88))
        logger.info("=" * 90)
        logger.info("  Model types:  %s", ", ".join(self.cfg.model_types))
        logger.info("  CV folds:     %d", self.cfg.cv_folds)
        logger.info("  Random iters: %d", self.cfg.n_iter_random)
        logger.info("  Saving models: %s", self.cfg.save_models)

    def _print_result(self, result: ModelResult) -> None:
        if not self.cfg.verbose:
            return
        logger.info("  ── %s complete ──", result.model_type)
        logger.info(
            "     Baseline:  log-loss=%.4f  accuracy=%.2f%%",
            result.baseline_val_log_loss,
            result.baseline_val_accuracy * 100,
        )
        logger.info(
            "     Tuned:     log-loss=%.4f  accuracy=%.2f%%",
            result.tuned_val_log_loss,
            result.tuned_val_accuracy * 100,
        )
        logger.info(
            "     Δ log-loss: %+.4f  |  Δ accuracy: %+.4f",
            result.improvement_log_loss,
            result.improvement_accuracy,
        )

    def _print_footer(self, best: ModelResult) -> None:
        if not self.cfg.verbose:
            return
        logger.info("")
        logger.info("=" * 90)
        logger.info("  TUNING COMPLETE".center(88))
        logger.info("=" * 90)
        logger.info("  Best model:          %s", best.model_type)
        logger.info(
            "  Validation log-loss: %.4f -> %.4f",
            best.baseline_val_log_loss,
            best.tuned_val_log_loss,
        )
        logger.info("  Δ log-loss:          %+.4f", best.improvement_log_loss)
        logger.info("  Tuned params:        %s", best.tuned_params)
        logger.info("  Models saved to:     %s", config.paths.models)
        logger.info("  Report saved to:     reports/hyperparameter_tuning_report.txt")
        logger.info("")
        logger.info("=" * 90)
        logger.info("")
