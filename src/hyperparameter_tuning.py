"""
Hyper-parameter Tuning — optimise every ML model, compare before/after, and save.

Orchestrates the full tuning pipeline::

    from src.hyperparameter_tuning import HyperTuner

    tuner = HyperTuner()
    results = tuner.run(X_train, y_train, X_val, y_val, X_test, y_test)

    print(results["report_text"])        # formatted comparison report
    print(results["summary_df"])         # pandas comparison table

Workflow for each model type
----------------------------
1. Train a **baseline** version using default ``config.train.*`` parameters.
2. Run **GridSearchCV** (LR) or **RandomizedSearchCV** (RF, XGB) to find
   optimal hyper-parameters with cross-validation.
3. Train an **optimised** version using the best found params.
4. Record validation log-loss, accuracy, and training time for both.
5. Save both models to ``models/{model_type}_baseline.joblib`` and
   ``models/{model_type}_tuned.joblib``.

After all models
----------------
6. Identify the single best model overall (lowest validation log-loss).
7. Generate a formatted text report and a pandas summary DataFrame.
8. Optionally evaluate on the held-out test set.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from config import HyperTuneConfig, config
from src.time_series_cv import create_time_series_folds

logger = logging.getLogger(__name__)

# ── Model type identifiers ──────────────────────────────
ModelType = Literal["logistic_regression", "random_forest", "xgboost"]

# ── Default model tuning list ───────────────────────────
_DEFAULT_MODEL_TYPES: tuple[ModelType, ...] = (
    "logistic_regression",
    "random_forest",
    "xgboost",
)


# ═══════════════════════════════════════════════════════════
#  Per-model result container
# ═══════════════════════════════════════════════════════════


@dataclass
class ModelResult:
    """Holds baseline and tuned results for a single model type."""

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
    cv_log_loss: float | None = None  # best CV score from search
    improvement_log_loss: float = 0.0
    improvement_accuracy: float = 0.0

    def __post_init__(self) -> None:
        self.improvement_log_loss = self.baseline_val_log_loss - self.tuned_val_log_loss
        self.improvement_accuracy = self.tuned_val_accuracy - self.baseline_val_accuracy


# ═══════════════════════════════════════════════════════════
#  Hyper-parameter grids
# ═══════════════════════════════════════════════════════════


def _lr_param_grid() -> dict[str, list[Any]]:
    """Small grid → use exact GridSearchCV."""
    return {
        "C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        "solver": ["lbfgs", "liblinear", "newton-cg"],
        "max_iter": [1000, 2000, 5000],
    }


def _rf_param_dist() -> dict[str, list[Any]]:
    """Moderate space → use RandomizedSearchCV."""
    return {
        "n_estimators": [100, 200, 300, 500, 800],
        "max_depth": [4, 6, 8, 10, 15, 20, None],
        "min_samples_leaf": [1, 2, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", None],
    }


def _xgb_param_dist() -> dict[str, list[Any]]:
    """Large space → use RandomizedSearchCV."""
    return {
        "n_estimators": [100, 200, 300, 500, 800, 1000],
        "max_depth": [3, 4, 5, 6, 8, 10, 12],
        "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
        "reg_alpha": [0.0, 0.001, 0.01, 0.1, 1.0],
        "min_child_weight": [1, 3, 5, 7, 10],
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.5],
    }


# ═══════════════════════════════════════════════════════════
#  Model factories
# ═══════════════════════════════════════════════════════════


def _build_baseline(model_type: str) -> Any:
    """Create a model with default ``config.train.*`` parameters.

    Parameters
    ----------
    model_type : str
        One of ``logistic_regression``, ``random_forest``, ``xgboost``.

    Returns
    -------
    Any
        Untrained model instance.
    """
    cfg = config.train

    if model_type == "logistic_regression":
        return LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=2000,
            random_state=cfg.seed,
            class_weight="balanced",
            C=1.0,
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    if model_type == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            reg_alpha=cfg.reg_alpha,
            random_state=cfg.seed,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def _build_with_params(model_type: str, params: dict[str, Any]) -> Any:
    """Create a model with given parameter overrides.

    Merges ``params`` into the baseline config, then builds the model.
    """
    cfg = config.train

    if model_type == "logistic_regression":
        return LogisticRegression(
            multi_class="multinomial",
            max_iter=params.get("max_iter", 2000),
            random_state=cfg.seed,
            class_weight="balanced",
            C=params.get("C", 1.0),
            solver=params.get("solver", "lbfgs"),
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", cfg.n_estimators),
            max_depth=params.get("max_depth", cfg.max_depth),
            min_samples_leaf=params.get("min_samples_leaf", cfg.min_samples_leaf),
            min_samples_split=params.get("min_samples_split", 2),
            max_features=params.get("max_features", "sqrt"),
            random_state=cfg.seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    if model_type == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=cfg.seed,
            n_jobs=-1,
            **params,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


# ═══════════════════════════════════════════════════════════
#  NaN handling helpers
# ═══════════════════════════════════════════════════════════


def _impute(X: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN with column means (safe for sklearn models)."""
    return X.fillna(X.mean().fillna(0))


def _needs_impute(model_type: str) -> bool:
    """XGBoost/LightGBM handle NaN natively — others need imputation."""
    return model_type not in ("xgboost", "lightgbm")


# ═══════════════════════════════════════════════════════════
#  Optimisation wrappers
# ═══════════════════════════════════════════════════════════


def _optimise_lr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int,
    verbose: bool,
) -> tuple[dict[str, Any], float]:
    """GridSearchCV for Logistic Regression (small param space).

    Uses ``TimeSeriesSplit`` (expanding window) to prevent future
    information from leaking into training folds.
    """
    logger.info("  GridSearchCV (Logistic Regression) — %d-fold time-series CV", cv)
    ts_cv = create_time_series_folds(n_splits=cv)
    base = LogisticRegression(
        multi_class="multinomial",
        random_state=config.train.seed,
        class_weight="balanced",
    )
    searcher = GridSearchCV(
        base, _lr_param_grid(),
        cv=ts_cv, scoring="neg_log_loss",
        n_jobs=-1, verbose=1 if verbose else 0,
    )
    searcher.fit(_impute(X_train), y_train)
    return searcher.best_params_, -searcher.best_score_


def _optimise_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int,
    n_iter: int,
    verbose: bool,
) -> tuple[dict[str, Any], float]:
    """RandomizedSearchCV for Random Forest.

    Uses ``TimeSeriesSplit`` (expanding window) to prevent future
    information from leaking into training folds.
    """
    logger.info(
        "  RandomizedSearchCV (Random Forest) — %d-fold time-series CV, %d iters",
        cv, n_iter,
    )
    ts_cv = create_time_series_folds(n_splits=cv)
    base = RandomForestClassifier(
        random_state=config.train.seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    searcher = RandomizedSearchCV(
        base, _rf_param_dist(),
        n_iter=n_iter, cv=ts_cv,
        scoring="neg_log_loss",
        n_jobs=-1, random_state=config.train.seed,
        verbose=1 if verbose else 0,
    )
    searcher.fit(_impute(X_train), y_train)
    return searcher.best_params_, -searcher.best_score_


def _optimise_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int,
    n_iter: int,
    verbose: bool,
) -> tuple[dict[str, Any], float]:
    """RandomizedSearchCV for XGBoost (large param space, NaN-native).

    Uses ``TimeSeriesSplit`` (expanding window) to prevent future
    information from leaking into training folds.
    """
    logger.info(
        "  RandomizedSearchCV (XGBoost) — %d-fold time-series CV, %d iters",
        cv, n_iter,
    )
    ts_cv = create_time_series_folds(n_splits=cv)
    import xgboost as xgb

    base = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=config.train.seed,
        n_jobs=-1,
    )
    searcher = RandomizedSearchCV(
        base, _xgb_param_dist(),
        n_iter=n_iter, cv=ts_cv,
        scoring="neg_log_loss",
        n_jobs=-1, random_state=config.train.seed,
        verbose=1 if verbose else 0,
    )
    # XGBoost handles NaN natively — no imputation
    searcher.fit(X_train, y_train)
    return searcher.best_params_, -searcher.best_score_


# ═══════════════════════════════════════════════════════════
#  Metric evaluation
# ═══════════════════════════════════════════════════════════


def _evaluate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
) -> tuple[float, float]:
    """Compute log-loss and accuracy on a dataset.

    Handles NaN imputation for models that need it.
    """
    X_eval = _impute(X) if _needs_impute(model_type) else X
    probs = model.predict_proba(X_eval)
    preds = model.predict(X_eval)
    return float(log_loss(y, probs)), float(accuracy_score(y, preds))


# ═══════════════════════════════════════════════════════════
#  Main orchestrator
# ═══════════════════════════════════════════════════════════


class HyperTuner:
    """Orchestrate hyper-parameter tuning across all model types.

    Parameters
    ----------
    config_override : HyperTuneConfig, optional
        Override default configuration.
    """

    def __init__(
        self,
        config_override: HyperTuneConfig | None = None,
    ) -> None:
        self.cfg = config_override or config.hyper_tune
        self.results: list[ModelResult] = []
        self.summary_df: pd.DataFrame | None = None
        self.report_text: str = ""

    # ── Main entry point ─────────────────────────────────

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Run the full tuning pipeline.

        Parameters
        ----------
        X_train, y_train : training data.
        X_val, y_val : validation data (metrics reported on this set).
        X_test, y_test : optional held-out test data.

        Returns
        -------
        dict[str, Any]
            Keys: ``results`` (list of ModelResult), ``summary_df``,
            ``report_text``, ``best_model``, ``best_model_type``,
            ``report_path``.
        """
        self.results = []
        self._print_header()

        for model_type in self.cfg.model_types:
            result = self._tune_one(model_type, X_train, y_train, X_val, y_val)
            self.results.append(result)
            self._print_result(result)

        # Build summary
        self.summary_df = self._build_summary_df()
        report = self._build_report()
        self.report_text = report["text"]

        if self.cfg.verbose:
            print(report["text"])

        # Identify overall best
        best_result = min(self.results, key=lambda r: r.tuned_val_log_loss)
        best_model = best_result.tuned_model
        best_model_type = best_result.model_type

        # Optionally evaluate on test
        test_results: dict[str, Any] = {}
        if X_test is not None and y_test is not None:
            test_results = self._evaluate_on_test(best_model, best_model_type, X_test, y_test)

        # Save report
        report_path: str | None = None
        if self.cfg.save_report:
            report_path = self._save_report(report["text"])
            if self.cfg.verbose:
                print(f"\n  Report saved to: {report_path}")

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

    # ── Tune a single model type ─────────────────────────

    def _tune_one(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> ModelResult:
        """Train baseline + tuned for one model type, return comparison."""
        cfg = self.cfg

        if cfg.verbose:
            print(f"\n  ┌─ {'=' * 60}")
            print(f"  │  MODEL: {model_type}")
            print(f"  └─ {'=' * 60}")

        # ── 1. Train baseline ─────────────────────────
        if cfg.verbose:
            print(f"  │  Training baseline ...")
        baseline = _build_baseline(model_type)
        t0 = time.time()
        if _needs_impute(model_type):
            baseline.fit(_impute(X_train), y_train)
        else:
            baseline.fit(X_train, y_train)
        baseline_time = time.time() - t0

        baseline_ll, baseline_acc = _evaluate(baseline, X_val, y_val, model_type)
        if cfg.verbose:
            print(f"  │    ✓ Baseline  |  log-loss: {baseline_ll:.4f}  |  accuracy: {baseline_acc:.2%}")

        # ── 2. Hyper-parameter search ─────────────────
        if model_type == "logistic_regression":
            best_params, cv_loss = _optimise_lr(X_train, y_train, cfg.cv_folds, cfg.verbose)
        elif model_type == "random_forest":
            best_params, cv_loss = _optimise_rf(
                X_train, y_train, cfg.cv_folds, cfg.n_iter_random, cfg.verbose,
            )
        elif model_type == "xgboost":
            best_params, cv_loss = _optimise_xgb(
                X_train, y_train, cfg.cv_folds, cfg.n_iter_random, cfg.verbose,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if cfg.verbose:
            print(f"  │    ✓ Best CV log-loss: {cv_loss:.4f}")

        # ── 3. Train tuned model ──────────────────────
        if cfg.verbose:
            print(f"  │  Training optimised ...")
        tuned = _build_with_params(model_type, best_params)
        t0 = time.time()
        if _needs_impute(model_type):
            tuned.fit(_impute(X_train), y_train)
        else:
            tuned.fit(X_train, y_train)
        tuned_time = time.time() - t0

        tuned_ll, tuned_acc = _evaluate(tuned, X_val, y_val, model_type)
        if cfg.verbose:
            print(f"  │    ✓ Tuned     |  log-loss: {tuned_ll:.4f}  |  accuracy: {tuned_acc:.2%}")
            imp_ll = baseline_ll - tuned_ll
            imp_acc = tuned_acc - baseline_acc
            print(f"  │    Δ log-loss: {imp_ll:+.4f}  |  Δ accuracy: {imp_acc:+.4f}")

        # ── 4. Save models ────────────────────────────
        if cfg.save_models:
            self._save_models(model_type, baseline, tuned)

        return ModelResult(
            model_type=model_type,
            baseline_model=baseline,
            tuned_model=tuned,
            baseline_params=_get_params(baseline),
            tuned_params=best_params,
            baseline_val_log_loss=baseline_ll,
            tuned_val_log_loss=tuned_ll,
            baseline_val_accuracy=baseline_acc,
            tuned_val_accuracy=tuned_acc,
            baseline_train_time=baseline_time,
            tuned_train_time=tuned_time,
            cv_log_loss=cv_loss,
        )

    # ── Persistence ──────────────────────────────────────

    @staticmethod
    def _save_models(model_type: str, baseline: Any, tuned: Any) -> None:
        """Save baseline and tuned models to ``models/``."""
        base_path = config.paths.models / f"{model_type}_baseline.joblib"
        tuned_path = config.paths.models / f"{model_type}_tuned.joblib"
        config.paths.models.mkdir(parents=True, exist_ok=True)
        joblib.dump(baseline, base_path)
        joblib.dump(tuned, tuned_path)
        logger.info("Saved baseline → %s", base_path)
        logger.info("Saved tuned   → %s", tuned_path)

    def _save_report(self, text: str) -> str:
        """Write the report to ``reports/hyperparameter_tuning_report.txt``."""
        report_dir = config.paths.data.parent / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "hyperparameter_tuning_report.txt"
        path.write_text(text, encoding="utf-8")
        return str(path)

    # ── Summary helpers ──────────────────────────────────

    def _build_summary_df(self) -> pd.DataFrame:
        """Build a pandas DataFrame with before/after comparison."""
        rows = []
        for r in self.results:
            rows.append({
                "Model": r.model_type,
                "Baseline LogLoss": round(r.baseline_val_log_loss, 4),
                "Tuned LogLoss": round(r.tuned_val_log_loss, 4),
                "LogLoss Δ": round(r.improvement_log_loss, 4),
                "Baseline Accuracy": round(r.baseline_val_accuracy, 4),
                "Tuned Accuracy": round(r.tuned_val_accuracy, 4),
                "Accuracy Δ": round(r.improvement_accuracy, 4),
                "CV LogLoss": round(r.cv_log_loss, 4) if r.cv_log_loss is not None else None,
                "Baseline Time (s)": round(r.baseline_train_time, 2),
                "Tuned Time (s)": round(r.tuned_train_time, 2),
            })
        return pd.DataFrame(rows)

    def _build_report(self) -> dict[str, str]:
        """Build the formatted comparison report."""
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
        lines.append(f"  Train size:  {df['Baseline LogLoss'].count()} models")
        lines.append("")

        # ── Before / After Table ──────────────────────────
        lines.append(f"  {'─' * 88}")
        lines.append(f"  {'Model':<22s} {'Baseline':>12s} {'Tuned':>12s} {'Δ LogLoss':>12s}  "
                      f"{'Baseline':>10s} {'Tuned':>10s} {'Δ Acc':>10s}")
        lines.append(f"  {'':<22s} {'LogLoss':>12s} {'LogLoss':>12s} {'':>12s}  "
                      f"{'Accuracy':>10s} {'Accuracy':>10s} {'':>10s}")
        lines.append(f"  {'─' * 88}")

        best_row = df.loc[df["Tuned LogLoss"].idxmin()]
        for _, row in df.iterrows():
            is_best = row["Model"] == best_row["Model"]
            marker = " ★" if is_best else "  "
            ll_delta_str = f"{row['LogLoss Δ']:+.4f}" if pd.notna(row.get("LogLoss Δ")) else "  N/A"
            acc_delta_str = f"{row['Accuracy Δ']:+.4f}" if pd.notna(row.get("Accuracy Δ")) else "  N/A"
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

        # ── Best model callout ────────────────────────────
        lines.append(f"  {'★' * 30}  BEST MODEL  {'★' * 30}")
        lines.append(f"")
        lines.append(f"    {best_row['Model']}")
        lines.append(f"      Validation log-loss: {best_row['Baseline LogLoss']:.4f} → "
                      f"{best_row['Tuned LogLoss']:.4f} "
                      f"(Δ = {best_row['LogLoss Δ']:+.4f})")
        lines.append(f"      Validation accuracy: {best_row['Baseline Accuracy']:.2%} → "
                      f"{best_row['Tuned Accuracy']:.2%} "
                      f"(Δ = {best_row['Accuracy Δ']:+.4f})")
        lines.append(f"")
        lines.append(f"  {'★' * 76}")

        # ── Parameter details ─────────────────────────────
        lines.append("")
        lines.append(f"  {'=' * 90}")
        lines.append("  PARAMETER DETAILS")
        lines.append(f"  {'=' * 90}")
        lines.append("")

        for r in self.results:
            lines.append(f"  ── {r.model_type} ──")
            lines.append(f"    Default params:  {r.baseline_params}")
            lines.append(f"    Tuned params:    {r.tuned_params}")
            lines.append(f"    CV log-loss:     {r.cv_log_loss:.4f}" if r.cv_log_loss else "")
            lines.append(f"    Train time:      {r.baseline_train_time:.2f}s → {r.tuned_train_time:.2f}s")
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
        """Evaluate the best model on the held-out test set."""
        test_ll, test_acc = _evaluate(model, X_test, y_test, model_type)

        # Also evaluate all tuned models on test
        all_test: dict[str, dict[str, float]] = {}
        for r in self.results:
            ll, acc = _evaluate(r.tuned_model, X_test, y_test, r.model_type)
            all_test[r.model_type] = {"log_loss": ll, "accuracy": acc}

        if self.cfg.verbose:
            print(f"\n  {'=' * 90}")
            print("  TEST SET EVALUATION")
            print(f"  {'=' * 90}")
            print(f"\n    Best model ({model_type}):")
            print(f"      Test log-loss: {test_ll:.4f}")
            print(f"      Test accuracy: {test_acc:.2%}")
            print(f"\n    All tuned models on test set:")
            for mt, m in all_test.items():
                marker = " ★" if mt == model_type else "  "
                print(f"      {mt:<22s}{marker}  log-loss: {m['log_loss']:.4f}  |  accuracy: {m['accuracy']:.2%}")

        return {
            "best_model_log_loss": test_ll,
            "best_model_accuracy": test_acc,
            "all_tuned_test_metrics": all_test,
        }

    # ── Print helpers ────────────────────────────────────

    def _print_header(self) -> None:
        if not self.cfg.verbose:
            return
        print()
        print("=" * 90)
        print("  HYPER-PARAMETER TUNING".center(88))
        print("=" * 90)
        print(f"\n  Model types:  {', '.join(self.cfg.model_types)}")
        print(f"  CV folds:     {self.cfg.cv_folds}")
        print(f"  Random iters: {self.cfg.n_iter_random}")
        print(f"  Saving models: {self.cfg.save_models}")

    def _print_result(self, result: ModelResult) -> None:
        if not self.cfg.verbose:
            return
        print(f"\n  ── {result.model_type} complete ──")
        print(f"     Baseline:  log-loss={result.baseline_val_log_loss:.4f}  "
              f"accuracy={result.baseline_val_accuracy:.2%}")
        print(f"     Tuned:     log-loss={result.tuned_val_log_loss:.4f}  "
              f"accuracy={result.tuned_val_accuracy:.2%}")
        print(f"     Δ log-loss: {result.improvement_log_loss:+.4f}  |  "
              f"Δ accuracy: {result.improvement_accuracy:+.4f}")

    def _print_footer(self, best: ModelResult) -> None:
        if not self.cfg.verbose:
            return
        print()
        print("=" * 90)
        print("  TUNING COMPLETE".center(88))
        print("=" * 90)
        print(f"\n  Best model:          {best.model_type}")
        print(f"  Validation log-loss: {best.baseline_val_log_loss:.4f} → {best.tuned_val_log_loss:.4f}")
        print(f"  Δ log-loss:          {best.improvement_log_loss:+.4f}")
        print(f"  Tuned params:        {best.tuned_params}")
        print(f"\n  Models saved to:     {config.paths.models}/")
        print(f"  Report saved to:     reports/hyperparameter_tuning_report.txt")
        print()
        print("=" * 90)
        print()


# ═══════════════════════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════════════════════


def _get_params(model: Any) -> dict[str, Any]:
    """Get the parameters of a fitted model, handling sklearn & XGBoost."""
    try:
        return model.get_params()
    except Exception:
        return {}
