"""
Walk-forward Validation — test models on chronologically out-of-sample data.

Simulates a rolling training window: trains on past data, predicts on future
data, records performance, and slides the window forward. This gives the
most realistic estimate of out-of-sample model performance.

Usage:
    python scripts/run_walk_forward_validation.py
    python scripts/run_walk_forward_validation.py --model-type xgboost --windows 6
    python scripts/run_walk_forward_validation.py --compare-all --output reports/walk_forward
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────

REPORTS_DIR = Path("reports/walk_forward")
"""Default output directory for walk-forward reports."""

DEFAULT_WINDOWS = 5
"""Default number of walk-forward windows."""

DEFAULT_TRAIN_PCT = 0.6
"""Default % of data used for initial training window."""


# ── Data structures ─────────────────────────────────────


@dataclass
class WindowResult:
    """Results for a single walk-forward window."""

    window_idx: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    n_train: int
    n_val: int
    log_loss: float
    accuracy: float
    brier_score: float
    roc_auc: float
    train_time: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window_idx,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "val_start": self.val_start,
            "val_end": self.val_end,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "log_loss": round(self.log_loss, 4),
            "accuracy": round(self.accuracy, 4),
            "brier_score": round(self.brier_score, 4),
            "roc_auc": round(self.roc_auc, 4),
            "train_time_s": round(self.train_time, 2),
        }


@dataclass
class WalkForwardResult:
    """Aggregated results across all windows."""

    model_type: str
    windows: list[WindowResult] = field(default_factory=list)

    @property
    def avg_log_loss(self) -> float:
        return float(np.mean([w.log_loss for w in self.windows])) if self.windows else 0.0

    @property
    def avg_accuracy(self) -> float:
        return float(np.mean([w.accuracy for w in self.windows])) if self.windows else 0.0

    @property
    def std_log_loss(self) -> float:
        return float(np.std([w.log_loss for w in self.windows])) if self.windows else 0.0

    @property
    def max_log_loss(self) -> float:
        return float(np.max([w.log_loss for w in self.windows])) if self.windows else 0.0

    @property
    def min_log_loss(self) -> float:
        return float(np.min([w.log_loss for w in self.windows])) if self.windows else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "n_windows": len(self.windows),
            "avg_log_loss": round(self.avg_log_loss, 4),
            "std_log_loss": round(self.std_log_loss, 4),
            "min_log_loss": round(self.min_log_loss, 4),
            "max_log_loss": round(self.max_log_loss, 4),
            "avg_accuracy": round(self.avg_accuracy, 4),
            "windows": [w.to_dict() for w in self.windows],
        }

    def summary(self) -> str:
        lines = [
            f"\n{'=' * 70}",
            f"  Walk-Forward: {self.model_type}",
            f"{'=' * 70}",
            f"  Windows:      {len(self.windows)}",
            f"  Avg LogLoss:  {self.avg_log_loss:.4f} ± {self.std_log_loss:.4f}",
            f"  Range:        [{self.min_log_loss:.4f}, {self.max_log_loss:.4f}]",
            f"  Avg Accuracy: {self.avg_accuracy:.2%}",
            f"{'─' * 70}",
        ]
        for w in self.windows:
            lines.append(
                f"  Window {w.window_idx}: train={w.train_end} "
                f"val={w.val_end} | "
                f"LL={w.log_loss:.4f} Acc={w.accuracy:.2%}"
            )
        lines.append(f"{'=' * 70}\n")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  Core validation logic
# ═══════════════════════════════════════════════════════════


def run_walk_forward(
    df: pd.DataFrame,
    model_type: str = "xgboost",
    n_windows: int = DEFAULT_WINDOWS,
    train_pct: float = DEFAULT_TRAIN_PCT,
    feature_cols: list[str] | None = None,
    target_col: str = "result_encoded",
    date_col: str = "date",
) -> WalkForwardResult:
    """Run walk-forward validation for a model type.

    Parameters
    ----------
    df : pd.DataFrame
        Chronologically sorted, preprocessed match data.
    model_type : str
        Model type to validate (default ``xgboost``).
    n_windows : int
        Number of walk-forward windows (default 5).
    train_pct : float
        Initial training fraction (default 0.6 = 60%).
    feature_cols : list[str], optional
        Feature columns to use. If None, all numeric columns except target/date.
    target_col : str
        Target column name (default ``result_encoded``).
    date_col : str
        Date column name for chronological splitting (default ``date``).

    Returns
    -------
    WalkForwardResult
        Per-window results with aggregated metrics.
    """
    # Determine feature columns
    if feature_cols is None:
        exclude = {target_col, date_col, "result", "season", "league",
                   "home_team", "away_team", "match_id"}
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude and not c.startswith("_")
        ]

    # Sort chronologically
    if date_col in df.columns and not df[date_col].is_monotonic_increasing:
        df = df.sort_values(date_col).reset_index(drop=True)

    n_total = len(df)
    n_train_initial = int(n_total * train_pct)

    # Determine window boundaries (expanding or sliding)
    n_val = (n_total - n_train_initial) // n_windows

    result = WalkForwardResult(model_type=model_type)

    for window_idx in range(n_windows):
        if window_idx == 0:
            train_end = n_train_initial
        else:
            # Expanding window: train_end grows each window
            train_end = n_train_initial + window_idx * n_val

        val_start = train_end
        val_end = min(val_start + n_val, n_total)

        if val_start >= n_total or val_end <= val_start:
            break

        # Split data
        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start:val_end]

        X_train = train_df[feature_cols].copy()
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols].copy()
        y_val = val_df[target_col].values

        logger.info(
            "Window %d: train=%d (%s → %s), val=%d (%s → %s)",
            window_idx + 1, len(train_df),
            train_df[date_col].iloc[0] if date_col in train_df.columns else "?",
            train_df[date_col].iloc[-1] if date_col in train_df.columns else "?",
            len(val_df),
            val_df[date_col].iloc[0] if date_col in val_df.columns else "?",
            val_df[date_col].iloc[-1] if date_col in val_df.columns else "?",
        )

        # Train model
        t0 = time.time()
        model = _train_model(model_type, X_train, y_train)
        train_time = time.time() - t0

        # Evaluate
        probs = _predict_proba(model, model_type, X_val)
        preds = np.argmax(probs, axis=1)

        ll = float(log_loss(y_val, probs))
        acc = float(accuracy_score(y_val, preds))

        # Brier score (multi-class version: average of one-vs-rest Brier)
        n_classes = probs.shape[1]
        y_onehot = np.eye(n_classes)[y_val]
        brier = float(np.mean([brier_score_loss(y_onehot[:, i], probs[:, i]) for i in range(n_classes)]))

        # ROC-AUC (one-vs-rest macro)
        try:
            roc_auc = float(roc_auc_score(y_val, probs, multi_class="ovr"))
        except Exception:
            roc_auc = 0.0

        window_result = WindowResult(
            window_idx=window_idx + 1,
            train_start=str(train_df[date_col].iloc[0]) if date_col in train_df.columns else "",
            train_end=str(train_df[date_col].iloc[-1]) if date_col in train_df.columns else "",
            val_start=str(val_df[date_col].iloc[0]) if date_col in val_df.columns else "",
            val_end=str(val_df[date_col].iloc[-1]) if date_col in val_df.columns else "",
            n_train=len(train_df),
            n_val=len(val_df),
            log_loss=ll,
            accuracy=acc,
            brier_score=brier,
            roc_auc=roc_auc,
            train_time=train_time,
        )
        result.windows.append(window_result)

        logger.info(
            "  → log-loss=%.4f, accuracy=%.2%%, brier=%.4f (%.1fs)",
            ll, acc, brier, train_time,
        )

    return result


def _train_model(model_type: str, X: pd.DataFrame, y: np.ndarray) -> Any:
    """Train a model of the given type."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    col_means = X.mean().fillna(0)

    if model_type == "xgboost":
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed")
        model = xgb.XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss",
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=config.train.seed, n_jobs=-1,
        )
        model.fit(X, y)

    elif model_type == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed")
        model = lgb.LGBMClassifier(
            objective="multiclass", metric="multi_logloss",
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            num_leaves=31, random_state=config.train.seed,
            n_jobs=-1, verbose=-1,
        )
        model.fit(X, y)

    elif model_type == "logistic_regression":
        model = LogisticRegression(
            solver="lbfgs", max_iter=2000, C=1.0,
            class_weight="balanced", random_state=config.train.seed,
            n_jobs=-1,
        )
        model.fit(X.fillna(col_means), y)

    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            class_weight="balanced_subsample", random_state=config.train.seed,
            n_jobs=-1,
        )
        model.fit(X.fillna(col_means), y)

    elif model_type == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise ImportError("catboost not installed")
        model = CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, random_seed=config.train.seed,
            loss_function="MultiClass", verbose=False, allow_writing_files=False,
        )
        model.fit(X, y)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model


def _predict_proba(model: Any, model_type: str, X: pd.DataFrame) -> np.ndarray:
    """Get probabilities from a model, handling NaN."""
    if model_type in ("xgboost", "lightgbm", "catboost"):
        probs = model.predict_proba(X)
    else:
        probs = model.predict_proba(X.fillna(X.mean().fillna(0)))
    return np.asarray(probs, dtype=np.float64)


# ═══════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════


def load_preprocessed_data(path: str | None = None) -> pd.DataFrame:
    """Load preprocessed match data with features.

    Tries common paths if none specified.
    """
    if path:
        p = Path(path)
        if p.exists():
            df = pd.read_csv(p)
            logger.info("Loaded %d rows from %s", len(df), p)
            return df

    candidates = [
        config.paths.processed / "results_clean.csv",
        config.paths.processed / "results_features.csv",
        config.paths.raw / "worldcup_all.csv",
    ]
    for c in candidates:
        if c.exists():
            df = pd.read_csv(c)
            logger.info("Loaded %d rows from %s", len(df), c)
            return df

    raise FileNotFoundError(
        "No preprocessed data found. Run collect_all_worldcups.py and "
        "train_worldcup.py first to generate feature-engineered data."
    )


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--input", type=str, default=None,
                        help="Input CSV with features (default: auto-find)")
    parser.add_argument("--model-type", type=str, default="xgboost",
                        choices=["xgboost", "lightgbm", "logistic_regression",
                                 "random_forest", "catboost"],
                        help="Model type to validate (default: xgboost)")
    parser.add_argument("--windows", type=int, default=DEFAULT_WINDOWS,
                        help=f"Number of walk-forward windows (default: {DEFAULT_WINDOWS})")
    parser.add_argument("--train-pct", type=float, default=DEFAULT_TRAIN_PCT,
                        help=f"Initial training fraction (default: {DEFAULT_TRAIN_PCT})")
    parser.add_argument("--compare-all", action="store_true",
                        help="Run validation for all available model types")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: reports/walk_forward)")
    parser.add_argument("--target", type=str, default="result_encoded",
                        help="Target column name (default: result_encoded)")
    args = parser.parse_args()

    # Load data
    df = load_preprocessed_data(args.input)
    if args.target not in df.columns:
        logger.error(
            "Target column '%s' not found in data. Available: %s",
            args.target, list(df.columns[:20]),
        )
        return

    output_dir = Path(args.output or REPORTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_types = ["xgboost", "lightgbm", "logistic_regression",
                   "random_forest", "catboost"] if args.compare_all else [args.model_type]

    all_results: dict[str, Any] = {}
    best_model = ""
    best_log_loss = float("inf")

    for mt in model_types:
        try:
            result = run_walk_forward(
                df=df,
                model_type=mt,
                n_windows=args.windows,
                train_pct=args.train_pct,
                target_col=args.target,
            )
            all_results[mt] = result.to_dict()
            print(result.summary())

            if result.avg_log_loss < best_log_loss:
                best_log_loss = result.avg_log_loss
                best_model = mt
        except (ImportError, ValueError) as e:
            logger.warning("Skipping %s: %s", mt, e)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"walk_forward_{timestamp}.json"
    report = {
        "timestamp": timestamp,
        "config": {
            "n_windows": args.windows,
            "train_pct": args.train_pct,
            "data_rows": len(df),
            "target": args.target,
        },
        "best_model": best_model,
        "best_avg_log_loss": round(best_log_loss, 4) if best_model else None,
        "models": all_results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", report_path)

    # Summary comparison table
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("  MODEL COMPARISON (walk-forward average)")
        print(f"{'=' * 70}")
        print(f"  {'Model':<25s} {'Avg LogLoss':<15s} {'Std':<10s} {'Avg Acc':<10s}")
        print(f"  {'─' * 60}")
        for mt, res in sorted(all_results.items(), key=lambda x: x[1]["avg_log_loss"]):
            marker = " ★" if mt == best_model else "  "
            print(f"  {mt:<23s}{marker} {res['avg_log_loss']:<15.4f} "
                  f"{res['std_log_loss']:<10.4f} {res['avg_accuracy']:.2%}")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
