"""
Training service — manages model training lifecycle.

Handles dataset splitting, feature pipeline execution, model
training, hyper-parameter tuning, and model versioning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.di_container import ConfigProvider, get_container

logger = logging.getLogger(__name__)


class TrainingService:
    """Service for training and managing ML models.

    Parameters
    ----------
    model_dir : Path, optional
        Directory where trained models are stored. Defaults to
        ``config.paths.models``.
    config : ConfigProvider, optional
        Config provider for dependency injection.  Defaults to the
        global container's ConfigProvider.
    """

    def __init__(self, model_dir: Path | None = None, config: ConfigProvider | None = None) -> None:
        self._config = config or get_container().resolve(ConfigProvider)
        self._model_dir = model_dir or self._config.paths.models
        self._model_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────

    def train(
        self,
        data_path: str | Path | None = None,
        model_type: str | None = None,
        tune_hyperparams: bool = False,
        cv_folds: int | None = None,
    ) -> dict:
        """Train a new model end-to-end.

        Loads data from *data_path*, builds the feature matrix via
        ``src.feature_engineering.build_features``, splits
        chronologically, optionally runs hyper-parameter tuning, trains
        the final model, saves it, and returns a report with metrics.

        Parameters
        ----------
        data_path : str | Path, optional
            Path to the raw match data CSV.  If omitted, the service
            tries ``config.paths.raw / "results.csv"`` then
            ``config.worldcup.data_path``.
        model_type : str, optional
            Model algorithm.  One of ``"xgboost"``, ``"lightgbm"``,
            ``"logistic_regression"``, ``"random_forest"``,
            ``"neural_network"``.  Defaults to ``config.train.model_type``.
        tune_hyperparams : bool
            Run randomised time-series CV before the final fit
            (default False).
        cv_folds : int, optional
            Number of CV folds for tuning (default
            ``config.train.cv_folds``).

        Returns
        -------
        dict
            Report with keys: ``model_type``, ``model_path``,
            ``metrics``, ``features``, ``splits``, and
            ``hyperparameter_tuning`` (if run).
        """
        cfg = self._config
        logger.info(
            "Training model: %s (tune=%s, cv_folds=%s)",
            model_type or cfg.train.model_type,
            tune_hyperparams,
            cv_folds,
        )

        # ── 1. Resolve data path ────────────────────────────
        data_path = resolve_data_path(data_path, config=cfg)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {data_path}. "
                "Run data collection first, or pass an explicit data_path."
            )

        # ── 2. Load & prepare data (pipeline) ─────────────
        df = load_and_prepare(data_path, add_temporal=True, config=cfg)

        # ── 2a. Validate data before training ───────────────
        from src.data_collection.cleaners import validate_data
        validation = validate_data(df)
        if not validation["is_valid"]:
            warnings_str = " | ".join(validation["warnings"])
            raise ValueError(
                f"Training data validation failed — {len(validation['warnings'])} issue(s): "
                f"{warnings_str}"
                ". Aborting training to avoid training on corrupted data."
            )
        logger.info(
            "Data validation passed: %d rows, %d cols, %.1f%% missing",
            validation["stats"]["rows"],
            validation["stats"]["columns"],
            validation["stats"]["missing_pct"],
        )

        completed = df[df["result"].notna()].copy()
        logger.info("Loaded %d completed matches from %s", len(completed), data_path)

        # ── 2.5. Fit SafeTargetEncoder BEFORE build_features ──
        # This ensures categorical encoding uses a training-only prior
        # instead of computing global_mean from ALL rows (which would
        # leak future labels into the encoding).
        #
        # The encoder is fitted on TRAINING rows only (first N rows after
        # chronological sort, where N is determined by the same split
        # ratios used later in train_val_test_split).
        from src.features.encoding import SafeTargetEncoder
        target_encoder = SafeTargetEncoder(cols=["home_team", "away_team"])
        if "target" in completed.columns:
            # Sort the same way build_features does so X.index aligns
            completed_sorted = completed.copy()
            if "date" in completed_sorted.columns:
                completed_sorted["date"] = pd.to_datetime(completed_sorted["date"])
                completed_sorted.sort_values(["date", "home_team"], inplace=True)
                completed_sorted.reset_index(drop=True, inplace=True)

            # Determine training boundary using same split ratios as downstream
            train_ratio = cfg.data.split_ratios[0]
            train_end = int(len(completed_sorted) * train_ratio)
            train_teams = completed_sorted.iloc[:train_end][["home_team", "away_team"]]
            train_targets = completed_sorted.iloc[:train_end]["target"]

            target_encoder.fit(train_teams, train_targets)
            logger.info(
                "SafeTargetEncoder fitted on %d training rows (first %.0f%% of %d) "
                "— prior=%.4f (%d categories)",
                train_end, train_ratio * 100, len(completed),
                target_encoder.prior,
                sum(len(v) for v in target_encoder._category_means.values()),
            )

        # ── 3. Build features (with leakage-free encoder) ──
        from src.feature_engineering import build_features

        X, y = build_features(
            completed, is_training=True, config=cfg,
            encoder=target_encoder,
        )
        logger.info("Feature matrix: %d rows x %d cols", *X.shape)

        if len(X) < 20:
            raise ValueError(
                f"Only {len(X)} rows after feature engineering — "
                "need at least 20 for a meaningful train."
            )

        # ── 4. Chronological split ─────────────────────────
        from src.feature_engineering import train_val_test_split

        splits = train_val_test_split(X, y)
        split_sizes = {
            "train": len(splits["X_train"]),
            "val": len(splits["X_val"]),
            "test": len(splits["X_test"]),
        }
        logger.info("Split: %s", split_sizes)

        # ── 3b. Feature selection (AFTER split — fit on X_train only!) ──
        _fitted_selector = None
        if cfg.feature_selection.enabled:
            X_train_fs, selector = self._apply_feature_selection_post_split(
                splits["X_train"], splits["y_train"],
            )
            splits["X_train"] = X_train_fs
            splits["X_val"] = selector.transform(splits["X_val"])
            splits["X_test"] = selector.transform(splits["X_test"])
            _fitted_selector = selector
            logger.info(
                "Feature selection (%s) fit on X_train only: %d -> %d features",
                cfg.feature_selection.method,
                splits["X_train"].shape[1] + (X.shape[1] - splits["X_train"].shape[1]),
                splits["X_train"].shape[1],
            )

        # ── 5. Optional hyper-parameter tuning ─────────────
        tuning_report: dict | None = None
        if tune_hyperparams:
            tuning_report = self._run_tuning(
                splits["X_train"], splits["y_train"],
                n_folds=cv_folds,
                model_type=model_type,
            )

        # ── 6. Train final model ───────────────────────────
        from src.train import train_model, save_model

        # Apply model type (save & restore original to avoid permanent config mutation)
        orig_model_type = cfg.train.model_type
        try:
            if model_type is not None:
                cfg.train.model_type = model_type

            model, history = train_model(
                splits["X_train"], splits["y_train"],
                splits["X_val"], splits["y_val"],
                config=cfg,
            )
        finally:
            cfg.train.model_type = orig_model_type

        # ── 7. Test-set evaluation ─────────────────────────
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            log_loss,
        )

        y_pred = model.predict(splits["X_test"])
        y_proba = model.predict_proba(splits["X_test"])

        test_accuracy = float(accuracy_score(splits["y_test"], y_pred))
        test_log_loss = float(log_loss(splits["y_test"], y_proba))
        cm = confusion_matrix(splits["y_test"], y_pred).tolist()
        class_report = classification_report(
            splits["y_test"], y_pred,
            target_names=["Away Win", "Draw", "Home Win"],
            output_dict=True, zero_division=0,
        )

        metrics = {
            "test_accuracy": round(test_accuracy, 4),
            "test_log_loss": round(test_log_loss, 4),
            "test_samples": int(len(splits["y_test"])),
            "train_log_loss": history.get("train_loss", [None])[0],
            "val_log_loss": history.get("val_loss", [None])[0],
            "val_accuracy": history.get("val_accuracy", [None])[0],
            "confusion_matrix": cm,
            "classification_report": class_report,
        }

        # ── 8. Save model as artifact (bundles feature names + metadata) ──
        all_feature_names = X.columns.tolist()
        selected_feature_names = (
            splits["X_train"].columns.tolist()
            if _fitted_selector is not None
            else all_feature_names
        )

        from datetime import datetime, timezone
        artifact_kwargs = {
            "feature_names": all_feature_names,
            "selected_feature_names": selected_feature_names,
            "model_type": cfg.train.model_type if model_type is None else model_type,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "target_encoder_state": target_encoder.get_state(),
        }

        artifact_path = save_model(
            model, config=cfg, **artifact_kwargs,
        )
        logger.info("Model artifact saved to %s", artifact_path)

        # Feature importance (if available)
        feature_importances = self._extract_importances(model, X)

        report = {
            "model_type": model_type or cfg.train.model_type,
            "model_path": artifact_path,
            "data_path": str(data_path),
            "metrics": metrics,
            "features": {
                "count": X.shape[1],
                "columns": X.columns.tolist(),
                "selected_count": len(selected_feature_names),
                "importances_top_15": feature_importances,
            },
            "splits": split_sizes,
            "hyperparameter_tuning": tuning_report,
        }

        return report

    def evaluate(self, model_path: str | Path) -> dict:
        """Evaluate a trained model on held-out test data.

        Loads the model, finds the most relevant dataset automatically,
        builds features, runs prediction, and computes standard metrics.

        Parameters
        ----------
        model_path : str | Path
            Path to the saved model file (``.joblib``), relative to
            ``models/`` or absolute.

        Returns
        -------
        dict
            Evaluation metrics: accuracy, log-loss, confusion matrix,
            classification report, and feature importances.
        """
        logger.info("Evaluating model: %s", model_path)

        # ── 1. Load model ───────────────────────────────────
        cfg_eval = self._config
        from src.train import load_model

        model_path_str = str(model_path)
        try:
            # If it's just a filename relative to models/ dir
            model = load_model(Path(model_path_str).name, config=cfg_eval)
        except FileNotFoundError:
            # Try absolute / explicit path
            import joblib
            path = Path(model_path_str)
            if path.exists():
                model = joblib.load(path)
            else:
                raise

        # ── 2. Load & prepare test data (pipeline) ─────────
        data_path = resolve_data_path(config=cfg_eval)
        if not data_path.exists():
            return {"error": f"No evaluation data found at {data_path}"}

        df = load_and_prepare(None, add_temporal=False, config=cfg_eval)

        completed = df[df["result"].notna()].copy()

        from src.feature_engineering import build_features, train_val_test_split
        X, y = build_features(completed, is_training=True, config=cfg_eval)
        splits = train_val_test_split(X, y, config=cfg_eval)
        X_test, y_test = splits["X_test"], splits["y_test"]

        if len(X_test) == 0:
            return {"error": "No test samples available after split."}

        # ── 3. Predict ─────────────────────────────────────
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # ── 4. Metrics ───────────────────────────────────────
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            log_loss,
        )

        accuracy = float(accuracy_score(y_test, y_pred))
        ll = float(log_loss(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()
        class_report = classification_report(
            y_test, y_pred,
            target_names=["Away Win", "Draw", "Home Win"],
            output_dict=True, zero_division=0,
        )

        result = {
            "accuracy": round(accuracy, 4),
            "log_loss": round(ll, 4),
            "samples": int(len(y_test)),
            "confusion_matrix": cm,
            "classification_report": class_report,
            "feature_importances": self._extract_importances(model, X),
            "model_path": str(model_path),
        }

        logger.info("Evaluation — accuracy=%.4f, log-loss=%.4f", accuracy, ll)
        return result

    def list_models(self) -> list[dict]:
        """List all trained models with metadata.

        Scans the ``models/`` directory for ``.joblib`` files and
        extracts basic metadata (file size, modification time).

        Returns
        -------
        list[dict]
            Each dict contains keys: ``file_name``, ``path``,
            ``size_bytes``, ``modified``.
        """
        if not self._model_dir.exists():
            logger.warning("Model directory %s does not exist.", self._model_dir)
            return []

        models: list[dict] = []
        for fpath in sorted(self._model_dir.glob("*.joblib")):
            stat = fpath.stat()
            models.append({
                "file_name": fpath.name,
                "path": str(fpath.relative_to(self._config.paths.models)),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime,
            })

        logger.info("Found %d models in %s", len(models), self._model_dir)
        return models

    # ── Internals ──────────────────────────────────────────

    def _run_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_folds: int | None = None,
        model_type: str | None = None,
    ) -> dict:
        """Run hyper-parameter tuning and return a summary.

        Saves and restores ``config.train.model_type`` so the caller's
        config is not mutated permanently.
        """
        cfg_tune = self._config
        from src.train import tune_hyperparameters

        orig_model_type = cfg_tune.train.model_type
        try:
            if model_type is not None:
                cfg_tune.train.model_type = model_type

            best_params = tune_hyperparameters(
                X_train, y_train,
                n_folds=n_folds or cfg_tune.train.cv_folds,
                n_iter=50, verbose=False,
                config=cfg_tune,
            )

            # Apply best params to config
            for key, val in best_params.items():
                if hasattr(cfg_tune.train, key):
                    setattr(cfg_tune.train, key, val)

            return {
                "performed": True,
                "best_params": {k: v for k, v in best_params.items()},
                "cv_folds": n_folds or cfg_tune.train.cv_folds,
            }
        except Exception as exc:
            logger.warning("Hyper-parameter tuning failed: %s — using defaults", exc)
            return {"performed": True, "best_params": {}, "error": str(exc)}
        finally:
            cfg_tune.train.model_type = orig_model_type

    # ── Feature Selection (LEAKAGE-FREE) ──────────────────

    def _apply_feature_selection_post_split(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> tuple[pd.DataFrame, Any]:
        """Fit feature selector **only** on training data to prevent leakage.

        Returns (X_train_reduced, fitted_selector).
        Caller must ``selector.transform(X_val)`` and ``selector.transform(X_test)``.

        Steps:
        1. Drop highly-correlated feature pairs (if configured).
        2. Select top-k features via the chosen method.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix (chronologically first).
        y_train : pd.Series
            Training target vector.

        Returns
        -------
        tuple[pd.DataFrame, object]
            Reduced training DataFrame and the fitted sklearn selector/pipeline.
        """
        fs = self._config.feature_selection

        # ── Step 1: Drop highly-correlated redundant pairs ──
        selector = self._build_selector(X_train, y_train)
        X_reduced = selector.fit_transform(X_train, y_train)
        return X_reduced, selector

    def _build_selector(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Build a sklearn-compatible selector pipeline (unfitted)."""
        from sklearn.pipeline import Pipeline
        fs = self._config.feature_selection
        steps: list[tuple[str, Any]] = []

        if fs.drop_redundant_first and fs.correlation_threshold < 1.0:
            # DropCorrelated — custom transformer
            from sklearn.base import BaseEstimator, TransformerMixin

            class DropCorrelated(BaseEstimator, TransformerMixin):
                def fit(self, X, y=None):
                    corr = X.corr().abs()
                    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                    to_drop = [
                        col for col in upper.columns
                        if any(upper[col] > fs.correlation_threshold)
                        and col in upper.columns
                    ]
                    self.to_drop_ = to_drop
                    return self

                def transform(self, X):
                    return X.drop(columns=[c for c in self.to_drop_ if c in X.columns], errors="ignore")

            steps.append(("drop_corr", DropCorrelated()))

        # Step 2: Select top-k features
        method = fs.method
        if method == "mutual_info":
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            n_target = min(fs.n_features, X.shape[1] - 1)
            if n_target > 0:
                steps.append(("select", SelectKBest(mutual_info_classif, k=n_target)))

        elif method == "l1":
            from sklearn.feature_selection import SelectFromModel
            from sklearn.linear_model import LogisticRegression
            l1 = LogisticRegression(
                penalty="l1", solver="saga", C=0.1,
                max_iter=2000, random_state=42, n_jobs=1,
            )
            steps.append(("select", SelectFromModel(l1, threshold="mean", max_features=fs.n_features)))

        elif method == "rfe":
            from sklearn.feature_selection import RFE
            from sklearn.linear_model import LogisticRegression
            estimator = LogisticRegression(
                solver="lbfgs", max_iter=1000,
                random_state=42, n_jobs=1,
            )
            n_target = min(fs.n_features, X.shape[1] - 1)
            if n_target > 0:
                steps.append(("rfe", RFE(estimator, n_features_to_select=n_target, step=0.2)))

        elif method == "threshold":
            from sklearn.base import BaseEstimator, TransformerMixin

            class ThresholdSelector(BaseEstimator, TransformerMixin):
                def fit(self, X, y=None):
                    from sklearn.ensemble import RandomForestClassifier
                    rf = RandomForestClassifier(
                        n_estimators=100, max_depth=6,
                        random_state=42, n_jobs=-1,
                    )
                    rf.fit(X, y)
                    importances = pd.Series(rf.feature_importances_, index=X.columns)
                    self.selected_cols_ = importances[
                        importances >= fs.importance_threshold
                    ].index.tolist()
                    if not self.selected_cols_:
                        # Fall back to top feature by importance
                        self.selected_cols_ = [importances.idxmax()]
                    return self

                def transform(self, X):
                    keep = [c for c in self.selected_cols_ if c in X.columns]
                    return X[keep]

            steps.append(("threshold", ThresholdSelector()))

        if not steps:
            # No-op passthrough
            from sklearn.base import BaseEstimator, TransformerMixin

            class Passthrough(BaseEstimator, TransformerMixin):
                def fit(self, X, y=None):
                    return self
                def transform(self, X):
                    return X

            steps.append(("passthrough", Passthrough()))

        return Pipeline(steps)

    def _apply_feature_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """[DEPRECATED] Old leakage-prone selection — kept for backward compat.

        Prefer ``_apply_feature_selection_post_split`` which fits on
        training data only.
        """
        logger.warning(
            "_apply_feature_selection is LEAKAGE-PRONE — use _apply_feature_selection_post_split"
        )
        X_reduced, _ = self._apply_feature_selection_post_split(X, y)
        return X_reduced
        fs = self._config.feature_selection
        original_n = X.shape[1]
        feature_names = X.columns.tolist()

        # ── Step 1: Drop highly-correlated redundant pairs ──
        if fs.drop_redundant_first and fs.correlation_threshold < 1.0:
            try:
                from src.feature_selection import find_redundant_pairs, drop_redundant
                pairs = find_redundant_pairs(X, threshold=fs.correlation_threshold)
                if pairs:
                    # Compute MI ranking for tie-breaking
                    from sklearn.feature_selection import mutual_info_classif
                    mi = mutual_info_classif(X, y, random_state=42)
                    ranking = [f for _, f in sorted(zip(mi, feature_names), reverse=True)]
                    to_drop = drop_redundant(pairs, ranking)
                    X = X.drop(columns=[c for c in to_drop if c in X.columns], errors="ignore")
                    logger.info(
                        "Dropped %d highly-correlated features (r>%.2f)",
                        len(to_drop), fs.correlation_threshold,
                    )
                    # Update feature_names after dropping columns
                    feature_names = X.columns.tolist()
            except Exception as exc:
                logger.warning("Redundant feature removal failed: %s — skipping", exc)

        # ── Step 2: Select top-k features ──────────────────
        try:
            n_target = min(fs.n_features, X.shape[1] - 1)
            if n_target <= 0:
                return X

            if fs.method in ("mutual_info",):
                from sklearn.feature_selection import SelectKBest, mutual_info_classif
                selector = SelectKBest(mutual_info_classif, k=n_target)
                selector.fit(X, y)
                mask = selector.get_support()
                selected = [f for f, s in zip(feature_names, mask) if s]
                keep = [c for c in selected if c in X.columns]
                if keep:
                    X = X[keep]

            elif fs.method == "l1":
                from sklearn.linear_model import LogisticRegression
                from sklearn.feature_selection import SelectFromModel
                l1 = LogisticRegression(
                    penalty="l1", solver="saga", C=0.1,
                    max_iter=2000, random_state=42, n_jobs=1,
                )
                selector = SelectFromModel(l1, threshold="mean", max_features=n_target)
                selector.fit(X, y)
                mask = selector.get_support()
                selected = [f for f, s in zip(feature_names, mask) if s]
                keep = [c for c in selected if c in X.columns]
                if keep:
                    X = X[keep]

            elif fs.method == "rfe":
                from sklearn.feature_selection import RFE
                from sklearn.linear_model import LogisticRegression
                estimator = LogisticRegression(
                    solver="lbfgs", max_iter=1000,
                    random_state=42, n_jobs=1,
                )
                rfe = RFE(estimator, n_features_to_select=n_target, step=0.2)
                rfe.fit(X, y)
                mask = rfe.support_
                selected = [f for f, s in zip(feature_names, mask) if s]
                keep = [c for c in selected if c in X.columns]
                if keep:
                    X = X[keep]

            elif fs.method == "threshold":
                # Use feature importances from a quick RandomForest
                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier(
                    n_estimators=100, max_depth=6,
                    random_state=42, n_jobs=-1,
                )
                rf.fit(X, y)
                importances = pd.Series(rf.feature_importances_, index=feature_names)
                keep = importances[importances >= fs.importance_threshold].index.tolist()
                keep = [c for c in keep if c in X.columns]
                if keep:
                    X = X[keep]

            logger.info(
                "Feature selection (%s): %d -> %d features",
                fs.method, original_n, X.shape[1],
            )
        except Exception as exc:
            logger.warning(
                "Feature selection method '%s' failed: %s — using original features",
                fs.method, exc,
            )

        return X

    @staticmethod
    def _extract_importances(model: Any, X: pd.DataFrame) -> list[dict] | None:
        """Extract top-15 feature importances if the model exposes them."""
        if not hasattr(model, "feature_importances_"):
            return None

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        return [
            {
                "rank": rank + 1,
                "feature": X.columns[idx],
                "importance": round(float(importances[idx]), 4),
            }
            for rank, idx in enumerate(indices)
        ]
