"""
Train — orchestrate model training with cross-validation and hyper-parameter tuning.

Typical usage::

    from src.train import train_model, tune_hyperparameters

    # Tune first, then train final model
    best_params = tune_hyperparameters(X_train, y_train)
    model, history = train_model(X_train, y_train, X_val, y_val)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV

from config import config
from src.time_series_cv import create_time_series_folds

logger = logging.getLogger(__name__)


# ── Public API ──────────────────────────────────────────


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> tuple[Any, dict[str, list[float]]]:
    """Train a model on the provided data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    X_val : pd.DataFrame, optional
        Validation feature matrix.
    y_val : pd.Series, optional
        Validation target vector.

    Returns
    -------
    model : Any
        Trained model object.
    history : dict[str, list[float]]
        Training history (loss, metrics).
    """
    logger.info("Starting training with model_type='%s'", config.train.model_type)

    model = _build_model()
    history: dict[str, list[float]] = {}

    if config.train.model_type in {"logistic_regression", "random_forest"}:
        model, history = _train_sklearn(model, X_train, y_train, X_val, y_val)
    elif config.train.model_type in {"xgboost", "lightgbm"}:
        model, history = _train_gbdt(model, X_train, y_train, X_val, y_val)
    elif config.train.model_type == "neural_network":
        model, history = _train_neural_net(model, X_train, y_train, X_val, y_val)
    else:
        raise ValueError(f"Unknown model_type: {config.train.model_type}")

    logger.info("Training complete.")
    return model, history


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_folds: int | None = None,
    n_iter: int = 80,
    verbose: bool = True,
) -> dict[str, Any]:
    """Randomised search cross-validation to find the best hyper-parameters.

    Uses ``RandomizedSearchCV`` instead of ``GridSearchCV`` to keep run
    times practical (the full grid for XGBoost would be 2,187+ combinations).

    **Time-series aware:** Uses ``TimeSeriesSplit`` instead of standard k-fold
    CV to prevent future data from leaking into training folds.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    n_folds : int, optional
        Number of CV folds.  Defaults to ``config.train.cv_folds``.
        These become ``TimeSeriesSplit`` folds — strictly chronological,
        no shuffle.
    n_iter : int
        Number of random parameter samples to try (default 80).
    verbose : bool
        Print progress.

    Returns
    -------
    dict[str, Any]
        Best hyper-parameters found by the search (does **not** mutate
        ``config`` — the caller must apply them).
    """
    model_type = config.train.model_type
    logger.info(
        "Hyper-parameter tuning '%s' — %s-fold CV, %d random samples",
        model_type, n_folds or config.train.cv_folds, n_iter,
    )

    if model_type == "logistic_regression":
        param_dist = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
        }
        base_model = LogisticRegression(
            max_iter=2000,
            random_state=config.train.seed, class_weight="balanced",
        )
        # Small grid — use exact search
        from sklearn.model_selection import GridSearchCV
        cv = create_time_series_folds(n_splits=n_folds or config.train.cv_folds)
        searcher = GridSearchCV(
            base_model, param_dist, cv=cv,
            scoring="neg_log_loss", n_jobs=-1, verbose=1 if verbose else 0,
        )
    elif model_type == "xgboost":
        import xgboost as xgb
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_lambda": [0.01, 0.1, 1.0, 5.0, 10.0],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0],
            "min_child_weight": [1, 3, 5, 7],
        }
        base_model = xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=config.train.seed,
            n_jobs=-1,
        )
        cv = create_time_series_folds(n_splits=n_folds or config.train.cv_folds)
        searcher = RandomizedSearchCV(
            base_model, param_dist, n_iter=n_iter,
            cv=cv,
            scoring="neg_log_loss", n_jobs=-1,
            random_state=config.train.seed,
            verbose=1 if verbose else 0,
        )
    elif model_type == "lightgbm":
        import lightgbm as lgb
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 8, -1],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_lambda": [0.01, 0.1, 1.0, 5.0, 10.0],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0],
            "num_leaves": [15, 31, 63, 127],
            "min_child_samples": [5, 10, 20, 50],
        }
        base_model = lgb.LGBMClassifier(
            objective="multiclass",
            metric="multi_logloss",
            random_state=config.train.seed,
            n_jobs=-1,
            verbose=-1,
        )
        cv = create_time_series_folds(n_splits=n_folds or config.train.cv_folds)
        searcher = RandomizedSearchCV(
            base_model, param_dist, n_iter=n_iter,
            cv=cv,
            scoring="neg_log_loss", n_jobs=-1,
            random_state=config.train.seed,
            verbose=1 if verbose else 0,
        )
    elif model_type == "random_forest":
        param_dist = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [4, 6, 8, 10, 15, None],
            "min_samples_leaf": [2, 5, 10, 20],
        }
        base_model = RandomForestClassifier(
            random_state=config.train.seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        cv = create_time_series_folds(n_splits=n_folds or config.train.cv_folds)
        searcher = RandomizedSearchCV(
            base_model, param_dist, n_iter=n_iter,
            cv=cv,
            scoring="neg_log_loss", n_jobs=-1,
            random_state=config.train.seed,
            verbose=1 if verbose else 0,
        )
    else:
        raise NotImplementedError(f"Tuning not implemented for '{model_type}'")

    # XGBoost/LightGBM handle NaN natively — no imputation needed
    if model_type in ("xgboost", "lightgbm"):
        searcher.fit(X_train, y_train)
    else:
        searcher.fit(X_train.fillna(X_train.mean().fillna(0)), y_train)

    logger.info(
        "Best CV log-loss: %.4f  with params: %s",
        -searcher.best_score_,
        searcher.best_params_,
    )

    return searcher.best_params_


def save_model(model: Any, file_name: str | None = None) -> str:
    """Serialise a trained model to ``models/`` via joblib.

    Parameters
    ----------
    model : Any
        Trained model object.
    file_name : str, optional
        Output file name.  Defaults to ``{model_type}_model.joblib``.

    Returns
    -------
    str
        Path to the saved model file.
    """
    if file_name is None:
        model_type = config.train.model_type
        file_name = f"{model_type}_model.joblib"

    path = config.paths.models / file_name
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)
    logger.info("Model saved to %s", path)
    return str(path)


def load_model(file_name: str) -> Any:
    """Load a serialised model from ``models/``.

    Parameters
    ----------
    file_name : str
        File name within the ``models/`` directory.

    Returns
    -------
    Any
        Deserialised model object.
    """
    path = config.paths.models / file_name
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model


# ── Internal: model factory ─────────────────────────────


def _build_model() -> Any:
    """Instantiate a fresh model per ``config.train.model_type``.

    Returns
    -------
    Any
        An untrained model instance.
    """
    cfg = config.train

    if cfg.model_type == "logistic_regression":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=cfg.seed,
            class_weight="balanced",
            C=1.0,
        )

    if cfg.model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )

    if cfg.model_type == "xgboost":
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

    if cfg.model_type == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            objective="multiclass",
            metric="multi_logloss",
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            reg_alpha=cfg.reg_alpha,
            num_leaves=31,
            min_child_samples=cfg.min_samples_leaf,
            random_state=cfg.seed,
            n_jobs=-1,
            verbose=-1,
        )

    raise NotImplementedError(f"_build_model for '{cfg.model_type}' is not yet implemented.")


def _train_sklearn(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | None,
) -> tuple[Any, dict[str, list[float]]]:
    """Train a scikit-learn model (LogisticRegression / RandomForest, etc.)."""
    col_means = X_train.mean().fillna(0)
    X_train_c = X_train.fillna(col_means)
    X_val_c = X_val.fillna(col_means) if X_val is not None else None

    model.fit(X_train_c, y_train)
    history = {"train_loss": [log_loss(y_train, model.predict_proba(X_train_c))]}

    if X_val_c is not None and y_val is not None:
        history["val_loss"] = [log_loss(y_val, model.predict_proba(X_val_c))]
        history["val_accuracy"] = [model.score(X_val_c, y_val)]

    return model, history


def _train_gbdt(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | None,
) -> tuple[Any, dict[str, list[float]]]:
    """Train XGBoost / LightGBM with early stopping.

    XGBoost/LightGBM handle NaN natively — no imputation needed.
    """
    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))

    # Different parameter names for XGBoost vs LightGBM
    is_lgbm = config.train.model_type == "lightgbm"
    if is_lgbm:
        model.set_params(
            metric="multi_logloss",
            early_stopping_round=10,
        )
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
    else:
        model.set_params(eval_metric="mlogloss", early_stopping_rounds=10)
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )

    # Use best_iteration if available (from early stopping)
    # XGBoost uses ``best_iteration`` (no underscore); LightGBM uses ``best_iteration_``
    best_attr = "best_iteration_" if is_lgbm else "best_iteration"
    if hasattr(model, best_attr) and getattr(model, best_attr) is not None:
        best_n = getattr(model, best_attr) + 1
        max_n = model.get_params().get("n_estimators", "?")
        logger.info("Early stopped at iteration %d / %s", best_n, max_n)

    history = {"train_loss": [log_loss(y_train, model.predict_proba(X_train))]}

    if X_val is not None and y_val is not None:
        history["val_loss"] = [log_loss(y_val, model.predict_proba(X_val))]
        history["val_accuracy"] = [model.score(X_val, y_val)]

    return model, history


def _train_neural_net(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | None,
) -> tuple[Any, dict[str, list[float]]]:
    """Train a feed-forward neural network (PyTorch).

    Architecture: ``input → 128 → ReLU → Dropout → 64 → ReLU → Dropout → 32 → ReLU → 3 (softmax)``
    Uses AdamW optimizer, reduces learning rate on plateau, early stopping.
    Returns a sklearn-compatible wrapper.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        raise ImportError(
            "PyTorch is required for neural network training. "
            "Install it with: pip install torch"
        )

    cfg = config.train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_features = X_train.shape[1]
    n_classes = len(y_train.unique())

    # Prepare data (fill NaN with 0 for neural net)
    X_train_clean = X_train.fillna(0).copy()
    X_val_clean = X_val.fillna(0).copy() if X_val is not None else None

    def _to_tensor(X, y=None):
        X_t = torch.tensor(
            X.values if hasattr(X, "values") else X,
            dtype=torch.float32, device=device,
        )
        if y is None:
            return X_t
        y_t = torch.tensor(
            y.values if hasattr(y, "values") else y,
            dtype=torch.long, device=device,
        )
        return X_t, y_t

    # Build network (define inline for config access)
    hidden = cfg.hidden_layers or (128, 64, 32)
    layers_list = []
    prev = n_features
    for h in hidden:
        layers_list.append(nn.Linear(prev, h))
        layers_list.append(nn.ReLU())
        if cfg.dropout > 0:
            layers_list.append(nn.Dropout(cfg.dropout))
        prev = h
    layers_list.append(nn.Linear(prev, n_classes))
    net = nn.Sequential(*layers_list).to(device)

    X_train_t, y_train_t = _to_tensor(X_train_clean, y_train)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=cfg.batch_size, shuffle=True,
    )

    val_loader = None
    if X_val_clean is not None and y_val is not None:
        X_val_t, y_val_t = _to_tensor(X_val_clean, y_val)
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t),
            batch_size=cfg.batch_size, shuffle=False,
        )

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=cfg.learning_rate or 0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )

    # Training loop
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop = cfg.early_stopping_patience or 10
    max_epochs = cfg.epochs or 100

    net.train()
    for epoch in range(max_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_train_loss)

        if val_loader is not None:
            net.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    loss = criterion(net(X_batch), y_batch)
                    val_loss += loss.item()
                    n_val += 1
            avg_val_loss = val_loss / n_val if n_val > 0 else float("inf")
            history["val_loss"].append(avg_val_loss)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop:
                    logger.info(
                        "Early stopping at epoch %d (val_loss=%.4f)",
                        epoch + 1, avg_val_loss,
                    )
                    break
            net.train()

        if (epoch + 1) % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.debug(
                "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f  lr=%.6f",
                epoch + 1, max_epochs, avg_train_loss,
                history["val_loss"][-1] if history["val_loss"] else float("nan"),
                lr,
            )

    if val_loader is not None:
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                _, predicted = torch.max(net(X_batch), 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        history["val_accuracy"] = [correct / total]

    logger.info(
        "Neural net trained — %d epochs, final val_loss=%.4f",
        len(history["train_loss"]),
        history["val_loss"][-1] if history["val_loss"] else float("nan"),
    )

    wrapped = TorchWrapper(net, device)
    return wrapped, history


class TorchWrapper:
    """Wraps a PyTorch model with sklearn's predict/predict_proba interface.

    Handles NaN in input by filling with zeros (like sklearn's pipelines).
    """
    def __init__(self, net: Any, device: Any):
        self.net = net
        self.device = device
        self.net.eval()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X.fillna(0) if hasattr(X, "fillna") else X
        X_t = torch.tensor(
            X_clean.values if hasattr(X_clean, "values") else X_clean,
            dtype=torch.float32, device=self.device,
        )
        with torch.no_grad():
            outputs = self.net(X_t)
            return torch.argmax(outputs, dim=1).cpu().numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X.fillna(0) if hasattr(X, "fillna") else X
        X_t = torch.tensor(
            X_clean.values if hasattr(X_clean, "values") else X_clean,
            dtype=torch.float32, device=self.device,
        )
        with torch.no_grad():
            outputs = self.net(X_t)
            return torch.softmax(outputs, dim=1).cpu().numpy()
