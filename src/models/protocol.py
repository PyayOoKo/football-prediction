"""
Model Protocol Definitions

This module defines standard interfaces for all prediction models,
eliminating the need for Phase 3 vs Phase 4 detection logic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class IModel(Protocol):
    """Standard interface for all prediction models.
    
    All models must implement predict_proba that accepts both feature
    matrices (X) and raw match data (df_raw), handling the appropriate
    prediction method internally.
    
    This protocol eliminates the need for Phase 3 vs Phase 4 detection
    by requiring a unified interface.
    """
    
    def predict_proba(
        self,
        X: pd.DataFrame | None = None,
        df_raw: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Predict match outcome probabilities.
        
        Parameters
        ----------
        X : pd.DataFrame, optional
            Feature matrix for ML models. Required if df_raw not provided.
        df_raw : pd.DataFrame, optional
            Raw match data for statistical models. Required if X not provided.
            
        Returns
        -------
        np.ndarray
            Probability array of shape (n, 3) with columns
            [away_prob, draw_prob, home_prob]. Rows sum to 1.0.
            
        Raises
        ------
        ValueError
            If neither X nor df_raw is provided, or if required
            input is missing for the model type.
        """
        ...
    
    def predict(
        self,
        X: pd.DataFrame | None = None,
        df_raw: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Predict hard class labels (0=Away, 1=Draw, 2=Home).
        
        Parameters
        ----------
        X : pd.DataFrame, optional
            Feature matrix for ML models.
        df_raw : pd.DataFrame, optional
            Raw match data for statistical models.
            
        Returns
        -------
        np.ndarray
            Array of predicted class labels.
        """
        ...


@runtime_checkable
class ITrainableModel(Protocol):
    """Extended interface for models that support training."""
    
    def fit(
        self,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        df_raw: pd.DataFrame | None = None,
    ) -> ITrainableModel:
        """Train the model.
        
        Parameters
        ----------
        X : pd.DataFrame, optional
            Feature matrix for ML models.
        y : pd.Series, optional
            Target labels.
        df_raw : pd.DataFrame, optional
            Raw match data for statistical models.
            
        Returns
        -------
        ITrainableModel
            Self for method chaining.
        """
        ...


def ensure_predict_proba(model: object) -> IModel:
    """Wrap a model to ensure it conforms to IModel protocol.
    
    This adapter allows legacy models with only predict_matches or
    only sklearn-style predict_proba to work with the unified interface.
    
    Parameters
    ----------
    model : object
        Model instance to wrap.
        
    Returns
    -------
    IModel
        Wrapped model conforming to IModel protocol.
        
    Raises
    ------
    TypeError
        If model has neither predict_proba nor predict_matches.
    """
    if isinstance(model, IModel):
        return model
    
    # Check what methods the model has
    has_predict_matches = hasattr(model, 'predict_matches')
    has_predict_proba = hasattr(model, 'predict_proba')
    
    if not has_predict_matches and not has_predict_proba:
        raise TypeError(
            f"Model {type(model).__name__} has neither predict_proba "
            f"nor predict_matches method"
        )
    
    # Return appropriate adapter
    if has_predict_matches:
        return _PredictMatchesAdapter(model)
    else:
        return _SklearnAdapter(model)


class _PredictMatchesAdapter:
    """Adapter for models with predict_matches method (legacy Phase 3)."""
    
    def __init__(self, model: object):
        self._model = model
    
    def predict_proba(
        self,
        X: pd.DataFrame | None = None,
        df_raw: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Delegate to model's predict_matches method."""
        if df_raw is None or df_raw.empty:
            raise ValueError(
                f"{type(self._model).__name__} requires df_raw parameter"
            )
        
        preds_df = self._model.predict_matches(df_raw)
        return np.column_stack([
            preds_df["away_win_prob"].values,
            preds_df["draw_prob"].values,
            preds_df["home_win_prob"].values,
        ])
    
    def predict(
        self,
        X: pd.DataFrame | None = None,
        df_raw: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Get probabilities and return argmax."""
        probs = self.predict_proba(X=X, df_raw=df_raw)
        return np.argmax(probs, axis=1)


class _SklearnAdapter:
    """Adapter for sklearn-style models (legacy Phase 4)."""
    
    def __init__(self, model: object):
        self._model = model
    
    def predict_proba(
        self,
        X: pd.DataFrame | None = None,
        df_raw: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Delegate to model's predict_proba method."""
        if X is None or X.empty:
            raise ValueError(
                f"{type(self._model).__name__} requires X parameter"
            )
        
        # Handle NaN values gracefully
        try:
            return np.asarray(self._model.predict_proba(X), dtype=np.float64)
        except Exception:
            col_means = X.mean().fillna(0) if hasattr(X, "mean") else 0
            X_clean = X.fillna(col_means) if hasattr(X, "fillna") else X
            return np.asarray(
                self._model.predict_proba(X_clean), dtype=np.float64
            )
    
    def predict(
        self,
        X: pd.DataFrame | None = None,
        df_raw: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Delegate to model's predict method."""
        if X is None:
            raise ValueError(
                f"{type(self._model).__name__} requires X parameter"
            )
        return np.asarray(self._model.predict(X))
