"""FeatureBuilder — constructs feature vectors from fixtures."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureBuilder:
    def __init__(self) -> None:
        self._historical_data: pd.DataFrame | None = None
        self._feature_cols: list[str] = []

    def load_historical_data(self) -> pd.DataFrame | None:
        if self._historical_data is not None:
            return self._historical_data

        from src.data_loader import load_clean_data  # type: ignore[attr-defined]

        df = load_clean_data()
        if df is not None and not df.empty:
            self._historical_data = df
            return df

        processed = Path("data/processed/results_clean.csv")
        if processed.exists():
            df = pd.read_csv(processed, low_memory=False)
            if not df.empty:
                self._historical_data = df
                return df

        raw = Path("data/raw/worldcup_all.csv")
        if raw.exists():
            df = pd.read_csv(raw, low_memory=False)
            if not df.empty:
                self._historical_data = df
                return df

        return None

    def build_features(
        self,
        fixtures: list[dict[str, Any]],
    ) -> pd.DataFrame | None:
        historical = self.load_historical_data()
        if historical is None:
            logger.warning("No historical data for feature engineering")
            return None

        try:
            from src.feature_engineering import build_features

            fixture_rows = []
            for fix in fixtures:
                row = {
                    "date": pd.Timestamp(fix.get("match_date", datetime.now().strftime("%Y-%m-%d"))),
                    "home_team": fix["home_team"],
                    "away_team": fix["away_team"],
                    "result": "H",
                    "home_goals": 0,
                    "away_goals": 0,
                }
                for col in historical.columns:
                    if col not in row:
                        row[col] = historical[col].iloc[-1] if len(historical) > 0 else 0
                fixture_rows.append(row)

            df_ext = pd.concat(
                [historical, pd.DataFrame(fixture_rows)],
                ignore_index=True,
            )
            X_full, _ = build_features(df_ext, is_training=False)
            n_hist = len(historical)
            X_fixtures = X_full.iloc[n_hist:]
            self._feature_cols = list(X_full.columns)
            return X_fixtures

        except Exception as exc:
            logger.warning("Feature engineering failed: %s", exc)
            return None
