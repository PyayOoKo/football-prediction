"""
Data preprocessor — cleaning and transformation pipeline.

Applies a configurable sequence of transformations:
- Column renaming and type casting
- Missing value imputation
- Categorical encoding
- Temporal feature extraction (year, month, day of week)
- Team name normalisation

Lifecycle
---------
``fit()`` learns normalisation statistics from training data.
``transform()`` applies learned state only — never recomputes from inference data.
``fit_transform()`` calls ``fit()`` then ``transform()``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import config as _global_config

logger = logging.getLogger(__name__)

_TEMPORAL_FEATURES = [
    "year",
    "month",
    "day_of_week",
    "day_of_year",
    "week_of_season",
]


class DataPreprocessor:
    """Configurable data preprocessing pipeline.

    Parameters
    ----------
    normalise_teams : bool
        Whether to normalise team name spelling (default ``True``).
    add_temporal_features : bool
        Whether to add year/month/day-of-week columns (default ``True``).
    missing_strategy : str
        Strategy for handling missing values: ``"drop"``, ``"fill_zero"``,
        or ``"fill_median"`` (default from config).
    max_missing_pct : float
        Drop columns with more than this percentage of missing values
        (default from config).
    """

    def __init__(
        self,
        normalise_teams: bool = True,
        add_temporal_features: bool = True,
        missing_strategy: str | None = None,
        max_missing_pct: float | None = None,
        config: Any | None = None,
    ) -> None:
        cfg = config or _global_config
        self._cfg = cfg
        self.normalise_teams = normalise_teams
        self.add_temporal_features = add_temporal_features
        self.missing_strategy = missing_strategy or cfg.data_collection.missing_strategy
        self.max_missing_pct = max_missing_pct or cfg.data_collection.max_missing_pct
        self._team_normalizer: Any = None

        # Fitted state (learned from training data only)
        self._fitted = False
        self._median_values: dict[str, float] = {}
        self._team_map: dict[str, str] = {}
        self._category_encodings: dict[str, dict[str, int]] = {}
        self._expected_columns: list[str] = []

    # ── Public API ─────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> DataPreprocessor:
        """Learn preprocessing statistics from training data.

        Computes and stores:
        - Median values for numeric columns (``fill_median`` strategy).
        - Team name mappings.
        - Category encodings.
        - Expected feature column set.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.

        Returns
        -------
        DataPreprocessor
            Fitted instance.
        """
        logger.info("Fitting preprocessor on %d rows", len(df))
        df = df.copy()

        df = self._normalise_columns(df)
        df = self._parse_dates(df)
        df = self._compute_target(df)

        if self.normalise_teams:
            df, team_map = self._learn_team_map(df)
            self._team_map = team_map

        if self.add_temporal_features:
            df = self._add_temporal_features(df)

        # Store median values for fill_median strategy
        if self.missing_strategy == "fill_median":
            numeric = df.select_dtypes(include=[np.number])
            self._median_values = numeric.median().to_dict()

        # Store high-missingness columns to drop
        missing_pct = df.isnull().mean() * 100
        self._high_missing_cols = missing_pct[
            missing_pct > self.max_missing_pct
        ].index.tolist()

        # Store expected columns (after all transformations)
        df = self._handle_missing(df, fitting=True)
        df = self._drop_high_missingness(df)
        self._expected_columns = df.columns.tolist()

        self._fitted = True
        logger.info(
            "Fitted: %d expected columns, %d median values, %d team mappings",
            len(self._expected_columns),
            len(self._median_values),
            len(self._team_map),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted preprocessing to new (inference) data.

        Uses stored statistics from ``fit()`` — never recomputes
        from inference data.

        Parameters
        ----------
        df : pd.DataFrame
            Inference data.

        Returns
        -------
        pd.DataFrame
            Transformed data with stable feature columns.

        Raises
        ------
        RuntimeError
            If ``transform()`` is called before ``fit()``.
        """
        if not self._fitted:
            raise RuntimeError(
                "DataPreprocessor.transform() called before fit(). "
                "Call .fit(train_df) first."
            )

        logger.info("Transforming %d rows with fitted preprocessor", len(df))
        df = df.copy()

        df = self._normalise_columns(df)
        df = self._parse_dates(df)
        df = self._compute_target(df)

        if self.normalise_teams:
            df = self._apply_team_map(df)

        if self.add_temporal_features:
            df = self._add_temporal_features(df)

        df = self._handle_missing(df, fitting=False)
        df = self._drop_high_missingness(df)

        # Align columns to training set (add missing, drop extra)
        for col in self._expected_columns:
            if col not in df.columns:
                df[col] = 0
                logger.debug("Added missing column %s with default 0", col)
        extra = [c for c in df.columns if c not in self._expected_columns]
        if extra:
            df.drop(columns=extra, inplace=True)
            logger.debug("Dropped %d extra columns not in training set", len(extra))

        # Ensure column order matches training
        df = df[self._expected_columns]

        logger.info("Transform complete: %d rows × %d cols", len(df), len(df.columns))
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    # ── Stage 1: Normalise column names ─────────────────

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase and strip column names, map known aliases."""
        col_map: dict[str, str] = {
            "div": "league",
            "hometeam": "home_team",
            "awayteam": "away_team",
            "fthg": "home_goals",
            "ftag": "away_goals",
            "ftr": "result",
            "hthg": "home_goals_ht",
            "htag": "away_goals_ht",
            "htr": "result_ht",
            "date": "date",
            "season": "season",
            "league": "league",
            "round": "round",
            "group": "group",
            "ground": "ground",
            "source": "source",
            "downloaded_at": "downloaded_at",
            "home_xg": "home_xg",
            "away_xg": "away_xg",
        }
        renamed = {}
        for col in df.columns:
            key = col.strip().lower()
            renamed[col] = col_map.get(key, key)
        df = df.rename(columns=renamed)
        logger.debug("Normalised %d column names", len(df.columns))
        return df

    # ── Stage 2: Parse dates ────────────────────────────

    @staticmethod
    def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Parse the date column to datetime."""
        if "date" not in df.columns:
            logger.warning("No 'date' column found — skipping date parsing")
            return df

        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        n_nat = df["date"].isna().sum()
        if n_nat > 0:
            logger.warning("%d date values could not be parsed (set to NaT)", n_nat)
        return df

    # ── Stage 3: Compute target column ──────────────────

    @staticmethod
    def _compute_target(df: pd.DataFrame) -> pd.DataFrame:
        """Create the target column from result (H=2, D=1, A=0).

        Missing or unknown results are encoded as -1 (kept distinct from draw=1).
        Future fixtures with no result are NOT filled with 0 (which is away-win).
        """
        if "result" not in df.columns:
            df["target"] = -1
            logger.warning("No 'result' column found — target set to -1")
            return df

        result_map = {"H": 2, "D": 1, "A": 0}
        df["target"] = df["result"].map(result_map).fillna(-1).astype("int8")

        for col in ["home_goals", "away_goals"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "home_goals" in df.columns and "away_goals" in df.columns:
            df["goal_diff"] = df["home_goals"] - df["away_goals"]
            df["total_goals"] = df["home_goals"] + df["away_goals"]

        n_valid = (df["target"] >= 0).sum()
        n_missing = (df["target"] < 0).sum()
        logger.debug(
            "Target computed: %d valid, %d missing",
            n_valid,
            n_missing,
        )
        return df

    # ── Stage 4: Normalise team names ───────────────────

    def _learn_team_map(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """Build a team-name canonicalisation map from training data."""
        team_map: dict[str, str] = {}

        try:
            if self._team_normalizer is None:
                from src.team_normalizer import TeamNormalizer

                self._team_normalizer = TeamNormalizer(
                    log_resolutions=False,
                    log_low_confidence=False,
                )
        except ImportError:
            logger.warning("TeamNormalizer not available — using fallback")
            return self._fallback_learn_team_map(df)

        for col in ["home_team", "away_team"]:
            if col not in df.columns:
                continue
            for name in df[col].dropna().unique():
                resolved = self._team_normalizer.resolve(str(name)).canonical
                team_map[str(name)] = resolved

        df = self._apply_team_map(df, team_map)
        logger.info("Learnt %d team name mappings", len(team_map))
        return df, team_map

    def _apply_team_map(
        self,
        df: pd.DataFrame,
        team_map: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Apply stored team name map to *df*."""
        mapping = team_map if team_map is not None else self._team_map
        if not mapping:
            return df

        changes = 0
        for col in ["home_team", "away_team"]:
            if col not in df.columns:
                continue
            original = df[col].astype(str)
            resolved = original.map(lambda x: mapping.get(x, x))
            changed = (original != resolved).sum()
            changes += changed
            df[col] = resolved
            if changed > 0:
                logger.debug("  %s: %d names normalised", col, changed)
        logger.info("Applied team name map: %d changes", changes)
        return df

    @staticmethod
    def _fallback_learn_team_map(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """Fallback: build a simple title-case map."""
        team_map: dict[str, str] = {}
        for col in ["home_team", "away_team"]:
            if col not in df.columns:
                continue
            for name in df[col].dropna().unique():
                cleaned = str(name).strip().title()
                team_map[str(name)] = cleaned
        return df, team_map

    # ── Stage 5: Add temporal features ──────────────────

    @staticmethod
    def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features derived from the date column.

        Features: year, month, day_of_week, day_of_year, week_of_season,
        is_midweek.
        """
        if "date" not in df.columns or df["date"].isnull().all():
            logger.warning("No valid dates — temporal features skipped")
            return df

        df["year"] = df["date"].dt.year.astype("Int64")
        df["month"] = df["date"].dt.month.astype("Int64")
        df["day_of_week"] = df["date"].dt.dayofweek.astype("Int64")
        df["day_of_year"] = df["date"].dt.dayofyear.astype("Int64")

        season_year = df["date"].dt.year.where(
            df["date"].dt.month >= 8,
            df["date"].dt.year - 1,
        )
        df["_aug_1st"] = pd.to_datetime(
            season_year.astype(str) + "-08-01", errors="coerce"
        )
        df["week_of_season"] = ((df["date"] - df["_aug_1st"]).dt.days // 7 + 1).astype(
            "Int64"
        )
        df.drop(columns=["_aug_1st"], inplace=True)

        df["is_midweek"] = df["day_of_week"].isin([1, 2, 3]).astype("int8")

        logger.debug(
            "Added %d temporal features: %s, is_midweek",
            len(_TEMPORAL_FEATURES),
            ", ".join(_TEMPORAL_FEATURES),
        )
        return df

    # ── Stage 6: Handle missing values ──────────────────

    def _handle_missing(self, df: pd.DataFrame, fitting: bool = False) -> pd.DataFrame:
        """Handle missing values per the configured strategy.

        During fitting this may store statistics; during transform
        it uses stored statistics only.
        """
        before = len(df)

        if self.missing_strategy == "drop":
            essential = ["home_team", "away_team", "result"]
            existing = [c for c in essential if c in df.columns]
            df = df.dropna(subset=existing)
            logger.info("Missing strategy 'drop': %d -> %d rows", before, len(df))

        elif self.missing_strategy == "fill_zero":
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(0)
            if "result" in df.columns:
                df["result"] = df["result"].fillna("")
            logger.info("Missing strategy 'fill_zero' applied")

        elif self.missing_strategy == "fill_median":
            if fitting:
                numeric = df.select_dtypes(include=[np.number]).columns
                for col in numeric:
                    self._median_values[col] = df[col].median()
            for col in df.columns:
                if col in self._median_values:
                    df[col] = df[col].fillna(self._median_values[col])
            logger.info("Missing strategy 'fill_median' applied")

        else:
            logger.warning(
                "Unknown missing strategy '%s' — skipping", self.missing_strategy
            )

        return df

    # ── Stage 7: Drop high-missingness columns ──────────

    def _drop_high_missingness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns with missing percentage above threshold."""
        if self.max_missing_pct >= 100:
            return df

        missing_pct = df.isnull().mean() * 100
        cols_to_drop = missing_pct[missing_pct > self.max_missing_pct].index.tolist()
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(
                "Dropped %d columns with >%.0f%% missing values: %s",
                len(cols_to_drop),
                self.max_missing_pct,
                cols_to_drop[:10],
            )
        return df

    # ── Serialisation support ──────────────────────────

    def get_state(self) -> dict[str, Any]:
        """Return serialisable state."""
        return {
            "fitted": self._fitted,
            "median_values": self._median_values,
            "team_map": self._team_map,
            "expected_columns": self._expected_columns,
            "missing_strategy": self.missing_strategy,
            "max_missing_pct": self.max_missing_pct,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> DataPreprocessor:
        """Restore from a state dict."""
        preprocessor = cls(
            missing_strategy=state.get("missing_strategy"),
            max_missing_pct=state.get("max_missing_pct"),
        )
        preprocessor._fitted = state.get("fitted", False)
        preprocessor._median_values = state.get("median_values", {})
        preprocessor._team_map = state.get("team_map", {})
        preprocessor._expected_columns = state.get("expected_columns", [])
        return preprocessor
