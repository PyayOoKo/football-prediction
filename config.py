"""
Project-wide configuration.

Centralises all paths, hyper-parameters, data sources,
and model settings in one place for easy experimentation.

Automatically loads ``.env`` file from the project root so
environment variables like ``THE_ODDS_API_KEY`` are always
available without manual setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Auto-load .env from project root — called once at import time
# so env vars are available project-wide without manual setup.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


# ── Data collection ─────────────────────────────────────
@dataclass
class DataCollectionConfig:
    """Settings for the data collection pipeline."""

    # League codes to collect (E0 = Premier League, E1 = Championship, etc.)
    leagues: tuple[str, ...] = ("E0",)

    # Number of most-recent seasons to download
    max_seasons: int = 10

    # Strategy for handling missing values
    missing_strategy: Literal["drop", "fill_zero", "fill_median"] = "fill_zero"

    # Output file name for the combined dataset
    output_file: str = "results.csv"

    # Max missing percentage before a column is dropped
    max_missing_pct: float = 50.0


# ── Preprocessing ───────────────────────────────────────
@dataclass
class PreprocessingConfig:
    """Settings for the data preprocessing pipeline."""

    # Input file (relative to ``data/raw/``)
    input_file: str = "results.csv"

    # Output file (relative to ``data/processed/``)
    output_file: str = "results_clean.csv"

    # Whether to normalise team names
    normalise_teams: bool = True

    # Whether to add temporal features (year, month, day_of_week, etc.)
    add_temporal_features: bool = True

    # Whether to save the cleaned dataset
    save_cleaned: bool = True


# ── Project root ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent


# ── Paths ───────────────────────────────────────────────
@dataclass
class Paths:
    """All managed directory & file paths."""

    # Data
    data: Path = PROJECT_ROOT / "data"
    raw: Path = data / "raw"
    processed: Path = data / "processed"
    external: Path = data / "external"

    # Models
    models: Path = PROJECT_ROOT / "models"

    # Notebooks
    notebooks: Path = PROJECT_ROOT / "notebooks"

    # Source
    src: Path = PROJECT_ROOT / "src"

    # App
    app: Path = PROJECT_ROOT / "app"

    def __post_init__(self) -> None:
        """Ensure all essential directories exist."""
        for d in (self.data, self.raw, self.processed, self.external,
                  self.models, self.notebooks, self.src, self.app):
            d.mkdir(parents=True, exist_ok=True)


# ── Data loading ────────────────────────────────────────
@dataclass
class DataConfig:
    """Settings related to data ingestion."""

    # Source: local CSV, API endpoint, or database connection string
    source: str = "local"  # "local" | "api" | "db"
    api_url: str = ""
    api_key_env: str = "FOOTBALL_DATA_API_KEY"

    # Local file name(s)
    fixtures_file: str = "fixtures.csv"
    results_file: str = "results.csv"
    teams_file: str = "teams.csv"

    # Train / validation / test split ratios (must sum to 1.0)
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)

    # Random seed for reproducibility
    seed: int = 42


# ── Feature engineering ─────────────────────────────────
@dataclass
class FeatureConfig:
    """Settings for feature-creation pipelines."""

    # Rolling window sizes (number of matches)
    form_window: int = 5
    rolling_windows: tuple[int, ...] = (5, 10, 20)
    rolling_avg_window: int = 10

    # Whether to include head-to-head features
    include_h2h: bool = True
    h2h_window: int = 6

    # Whether to include league-table position features
    include_league_position: bool = True

    # Encoding strategy for categorical columns
    categorical_encoding: Literal["label", "onehot", "target"] = "label"

    # Time decay halflife for rolling features (None = equal weight)
    # Set to a positive integer (e.g. 5 or 10) to give recent matches
    # exponentially more weight than older ones.
    time_decay_halflife: int | None = None

    # Whether to reset rolling features per season boundary
    # (avoids pre-season including stats from previous seasons)
    reset_per_season: bool = False


# ── Training ────────────────────────────────────────────
@dataclass
class TrainConfig:
    """Model training hyper-parameters."""

    # Algorithm
    model_type: Literal[
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "neural_network",
    ] = "xgboost"

    # Random forest / tree-specific
    n_estimators: int = 300
    max_depth: int = 8
    min_samples_leaf: int = 10

    # XGBoost / LightGBM
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.1

    # Neural-network specific
    hidden_layers: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.3
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10

    # Cross-validation
    cv_folds: int = 5

    # Objective — name of the target column
    target_column: str = "result"  # "home_win" | "draw" | "away_win"

    seed: int = 42


# ── Prediction ──────────────────────────────────────────
@dataclass
class PredictConfig:
    """Prediction-time settings."""

    # Probability threshold for binary classification
    probability_threshold: float = 0.5

    # Number of top predictions to display
    top_k: int = 10

    # Output format: "csv", "json", or "console"
    output_format: Literal["csv", "json", "console"] = "console"


# ── Odds API (Live Odds) ──────────────────────────────
@dataclass
class OddsAPIConfig:
    """Settings for The Odds API integration.

    Attributes
    ----------
    api_key_env : str
        Environment variable name for the API key (default ``THE_ODDS_API_KEY``).
    regions : str
        Bookmaker regions to query (default ``"uk,ie,eu"``).
    markets : str
        Markets to fetch (default ``"h2h"`` = head-to-head).
    cache_ttl : int
        Cache lifetime in seconds (default 3600 = 1 hour).
    request_timeout : int
        HTTP request timeout in seconds (default 15).
    sport_key_wc : str
        Sport key for World Cup / international soccer
        (default ``"soccer_fifa_world_cup"``).
    fallback_to_hardcoded : bool
        If no API key or API fails, fall back to hardcoded odds
        (default True).
    """

    api_key_env: str = "THE_ODDS_API_KEY"
    regions: str = "us,uk,eu"
    markets: str = "h2h"
    cache_ttl: int = 3600
    request_timeout: int = 15
    sport_key_wc: str = "soccer_fifa_world_cup"
    fallback_to_hardcoded: bool = True


# ── Value Betting ──────────────────────────────────────
@dataclass
class ValueBetConfig:
    """Settings for the value betting module."""

    # Default bankroll for Kelly stake calculation
    bankroll: float = 1000.0

    # Fraction of full Kelly to use (0.25 = 25% Kelly — conservative)
    kelly_fraction: float = 0.25

    # Minimum EV threshold to flag as a value bet
    min_ev: float = 0.0


# ── Odds Processing ────────────────────────────────────
@dataclass
class OddsConfig:
    """Settings for the odds processing module.

    Attributes
    ----------
    opening_odds_cols : tuple[str, str, str]
        Column names for opening odds ``(home, draw, away)``.
        Default: ``("BbMxH", "BbMxD", "BbMxA")``.
    closing_odds_cols : tuple[str, str, str]
        Column names for closing odds ``(home, draw, away)``.
        Default: ``("BbAvH", "BbAvD", "BbAvA")``.
    compute_consensus : bool
        Whether to compute multi-bookmaker consensus probabilities
        (default True).
    warn_missing : bool
        Log a warning when odds columns not found (default True).
    """

    opening_odds_cols: tuple[str, str, str] = ("BbMxH", "BbMxD", "BbMxA")
    closing_odds_cols: tuple[str, str, str] = ("BbAvH", "BbAvD", "BbAvA")
    compute_consensus: bool = True
    warn_missing: bool = True


# ── Player Information ─────────────────────────────────
@dataclass
class PlayerInfoConfig:
    """Settings for the player information module.

    This module is **optional** — set ``enabled=False`` to skip it
    entirely (no columns added, no processing time).

    Attributes
    ----------
    enabled : bool
        Whether to run the player info feature module (default True).
    default_age : float
        Neutral placeholder age when no player data (default 25).
    placeholder_value : float
        Neutral placeholder value for counts/flags when no data (default 0).
    warn_missing : bool
        Log a warning when no player data is provided (default True).
    """

    enabled: bool = False
    default_age: float = 25.0
    placeholder_value: float = 0.0
    warn_missing: bool = True


# ── Expected Goals (xG) Features ──────────────────────
@dataclass
class XgConfig:
    """Settings for the Expected Goals feature module."""

    # Rolling window sizes for xG/xGA averages
    rolling_windows: tuple[int, ...] = (5, 10)

    # Whether to compute Expected Points from xG using Poisson
    compute_xpts: bool = True

    # Max goals per team for the xPts probability table
    max_goals_table: int = 8

    # Placeholder value when no xG data is available (0 = all-zero placeholders)
    placeholder_value: float = 0.0

    # Whether to log a warning when xG columns are not found
    warn_missing: bool = True


# ── Poisson Model ─────────────────────────────────────
@dataclass
class PoissonConfig:
    """Settings for the Poisson regression model.

    Attributes
    ----------
    min_matches : int
        Minimum matches a team must play before its strengths are used.
        Teams with fewer matches default to league average (strength = 1.0).
    max_goals : int
        Maximum goals per team to consider in the probability table (0–*n*).
    """

    min_matches: int = 0
    max_goals: int = 8


# ── Elo Rating System ──────────────────────────────────
@dataclass
class EloConfig:
    """Settings for the dynamic Elo rating system.

    Attributes
    ----------
    k : int
        Base K-factor — how much a single match changes ratings (default 32).
    home_advantage : int
        Home advantage bonus in Elo points (default 100).
    initial_rating : int
        Starting Elo rating for unseen teams (default 1500).
    regress_to_mean : bool
        Regress ratings towards the mean between seasons (default True).
    regress_factor : float
        Fraction of distance to mean to regress each season (default 1/3).
    use_goal_margin : bool
        Scale K-factor by goal margin so bigger wins cause bigger changes
        (default True).
    max_goal_margin : int
        Cap on goal margin in K-factor adjustment (default 5).
    adjustments : dict[str, int]
        Manual Elo adjustments per team (e.g. ``{"Morocco": 100}`` subtracts 100
        from Morocco's rating after computation). Useful for applying domain
        knowledge or skepticism about a team's performance.
    """

    k: int = 32
    home_advantage: int = 100
    initial_rating: int = 1500
    regress_to_mean: bool = True
    regress_factor: float = 1 / 3
    use_goal_margin: bool = True
    max_goal_margin: int = 5
    adjustments: dict[str, int] = field(default_factory=dict)


# ── Hyper-parameter Tuning ───────────────────────────
@dataclass
class HyperTuneConfig:
    """Settings for the hyper-parameter tuning orchestrator.

    Attributes
    ----------
    model_types : tuple[str, ...]
        Which model types to tune (default: all three).
    n_iter_random : int
        Number of random parameter samples for RandomizedSearchCV (default 50).
    cv_folds : int
        Cross-validation folds for the search (default 5).
    save_models : bool
        Save baseline and tuned models to ``models/`` (default True).
    save_report : bool
        Write the comparison report to ``reports/`` (default True).
    verbose : bool
        Print progress during tuning (default True).
    """

    model_types: tuple[str, ...] = (
        "logistic_regression", "random_forest", "xgboost"
    )
    n_iter_random: int = 50
    cv_folds: int = 5
    save_models: bool = True
    save_report: bool = True
    verbose: bool = True


# ── Confidence Scoring ────────────────────────────────
@dataclass
class ConfidenceConfig:
    """Settings for the confidence scoring system.

    Attributes
    ----------
    weight_spread : float
        Weight for the probability spread component (default 0.40).
    weight_agreement : float
        Weight for the model agreement component (default 0.35).
    weight_calibration : float
        Weight for the historical calibration component (default 0.25).
    default_agreement : float
        Default agreement score when no ensemble predictions are provided
        (default 50 — neutral).
    default_calibration : float
        Default calibration score when no calibration data is available
        (default 50 — neutral).
    calibration_brier_default : float
        Fallback Brier score if none is provided (default 0.25 — moderate).
    """
    weight_spread: float = 0.40
    weight_agreement: float = 0.35
    weight_calibration: float = 0.25
    default_agreement: float = 50.0
    default_calibration: float = 50.0
    calibration_brier_default: float = 0.25


# ── Ensemble Model ────────────────────────────────────
@dataclass
class EnsembleConfig:
    """Settings for the ensemble prediction model.

    Attributes
    ----------
    model_names : tuple[str, ...]
        Which models to include in the ensemble.
        Default includes 5 diverse models for robust predictions.
    weight_grid_step : float
        Step size for grid search weight optimisation (default 0.05).
    tune_base_models : bool
        If True, run hyper-parameter tuning on base models before
        fitting the ensemble (default False).
    model_weight_ranges : dict[str, tuple[float, float]]
        Minimum and maximum weight boundaries for each model in the
        ensemble (as fractions of total weight, sum to 1.0).
        These ensure no single model dominates and each model type
        contributes meaningfully based on its strengths.
    """

    model_names: tuple[str, ...] = (
        "xgboost", "logistic_regression", "poisson"
    )
    weight_grid_step: float = 0.10
    tune_base_models: bool = False
    model_weight_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


# ── Backtesting ────────────────────────────────────────
@dataclass
class BacktestConfig:
    """Settings for the backtesting engine."""

    # Starting bankroll
    initial_bankroll: float = 1000.0

    # Fraction of full Kelly to use
    kelly_fraction: float = 0.25

    # Minimum EV to consider a bet
    min_ev: float = 0.0

    # Odds columns to use (preference order; engine tries each)
    odds_column_sets: tuple[tuple[str, str, str], ...] = (
        ("BbAvA", "BbAvD", "BbAvH"),
        ("B365A", "B365D", "B365H"),
        ("BWA", "BWD", "BWH"),
    )

    # Output directory for charts (relative to project root)
    output_dir: str = "reports/backtest"


# ── Evaluation ──────────────────────────────────────────
@dataclass
class EvalConfig:
    """Evaluation metrics and visualisation settings."""

    metrics: tuple[str, ...] = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "log_loss",
    )
    plot_confusion_matrix: bool = True
    plot_feature_importance: bool = True
    plot_roc_curve: bool = True
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ── Dixon-Coles Model ─────────────────────────────────
@dataclass
class DixonColesConfig:
    """Settings for the Dixon-Coles MLE model.

    Attributes
    ----------
    enabled : bool
        Whether to compute Dixon-Coles features (default False).
        **Disabled by default** — the MLE optimisation is extremely slow
        on large datasets (~1,800 refits over 17k rows at refit_every=10).
        Enable only for small datasets or when DC-specific features are
        critical (e.g. international tournaments with sparse H2H data).
    refit_every : int
        How often to refit the MLE model when adding features (default 500).
        Higher = faster but less responsive to recent form.
        500 means ~34 refits over 17k rows instead of ~1,794 at refit_every=10.
    decay_halflife_days : float
        Recency decay halflife in days. A match this many days ago gets 50%
        weight. Default 1460 (~4 years). Set to 0 to disable.
    use_importance : bool
        Apply tournament importance weighting (default True).
    rho_fixed : float | None
        Fix the tau-correction parameter (default None = estimate via MLE).
        Set to 0.0 for standard independent Poisson.
    regress_prior : bool
        Apply L2 prior on attack/defence parameters (default True).
    prior_strength : float
        Strength of the L2 prior (default 0.01).
    fit_intercept_only : bool
        Only estimate home advantage and rho (default False).
    """

    enabled: bool = False
    refit_every: int = 500
    decay_halflife_days: float = 1460.0
    use_importance: bool = True
    rho_fixed: float | None = None
    regress_prior: bool = True
    prior_strength: float = 0.01
    fit_intercept_only: bool = False


# ── Convenience singleton ───────────────────────────────
@dataclass
class Config:
    """Top-level config aggregating all sub-configs."""

    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    odds_api: OddsAPIConfig = field(default_factory=OddsAPIConfig)
    value_betting: ValueBetConfig = field(default_factory=ValueBetConfig)
    odds: OddsConfig = field(default_factory=OddsConfig)
    player_info: PlayerInfoConfig = field(default_factory=PlayerInfoConfig)
    xg: XgConfig = field(default_factory=XgConfig)
    poisson: PoissonConfig = field(default_factory=PoissonConfig)
    dixon_coles: DixonColesConfig = field(default_factory=DixonColesConfig)
    elo: EloConfig = field(default_factory=EloConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    hyper_tune: HyperTuneConfig = field(default_factory=HyperTuneConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    backtesting: BacktestConfig = field(default_factory=BacktestConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Global toggle
    verbose: bool = True


# Single importable instance
config = Config()
