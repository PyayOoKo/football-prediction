"""
Application settings — centralised configuration hierarchy.

All configuration is loaded from environment variables (via ``.env``)
and organised into typed dataclasses. Import the singleton ``config``
instance to access settings anywhere in the application.

Environment
-----------
``.env`` files are loaded automatically from the project root.
See ``.env.example`` for the full list of supported variables.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv

# ── Auto-load .env from project root ────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")


# ── Environment detection & validation ──────────────────
APP_ENV = os.environ.get("APP_ENV", "production").lower()
IS_PRODUCTION = APP_ENV == "production"

_REQUIRED_IN_PRODUCTION = [
    ("PREDICTION_API_KEY", "API authentication key"),
    ("DATABASE_URL", "Database connection string"),
]

if IS_PRODUCTION:
    missing = []
    for var_name, description in _REQUIRED_IN_PRODUCTION:
        if not os.environ.get(var_name):
            missing.append(f"  {var_name} ({description})")
    if missing:
        warnings.warn(
            "PRODUCTION MODE: Required environment variables are not set:\n"
            + "\n".join(missing)
            + "\nSet these in your .env file or environment before running.",
            stacklevel=2,
        )


# ── Helpers ─────────────────────────────────────────────
def _env_str(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


# ============================================================
#  DATA COLLECTION
# ============================================================
@dataclass
class DataCollectionConfig:
    """Settings for the data collection pipeline."""

    leagues: tuple[str, ...] = (
        "E0",
        "E1",
        "E2",
        "E3",
        "SC0",
        "SC1",
        "D1",
        "D2",
        "I1",
        "I2",
        "SP1",
        "SP2",
        "F1",
        "F2",
        "N1",
        "B1",
        "P1",
        "T1",
        "G1",
        "IRL",
    )
    max_seasons: int = 8
    missing_strategy: Literal["drop", "fill_zero", "fill_median"] = "fill_zero"
    output_file: str = "results.csv"
    max_missing_pct: float = 90.0


# ============================================================
#  PREPROCESSING
# ============================================================
@dataclass
class PreprocessingConfig:
    """Settings for the data preprocessing pipeline."""

    input_file: str = "results.csv"
    output_file: str = "results_clean.csv"
    normalise_teams: bool = True
    add_temporal_features: bool = True
    save_cleaned: bool = True


# ============================================================
#  PATHS
# ============================================================
@dataclass
class Paths:
    """Managed directory and file paths.

    Directories are created on instantiation if they don't exist.
    """

    root: Path = _PROJECT_ROOT

    # Data
    data: Path = root / "data"
    raw: Path = data / "raw"
    processed: Path = data / "processed"
    external: Path = data / "external"

    # Models
    models: Path = root / "models"

    # Logs
    logs: Path = root / "logs"

    # Reports
    reports: Path = root / "reports"

    # Notebooks
    notebooks: Path = root / "notebooks"

    # Source
    src: Path = root / "src"

    # App
    app: Path = root / "app"

    def __post_init__(self) -> None:
        for d in (
            self.data,
            self.raw,
            self.processed,
            self.external,
            self.models,
            self.logs,
            self.reports,
            self.notebooks,
            self.src,
            self.app,
        ):
            d.mkdir(parents=True, exist_ok=True)


# ============================================================
#  DATA LOADING
# ============================================================
@dataclass
class DataConfig:
    """Settings related to data ingestion."""

    source: str = "local"
    api_url: str = ""
    api_key_env: str = "FOOTBALL_DATA_API_KEY"
    fixtures_file: str = "fixtures.csv"
    results_file: str = "results.csv"
    teams_file: str = "teams.csv"
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    seed: int = 42


# ============================================================
#  FEATURE ENGINEERING
# ============================================================
@dataclass
class FeatureConfig:
    """Settings for feature-creation pipelines."""

    form_window: int = 5
    rolling_windows: tuple[int, ...] = (5, 10, 20)
    rolling_avg_window: int = 10
    include_h2h: bool = True
    h2h_window: int = 6
    include_league_position: bool = True
    categorical_encoding: Literal["label", "onehot", "target"] = "label"
    time_decay_halflife: int | None = 5
    reset_per_season: bool = False


# ============================================================
#  TRAINING
# ============================================================
@dataclass
class TrainConfig:
    """Model training hyper-parameters."""

    model_type: Literal[
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "neural_network",
    ] = "lightgbm"
    sample_weight_halflife_days: float = 730.0
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 2000
    n_estimators: int = 300
    max_depth: int = 8
    min_samples_leaf: int = 10
    min_samples_split: int = 2
    max_features: str | None = "sqrt"
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.1
    gamma: float = 0.0
    min_child_weight: float = 1.0
    num_leaves: int = 31
    min_child_samples: int = 10
    hidden_layers: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.3
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10
    cv_folds: int = 5
    target_column: str = "result"
    seed: int = 42


# ============================================================
#  PREDICTION
# ============================================================
@dataclass
class PredictConfig:
    """Prediction-time settings."""

    probability_threshold: float = 0.5
    top_k: int = 10
    output_format: Literal["csv", "json", "console"] = "console"


# ============================================================
#  ODDS API
# ============================================================
@dataclass
class OddsAPIConfig:
    """Settings for The Odds API integration."""

    api_key_env: str = "THE_ODDS_API_KEY"
    regions: str = "us,uk,eu"
    markets: str = "h2h"
    cache_ttl: int = 3600
    request_timeout: int = 15
    sport_key_wc: str = "soccer_fifa_world_cup"
    fallback_to_hardcoded: bool = True


# ============================================================
#  VALUE BETTING
# ============================================================
@dataclass
class ValueBetConfig:
    """Settings for the value betting module."""

    bankroll: float = 1000.0
    kelly_fraction: float = 0.25
    min_ev: float = 0.05
    max_odds: float = 30.0
    max_stake_pct: float = 0.10


# ============================================================
#  ODDS PROCESSING
# ============================================================
@dataclass
class OddsConfig:
    """Settings for the odds processing module."""

    opening_odds_cols: tuple[str, str, str] = ("maxh", "maxd", "maxa")
    closing_odds_cols: tuple[str, str, str] = ("avgh", "avgd", "avga")
    compute_consensus: bool = True
    warn_missing: bool = True


# ============================================================
#  PLAYER INFO
# ============================================================
@dataclass
class PlayerInfoConfig:
    """Settings for the player information module."""

    enabled: bool = True
    default_age: float = 25.0
    placeholder_value: float = 0.0
    warn_missing: bool = True


# ============================================================
#  EXPECTED GOALS (xG) FEATURES
# ============================================================
@dataclass
class XgConfig:
    """Settings for the Expected Goals feature module."""

    rolling_windows: tuple[int, ...] = (5, 10)
    compute_xpts: bool = True
    max_goals_table: int = 8
    placeholder_value: float = 0.0
    warn_missing: bool = True


# ============================================================
#  ENHANCED PLAYER FEATURES
# ============================================================
@dataclass
class PlayerFeaturesConfig:
    """Settings for the enhanced player features module."""

    enabled: bool = False
    rolling_windows: tuple[int, ...] = (5, 10)
    warn_missing: bool = True


# ============================================================
#  POISSON MODEL
# ============================================================
@dataclass
class PoissonConfig:
    """Settings for the Poisson regression model."""

    min_matches: int = 0
    max_goals: int = 8
    decay_halflife_days: float = 1460.0


# ============================================================
#  ELO RATING SYSTEM
# ============================================================
@dataclass
class EloConfig:
    """Settings for the dynamic Elo rating system."""

    k: int = 32
    home_advantage: int = 100
    initial_rating: int = 1500
    regress_to_mean: bool = True
    regress_factor: float = 1 / 3
    use_goal_margin: bool = True
    max_goal_margin: int = 5
    adjustments: dict[str, int] = field(default_factory=dict)
    per_league: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "SE1": {"k": 48, "home_advantage": 70},
            "NO2": {"k": 48, "home_advantage": 70},
            "FI2": {"k": 48, "home_advantage": 70},
        }
    )


# ============================================================
#  HYPER-PARAMETER TUNING
# ============================================================
@dataclass
class HyperTuneConfig:
    """Settings for the hyper-parameter tuning orchestrator."""

    model_types: tuple[str, ...] = ("logistic_regression", "random_forest", "xgboost")
    n_iter_random: int = 50
    cv_folds: int = 5
    save_models: bool = True
    save_report: bool = True
    verbose: bool = True


# ============================================================
#  CONFIDENCE SCORING
# ============================================================
@dataclass
class ConfidenceConfig:
    """Settings for the confidence scoring system."""

    weight_spread: float = 0.40
    weight_agreement: float = 0.35
    weight_calibration: float = 0.25
    default_agreement: float = 50.0
    default_calibration: float = 50.0
    calibration_brier_default: float = 0.25


# ============================================================
#  ENSEMBLE MODEL
# ============================================================
@dataclass
class EnsembleConfig:
    """Settings for the ensemble prediction model."""

    model_names: tuple[str, ...] = ("xgboost", "logistic_regression", "poisson")
    weight_grid_step: float = 0.10
    tune_base_models: bool = False
    model_weight_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


# ============================================================
#  BACKTESTING
# ============================================================
@dataclass
class BacktestConfig:
    """Settings for the backtesting engine."""

    initial_bankroll: float = 1000.0
    kelly_fraction: float = 0.25
    min_ev: float = 0.0
    odds_column_sets: tuple[tuple[str, str, str], ...] = (
        ("BbAvA", "BbAvD", "BbAvH"),
        ("B365A", "B365D", "B365H"),
        ("BWA", "BWD", "BWH"),
    )
    output_dir: str = "reports/backtest"


# ============================================================
#  EVALUATION
# ============================================================
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
        "brier_score",
    )
    plot_confusion_matrix: bool = True
    plot_feature_importance: bool = True
    plot_roc_curve: bool = True
    output_dir: Path = _PROJECT_ROOT / "reports"

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
#  WEATHER COLLECTOR
# ============================================================
@dataclass
class WeatherCollectorConfig:
    """Settings for the OpenWeatherMap API collector."""

    enabled: bool = False
    api_key_env: str = "OPENWEATHER_API_KEY"
    cache_ttl: int = 86400
    request_delay: float = 1.0
    output_file: str = "weather.csv"


# ============================================================
#  REFEREE COLLECTOR
# ============================================================
@dataclass
class RefereeCollectorConfig:
    """Settings for the referee statistics collector."""

    enabled: bool = False
    delay: float = 2.0
    output_file: str = "referees.csv"


# ============================================================
#  TRANSFER COLLECTOR
# ============================================================
@dataclass
class TransferCollectorConfig:
    """Settings for the Transfermarkt transfer scraper."""

    enabled: bool = False
    delay: float = 1.5
    output_file: str = "transfers.csv"
    max_windows: int = 5


# ============================================================
#  STATSBOMB COLLECTOR
# ============================================================
@dataclass
class StatsBombCollectorConfig:
    """Settings for the StatsBomb open data reader."""

    enabled: bool = False
    repo_url: str = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
    competitions: tuple[str, ...] = ("World Cup", "Champions League", "Premier League")
    output_dir: str = "data/scrapers/statsbomb"


# ============================================================
#  WEATHER FEATURES
# ============================================================
@dataclass
class WeatherConfig:
    """Settings for weather-based features."""

    enabled: bool = False
    default_temp: float = 15.0
    placeholder_value: float = 0.0
    warn_missing: bool = True


# ============================================================
#  REFEREE FEATURES
# ============================================================
@dataclass
class RefereeConfig:
    """Settings for referee-based features."""

    enabled: bool = True
    window: int = 20
    placeholder_value: float = 0.0
    warn_missing: bool = True


# ============================================================
#  SCHEDULE / CONGESTION FEATURES
# ============================================================
@dataclass
class ScheduleConfig:
    """Settings for schedule/congestion features."""

    enabled: bool = True
    include_travel_distance: bool = True
    include_fatigue: bool = True


# ============================================================
#  EXTENDED FEATURES
# ============================================================
@dataclass
class ExtendedFeaturesConfig:
    """Toggle advanced/extended feature sets."""

    enabled: bool = False
    include_extended_h2h: bool = True
    include_extended_form: bool = True
    h2h_windows: tuple[int, ...] = (3, 5, 10)
    form_windows: tuple[int, ...] = (3, 5, 10, 20)


# ============================================================
#  BLEND
# ============================================================
@dataclass
class BlendConfig:
    """Settings for the multi-model blend integration."""

    enabled: bool = True
    markets: tuple[str, ...] = ("Over2.5", "BTTS", "Over3.5")
    weights_path: str | None = "config/three_model_weights.json"
    use_blend_for_1x2: bool = False


# ============================================================
#  WORLD CUP
# ============================================================
@dataclass
class WorldCupConfig:
    """Paths for World Cup data, predictions, and model artifacts."""

    data_path: str = "data/raw/worldcup_all.csv"
    predictions_dir: str = "reports/predictions_worldcup"
    predictions_file: str = "worldcup_predictions.csv"
    model_save_name: str = "worldcup_lightgbm.joblib"


# ============================================================
#  FEATURE SELECTION
# ============================================================
@dataclass
class FeatureSelectionConfig:
    """Settings for the feature selection module."""

    enabled: bool = False
    method: Literal["mutual_info", "rfe", "l1", "threshold"] = "mutual_info"
    n_features: int = 30
    importance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    drop_redundant_first: bool = True


# ============================================================
#  DIXON-COLES MODEL
# ============================================================
@dataclass
class DixonColesConfig:
    """Settings for the Dixon-Coles MLE model."""

    enabled: bool = True
    refit_every: int = 2000
    decay_halflife_days: float = 1460.0
    use_importance: bool = True
    rho_fixed: float | None = None
    regress_prior: bool = True
    prior_strength: float = 0.01
    fit_intercept_only: bool = False
    per_league: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "SE1": {"decay_halflife_days": 730, "rho_fixed": -0.05},
            "NO2": {"decay_halflife_days": 730, "rho_fixed": -0.05},
            "FI2": {"decay_halflife_days": 730, "rho_fixed": -0.05},
            "IRL": {"decay_halflife_days": 730, "rho_fixed": -0.05},
        }
    )


# ============================================================
#  DATABASE
# ============================================================
@dataclass
class DatabaseConfig:
    """PostgreSQL / SQLAlchemy connection settings."""

    url: str = field(
        default_factory=lambda: _env_str(
            "DATABASE_URL",
            f"postgresql+psycopg2://{_env_str('DB_USER', 'postgres')}:"
            f"{_env_str('DB_PASSWORD', 'postgres')}@"
            f"{_env_str('DB_HOST', 'localhost')}:"
            f"{_env_int('DB_PORT', 5432)}/"
            f"{_env_str('DB_NAME', 'football_prediction')}",
        )
    )
    host: str = field(default_factory=lambda: _env_str("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: _env_int("DB_PORT", 5432))
    name: str = field(
        default_factory=lambda: _env_str("DB_NAME", "football_prediction")
    )
    user: str = field(default_factory=lambda: _env_str("DB_USER", "postgres"))
    password: str = field(default_factory=lambda: _env_str("DB_PASSWORD", "postgres"))
    pool_size: int = field(default_factory=lambda: _env_int("DB_POOL_SIZE", 10))
    max_overflow: int = field(default_factory=lambda: _env_int("DB_MAX_OVERFLOW", 20))
    pool_recycle: int = field(default_factory=lambda: _env_int("DB_POOL_RECYCLE", 3600))
    pool_pre_ping: bool = field(
        default_factory=lambda: _env_bool("DB_POOL_PRE_PING", True)
    )
    echo: bool = field(default_factory=lambda: _env_bool("DB_ECHO", False))

    @property
    def sa_url(self) -> str:
        url = self.url
        if os.environ.get("USE_PGBOUNCER", "").lower() in ("1", "true", "yes"):
            pgbouncer_port = _env_int("PGBOUNCER_PORT", 6432)
            if pgbouncer_port > 0:
                url = _replace_url_port(url, pgbouncer_port)
            if "?" not in url:
                url += "?"
            else:
                url += "&"
            url += "prepared_statement_cache_size=0&keepalives=1"
        return url


# ============================================================
#  LOGGING
# ============================================================
@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO"))
    format: str = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    )
    file: bool = field(default_factory=lambda: _env_bool("LOG_FILE", True))
    file_path: Path = _PROJECT_ROOT / "logs" / "football_prediction.log"
    rotation: str = "midnight"
    retention: int = 30


# ============================================================
#  API KEYS
# ============================================================
@dataclass
class APIConfig:
    """External API configuration."""

    football_data_key: str = field(
        default_factory=lambda: _env_str("FOOTBALL_DATA_API_KEY", "")
    )
    odds_api_key: str = field(default_factory=lambda: _env_str("THE_ODDS_API_KEY", ""))


# ============================================================
#  APPLICATION
# ============================================================
@dataclass
class AppConfig:
    """Application-level settings."""

    debug: bool = field(default_factory=lambda: _env_bool("APP_DEBUG", False))
    environment: str = field(default_factory=lambda: _env_str("APP_ENV", "development"))
    secret_key: str = field(
        default_factory=lambda: _env_str("SECRET_KEY", "change-me-in-production")
    )


# ============================================================
#  URL HELPERS
# ============================================================


def _replace_url_port(url: str, new_port: int) -> str:
    """Replace the port in a database URL with *new_port*."""
    parsed = urlparse(url)
    if parsed.hostname is None:
        return url  # not a host-based URL (e.g. sqlite:///path)
    new_netloc = f"{parsed.hostname}:{new_port}"
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth = f"{parsed.username}:{parsed.password}"
        new_netloc = f"{auth}@{new_netloc}"
    return urlunparse(parsed._replace(netloc=new_netloc))


# ============================================================
#  TOP-LEVEL CONFIG
# ============================================================
@dataclass
class Config:
    """Root configuration object aggregating all sub-configs."""

    # Infrastructure
    app: AppConfig = field(default_factory=AppConfig)
    paths: Paths = field(default_factory=Paths)
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)

    # Data
    data: DataConfig = field(default_factory=DataConfig)
    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    # Features
    features: FeatureConfig = field(default_factory=FeatureConfig)
    xg: XgConfig = field(default_factory=XgConfig)
    player_info: PlayerInfoConfig = field(default_factory=PlayerInfoConfig)
    player_features: PlayerFeaturesConfig = field(default_factory=PlayerFeaturesConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    referee: RefereeConfig = field(default_factory=RefereeConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    extended_features: ExtendedFeaturesConfig = field(
        default_factory=ExtendedFeaturesConfig
    )
    feature_selection: FeatureSelectionConfig = field(
        default_factory=FeatureSelectionConfig
    )

    # Models
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    poisson: PoissonConfig = field(default_factory=PoissonConfig)
    dixon_coles: DixonColesConfig = field(default_factory=DixonColesConfig)
    elo: EloConfig = field(default_factory=EloConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    hyper_tune: HyperTuneConfig = field(default_factory=HyperTuneConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)

    # Betting
    odds_api: OddsAPIConfig = field(default_factory=OddsAPIConfig)
    odds: OddsConfig = field(default_factory=OddsConfig)
    value_betting: ValueBetConfig = field(default_factory=ValueBetConfig)
    backtesting: BacktestConfig = field(default_factory=BacktestConfig)

    # Evaluation
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Data collectors
    weather_collector: WeatherCollectorConfig = field(
        default_factory=WeatherCollectorConfig
    )
    referee_collector: RefereeCollectorConfig = field(
        default_factory=RefereeCollectorConfig
    )
    transfer_collector: TransferCollectorConfig = field(
        default_factory=TransferCollectorConfig
    )
    statsbomb_collector: StatsBombCollectorConfig = field(
        default_factory=StatsBombCollectorConfig
    )

    # Legacy
    worldcup: WorldCupConfig = field(default_factory=WorldCupConfig)

    # Global toggles
    verbose: bool = True


# Singleton instance — import this anywhere in the application.
config = Config()
