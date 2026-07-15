"""
ETL Framework — reusable data pipeline for football analytics.

Pipeline stages (in order)
--------------------------
1. Extract    — fetch raw data from APIs, CSVs, or databases
2. Validate   — check schema compliance, data quality rules
3. Clean      — fix missing values, type coercion, deduplication
4. Normalize  — standardise team names, dates, categorical values
5. Transform  — build features, join related data, aggregate
6. Store      — upsert to database, write CSV/Parquet, commit

Each stage is a standalone class that any scraper or pipeline
can compose into its own workflow.

Cross-cutting features
----------------------
- RetryWithBackoff    — exponential backoff for API calls
- JobTracker          — checkpoint/resume for failed jobs
- ParallelProcessor   — thread/process pool for batch work
- ValidationReport    — structured pass/fail/warn results
- ProgressReporter    — tqdm bars + throughput stats
- StageTimer          — per-stage timing and logging
- ETLConfig           — dict-driven pipeline definitions
"""

from src.etl.models import (
    ETLConfig,
    ETLResult,
    PipelineStage,
    StageResult,
    ValidationReport,
    ValidationRuleResult,
)
from src.etl.pipeline import ETLPipeline
from src.etl.extract import BaseExtractor, RetryWithBackoff
from src.etl.validate import DataValidator, SchemaValidator
from src.etl.clean import DataCleaner
from src.etl.normalize import DataNormalizer
from src.etl.transform import DataTransformer
from src.etl.store import DataStore, DatabaseStore, FileStore
from src.etl.tracker import JobTracker, JobState
from src.etl.progress import ProgressReporter
from src.etl.extractors import (
    TransferExtractor,
    WeatherExtractor,
    RefereeExtractor,
    StatsBombExtractor,
    EXTRACTOR_REGISTRY,
    get_extractor,
)

__all__ = [
    "ETLConfig",
    "ETLResult",
    "ETLPipeline",
    "PipelineStage",
    "StageResult",
    "ValidationReport",
    "ValidationRuleResult",
    "BaseExtractor",
    "RetryWithBackoff",
    "DataValidator",
    "SchemaValidator",
    "DataCleaner",
    "DataNormalizer",
    "DataTransformer",
    "DataStore",
    "DatabaseStore",
    "FileStore",
    "JobTracker",
    "JobState",
    "ProgressReporter",
    "TransferExtractor",
    "WeatherExtractor",
    "RefereeExtractor",
    "StatsBombExtractor",
    "EXTRACTOR_REGISTRY",
    "get_extractor",
]
