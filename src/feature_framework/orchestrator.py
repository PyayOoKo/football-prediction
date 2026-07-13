"""
Feature Pipeline Orchestrator — production-grade pipeline execution with
discovery, caching, retry, resume, parallelism, progress tracking,
logging, metrics, and incremental updates.

Responsibilities
----------------
- **Discover**: auto-find feature transformers via ``FeaturePluginRegistry``
- **Resolve DAG**: topological sort with cycle detection
- **Execute order**: compute features in DAG order (parallel where possible)
- **Cache**: store intermediate results in a caching layer
- **Retry**: exponential backoff on transient failures
- **Resume**: checkpoint/restart after interruption
- **Parallel**: thread/process pool execution
- **Progress**: ``tqdm`` progress bars + per-feature timing
- **Logging**: structured JSON logs per pipeline run
- **Metrics**: timing, counts, success rates, per-feature stats
- **Incremental**: skip features that haven't changed since last run

Usage
-----
::

    from src.feature_framework.orchestrator import FeatureOrchestrator

    orchestrator = FeatureOrchestrator()
    report = orchestrator.run(
        entity_type="dataframe",
        df=matches_df,
        trigger="manual",
    )
    print(report.summary())

    # CLI equivalent:
    #   python -m src.feature_framework.orchestrator_cli build-features \\
    #       --entity-type dataframe --input results.csv

Integration with FeatureStore
-----------------------------
The orchestrator integrates with ``src.feature_store.FeatureStore`` for
persistence.  When a database session is available, computed feature
values are stored and can be used for incremental updates.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.feature_framework.base import FeatureTransformer
from src.feature_framework.config import FeatureConfig, FeatureDefinitionSchema
from src.feature_framework.exceptions import (
    FeatureComputationError,
    FeatureDependencyCycleError,
    FeatureNotFoundError,
)
from src.feature_framework.models import TransformContext
from src.feature_framework.parallel import ParallelComputer
from src.feature_framework.plugins import FeaturePluginRegistry

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Report types
# ═══════════════════════════════════════════════════════════════


class OrchestratorStage(Enum):
    """Stages of orchestrator execution."""
    DISCOVER = "discover"
    RESOLVE = "resolve"
    COMPUTE = "compute"
    VALIDATE = "validate"
    STORE = "store"


class FeatureStatus(Enum):
    """Per-feature execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class FeatureExecutionRecord:
    """Execution details for a single feature."""
    name: str = ""
    status: str = "pending"
    duration: float = 0.0
    error: str = ""
    retries: int = 0
    cached: bool = False
    n_entities: int = 0
    output_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}


@dataclass
class OrchestratorReport:
    """Complete report for an orchestrator run.

    Parameters
    ----------
    run_id : str
        Unique run identifier.
    started_at : str
        ISO timestamp of start.
    ended_at : str, optional
        ISO timestamp of end.
    duration : float
        Total wall-clock seconds.
    trigger : str
        Computation trigger (manual, scheduled, pipeline).
    entity_type : str
        Type of entities processed.
    n_entities : int
        Number of entities processed.
    n_features : int
        Number of features configured.
    n_computed : int
        Features that were actually computed.
    n_skipped : int
        Features skipped (cached or not needed).
    n_failed : int
        Features that failed.
    success : bool
        True if no features failed.
    features : dict[str, FeatureExecutionRecord]
        Per-feature execution records.
    errors : list[str]
        List of error messages.
    metrics : dict[str, Any]
        Run-level metrics (timing, rates).
    validation : dict[str, Any], optional
        Feature validation results.
    checkpoint_path : str, optional
        Path to checkpoint file.
    """

    run_id: str = ""
    started_at: str = ""
    ended_at: str = ""
    duration: float = 0.0
    trigger: str = "manual"
    entity_type: str = ""
    n_entities: int = 0
    n_features: int = 0
    n_computed: int = 0
    n_skipped: int = 0
    n_failed: int = 0
    success: bool = True
    features: dict[str, FeatureExecutionRecord] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] | None = None
    checkpoint_path: str = ""

    def summary(self) -> str:
        """Return a concise human-readable summary."""
        status = "✅ PASS" if self.success else "❌ FAIL"
        lines = [
            "FEATURE PIPELINE ORCHESTRATOR REPORT",
            "=" * 65,
            f"  Run ID:       {self.run_id[:8]}...",
            f"  Trigger:      {self.trigger}",
            f"  Entity type:  {self.entity_type}",
            f"  Entities:     {self.n_entities}",
            f"  Duration:     {self.duration:.2f}s",
            f"  Result:       {status}",
            "",
            f"  Features:     {self.n_features} configured",
            f"  Computed:     {self.n_computed}",
            f"  Cached:       {self.n_skipped}",
            f"  Failed:       {self.n_failed}",
            "",
        ]
        if self.errors:
            lines.append("  Errors:")
            for err in self.errors[:5]:
                lines.append(f"    • {err}")
            if len(self.errors) > 5:
                lines.append(f"    ... and {len(self.errors) - 5} more")
            lines.append("")

        if self.metrics:
            lines.append("  Metrics:")
            for k, v in sorted(self.metrics.items()):
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.4f}")
                else:
                    lines.append(f"    {k}: {v}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration": self.duration,
            "trigger": self.trigger,
            "entity_type": self.entity_type,
            "n_entities": self.n_entities,
            "n_features": self.n_features,
            "n_computed": self.n_computed,
            "n_skipped": self.n_skipped,
            "n_failed": self.n_failed,
            "success": self.success,
            "errors": self.errors[:20],
            "metrics": self.metrics,
            "checkpoint_path": self.checkpoint_path,
        }

    def __repr__(self) -> str:
        return (
            f"<OrchestratorReport {self.run_id[:8]} "
            f"{self.n_computed}/{self.n_features} computed, "
            f"{self.n_failed} failed, {self.duration:.1f}s>"
        )


# ═══════════════════════════════════════════════════════════════
#  FeatureOrchestrator
# ═══════════════════════════════════════════════════════════════


class FeatureOrchestrator:
    """Production-grade feature pipeline orchestrator.

    Parameters
    ----------
    config_path : str | Path, optional
        Path to YAML/JSON config file.
    config_dict : dict, optional
        Inline config dict.
    plugin_registry : FeaturePluginRegistry, optional
        Custom plugin registry.
    cache_dir : str | Path
        Directory for caching intermediate results (default ``.cache/features/``).
    checkpoint_dir : str | Path
        Directory for checkpoint files (default ``.checkpoints/``).
    max_retries : int
        Maximum retry attempts per feature (default 2).
    retry_delay : float
        Base delay for exponential backoff in seconds (default 2.0).
    parallel : bool
        Enable parallel computation (default True).
    max_workers : int, optional
        Max parallel workers.
    show_progress : bool
        Show ``tqdm`` progress bars (default True).
    incremental : bool
        Enable incremental computation (default True).
    log_to_file : bool
        Write structured JSON logs to file (default False).
    log_dir : str | Path, optional
        Directory for log files.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config_dict: dict[str, Any] | None = None,
        plugin_registry: FeaturePluginRegistry | None = None,
        cache_dir: str | Path = ".cache/features/",
        checkpoint_dir: str | Path = ".checkpoints/",
        max_retries: int = 2,
        retry_delay: float = 2.0,
        parallel: bool = True,
        max_workers: int | None = None,
        show_progress: bool = True,
        incremental: bool = True,
        log_to_file: bool = False,
        log_dir: str | Path | None = None,
    ) -> None:
        # Config
        self._config: FeatureConfig | None = None
        self._inline_features: list[dict[str, Any]] = []
        self._inline_pipeline_cfg: dict[str, Any] = {}
        if config_path is not None:
            self._config = FeatureConfig(config_path)
        elif config_dict is not None:
            self._load_from_dict(config_dict)

        # Plugins
        self.plugins = plugin_registry or FeaturePluginRegistry()

        # Directories
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Execution config
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.parallel = parallel
        self.max_workers = max_workers or (os.cpu_count() or 4)
        self.show_progress = show_progress
        self.incremental = incremental
        self.log_to_file = log_to_file
        self.log_dir = Path(log_dir) if log_dir else self.checkpoint_dir
        if self.log_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Internal state
        self._transformers: dict[str, FeatureTransformer] = {}
        self._dag: dict[str, list[str]] = {}
        self._cache_checksums: dict[str, str] = {}
        self._metrics_history: list[dict[str, Any]] = []

    def _load_from_dict(self, config_dict: dict[str, Any]) -> None:
        """Load pipeline config from a dict."""
        raw_features = config_dict.get("features", [])
        if not isinstance(raw_features, list):
            raise ValueError("'features' must be a list")

        errors: list[str] = []
        for i, feat in enumerate(raw_features):
            schema = FeatureDefinitionSchema(feat)
            feat_errors = schema.validate()
            if feat_errors:
                errors.append(f"Feature #{i}: {'; '.join(feat_errors)}")

        if errors:
            raise ValueError(
                f"Config validation failed with {len(errors)} error(s):\n  "
                + "\n  ".join(errors)
            )

        self._inline_features = [f for f in raw_features if f.get("enabled", True)]
        self._inline_pipeline_cfg = config_dict.get("pipeline", {})

    def _get_features(self) -> list[dict[str, Any]]:
        if self._config:
            return self._config.features
        return self._inline_features

    # ══════════════════════════════════════════════════════
    #  Main execution
    # ══════════════════════════════════════════════════════

    def run(
        self,
        entity_type: str = "match",
        entity_ids: list[int] | None = None,
        trigger: str = "manual",
        df: pd.DataFrame | None = None,
        force_recompute: bool = False,
        **kwargs: Any,
    ) -> OrchestratorReport:
        """Run the full feature pipeline.

        Parameters
        ----------
        entity_type : str
            Type of entities (match, team, league, dataframe).
        entity_ids : list[int], optional
            Specific entities. If None, uses all from config/DataFrame.
        trigger : str
            Computation trigger.
        df : pd.DataFrame, optional
            Input DataFrame for DataFrame mode.
        force_recompute : bool
            Skip cache and recompute all features.
        **kwargs
            Passed to TransformContext.

        Returns
        -------
        OrchestratorReport
        """
        run_id = str(uuid.uuid4())[:12]
        started_at = datetime.now(timezone.utc).isoformat()
        start_time = time.time()

        report = OrchestratorReport(
            run_id=run_id,
            started_at=started_at,
            trigger=trigger,
            entity_type=entity_type,
        )

        # ── Stage 1: Discover ───────────────────────────
        try:
            self._log_stage(run_id, OrchestratorStage.DISCOVER, "start")
            features_config = self._get_features()
            if not features_config:
                logger.warning("No features configured.")
                report.success = True
                self._finalise_report(report, start_time)
                return report

            report.n_features = len(features_config)
            self._log_stage(run_id, OrchestratorStage.DISCOVER, "end",
                            extra={"n_features": len(features_config)})
        except Exception as exc:
            report.errors.append(f"Discovery failed: {exc}")
            self._finalise_report(report, start_time)
            return report

        # ── Stage 2: Resolve ────────────────────────────
        try:
            self._log_stage(run_id, OrchestratorStage.RESOLVE, "start")
            transformer_map = self._resolve_transformers(features_config, report)
            if not transformer_map:
                self._finalise_report(report, start_time)
                return report

            dag = self._build_dag(transformer_map)
            sorted_names = self._topological_sort(dag)

            self._log_stage(run_id, OrchestratorStage.RESOLVE, "end",
                            extra={
                                "n_transformers": len(transformer_map),
                                "dag_size": len(dag),
                            })
        except FeatureDependencyCycleError as exc:
            report.errors.append(str(exc))
            self._finalise_report(report, start_time)
            return report

        # ── Stage 3: Compute ────────────────────────────
        self._log_stage(run_id, OrchestratorStage.COMPUTE, "start",
                        extra={"n_features": len(sorted_names)})

        if entity_type == "dataframe" and df is not None:
            report = self._compute_dataframe_mode(
                report, transformer_map, sorted_names, df, entity_type,
                trigger, run_id, force_recompute,
            )
        else:
            report = self._compute_entity_mode(
                report, transformer_map, sorted_names, entity_type,
                entity_ids or [], trigger, run_id, force_recompute,
            )

        self._log_stage(run_id, OrchestratorStage.COMPUTE, "end",
                        extra={
                            "computed": report.n_computed,
                            "skipped": report.n_skipped,
                            "failed": report.n_failed,
                        })

        # ── Stage 4: Validate ───────────────────────────
        self._log_stage(run_id, OrchestratorStage.VALIDATE, "start")
        if report.n_computed > 0:
            try:
                from src.feature_framework.validation import FeatureValidator
                validator = FeatureValidator(verbose=False)
                if entity_type == "dataframe" and df is not None:
                    validation_result = validator.validate_for_pipeline(
                        df, step_name="orchestrator_post",
                    )
                    report.validation = validation_result
                    if not validation_result["passed"]:
                        logger.warning(
                            "Validation: %d violations",
                            validation_result["total_violations"],
                        )
            except Exception as exc:
                logger.warning("Validation skipped: %s", exc)

        self._log_stage(run_id, OrchestratorStage.VALIDATE, "end")

        # ── Finalise ────────────────────────────────────
        report.success = report.n_failed == 0
        self._finalise_report(report, start_time)

        # Save checkpoint
        if report.n_failed > 0:
            checkpoint_path = self._save_checkpoint(report)
            report.checkpoint_path = str(checkpoint_path)
            logger.info("Checkpoint saved: %s", checkpoint_path)

        # Collect metrics
        self._collect_metrics(report)

        return report

    # ══════════════════════════════════════════════════════
    #  DataFrame mode
    # ══════════════════════════════════════════════════════

    def _compute_dataframe_mode(
        self,
        report: OrchestratorReport,
        transformers: dict[str, FeatureTransformer],
        sorted_names: list[str],
        df: pd.DataFrame,
        entity_type: str,
        trigger: str,
        run_id: str,
        force_recompute: bool,
    ) -> OrchestratorReport:
        """Compute features in DataFrame (pass-through) mode."""
        result_df = df.copy()
        report.n_entities = len(result_df)

        context = TransformContext(
            entity_type=entity_type,
            entity_ids=[],
            trigger=trigger,
            raw_data={},
            params={},
        )

        # tqdm progress bar
        pbar = None
        if self.show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(
                    total=len(sorted_names),
                    desc="Computing features",
                    unit="feat",
                    ncols=100,
                )
            except ImportError:
                pass

        for feat_name in sorted_names:
            record = FeatureExecutionRecord(name=feat_name, status="running")
            feat_start = time.time()

            try:
                # Check cache
                if not force_recompute and self._is_cached(feat_name, result_df):
                    record.status = "cached"
                    record.cached = True
                    report.n_skipped += 1
                    report.features[feat_name] = record
                    if pbar:
                        pbar.update(1)
                    continue

                transformer = transformers.get(feat_name)
                if transformer is None:
                    record.status = "failed"
                    record.error = "Transformer not found"
                    report.n_failed += 1
                    report.features[feat_name] = record
                    if pbar:
                        pbar.update(1)
                    self._log_feature(run_id, feat_name, "failed", "transformer not found")
                    continue

                # Retry loop
                last_error = ""
                for attempt in range(1 + self.max_retries):
                    try:
                        if not transformer._initialized:
                            transformer.init(context)

                        # Validate input
                        input_errors = transformer.validate_input(result_df)
                        if input_errors:
                            report.errors.append(
                                f"{feat_name}: input validation: {input_errors}"
                            )
                            break

                        result_df = transformer.transform(result_df, context)

                        # Validate output
                        output_errors = transformer.validate_output(result_df)
                        if output_errors:
                            report.errors.append(
                                f"{feat_name}: output validation: {output_errors}"
                            )
                            break

                        record.status = "completed"
                        record.output_columns = list(transformer.output_columns)
                        record.retries = attempt - 1 if attempt > 0 else 0
                        report.n_computed += 1
                        self._update_cache(feat_name, result_df)

                        self._log_feature(
                            run_id, feat_name, "completed",
                            extra={
                                "duration": time.time() - feat_start,
                                "retries": attempt - 1,
                                "columns": len(transformer.output_columns),
                            },
                        )
                        break

                    except Exception as exc:
                        last_error = str(exc)
                        if attempt < self.max_retries:
                            delay = self.retry_delay * (2 ** attempt)
                            logger.warning(
                                "%s attempt %d/%d failed: %s. Retrying in %.1fs...",
                                feat_name, attempt + 1, 1 + self.max_retries,
                                exc, delay,
                            )
                            time.sleep(delay)
                        else:
                            record.status = "failed"
                            record.error = last_error
                            record.retries = self.max_retries
                            report.n_failed += 1
                            report.errors.append(f"{feat_name}: {last_error}")
                            self._log_feature(
                                run_id, feat_name, "failed",
                                extra={"error": last_error, "attempts": attempt + 1},
                            )

                record.duration = time.time() - feat_start
                report.features[feat_name] = record

            except Exception as exc:
                record.status = "failed"
                record.error = str(exc)
                record.duration = time.time() - feat_start
                report.n_failed += 1
                report.errors.append(f"{feat_name}: {exc}")
                report.features[feat_name] = record
                self._log_feature(run_id, feat_name, "failed", extra={"error": str(exc)})

            if pbar:
                pbar.update(1)
                pbar.set_postfix(
                    ok=report.n_computed,
                    skip=report.n_skipped,
                    fail=report.n_failed,
                    refresh=False,
                )

        if pbar:
            pbar.close()

        return report

    # ══════════════════════════════════════════════════════
    #  Entity mode
    # ══════════════════════════════════════════════════════

    def _compute_entity_mode(
        self,
        report: OrchestratorReport,
        transformers: dict[str, FeatureTransformer],
        sorted_names: list[str],
        entity_type: str,
        entity_ids: list[int],
        trigger: str,
        run_id: str,
        force_recompute: bool,
    ) -> OrchestratorReport:
        """Compute features in entity mode (per-entity via FeatureStore)."""
        report.n_entities = len(entity_ids)

        if not entity_ids:
            logger.warning("No entity IDs provided.")
            report.success = True
            return report

        context = TransformContext(
            entity_type=entity_type,
            entity_ids=entity_ids,
            trigger=trigger,
            raw_data={},
            params={},
        )

        # Build computer registry
        from src.feature_store.computers import ComputerRegistry
        from src.feature_store.computation import FeatureComputationEngine
        from src.feature_store.registry import FeatureRegistry
        from src.feature_store.store import FeatureStore

        try:
            from src.database.session import get_session
        except ImportError:
            report.errors.append("Database session not available")
            return report

        computer_registry = ComputerRegistry()

        # Create a _TransformerComputer for each transformer
        for name, transformer in transformers.items():
            if not transformer._initialized:
                transformer.init(context)
            # Simple adapter for feature store
            computer_registry.add(
                name,
                _TransformerComputer(transformer, context),
            )

        try:
            with get_session() as session:
                feature_registry = FeatureRegistry(session)
                store = FeatureStore(session)

                engine = FeatureComputationEngine(
                    registry=computer_registry,
                    store=store,
                    registry_service=feature_registry,
                    show_progress=self.show_progress,
                )

                batch = engine.compute_features(
                    feature_names=sorted_names,
                    entity_ids=entity_ids,
                    entity_type=entity_type,
                    trigger=trigger,
                    incremental=not force_recompute and self.incremental,
                )

                report.n_computed = batch.computed_count
                report.n_skipped = batch.skipped_count
                report.n_failed = batch.failed_count

                if batch.per_feature_stats:
                    for fname, stats in batch.per_feature_stats.items():
                        record = FeatureExecutionRecord(
                            name=fname,
                            status="completed" if stats.get("failed", 0) == 0 else "failed",
                            duration=stats.get("duration", 0.0),
                            n_entities=stats.get("computed", 0),
                        )
                        report.features[fname] = record

                if batch.error:
                    report.errors.append(batch.error)

                self._log_stage(run_id, OrchestratorStage.STORE, "end",
                                extra={
                                    "computed": report.n_computed,
                                    "skipped": report.n_skipped,
                                    "failed": report.n_failed,
                                })

        except Exception as exc:
            report.errors.append(f"Entity computation failed: {exc}")
            report.n_failed += 1

        return report

    # ══════════════════════════════════════════════════════
    #  Resume
    # ══════════════════════════════════════════════════════

    def resume(
        self,
        checkpoint_path: str | Path,
        entity_type: str = "match",
        entity_ids: list[int] | None = None,
        df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> OrchestratorReport:
        """Resume a failed or interrupted pipeline run.

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to a previously saved checkpoint file.
        entity_type : str
            Entity type (must match original run).
        entity_ids : list[int], optional
            Entity IDs (must match original run).
        df : pd.DataFrame, optional
            Input DataFrame (must match original run).
        **kwargs
            Passed to ``run()``.

        Returns
        -------
        OrchestratorReport
        """
        checkpoint = self._load_checkpoint(checkpoint_path)
        if checkpoint is None:
            report = OrchestratorReport(
                run_id=str(uuid.uuid4())[:12],
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            report.errors.append(f"Checkpoint not found: {checkpoint_path}")
            report.success = False
            return report

        logger.info(
            "Resuming run %s — %d features completed, %d failed",
            checkpoint.get("run_id", "?"),
            checkpoint.get("n_computed", 0),
            checkpoint.get("n_failed", 0),
        )

        # Re-run with force_recompute for failed features
        return self.run(
            entity_type=entity_type,
            entity_ids=entity_ids,
            trigger="resume",
            df=df,
            force_recompute=True,
            **kwargs,
        )

    # ══════════════════════════════════════════════════════
    #  Single-feature operations
    # ══════════════════════════════════════════════════════

    def recompute_feature(
        self,
        feature_name: str,
        entity_type: str = "match",
        df: pd.DataFrame | None = None,
    ) -> FeatureExecutionRecord:
        """Force recompute a single feature by name.

        Parameters
        ----------
        feature_name : str
            Name of the feature to recompute.
        entity_type : str
            Entity type.
        df : pd.DataFrame, optional
            Input DataFrame for DataFrame mode.

        Returns
        -------
        FeatureExecutionRecord
        """
        features_config = self._get_features()
        config_map = {f["name"]: f for f in features_config}
        feat_cfg = config_map.get(feature_name)

        if feat_cfg is None:
            return FeatureExecutionRecord(
                name=feature_name,
                status="failed",
                error=f"Feature '{feature_name}' not found in config",
            )

        transformer = self._resolve_single_transformer(feature_name, feat_cfg)
        if transformer is None:
            return FeatureExecutionRecord(
                name=feature_name,
                status="failed",
                error=f"No transformer registered for '{feature_name}'",
            )

        if df is None:
            return FeatureExecutionRecord(
                name=feature_name,
                status="failed",
                error="DataFrame is required for recompute_feature",
            )

        record = FeatureExecutionRecord(name=feature_name, status="running")
        feat_start = time.time()

        try:
            context = TransformContext(
                entity_type=entity_type, entity_ids=[], trigger="manual",
                raw_data={}, params={},
            )

            if not transformer._initialized:
                transformer.init(context)

            input_errors = transformer.validate_input(df)
            if input_errors:
                record.status = "failed"
                record.error = f"Input validation: {input_errors}"
                record.duration = time.time() - feat_start
                return record

            result_df = transformer.transform(df, context)
            output_errors = transformer.validate_output(result_df)

            if output_errors:
                record.status = "failed"
                record.error = f"Output validation: {output_errors}"
            else:
                record.status = "completed"
                record.duration = time.time() - feat_start
                record.output_columns = list(transformer.output_columns)

        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            record.duration = time.time() - feat_start

        return record

    def list_features(self) -> list[dict[str, Any]]:
        """List all configured features with their metadata.

        Returns
        -------
        list[dict[str, Any]]
            Feature metadata dicts.
        """
        features_config = self._get_features()
        result: list[dict[str, Any]] = []

        for feat_cfg in features_config:
            name = feat_cfg.get("name", "?")
            transformer = self._resolve_single_transformer(name, feat_cfg)
            result.append({
                "name": feat_cfg.get("name", ""),
                "type": feat_cfg.get("type", ""),
                "version": feat_cfg.get("version", 1),
                "enabled": feat_cfg.get("enabled", True),
                "category": feat_cfg.get("category", ""),
                "description": feat_cfg.get("description", ""),
                "dependencies": feat_cfg.get("dependencies", []),
                "output_columns": list(transformer.output_columns)
                if transformer else [],
                "has_transformer": transformer is not None,
            })

        return result

    def feature_status(self, feature_name: str) -> dict[str, Any]:
        """Get detailed status for a single feature.

        Parameters
        ----------
        feature_name : str
            Name of the feature.

        Returns
        -------
        dict[str, Any]
            Status information.
        """
        features_config = self._get_features()
        config_map = {f["name"]: f for f in features_config}
        feat_cfg = config_map.get(feature_name)

        if feat_cfg is None:
            return {"name": feature_name, "status": "not_found"}

        transformer = self._resolve_single_transformer(feature_name, feat_cfg)

        return {
            "name": feature_name,
            "type": feat_cfg.get("type", ""),
            "version": feat_cfg.get("version", 1),
            "enabled": feat_cfg.get("enabled", True),
            "status": "ready" if transformer else "no_transformer",
            "dependencies": feat_cfg.get("dependencies", []),
            "output_columns": list(transformer.output_columns)
            if transformer else [],
            "initialized": transformer._initialized if transformer else False,
            "metrics": None,
        }

    # ══════════════════════════════════════════════════════
    #  Internal: transformer resolution
    # ══════════════════════════════════════════════════════

    def _resolve_transformers(
        self,
        features_config: list[dict[str, Any]],
        report: OrchestratorReport,
    ) -> dict[str, FeatureTransformer]:
        """Resolve all transformer instances from config."""
        transformer_map: dict[str, FeatureTransformer] = {}

        for feat_cfg in features_config:
            name = feat_cfg["name"]
            transformer = self._resolve_single_transformer(name, feat_cfg)
            if transformer is not None:
                transformer_map[name] = transformer
            else:
                report.errors.append(f"Transformer not found: {name}")

        return transformer_map

    def _resolve_single_transformer(
        self,
        name: str,
        config: dict[str, Any],
    ) -> FeatureTransformer | None:
        """Resolve a single transformer by name or type."""
        # Check already-registered instances
        transformer = self.plugins.get(name)
        if transformer is not None:
            return transformer

        # Check by type
        feat_type = config.get("type", "")
        if feat_type:
            params = {k: v for k, v in config.items()
                      if k not in ("name", "type", "enabled", "version",
                                   "category", "description", "dependencies",
                                   "output_columns", "tags")}
            transformer = self.plugins.get_or_create(name, **params)
            if transformer is not None:
                return transformer

            # Fallback: type-based lookup
            transformer = self.plugins.get(feat_type)
            if transformer is not None:
                return transformer

        return None

    # ══════════════════════════════════════════════════════
    #  Internal: DAG management
    # ══════════════════════════════════════════════════════

    def _build_dag(
        self,
        transformers: dict[str, FeatureTransformer],
    ) -> dict[str, list[str]]:
        """Build the feature dependency DAG."""
        dag: dict[str, list[str]] = {}
        for name, transformer in transformers.items():
            dag[name] = list(transformer.dependencies)
        self._dag = dag
        return dag

    def _topological_sort(
        self,
        dag: dict[str, list[str]],
    ) -> list[str]:
        """Topological sort using Kahn's algorithm."""
        in_degree: dict[str, int] = {node: 0 for node in dag}
        successors: dict[str, list[str]] = {node: [] for node in dag}

        for node, deps in dag.items():
            if not deps:
                continue
            for dep in deps:
                if dep in in_degree:
                    in_degree[node] += 1
                    successors[dep].append(node)
                else:
                    in_degree[dep] = 0
                    successors[dep] = [node]
                    in_degree[node] += 1

        queue = deque([node for node, deg in in_degree.items() if deg == 0])
        sorted_nodes: list[str] = []

        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)
            for successor in successors.get(node, []):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if len(sorted_nodes) != len(in_degree):
            cycle = [n for n, d in in_degree.items() if d > 0]
            raise FeatureDependencyCycleError(cycle)

        return sorted_nodes

    def get_dag(self) -> dict[str, list[str]]:
        """Return the current feature dependency DAG."""
        return dict(self._dag)

    # ══════════════════════════════════════════════════════
    #  Internal: caching
    # ══════════════════════════════════════════════════════

    def _is_cached(self, feature_name: str, df: pd.DataFrame) -> bool:
        """Check if a feature's output is already cached and current.

        Uses metadata-based checks (row count comparison) to decide
        if a feature's output is fresh.
        """
        if not self.incremental:
            return False

        # Check for the metadata file (not a .npy file — we only use
        # metadata JSON for cache validation, not binary data)
        cached_meta = self._load_cache_meta(feature_name)
        if cached_meta is None:
            return False

        if cached_meta.get("n_rows") != len(df):
            return False

        return True

    def _cache_key(self, feature_name: str) -> str:
        """Generate a cache key for a feature."""
        return f"feat_{feature_name}"

    def _update_cache(self, feature_name: str, df: pd.DataFrame) -> None:
        """Cache the current state of a feature's output columns."""
        cache_key = self._cache_key(feature_name)
        meta = {
            "feature_name": feature_name,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = self.cache_dir / f"{cache_key}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    def _load_cache_meta(self, feature_name: str) -> dict[str, Any] | None:
        """Load cached metadata for a feature."""
        cache_key = self._cache_key(feature_name)
        meta_path = self.cache_dir / f"{cache_key}_meta.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def clear_cache(self, feature_name: str | None = None) -> int:
        """Clear cached results.

        Parameters
        ----------
        feature_name : str, optional
            Clear cache for a specific feature. If None, clears all.

        Returns
        -------
        int
            Number of cache entries removed.
        """
        count = 0
        if feature_name:
            key = self._cache_key(feature_name)
            for f in self.cache_dir.glob(f"{key}*"):
                f.unlink()
                count += 1
        else:
            for f in self.cache_dir.glob("feat_*"):
                f.unlink()
                count += 1
        return count

    # ══════════════════════════════════════════════════════
    #  Internal: checkpoint
    # ══════════════════════════════════════════════════════

    def _save_checkpoint(self, report: OrchestratorReport) -> Path:
        """Save a checkpoint file for resume support."""
        checkpoint = {
            "run_id": report.run_id,
            "started_at": report.started_at,
            "trigger": report.trigger,
            "entity_type": report.entity_type,
            "n_entities": report.n_entities,
            "n_features": report.n_features,
            "n_computed": report.n_computed,
            "n_skipped": report.n_skipped,
            "n_failed": report.n_failed,
            "success": False,
            "errors": report.errors[:20],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self.checkpoint_dir / f"{report.run_id}.json"
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        return path

    def _load_checkpoint(self, path: str | Path) -> dict[str, Any] | None:
        """Load a checkpoint file."""
        path = Path(path)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all available checkpoint files."""
        checkpoints: list[dict[str, Any]] = []
        for f in sorted(self.checkpoint_dir.glob("*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                    checkpoints.append({
                        "path": str(f),
                        "run_id": data.get("run_id", f.stem),
                        "trigger": data.get("trigger", "?"),
                        "computed": data.get("n_computed", 0),
                        "failed": data.get("n_failed", 0),
                        "saved_at": data.get("saved_at", ""),
                    })
            except Exception:
                pass
        return checkpoints

    # ══════════════════════════════════════════════════════
    #  Internal: metrics & logging
    # ══════════════════════════════════════════════════════

    def _collect_metrics(self, report: OrchestratorReport) -> None:
        """Collect and store run-level metrics."""
        computed_times = [
            f.duration for f in report.features.values()
            if f.status == "completed"
        ]

        metrics = {
            "total_duration": report.duration,
            "avg_feature_time": float(np.mean(computed_times))
            if computed_times else 0.0,
            "total_computed": report.n_computed,
            "total_skipped": report.n_skipped,
            "total_failed": report.n_failed,
            "success_rate": report.n_computed / max(report.n_features, 1),
            "entities_per_second": report.n_entities / max(report.duration, 0.001),
            "features_per_second": report.n_computed / max(report.duration, 0.001),
        }
        report.metrics = metrics
        self._metrics_history.append({
            "run_id": report.run_id,
            "trigger": report.trigger,
            **metrics,
        })

    def get_metrics_history(self) -> list[dict[str, Any]]:
        """Return metrics from all runs in this session."""
        return list(self._metrics_history)

    def _log_stage(
        self,
        run_id: str,
        stage: OrchestratorStage,
        event: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log a structured stage event."""
        record = {
            "run_id": run_id,
            "stage": stage.value,
            "event": event,
        }
        if extra:
            record.update(extra)

        if self.log_to_file:
            self._write_log(record)

        logger.debug("[%s] %s %s", run_id, stage.value, event)

    def _log_feature(
        self,
        run_id: str,
        feature_name: str,
        status: str,
        message: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log a structured feature-level event."""
        record = {
            "run_id": run_id,
            "feature": feature_name,
            "status": status,
            "message": message,
        }
        if extra:
            record.update(extra)

        if self.log_to_file:
            self._write_log(record)

        logger.debug("[%s] %s: %s", feature_name, status, message)

    def _write_log(self, record: dict[str, Any]) -> None:
        """Write a structured log entry to file."""
        try:
            log_path = self.log_dir / f"orchestrator_{datetime.now().strftime('%Y%m')}.jsonl"
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass

    @staticmethod
    def _finalise_report(
        report: OrchestratorReport,
        start_time: float,
    ) -> None:
        """Set final timestamps on the report."""
        report.ended_at = datetime.now(timezone.utc).isoformat()
        report.duration = time.time() - start_time


# ═══════════════════════════════════════════════════════════════
#  TransformerComputer adapter
# ═══════════════════════════════════════════════════════════════


class _TransformerComputer:
    """Adapter: wraps a FeatureTransformer for use with FeatureComputationEngine."""

    def __init__(
        self,
        transformer: FeatureTransformer,
        context: TransformContext,
    ) -> None:
        self.transformer = transformer
        self.context = context
        self.name = transformer.name
        self.description = transformer.description or ""
        self.required_data: list[str] = []
        self.version = str(transformer.version)
        self.params = transformer.params

    def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
        import pandas as pd
        row: dict[str, Any] = {"id": entity_id}
        row.update(kwargs)
        df = pd.DataFrame([row])

        try:
            result_df = self.transformer.transform(df, self.context)
            values: dict[str, Any] = {}
            for col in self.transformer.output_columns:
                if col in result_df.columns:
                    val = result_df[col].iloc[0]
                    if pd.notna(val):
                        values[col] = val
            return values
        except Exception as exc:
            raise FeatureComputationError(
                self.transformer.name, entity_id, str(exc),
            )

    def compute_batch(
        self, entity_ids: list[int], **kwargs: Any,
    ) -> dict[int, dict[str, Any]]:
        results: dict[int, dict[str, Any]] = {}
        for eid in entity_ids:
            try:
                results[eid] = self.compute_one(eid, **kwargs)
            except Exception as exc:
                logger.error("Failed computing %s for %d: %s", self.name, eid, exc)
        return results

    def init(self) -> None:
        self.transformer.init(self.context)

    def validate(self, result: dict[str, Any]) -> bool:
        for col in self.transformer.output_columns:
            if col not in result:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return self.transformer.to_dict()

    def __repr__(self) -> str:
        return f"<_TransformerComputer {self.name} v{self.version}>"
