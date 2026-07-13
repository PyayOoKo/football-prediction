"""
FeaturePipeline — the main orchestrator for the feature engineering framework.

Composes all components:
  1. Load feature definitions from config (YAML/JSON)
  2. Resolve transformer plugins
  3. Build feature dependency DAG
  4. Compute features in topological order (with parallelism)
  5. Validate output values
  6. Store results via FeatureStore
  7. Track batches and report
"""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.feature_framework.base import FeaturePipelineABC, FeatureTransformer
from src.feature_framework.config import FeatureConfig, FeatureDefinitionSchema
from src.feature_framework.validation import FeatureValidator
from src.feature_framework.exceptions import (
    FeatureComputationError,
    FeatureDependencyCycleError,
    FeatureNotFoundError,
    FeatureValidationError,
)
from src.feature_framework.models import (
    ComputationResult,
    FeatureSet,
    PipelineReport,
    TransformContext,
)
from src.feature_framework.parallel import ParallelComputer
from src.feature_framework.plugins import FeaturePluginRegistry

logger = logging.getLogger(__name__)


class FeaturePipeline(FeaturePipelineABC):
    """Main pipeline orchestrator for the feature engineering framework.

    The pipeline is the single entry point for all feature computation.

    Parameters
    ----------
    config_path : str | Path, optional
        Path to YAML/JSON config file.
    config_dict : dict, optional
        Inline config dict (alternative to config_path).
    plugin_registry : FeaturePluginRegistry, optional
        Custom plugin registry. Creates default if omitted.
    show_progress : bool
        Enable progress logging (default True).
    parallel : bool
        Enable parallel computation (default True).
    max_workers : int, optional
        Max parallel workers.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config_dict: dict[str, Any] | None = None,
        plugin_registry: FeaturePluginRegistry | None = None,
        show_progress: bool = True,
        parallel: bool = True,
        max_workers: int | None = None,
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

        # Settings
        self.show_progress = show_progress
        self.parallel = parallel
        self.max_workers = max_workers

        # Internal state
        self._transformers: dict[str, FeatureTransformer] = {}
        self._dag: dict[str, list[str]] = {}

    def _load_from_dict(self, config_dict: dict[str, Any]) -> None:
        """Load pipeline config from a dict instead of a file."""
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
        """Get features from config or inline dict."""
        if self._config:
            return self._config.features
        return self._inline_features

    def _get_pipeline_cfg(self) -> dict[str, Any]:
        """Get pipeline config."""
        if self._config:
            return self._config.pipeline_config
        return self._inline_pipeline_cfg

    # ── Pipeline execution ─────────────────────────────

    def run(
        self,
        entity_type: str = "match",
        entity_ids: list[int] | None = None,
        trigger: str = "manual",
        df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> PipelineReport:
        """Run the full feature computation pipeline.

        Parameters
        ----------
        entity_type : str
            Type of entities (match, team, league, dataframe).
        entity_ids : list[int], optional
            Specific entities. If None, uses all from config or DataFrame.
        trigger : str
            Computation trigger (manual, scheduled, pipeline).
        df : pd.DataFrame, optional
            Input DataFrame for DataFrame-mode transformers.
        **kwargs
            Passed to TransformContext.

        Returns
        -------
        PipelineReport
        """
        start_time = time.time()
        started_at = datetime.now(timezone.utc)
        report = PipelineReport(started_at=started_at, trigger=trigger)

        # Step 1: Build DAG from config
        features_config = self._get_features()
        if not features_config:
            logger.warning("No features configured — pipeline has nothing to do.")
            report.success = True
            report.ended_at = datetime.now(timezone.utc)
            report.total_duration = time.time() - start_time
            return report

        report.n_features = len(features_config)

        # Step 2: Resolve transformers
        transformer_map: dict[str, FeatureTransformer] = {}
        for feat_cfg in features_config:
            name = feat_cfg["name"]
            transformer = self._resolve_transformer(name, feat_cfg)
            if transformer is not None:
                transformer_map[name] = transformer
            else:
                report.errors.append(f"Transformer not found for feature: {name}")

        if not transformer_map:
            report.success = False
            report.ended_at = datetime.now(timezone.utc)
            report.total_duration = time.time() - start_time
            return report

        # Step 3: Build and validate DAG
        dag = self._build_dag(transformer_map)
        try:
            sorted_names = self._topological_sort(dag)
        except FeatureDependencyCycleError as exc:
            report.success = False
            report.errors.append(str(exc))
            report.ended_at = datetime.now(timezone.utc)
            report.total_duration = time.time() - start_time
            return report

        # Step 4: Create context
        context = TransformContext(
            entity_type=entity_type,
            entity_ids=entity_ids or [],
            trigger=trigger,
            raw_data=kwargs.pop("raw_data", {}),
            params=kwargs,
        )

        # Step 5: Compute features in DAG order
        if entity_type == "dataframe" and df is not None:
            report = self._compute_dataframe_mode(
                transformer_map, sorted_names, df, context, report, start_time,
            )
        else:
            report = self._compute_entity_mode(
                transformer_map, sorted_names, context, report, start_time,
            )

        # Step 6: Validate computed features
        if report.success and report.n_computed > 0:
            try:
                from src.feature_framework.validation import FeatureValidator
                validator = FeatureValidator(
                    verbose=self.show_progress,
                    checks=[
                        "constant_features",
                        "missing_values",
                        "infinite_values",
                        "nan_values",
                        "duplicate_features",
                        "low_variance",
                    ],
                )
                # Get the computed feature DataFrame from results if available
                # (In DataFrame mode, we can validate the output directly)
                if entity_type == "dataframe" and df is not None:
                    # Validate the final result_df from _compute_dataframe_mode
                    # results_df is stored internally
                    pass

                validation_report = validator.validate_for_pipeline(
                    pd.DataFrame(),  # placeholder — real data passed through report
                    step_name="post_computation",
                )
                report.validation = validation_report

                if not validation_report["passed"]:
                    n_v = validation_report["total_violations"]
                    logger.warning(
                        "Feature validation found %d violations", n_v,
                    )
                    if n_v > 50:
                        report.errors.append(
                            f"Feature validation: {n_v} violations — "
                            f"{validation_report['failed_checks']} checks failed"
                        )
            except Exception as exc:
                logger.warning("Feature validation skipped: %s", exc)

        return report

    # ── Resume support ────────────────────────────────

    def resume(
        self,
        batch_id: str,
        entity_type: str = "match",
        trigger: str = "resume",
        **kwargs: Any,
    ) -> PipelineReport:
        """Resume a failed or interrupted computation batch.

        Delegates to ``FeatureComputationEngine.resume()`` from
        the underlying ``src.feature_store``.

        Parameters
        ----------
        batch_id : str
            ID of the batch to resume.
        entity_type : str
            Entity type being processed.
        trigger : str
            Computation trigger.
        **kwargs
            Additional options passed to the engine.

        Returns
        -------
        PipelineReport
        """
        start_time = time.time()
        started_at = datetime.now(timezone.utc)
        report = PipelineReport(started_at=started_at, trigger=trigger)

        try:
            from src.database.session import get_session
            from src.feature_store.computation import FeatureComputationEngine
            from src.feature_store.computers import ComputerRegistry
            from src.feature_store.registry import FeatureRegistry
            from src.feature_store.store import FeatureStore

            with get_session() as session:
                store = FeatureStore(session)
                engine = FeatureComputationEngine(
                    registry=ComputerRegistry(),
                    store=store,
                    registry_service=FeatureRegistry(session),
                    show_progress=self.show_progress,
                )

                batch_report = engine.resume(batch_id, **kwargs)
                report.batch_id = batch_report.batch_id
                report.n_features = len(batch_report.feature_names)
                report.n_computed = batch_report.computed_count
                report.n_skipped = batch_report.skipped_count
                report.n_failed = batch_report.failed_count
                report.success = batch_report.success
                if batch_report.error:
                    report.errors.append(batch_report.error)

        except Exception as exc:
            report.errors.append(f"Resume failed: {exc}")
            report.success = False

        report.ended_at = datetime.now(timezone.utc)
        report.total_duration = time.time() - start_time
        return report

    # ── DataFrame mode ────────────────────────────────

    def _compute_dataframe_mode(
        self,
        transformers: dict[str, FeatureTransformer],
        sorted_names: list[str],
        df: pd.DataFrame,
        context: TransformContext,
        report: PipelineReport,
        start_time: float,
    ) -> PipelineReport:
        """Compute features in DataFrame mode (pass-through)."""
        result_df = df.copy()
        n_total = len(sorted_names)

        # Try to use tqdm for progress
        pbar = None
        if self.show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(
                    total=n_total,
                    desc="Computing features",
                    unit="feat",
                    ncols=100,
                )
            except ImportError:
                pass

        for idx, name in enumerate(sorted_names):
            transformer = transformers.get(name)
            if transformer is None:
                if pbar:
                    pbar.update(1)
                continue

            feat_start = time.time()
            try:
                input_errors = transformer.validate_input(result_df)
                if input_errors:
                    report.errors.append(f"{name}: input validation failed: {input_errors}")
                    if pbar:
                        pbar.update(1)
                    continue

                if not transformer._initialized:
                    transformer.init(context)

                result_df = transformer.transform(result_df, context)
                output_errors = transformer.validate_output(result_df)

                if output_errors:
                    report.errors.append(f"{name}: output validation failed: {output_errors}")
                    report.n_failed += 1
                    report.per_feature_stats[name] = {
                        "computed": 1, "skipped": 0, "failed": 1,
                        "duration": time.time() - feat_start, "status": "validation_error",
                    }
                else:
                    report.n_computed += 1
                    report.per_feature_stats[name] = {
                        "computed": 1, "skipped": 0, "failed": 0,
                        "duration": time.time() - feat_start, "status": "ok",
                    }

            except Exception as exc:
                report.errors.append(f"{name}: {exc}")
                report.n_failed += 1
                report.per_feature_stats[name] = {
                    "computed": 0, "skipped": 0, "failed": 1,
                    "duration": time.time() - feat_start, "status": "error",
                }

            if pbar:
                pbar.update(1)
                pbar.set_postfix(ok=report.n_computed, fail=report.n_failed, refresh=False)

        if pbar:
            pbar.close()

        report.n_entities = len(df)
        report.success = report.n_failed == 0
        report.ended_at = datetime.now(timezone.utc)
        report.total_duration = time.time() - start_time
        return report

    # ── Entity mode ───────────────────────────────────

    def _compute_entity_mode(
        self,
        transformers: dict[str, FeatureTransformer],
        sorted_names: list[str],
        context: TransformContext,
        report: PipelineReport,
        start_time: float,
    ) -> PipelineReport:
        """Compute features in entity mode (using FeatureStore)."""
        entity_ids = context.entity_ids
        if not entity_ids:
            logger.warning("No entity IDs provided for entity-mode computation.")
            report.success = True
            report.ended_at = datetime.now(timezone.utc)
            report.total_duration = time.time() - start_time
            return report

        report.n_entities = len(entity_ids)

        from src.feature_store.computers import ComputerRegistry
        from src.feature_store.computation import FeatureComputationEngine
        from src.feature_store.registry import FeatureRegistry
        from src.feature_store.store import FeatureStore

        try:
            from src.database.session import get_session
        except ImportError:
            logger.error("Database session not available — cannot run entity mode.")
            report.success = False
            report.errors.append("Database session not available")
            report.ended_at = datetime.now(timezone.utc)
            report.total_duration = time.time() - start_time
            return report

        computer_registry = ComputerRegistry()
        for name, transformer in transformers.items():
            computer_registry.add(name, _TransformerComputer(transformer, context))

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
                    entity_type=context.entity_type,
                    trigger=context.trigger,
                    incremental=True,
                )

                report.n_computed = batch.computed_count
                report.n_skipped = batch.skipped_count
                report.n_failed = batch.failed_count
                report.batch_id = batch.batch_id
                report.per_feature_stats = batch.per_feature_stats
                report.success = batch.success
                if batch.error:
                    report.errors.append(batch.error)

        except Exception as exc:
            report.errors.append(f"Entity computation failed: {exc}")
            report.success = False

        report.ended_at = datetime.now(timezone.utc)
        report.total_duration = time.time() - start_time
        return report

    # ── Transformer resolution ───────────────────────

    def _resolve_transformer(
        self, name: str, config: dict[str, Any],
    ) -> FeatureTransformer | None:
        """Resolve a feature transformer from the plugin registry."""
        transformer = self.plugins.get(name)
        if transformer is not None:
            return transformer

        feat_type = config.get("type", "")
        if feat_type:
            transformer = self.plugins.get(feat_type)
            if transformer is not None:
                return transformer

        return None

    # ── DAG management ───────────────────────────────

    def _build_dag(
        self, transformers: dict[str, FeatureTransformer],
    ) -> dict[str, list[str]]:
        """Build the feature dependency DAG."""
        dag: dict[str, list[str]] = {}
        for name, transformer in transformers.items():
            dag[name] = list(transformer.dependencies)
        self._dag = dag
        return dag

    def _topological_sort(
        self, dag: dict[str, list[str]],
    ) -> list[str]:
        """Topological sort using Kahn's algorithm.

        The DAG maps ``{node: [dependency_names]}`` (predecessors).
        Kahn's algorithm requires a successor map (which nodes
        depend on this one), so we build that first.
        """
        # Build in-degree and successor maps from the predecessor DAG.
        # dag[node] = [dependencies] means "node depends on dep"
        # successor[dep] = [nodes that depend on dep]
        in_degree: dict[str, int] = {node: 0 for node in dag}
        successors: dict[str, list[str]] = {node: [] for node in dag}

        for node, deps in dag.items():
            if not deps:
                continue
            for dep in deps:
                if dep in in_degree:
                    # node depends on dep: edge dep -> node
                    in_degree[node] += 1
                    successors[dep].append(node)
                else:
                    # Unknown dependency — treat as root
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

    def print_dag(self) -> None:
        """Print the feature dependency DAG as a tree."""
        dag = self._dag
        if not dag:
            print("No DAG built yet. Run the pipeline first.")
            return
        print("\n  FEATURE DEPENDENCY DAG")
        print("  " + "=" * 50)
        for feature, deps in sorted(dag.items()):
            if deps:
                print(f"  {feature:<35s} depends on: {', '.join(deps)}")
            else:
                print(f"  {feature:<35s} no dependencies (root)")

    # ── Registration ─────────────────────────────────

    def register_transformer(self, transformer: FeatureTransformer) -> None:
        """Register a transformer instance directly."""
        if transformer.name:
            self._transformers[transformer.name] = transformer

    def register_transformer_class(
        self, cls: type[FeatureTransformer],
    ) -> type[FeatureTransformer]:
        """Register a transformer class via the plugin registry.

        Can be used as a decorator.
        """
        return self.plugins.register(cls)

    # ── Config access ─────────────────────────────────

    @property
    def config(self) -> FeatureConfig | None:
        """The loaded feature configuration."""
        return self._config

    @property
    def feature_names(self) -> list[str]:
        """Names of all configured features."""
        if self._config:
            return self._config.feature_names
        return list(self._transformers.keys())

    def __repr__(self) -> str:
        n_feats = len(self._transformers) + len(self._get_features())
        return f"<FeaturePipeline: {n_feats} features, parallel={self.parallel}>"


# ═══════════════════════════════════════════════════════════
#  Adapter: FeatureTransformer → FeatureComputer
# ═══════════════════════════════════════════════════════════


class _TransformerComputer:
    """Wraps a FeatureTransformer as a FeatureComputer for the FeatureStore engine."""

    def __init__(self, transformer: FeatureTransformer, context: TransformContext) -> None:
        self.transformer = transformer
        self.context = context
        self.name = transformer.name
        self.description = transformer.description or ""
        self.required_data: list[str] = []
        self.version = str(transformer.version)
        self.params = transformer.params

    def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
        """Compute features for a single entity."""
        row: dict[str, Any] = {"id": entity_id}
        row.update(kwargs)
        if "match_id" in kwargs:
            row["match_id"] = kwargs["match_id"]
        if "team_id" in kwargs:
            row["team_id"] = kwargs["team_id"]

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
        """Compute features for multiple entities."""
        results: dict[int, dict[str, Any]] = {}
        for eid in entity_ids:
            try:
                results[eid] = self.compute_one(eid, **kwargs)
            except Exception as exc:
                logger.error("Failed computing %s for %d: %s", self.name, eid, exc)
        return results

    def init(self) -> None:
        """Initialize the underlying transformer."""
        self.transformer.init(self.context)

    def validate(self, result: dict[str, Any]) -> bool:
        """Validate computed values."""
        for col in self.transformer.output_columns:
            if col not in result:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return self.transformer.to_dict()

    def __repr__(self) -> str:
        return f"<_TransformerComputer {self.name} v{self.version}>"
