"""
Abstract Base Classes — FeatureTransformer and FeaturePipelineABC.

Every feature in the framework is a ``FeatureTransformer`` with:
- A unique name, version, and description
- Declared dependencies on other features
- A ``transform()`` method that produces output columns
- Optional validation rules and metadata
"""

from __future__ import annotations

import abc
import logging
from typing import Any

import pandas as pd

from src.feature_framework.models import FeatureMetadata, TransformContext

logger = logging.getLogger(__name__)


class FeatureTransformer(abc.ABC):
    """Abstract base class for all feature transformers.

    Every feature in the system extends this class. Subclasses must
    implement ``transform()`` and declare class-level metadata.

    Lifecycle
    ---------
    1. ``__init__`` — store parameters from config
    2. ``validate_input(df)`` — optional input validation
    3. ``transform(df, context)`` — core computation (must implement)
    4. ``validate_output(df)`` — optional output validation

    Parameters
    ----------
    name : str
        Unique feature name (e.g. ``home_attack_strength``).
    version : int
        Feature version (incremented on breaking changes).
    description : str
        Human-readable description.
    dependencies : list[str]
        Names of features this transformer depends on.
    output_columns : list[str]
        Column names this transformer produces.
    data_type : str
        Expected data type of the output (float, int, str, bool, categorical).
    computation_time : str
        Expected speed: ``fast`` (< 1s), ``medium`` (1-10s), ``slow`` (> 10s).
    **params : Any
        Additional parameters passed from config.
    """

    # ── Class-level metadata (override in subclass) ─────
    name: str = ""
    version: int = 1
    description: str = ""
    dependencies: list[str] = []
    output_columns: list[str] = []
    data_type: str = "float"
    computation_time: str = "fast"
    category: str = "other"
    author: str = "system"
    tags: list[str] = []
    source: str = ""

    def __init__(self, **params: Any) -> None:
        # Instance-level params (from config)
        self.params: dict[str, Any] = params
        self._initialized: bool = False

    @property
    def metadata(self) -> FeatureMetadata:
        """Return rich metadata for this feature."""
        return FeatureMetadata(
            name=self.name,
            version=self.version,
            description=self.description or self.__doc__ or "",
            dependencies=list(self.dependencies),
            output_columns=list(self.output_columns),
            data_type=self.data_type,
            computation_time=self.computation_time,
            category=self.category,
            author=self.author,
            tags=list(self.tags),
            source=self.source,
        )

    # ── Lifecycle hooks ─────────────────────────────────

    def init(self, context: TransformContext | None = None) -> None:
        """Initialize the transformer (load data, set up state).

        Called once before any ``transform()`` calls.
        Override in subclasses if setup is required.
        """
        self._initialized = True
        logger.debug("Initialized transformer: %s v%d", self.name, self.version)

    def validate_input(self, df: pd.DataFrame) -> list[str]:
        """Validate input DataFrame before transformation.

        Override to add custom input checks.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.

        Returns
        -------
        list[str]
            List of validation errors (empty if valid).
        """
        return []

    @abc.abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
        context: TransformContext | None = None,
    ) -> pd.DataFrame:
        """Compute feature values and add them to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input data (includes columns from upstream features).
        context : TransformContext, optional
            Pipeline context with shared data.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with new feature columns added.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclass.
        """
        ...

    def validate_output(self, df: pd.DataFrame) -> list[str]:
        """Validate output columns after transformation.

        Checks that all ``output_columns`` exist and have valid types.
        Override to add custom output validation.

        Parameters
        ----------
        df : pd.DataFrame
            Output data.

        Returns
        -------
        list[str]
            List of validation errors (empty if valid).
        """
        errors: list[str] = []
        for col in self.output_columns:
            if col not in df.columns:
                errors.append(f"Missing output column: {col}")
            elif self.data_type == "float" and not pd.api.types.is_float_dtype(df[col]):
                errors.append(f"Column {col} should be float, got {df[col].dtype}")
            elif self.data_type == "int" and not pd.api.types.is_integer_dtype(df[col]):
                errors.append(f"Column {col} should be int, got {df[col].dtype}")
        return errors

    # ── Serialization ──────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize transformer configuration."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "output_columns": list(self.output_columns),
            "data_type": self.data_type,
            "computation_time": self.computation_time,
            "category": self.category,
            "params": self.params,
        }

    def __repr__(self) -> str:
        return f"<FeatureTransformer {self.name!r} v{self.version}>"


class FeaturePipelineABC(abc.ABC):
    """Abstract base class for feature pipeline orchestrators.

    Subclasses implement ``run()`` to execute the full pipeline:
    resolve DAG → compute features → validate → store.
    """

    @abc.abstractmethod
    def run(
        self,
        entity_type: str = "match",
        entity_ids: list[int] | None = None,
        trigger: str = "manual",
        **kwargs: Any,
    ) -> Any:
        """Run the full feature computation pipeline.

        Parameters
        ----------
        entity_type : str
            Type of entities to compute for.
        entity_ids : list[int], optional
            Specific entity IDs. If None, computes for all.
        trigger : str
            Computation trigger.
        **kwargs
            Additional pipeline parameters.

        Returns
        -------
        PipelineReport
            Report of the pipeline run.
        """
        ...
