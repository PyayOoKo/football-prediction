"""
Transformation stage — builds features, joins related data, aggregates.

Applies a sequence of user-defined transform functions to the
normalised data. Each transform receives ``(data, context)`` and
returns modified data.

Built-in transforms:
- ColumnSelector — keep/drop/rename columns
- FieldMapper — apply a function to a specific column
- Aggregator — group by key columns and aggregate
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from src.etl.models import PipelineStage, StageResult, StageStatus

logger = logging.getLogger(__name__)

# Type alias for a transform function
TransformFn = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]


class ColumnSelector:
    """Keep, drop, or rename columns.

    Parameters
    ----------
    keep : list[str], optional
        Columns to keep (all others dropped). Mutually exclusive with ``drop``.
    drop : list[str], optional
        Columns to drop.
    rename : dict[str, str], optional
        Column renames: ``{old_name: new_name}``.
    """

    def __init__(
        self,
        keep: list[str] | None = None,
        drop: list[str] | None = None,
        rename: dict[str, str] | None = None,
    ) -> None:
        if keep and drop:
            raise ValueError("'keep' and 'drop' are mutually exclusive")
        self.keep = set(keep) if keep else None
        self.drop = set(drop) if drop else None
        self.rename = rename or {}

    def __call__(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for row in data:
            if self.keep is not None:
                row = {k: v for k, v in row.items() if k in self.keep}
            elif self.drop is not None:
                row = {k: v for k, v in row.items() if k not in self.drop}

            for old_name, new_name in self.rename.items():
                if old_name in row:
                    row[new_name] = row.pop(old_name)
            result.append(row)
        return result


class FieldMapper:
    """Apply a function to a specific column.

    Parameters
    ----------
    column : str
        Column to transform.
    func : Callable
        Function that takes a single value and returns a transformed value.
    """

    def __init__(self, column: str, func: Callable[[Any], Any]) -> None:
        self.column = column
        self.func = func

    def __call__(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for row in data:
            if self.column in row:
                row[self.column] = self.func(row[self.column])
        return data


class Aggregator:
    """Group by key columns and compute aggregations.

    Parameters
    ----------
    group_by : list[str]
        Columns to group by.
    aggregations : dict[str, str]
        ``{col: agg_function}`` — agg functions: ``sum``, ``mean``, ``count``, ``min``, ``max``.
    """

    def __init__(
        self,
        group_by: list[str],
        aggregations: dict[str, str],
    ) -> None:
        self.group_by = group_by
        self.aggregations = aggregations

    def __call__(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from collections import defaultdict

        groups: dict[tuple[Any, ...], dict[str, list[Any]]] = defaultdict(
            lambda: {col: [] for col in self.aggregations}
        )

        for row in data:
            key = tuple(row.get(k) for k in self.group_by)
            for col in self.aggregations:
                val = row.get(col)
                if val is not None:
                    groups[key][col].append(val)

        result = []
        for key, agg_values in groups.items():
            out = dict(zip(self.group_by, key))
            for col, agg_fn in self.aggregations.items():
                values = agg_values[col]
                if not values:
                    out[col] = None
                elif agg_fn == "sum":
                    out[col] = sum(values)
                elif agg_fn == "mean":
                    out[col] = sum(values) / len(values)
                elif agg_fn == "count":
                    out[col] = len(values)
                elif agg_fn == "min":
                    out[col] = min(values)
                elif agg_fn == "max":
                    out[col] = max(values)
                else:
                    out[col] = values
            result.append(out)

        return result


class DataTransformer:
    """Pluggable transformation stage.

    Applies a sequence of transform functions in order.

    Parameters
    ----------
    transforms : list[TransformFn]
        Ordered list of transform functions/callables.
    """

    def __init__(self, transforms: list[TransformFn] | None = None) -> None:
        self.transforms = transforms or []

    def add(self, transform: TransformFn) -> DataTransformer:
        """Append a transform to the pipeline."""
        self.transforms.append(transform)
        return self

    def run(self, data: list[dict[str, Any]]) -> StageResult:
        """Execute all transforms in sequence.

        Parameters
        ----------
        data : list[dict]
            Normalised data.

        Returns
        -------
        StageResult
            Transformed data with metrics.
        """
        stage = PipelineStage.TRANSFORM
        result = StageResult(stage=stage, status=StageStatus.RUNNING)
        start = time.perf_counter()

        result.records_in = len(data)

        try:
            current = data
            for i, transform_fn in enumerate(self.transforms):
                name = getattr(transform_fn, "__name__", f"transform_{i}")
                logger.debug("Applying transform %d/%d: %s", i + 1, len(self.transforms), name)
                current = transform_fn(current)

            result.data = current
            result.records_out = len(current)
            result.metrics["transforms_applied"] = len(self.transforms)
            result.status = StageStatus.SUCCESS if current else StageStatus.WARNING

            if not current:
                result.errors.append("Transform stage produced zero rows")

        except Exception as exc:
            logger.exception("Transform failed: %s", exc)
            result.status = StageStatus.FAILED
            result.errors.append(str(exc))

        result.duration_seconds = time.perf_counter() - start
        logger.info(
            "Transform: %d rows, %d transforms in %.1fs",
            result.records_out,
            len(self.transforms),
            result.duration_seconds,
        )
        return result
