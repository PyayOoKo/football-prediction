"""
Cleaning stage — fixes common data quality issues.

Operations (applied in order):
1. Strip whitespace from string columns
2. Coerce column types (int, float, str, date)
3. Drop or fill missing values (only on key columns for drop)
4. Remove duplicate rows
5. Clip outliers to valid ranges
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

from src.etl.models import PipelineStage, StageResult, StageStatus

logger = logging.getLogger(__name__)

FillStrategy = Literal["drop", "fill_zero", "fill_mean", "fill_median", "fill_mode"]


class DataCleaner:
    """Configurable data cleaning stage.

    Parameters
    ----------
    trim_strings : bool
        Strip whitespace from string columns (default True).
    drop_duplicates : bool
        Remove duplicate rows (default True).
    duplicate_keys : list[str] | None
        Columns to check for duplicates (default None = all cols).
    fill_strategy : FillStrategy
        Missing value strategy (default ``drop``).
        When ``drop``, only rows where **key_columns** are null are dropped.
    key_columns : list[str]
        Columns to check for missing values when using ``drop`` strategy.
        Defaults to all columns if not specified. Set to a short list like
        ``[\"match_id\", \"home_team\", \"away_team\", \"date\"]`` to avoid
        dropping rows with optional stats missing.
    fill_columns : dict[str, Any] | None
        Per-column missing value overrides: ``{col: value}``.
    type_coercions : dict[str, type] | None
        Column type casts: ``{col: int}``, ``{col: str}``, etc.
    clip_ranges : dict[str, tuple[float, float]] | None
        Min/max clip ranges: ``{col: (0, 100)}``.
    """

    def __init__(
        self,
        trim_strings: bool = True,
        drop_duplicates: bool = True,
        duplicate_keys: list[str] | None = None,
        fill_strategy: FillStrategy = "drop",
        key_columns: list[str] | None = None,
        fill_columns: dict[str, Any] | None = None,
        type_coercions: dict[str, type] | None = None,
        clip_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.trim_strings = trim_strings
        self.drop_duplicates = drop_duplicates
        self.duplicate_keys = duplicate_keys
        self.fill_strategy = fill_strategy
        self.key_columns = key_columns  # None = all columns (legacy behaviour)
        self.fill_columns = fill_columns or {}
        self.type_coercions = type_coercions or {}
        self.clip_ranges = clip_ranges or {}

    def run(self, data: list[dict[str, Any]]) -> StageResult:
        """Execute the cleaning pipeline.

        Parameters
        ----------
        data : list[dict]
            Data to clean.

        Returns
        -------
        StageResult
            Cleaned data with cleaning metrics.
        """
        stage = PipelineStage.CLEAN
        result = StageResult(stage=stage, status=StageStatus.RUNNING)
        start = time.perf_counter()

        n_before = len(data)
        metrics: dict[str, float] = {}

        try:
            # 1. Trim strings
            if self.trim_strings:
                data = self._trim_strings(data)

            # 2. Type coercion
            if self.type_coercions:
                data, metrics["coercion_errors"] = self._coerce_types(data)

            # 3. Fill / Drop missing (only drops rows where KEY columns are null)
            n_missing_before = self._count_null(data, self.key_columns)
            data = self._handle_missing(data)
            n_missing_after = self._count_null(data, self.key_columns)
            metrics["nulls_filled"] = float(n_missing_before - n_missing_after)

            # 4. Deduplicate
            if self.drop_duplicates:
                n_dupes_before = len(data)
                data = self._deduplicate(data)
                metrics["duplicates_removed"] = float(n_dupes_before - len(data))

            # 5. Clip ranges
            if self.clip_ranges:
                data, metrics["clipped_values"] = self._clip_ranges(data)

            result.data = data
            result.records_in = n_before
            result.records_out = len(data)
            result.metrics = metrics
            result.status = StageStatus.SUCCESS if data else StageStatus.WARNING

            if not data:
                result.errors.append("All rows removed during cleaning")

        except Exception as exc:
            logger.exception("Cleaning failed: %s", exc)
            result.status = StageStatus.FAILED
            result.errors.append(str(exc))

        result.duration_seconds = time.perf_counter() - start
        logger.info(
            "Cleaning: %d -> %d rows in %.1fs",
            n_before,
            result.records_out,
            result.duration_seconds,
        )
        return result

    # ── Internal helpers ────────────────────────────────

    def _trim_strings(self, data: list[dict]) -> list[dict]:
        trimmed = []
        for row in data:
            new_row = {}
            for k, v in row.items():
                if isinstance(v, str):
                    new_row[k] = v.strip()
                else:
                    new_row[k] = v
            trimmed.append(new_row)
        return trimmed

    def _coerce_types(
        self, data: list[dict]
    ) -> tuple[list[dict], float]:
        errors = 0
        coerced = []
        for row in data:
            new_row = dict(row)
            for col, target_type in self.type_coercions.items():
                if col in new_row and new_row[col] is not None:
                    try:
                        new_row[col] = target_type(new_row[col])
                    except (ValueError, TypeError):
                        errors += 1
                        new_row[col] = None
            coerced.append(new_row)
        return coerced, float(errors)

    def _handle_missing(self, data: list[dict]) -> list[dict]:
        if self.fill_strategy == "drop":
            # Only drop rows where KEY columns are null (or all cols if key_columns not set)
            cols_to_check = self.key_columns or (list(data[0].keys()) if data else [])
            kept = []
            dropped = 0
            for row in data:
                is_null_key = any(row.get(c) is None for c in cols_to_check)
                if is_null_key:
                    dropped += 1
                else:
                    kept.append(row)
            if dropped > 0:
                logger.info(
                    "Dropped %d rows with null values in key columns: %s",
                    dropped,
                    cols_to_check[:5],
                )
            return kept

        # Per-column fill values take priority
        fill_values = dict(self.fill_columns)

        if self.fill_strategy == "fill_zero":
            for row in data:
                for col in row:
                    if row[col] is None:
                        row[col] = fill_values.get(col, 0)
        elif self.fill_strategy == "fill_mean":
            means = self._compute_means(data)
            filled_with_default = False
            for row in data:
                for col in row:
                    if row[col] is None:
                        val = fill_values.get(col, means.get(col))
                        if val is None:
                            val = 0
                            filled_with_default = True
                        row[col] = val
            if filled_with_default:
                logger.warning(
                    "fill_mean: some columns had no numeric data, "
                    "filled with 0. Check your data for all-null numeric columns."
                )
        else:
            # Default: fill with 0
            for row in data:
                for col in row:
                    if row[col] is None:
                        row[col] = fill_values.get(col, 0)
        return data

    def _deduplicate(self, data: list[dict]) -> list[dict]:
        if not data:
            return data
        keys = self.duplicate_keys or list(data[0].keys())
        seen: set[tuple[Any, ...]] = set()
        unique = []
        for row in data:
            key = tuple(row.get(k) for k in keys)
            if key not in seen:
                seen.add(key)
                unique.append(row)
        return unique

    def _clip_ranges(
        self, data: list[dict]
    ) -> tuple[list[dict], float]:
        clipped = 0
        for row in data:
            for col, (lo, hi) in self.clip_ranges.items():
                val = row.get(col)
                if isinstance(val, (int, float)):
                    if val < lo:
                        row[col] = lo
                        clipped += 1
                    elif val > hi:
                        row[col] = hi
                        clipped += 1
        return data, float(clipped)

    @staticmethod
    def _compute_means(data: list[dict]) -> dict[str, float]:
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for row in data:
            for col, val in row.items():
                if isinstance(val, (int, float)):
                    sums[col] = sums.get(col, 0.0) + val
                    counts[col] = counts.get(col, 0) + 1
        return {col: sums[col] / counts[col] for col in sums if counts[col] > 0}

    @staticmethod
    def _count_null(data: list[dict], columns: list[str] | None = None) -> int:
        cols = columns or (list(data[0].keys()) if data else [])
        return sum(1 for row in data for c in cols if row.get(c) is None)
