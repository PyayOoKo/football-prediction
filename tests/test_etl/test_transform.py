"""
Tests for the ETL transformation stage — DataTransformer, ColumnSelector, FieldMapper, Aggregator.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.etl.transform import (
    Aggregator,
    ColumnSelector,
    DataTransformer,
    FieldMapper,
)
from src.etl.models import PipelineStage, StageStatus


# ═══════════════════════════════════════════════════════════
#  ColumnSelector
# ═══════════════════════════════════════════════════════════

class TestColumnSelector:
    def test_keep_only(self) -> None:
        selector = ColumnSelector(keep=["id", "name"])
        data: list[dict[str, Any]] = [
            {"id": 1, "name": "A", "secret": "hidden"},
            {"id": 2, "name": "B", "secret": "hidden2"},
        ]
        result = selector(data)
        assert result == [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]

    def test_drop(self) -> None:
        selector = ColumnSelector(drop=["secret"])
        data = [{"id": 1, "name": "A", "secret": "x"}]
        result = selector(data)
        assert result == [{"id": 1, "name": "A"}]

    def test_rename(self) -> None:
        selector = ColumnSelector(rename={"old_name": "new_name"})
        data = [{"old_name": "value", "other": 1}]
        result = selector(data)
        assert result == [{"new_name": "value", "other": 1}]
        assert "old_name" not in result[0]

    def test_keep_and_drop_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            ColumnSelector(keep=["a"], drop=["b"])

    def test_no_op(self) -> None:
        selector = ColumnSelector()
        data = [{"a": 1, "b": 2}]
        result = selector(data)
        assert result == data

    def test_keep_drops_unlisted(self) -> None:
        selector = ColumnSelector(keep=["x"])
        data = [{"x": 10, "y": 20, "z": 30}]
        result = selector(data)
        assert result == [{"x": 10}]

    def test_rename_preserves_unrelated(self) -> None:
        selector = ColumnSelector(rename={"a": "alpha"})
        data = [{"a": 1, "b": 2}]
        result = selector(data)
        assert result == [{"alpha": 1, "b": 2}]


# ═══════════════════════════════════════════════════════════
#  FieldMapper
# ═══════════════════════════════════════════════════════════

class TestFieldMapper:
    def test_multiply_column(self) -> None:
        mapper = FieldMapper("goals", lambda x: x * 2)
        data: list[dict[str, Any]] = [{"goals": 3}]
        result = mapper(data)
        assert result[0]["goals"] == 6

    def test_string_upper(self) -> None:
        mapper = FieldMapper("name", lambda x: x.upper() if isinstance(x, str) else x)
        data = [{"name": "hello"}]
        result = mapper(data)
        assert result[0]["name"] == "HELLO"

    def test_nonexistent_column_ignored(self) -> None:
        mapper = FieldMapper("missing", lambda x: "changed")
        data = [{"name": "test"}]
        result = mapper(data)
        assert result == data  # unchanged

    def test_preserves_other_columns(self) -> None:
        mapper = FieldMapper("goals", lambda x: 0)
        data = [{"goals": 5, "assists": 2}]
        result = mapper(data)
        assert result[0]["goals"] == 0
        assert result[0]["assists"] == 2

    def test_mapper_is_callable(self) -> None:
        mapper = FieldMapper("col", lambda x: x)
        assert callable(mapper)


# ═══════════════════════════════════════════════════════════
#  Aggregator
# ═══════════════════════════════════════════════════════════

class TestAggregator:
    def test_sum_aggregation(self) -> None:
        agg = Aggregator(group_by=["team"], aggregations={"goals": "sum"})
        data: list[dict[str, Any]] = [
            {"team": "A", "goals": 3},
            {"team": "A", "goals": 2},
            {"team": "B", "goals": 1},
        ]
        result = agg(data)
        assert len(result) == 2
        team_a = next(r for r in result if r["team"] == "A")
        team_b = next(r for r in result if r["team"] == "B")
        assert team_a["goals"] == 5
        assert team_b["goals"] == 1

    def test_mean_aggregation(self) -> None:
        agg = Aggregator(group_by=["team"], aggregations={"rating": "mean"})
        data = [
            {"team": "A", "rating": 8.0},
            {"team": "A", "rating": 6.0},
        ]
        result = agg(data)
        assert result[0]["rating"] == 7.0

    def test_count_aggregation(self) -> None:
        agg = Aggregator(group_by=["team"], aggregations={"id": "count"})
        data = [
            {"team": "A", "id": 1},
            {"team": "A", "id": 2},
            {"team": "A", "id": 3},
        ]
        result = agg(data)
        assert result[0]["id"] == 3

    def test_min_max(self) -> None:
        agg = Aggregator(group_by=["team"], aggregations={"score": "min", "score": "max"})
        data = [
            {"team": "A", "score": 10},
            {"team": "A", "score": 20},
            {"team": "A", "score": 30},
        ]
        result = agg(data)
        # Last aggregator wins for duplicate cols
        assert "score" in result[0]

    def test_multiple_group_columns(self) -> None:
        agg = Aggregator(group_by=["league", "season"], aggregations={"goals": "sum"})
        data = [
            {"league": "E0", "season": "2024", "goals": 10},
            {"league": "E0", "season": "2024", "goals": 5},
            {"league": "E0", "season": "2025", "goals": 8},
        ]
        result = agg(data)
        assert len(result) == 2

    def test_none_values_skipped_in_agg(self) -> None:
        agg = Aggregator(group_by=["team"], aggregations={"goals": "sum"})
        data = [
            {"team": "A", "goals": 5},
            {"team": "A", "goals": None},
        ]
        result = agg(data)
        assert result[0]["goals"] == 5  # None skipped

    def test_all_none_values(self) -> None:
        agg = Aggregator(group_by=["team"], aggregations={"goals": "sum"})
        data = [
            {"team": "A", "goals": None},
        ]
        result = agg(data)
        # When all values are None, the group key is never created
        # because val is not appended to the group dict
        assert len(result) == 0

    def test_unknown_agg_fn_returns_list(self) -> None:
        agg = Aggregator(group_by=["team"], aggregations={"goals": "custom_fn"})
        data = [{"team": "A", "goals": 3}]
        result = agg(data)
        assert isinstance(result[0]["goals"], list)


# ═══════════════════════════════════════════════════════════
#  DataTransformer
# ═══════════════════════════════════════════════════════════

class TestDataTransformer:
    def test_no_transforms(self) -> None:
        transformer = DataTransformer()
        data: list[dict[str, Any]] = [{"a": 1}]
        result = transformer.run(data)
        assert result.status == StageStatus.SUCCESS
        assert result.records_out == 1
        assert result.data == data

    def test_single_transform(self) -> None:
        transformer = DataTransformer()
        transformer.add(ColumnSelector(keep=["id"]))
        data = [{"id": 1, "name": "A"}]
        result = transformer.run(data)
        assert result.data == [{"id": 1}]

    def test_multiple_transforms_in_order(self) -> None:
        transformer = DataTransformer()
        transformer.add(ColumnSelector(keep=["value"]))
        transformer.add(FieldMapper("value", lambda x: x * 2))
        data = [{"value": 5, "ignored": "x"}]
        result = transformer.run(data)
        assert result.data == [{"value": 10}]

    def test_transform_applied_count_in_metrics(self) -> None:
        transformer = DataTransformer()
        transformer.add(ColumnSelector(keep=["a"]))
        transformer.add(FieldMapper("a", lambda x: x))
        result = transformer.run([{"a": 1}])
        assert result.metrics["transforms_applied"] == 2

    def test_empty_data_warning(self) -> None:
        transformer = DataTransformer()
        result = transformer.run([])
        assert result.status == StageStatus.WARNING
        assert "zero rows" in result.errors[0]

    def test_transform_error_handling(self) -> None:
        def broken(data):
            raise RuntimeError("Transform crashed")

        transformer = DataTransformer(transforms=[broken])
        result = transformer.run([{"a": 1}])
        assert result.status == StageStatus.FAILED
        assert "Transform crashed" in result.errors[0]

    def test_add_returns_self_for_chaining(self) -> None:
        transformer = DataTransformer()
        result = transformer.add(ColumnSelector(keep=["a"]))
        assert result is transformer

    def test_records_in_out(self) -> None:
        transformer = DataTransformer(transforms=[lambda d: d])
        data = [{"x": 1}, {"x": 2}, {"x": 3}]
        result = transformer.run(data)
        assert result.records_in == 3
        assert result.records_out == 3
