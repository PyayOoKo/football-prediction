"""Tests for FeatureRegistry."""

from __future__ import annotations

import pytest

from src.feature_store.models import (
    FeatureCategory,
    FeatureStatus,
)
from src.feature_store.registry import FEATURE_TYPES


class TestFeatureRegistry:
    """Test feature definition registry."""

    def test_register(self, registry) -> None:
        fd = registry.register(
            name="home_attack_strength",
            feature_type="attack_strength",
            category=FeatureCategory.ATTACK_STRENGTH,
            entity_type="team",
            description="Rolling attack strength ratio",
            computation_params={"window": 5},
            status=FeatureStatus.ACTIVE,
        )
        assert fd.name == "home_attack_strength"
        assert fd.version == 1
        assert fd.is_active is True
        assert fd.id is not None

    def test_register_with_initial_dependencies(self, registry) -> None:
        fd_a = registry.register(
            name="base_elo", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
        )
        fd_b = registry.register(
            name="derived_form", feature_type="team_form",
            category=FeatureCategory.TEAM_FORM, entity_type="match",
            dependencies=["base_elo"],
        )
        deps = registry.get_dependencies(fd_b.id)
        assert len(deps) == 1
        assert deps[0].id == fd_a.id

    def test_register_duplicate_raises(self, registry) -> None:
        registry.register(
            name="dup", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
        )
        with pytest.raises(ValueError, match="already exists"):
            registry.register(
                name="dup", feature_type="elo",
                category=FeatureCategory.ELO_RATING, entity_type="team",
            )

    def test_register_empty_name(self, registry) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            registry.register(
                name="", feature_type="elo",
                category=FeatureCategory.ELO_RATING, entity_type="team",
            )

    def test_register_invalid_type(self, registry) -> None:
        with pytest.raises(ValueError, match="Unknown feature type"):
            registry.register(
                name="bad", feature_type="nonexistent_type",
                category=FeatureCategory.ELO_RATING, entity_type="team",
            )

    def test_register_invalid_entity_type(self, registry) -> None:
        with pytest.raises(ValueError, match="Unknown entity type"):
            registry.register(
                name="bad", feature_type="elo",
                category=FeatureCategory.ELO_RATING, entity_type="spaceship",
            )

    # ── Lookup ────────────────────────────────────────────

    def test_get_by_id(self, registry) -> None:
        fd = registry.register(
            name="test", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
        )
        found = registry.get(fd.id)
        assert found is not None
        assert found.id == fd.id

    def test_get_nonexistent_id(self, registry) -> None:
        found = registry.get("nonexistent-id-12345678901234567890")
        assert found is None

    def test_latest(self, registry) -> None:
        registry.register(
            name="multi", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
            version=1,
        )
        registry.register(
            name="multi", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
            version=2,
        )
        latest = registry.latest("multi")
        assert latest is not None
        assert latest.version == 2

    def test_latest_nonexistent(self, registry) -> None:
        assert registry.latest("nonexistent") is None

    def test_get_by_name_version(self, registry) -> None:
        registry.register(
            name="test", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
            version=1,
        )
        found = registry.get_by_name_version("test", 1)
        assert found is not None
        assert found.version == 1
        assert registry.get_by_name_version("test", 999) is None

    # ── Listing ───────────────────────────────────────────

    def test_list_empty(self, registry) -> None:
        assert registry.list() == []

    def test_list_all(self, registry) -> None:
        registry.register(
            name="a", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
        )
        registry.register(
            name="b", feature_type="rolling_stat",
            category=FeatureCategory.ROLLING_STAT, entity_type="match",
        )
        assert len(registry.list()) == 2

    def test_list_filter_by_category(self, registry) -> None:
        registry.register(
            name="a", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
        )
        registry.register(
            name="b", feature_type="rolling_stat",
            category=FeatureCategory.ROLLING_STAT, entity_type="match",
        )
        results = registry.list(category=FeatureCategory.ELO_RATING)
        assert len(results) == 1
        assert results[0].name == "a"

    def test_list_filter_by_active(self, registry) -> None:
        registry.register(
            name="active_feat", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
            status=FeatureStatus.ACTIVE,
        )
        registry.register(
            name="draft_feat", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
            status=FeatureStatus.DRAFT,
        )
        assert len(registry.list(is_active=True)) == 1
        assert len(registry.list(is_active=False)) == 1

    def test_count(self, registry) -> None:
        assert registry.count() == 0
        registry.register(
            name="a", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
        )
        registry.register(
            name="b", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
        )
        assert registry.count() == 2

    # ── Versioning ────────────────────────────────────────

    def test_new_version(self, registry) -> None:
        registry.register(
            name="test", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
        )
        v2 = registry.new_version(
            "test",
            changelog="Updated K-factor",
            computation_params={"k": 64},
        )
        assert v2.version == 2
        assert v2.computation_params["k"] == 64

    def test_new_version_nonexistent(self, registry) -> None:
        with pytest.raises(ValueError, match="No active feature"):
            registry.new_version("nonexistent")

    def test_get_history(self, registry) -> None:
        registry.register(
            name="test", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
        )
        registry.new_version("test", changelog="v2")
        registry.new_version("test", changelog="v3")
        history = registry.get_history("test")
        assert len(history) == 3
        assert history[0].version == 3
        assert history[0].is_current is True

    # ── Lifecycle ─────────────────────────────────────────

    def test_activate(self, registry) -> None:
        fd = registry.register(
            name="test", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
            status=FeatureStatus.DRAFT,
        )
        assert fd.status == FeatureStatus.DRAFT
        activated = registry.activate("test")
        assert activated.status == FeatureStatus.ACTIVE
        assert activated.is_active is True

    def test_activate_nonexistent(self, registry) -> None:
        with pytest.raises(ValueError, match="not found"):
            registry.activate("nonexistent")

    def test_deprecate(self, registry) -> None:
        fd = registry.register(
            name="test", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
            status=FeatureStatus.ACTIVE,
        )
        deprecated = registry.deprecate("test", reason="Replaced by v2")
        assert deprecated.status == FeatureStatus.DEPRECATED
        assert deprecated.is_active is False

    def test_retire(self, registry) -> None:
        registry.register(
            name="test", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
        )
        retired = registry.retire("test", reason="No longer useful")
        assert retired.status == FeatureStatus.RETIRED

    # ── Dependencies ──────────────────────────────────────

    def test_dependency_dag(self, registry) -> None:
        a = registry.register(
            name="elo_rating", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
        )
        b = registry.register(
            name="attack_strength", feature_type="attack_strength",
            category=FeatureCategory.ATTACK_STRENGTH, entity_type="team",
            dependencies=["elo_rating"],
        )
        c = registry.register(
            name="team_form", feature_type="team_form",
            category=FeatureCategory.TEAM_FORM, entity_type="match",
            dependencies=["attack_strength"],
        )

        # Dependencies
        assert len(registry.get_dependencies(a.id)) == 0
        assert len(registry.get_dependencies(b.id)) == 1
        assert registry.get_dependencies(b.id)[0].name == "elo_rating"
        assert len(registry.get_dependencies(c.id)) == 1
        assert registry.get_dependencies(c.id)[0].name == "attack_strength"

        # Dependents
        assert len(registry.get_dependents(a.id)) == 1
        assert registry.get_dependents(a.id)[0].name == "attack_strength"
        assert len(registry.get_dependents(c.id)) == 0

    def test_topological_sort(self, registry) -> None:
        a = registry.register(
            name="a", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
        )
        b = registry.register(
            name="b", feature_type="attack_strength",
            category=FeatureCategory.ATTACK_STRENGTH, entity_type="team",
            dependencies=["a"],
        )
        c = registry.register(
            name="c", feature_type="team_form",
            category=FeatureCategory.TEAM_FORM, entity_type="match",
            dependencies=["b"],
        )

        sorted_defs = registry.topological_sort()
        names = [d.name for d in sorted_defs]
        # Dependencies should come before dependents
        assert names.index("a") < names.index("b")
        assert names.index("b") < names.index("c")

    def test_topological_sort_empty(self, registry) -> None:
        assert registry.topological_sort() == []

    def test_has_cycle_no_cycle(self, registry) -> None:
        assert registry.has_cycle() is False

    def test_search(self, registry) -> None:
        registry.register(
            name="home_attack", feature_type="attack_strength",
            category=FeatureCategory.ATTACK_STRENGTH, entity_type="team",
        )
        registry.register(
            name="away_attack", feature_type="attack_strength",
            category=FeatureCategory.ATTACK_STRENGTH, entity_type="team",
        )
        registry.register(
            name="elo_rating", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
        )
        results = registry.search("attack")
        assert len(results) == 2
        assert set(r.name for r in results) == {"home_attack", "away_attack"}

    def test_to_dict(self, registry) -> None:
        registry.register(
            name="test", feature_type="elo",
            category=FeatureCategory.ELO_RATING, entity_type="team",
        )
        d_list = registry.to_dict()
        assert len(d_list) == 1
        assert d_list[0]["name"] == "test"
        assert "id" in d_list[0]
