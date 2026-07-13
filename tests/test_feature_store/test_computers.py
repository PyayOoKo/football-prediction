"""Tests for FeatureComputer base class and ComputerRegistry."""

from __future__ import annotations

from typing import Any

import pytest

from src.feature_store.computers import ComputerRegistry, FeatureComputer


class TestFeatureComputer:
    """Test the abstract FeatureComputer base class."""

    def test_cannot_instantiate_abstract(self) -> None:
        """Can't instantiate FeatureComputer directly since compute_one is abstract."""
        with pytest.raises(TypeError):
            FeatureComputer()  # type: ignore

    def test_concrete_subclass(self) -> None:
        class MyComputer(FeatureComputer):
            def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
                return {"feature": float(entity_id * 2)}

        comp = MyComputer(name="test", description="Test computer", version="1.0.0")
        assert comp.name == "test"
        assert comp.description == "Test computer"
        assert comp.version == "1.0.0"
        assert comp.required_data == []

    def test_compute_one(self) -> None:
        class MyComputer(FeatureComputer):
            def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
                return {"value": float(entity_id * 2)}

        comp = MyComputer(name="test")
        result = comp.compute_one(5)
        assert result["value"] == 10.0

    def test_compute_batch(self) -> None:
        class MyComputer(FeatureComputer):
            def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
                return {"value": float(entity_id)}

        comp = MyComputer(name="test")
        results = comp.compute_batch([1, 2, 3])
        assert len(results) == 3
        assert results[1]["value"] == 1.0
        assert results[2]["value"] == 2.0
        assert results[3]["value"] == 3.0

    def test_init_hook(self) -> None:
        class InitComputer(FeatureComputer):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.initialized = False

            def init(self) -> None:
                self.initialized = True

            def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
                return {"val": float(entity_id)}

        comp = InitComputer(name="init_test")
        assert comp.initialized is False
        comp.init()
        assert comp.initialized is True

    def test_validate_hook(self) -> None:
        class ValidComputer(FeatureComputer):
            def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
                return {"val": float(entity_id)}

            def validate(self, result: dict[str, Any]) -> bool:
                return result.get("val", 0) >= 0

        comp = ValidComputer(name="validate_test")
        assert comp.validate({"val": 5.0}) is True
        assert comp.validate({"val": -1.0}) is False

    def test_required_data_init(self) -> None:
        class DataComputer(FeatureComputer):
            def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
                return {}

        comp = DataComputer(
            name="data_test",
            required_data=["matches", "teams", "odds"],
        )
        assert "matches" in comp.required_data
        assert "odds" in comp.required_data

    def test_to_dict(self) -> None:
        class MyComputer(FeatureComputer):
            def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
                return {}

        comp = MyComputer(
            name="test", description="A test",
            version="2.0.0", window=10, k=32,
        )
        d = comp.to_dict()
        assert d["name"] == "test"
        assert d["version"] == "2.0.0"
        assert d["params"]["window"] == 10
        assert d["params"]["k"] == 32

    def test_repr(self) -> None:
        class MyComputer(FeatureComputer):
            def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
                return {}

        comp = MyComputer(name="my_comp", version="1.0")
        assert "MyComputer" in repr(comp)
        assert "my_comp" in repr(comp)


class TestComputerRegistry:
    """Test ComputerRegistry."""

    def test_empty_registry(self) -> None:
        reg = ComputerRegistry()
        assert reg.list_types() == []
        assert reg.has_type("anything") is False
        assert reg.get("anything") is None

    def test_register_decorator(self) -> None:
        reg = ComputerRegistry()

        @reg.register("elo")
        class EloComputer(FeatureComputer):
            def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
                return {"elo": 1500.0}

        assert reg.has_type("elo") is True
        assert "elo" in reg.list_types()

        comp = reg.get("elo")
        assert comp is not None
        assert comp.compute_one(1) == {"elo": 1500.0}

    def test_add_instance(self) -> None:
        reg = ComputerRegistry()

        class MyComputer(FeatureComputer):
            def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
                return {"val": float(entity_id)}

        comp = MyComputer(name="my_comp")
        reg.add("my_type", comp)
        assert reg.has_type("my_type") is True
        got = reg.get("my_type")
        assert got is comp  # Same instance

    def test_get_returns_same_instance(self) -> None:
        reg = ComputerRegistry()

        @reg.register("elo")
        class EloComputer(FeatureComputer):
            def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
                return {}

        c1 = reg.get("elo")
        c2 = reg.get("elo")
        assert c1 is c2  # Same cached instance

    def test_remove(self) -> None:
        reg = ComputerRegistry()

        @reg.register("temp")
        class TempComputer(FeatureComputer):
            def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
                return {}

        assert reg.has_type("temp") is True
        reg.remove("temp")
        assert reg.has_type("temp") is False
        assert reg.get("temp") is None

    def test_list_types(self) -> None:
        reg = ComputerRegistry()

        @reg.register("elo")
        class EloComputer(FeatureComputer):
            def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
                return {}

        class FormComputer(FeatureComputer):
            def compute_one(self, entity_id: int, **kwargs: Any) -> dict[str, Any]:
                return {}

        reg.add("team_form", FormComputer(name="form"))

        types = reg.list_types()
        assert "elo" in types
        assert "team_form" in types
