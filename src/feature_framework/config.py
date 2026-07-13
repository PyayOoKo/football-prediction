"""
Feature Configuration — load feature definitions from YAML/JSON files
and validate them against the schema.

This module enables **declarative feature engineering**: define features
in ``features.yaml``, and the framework handles registration, dependency
resolution, and computation automatically.

Example YAML
------------
.. code-block:: yaml

    version: "1.0"
    pipeline:
      default_entity_type: match
      show_progress: true
      max_retries: 2
      parallel: true
      max_workers: 4

    features:
      - name: elo_rating
        version: 1
        description: "Elo ratings for home and away teams"
        type: elo
        category: elo_rating
        data_type: float
        computation_time: medium
        output_columns: [h_elo, a_elo, elo_difference]
        params:
          k: 32
          home_advantage: 100
        validation:
          min: 1000
          max: 2500
        dependencies: []
        tags: [rating, historical]

      - name: home_attack_strength
        version: 2
        description: "Rolling average home goals scored"
        type: rolling_stat
        category: attack_strength
        data_type: float
        computation_time: fast
        output_columns: [h_goals_scored_avg5]
        params:
          window: 5
          stat: goals_scored
          role: home
        validation:
          min: 0
          max: 5
        dependencies: []
        tags: [form, attack]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.feature_framework.exceptions import FeatureConfigError

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  Schema
# ═══════════════════════════════════════════════════════════

REQUIRED_FEATURE_FIELDS = {"name", "type", "category"}
OPTIONAL_FEATURE_FIELDS = {
    "version", "description", "data_type", "computation_time",
    "output_columns", "params", "validation", "dependencies",
    "tags", "author", "source", "enabled",
}

ALLOWED_DATA_TYPES = {"float", "int", "str", "bool", "categorical", "datetime"}
ALLOWED_COMPUTATION_TIMES = {"fast", "medium", "slow"}
ALLOWED_TRIGGERS = {"manual", "scheduled", "pipeline", "all"}

DEFAULT_PIPELINE_CONFIG = {
    "default_entity_type": "match",
    "show_progress": True,
    "max_retries": 0,
    "parallel": True,
    "max_workers": 4,
}


class FeatureDefinitionSchema:
    """Schema for validating feature definition dicts.

    Parameters
    ----------
    definition : dict
        Raw feature definition from config.
    """

    def __init__(self, definition: dict[str, Any]) -> None:
        self._def = definition

    def validate(self) -> list[str]:
        """Validate a feature definition against the schema.

        Returns
        -------
        list[str]
            Validation errors (empty if valid).
        """
        errors: list[str] = []

        # Required fields
        for field in REQUIRED_FEATURE_FIELDS:
            if field not in self._def:
                errors.append(f"Missing required field: {field}")

        if errors:
            return errors

        # Type checks
        if self._def.get("data_type") and self._def["data_type"] not in ALLOWED_DATA_TYPES:
            errors.append(
                f"Invalid data_type: {self._def['data_type']}. "
                f"Must be one of {ALLOWED_DATA_TYPES}"
            )

        if self._def.get("computation_time") and \
           self._def["computation_time"] not in ALLOWED_COMPUTATION_TIMES:
            errors.append(
                f"Invalid computation_time: {self._def['computation_time']}. "
                f"Must be one of {ALLOWED_COMPUTATION_TIMES}"
            )

        # Params validation
        if self._def.get("params") and not isinstance(self._def["params"], dict):
            errors.append("params must be a dict")

        # Dependencies validation
        if self._def.get("dependencies") and not isinstance(self._def["dependencies"], list):
            errors.append("dependencies must be a list of strings")

        # Version check
        version = self._def.get("version", 1)
        if not isinstance(version, int) or version < 1:
            errors.append(f"version must be a positive integer, got {version}")

        return errors


# ═══════════════════════════════════════════════════════════
#  FeatureConfig — loads and validates feature YAML/JSON
# ═══════════════════════════════════════════════════════════


class FeatureConfig:
    """Configuration-driven feature definitions.

    Loads feature definitions from a YAML/JSON file, validates them,
    and provides access to all configured features and pipeline settings.

    Parameters
    ----------
    path : str | Path
        Path to the YAML/JSON config file.

    Raises
    ------
    FeatureConfigError
        If the file cannot be loaded or validated.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[str, Any] = {}
        self._features: list[dict[str, Any]] = []
        self._pipeline_cfg: dict[str, Any] = {}
        self.load()

    # ── Loading ─────────────────────────────────────────

    def load(self) -> None:
        """Load and validate the configuration file."""
        if not self._path.exists():
            raise FeatureConfigError(f"Config file not found: {self._path}", str(self._path))

        suffix = self._path.suffix.lower()
        try:
            with open(self._path) as f:
                if suffix in (".yaml", ".yml"):
                    import yaml
                    self._data = yaml.safe_load(f) or {}
                elif suffix == ".json":
                    self._data = json.load(f)
                else:
                    raise FeatureConfigError(
                        f"Unsupported file format: {suffix}. Use .yaml, .yml, or .json.",
                        str(self._path),
                    )
        except ImportError as exc:
            raise FeatureConfigError(
                f"PyYAML is required for .yaml files. Install: pip install pyyaml. Error: {exc}",
                str(self._path),
            )
        except json.JSONDecodeError as exc:
            raise FeatureConfigError(f"Invalid JSON: {exc}", str(self._path))

        # Extract pipeline config
        self._pipeline_cfg = {**DEFAULT_PIPELINE_CONFIG, **self._data.get("pipeline", {})}

        # Extract and validate features
        raw_features = self._data.get("features", [])
        errors: list[str] = []
        seen_names: set[str] = set()
        for i, feat in enumerate(raw_features):
            feat_name = feat.get("name", f"<unnamed #{i}>")
            if feat_name in seen_names:
                errors.append(f"Duplicate feature name: {feat_name!r} (appears multiple times)")
            seen_names.add(feat_name)
            schema = FeatureDefinitionSchema(feat)
            feat_errors = schema.validate()
            if feat_errors:
                errors.append(f"Feature #{i}: {'; '.join(feat_errors)}")
            elif feat.get("enabled", True):
                self._features.append(feat)

        if errors:
            raise FeatureConfigError(
                f"Config validation failed with {len(errors)} error(s):\n  "
                + "\n  ".join(errors),
                str(self._path),
            )

        logger.info(
            "Loaded %d features from %s (pipeline: %s)",
            len(self._features), self._path, self._pipeline_cfg,
        )

    # ── Accessors ──────────────────────────────────────

    @property
    def features(self) -> list[dict[str, Any]]:
        """All configured feature definitions."""
        return list(self._features)

    @property
    def pipeline_config(self) -> dict[str, Any]:
        """Pipeline-level configuration."""
        return dict(self._pipeline_cfg)

    @property
    def feature_names(self) -> list[str]:
        """Names of all configured features."""
        return [f["name"] for f in self._features]

    def get(self, name: str) -> dict[str, Any] | None:
        """Get a specific feature definition by name."""
        for feat in self._features:
            if feat["name"] == name:
                return feat
        return None

    def get_by_type(self, feature_type: str) -> list[dict[str, Any]]:
        """Get all features of a given type."""
        return [f for f in self._features if f.get("type") == feature_type]

    def get_by_category(self, category: str) -> list[dict[str, Any]]:
        """Get all features in a given category."""
        return [f for f in self._features if f.get("category") == category]

    def get_enabled(self) -> list[dict[str, Any]]:
        """Get all enabled feature definitions."""
        return [f for f in self._features if f.get("enabled", True)]

    # ── Serialization ──────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Export entire configuration as a dict."""
        return {
            "version": self._data.get("version", "1.0"),
            "pipeline": self._pipeline_cfg,
            "features": self._features,
        }

    def __repr__(self) -> str:
        return (
            f"<FeatureConfig: {len(self._features)} features, "
            f"pipeline={self._pipeline_cfg.get('default_entity_type', 'match')}>"
        )


# ═══════════════════════════════════════════════════════════
#  Convenience factory
# ═══════════════════════════════════════════════════════════


def load_feature_config(
    path: str | Path,
) -> FeatureConfig:
    """Load and return a FeatureConfig from a file.

    Parameters
    ----------
    path : str | Path
        Path to YAML/JSON config file.

    Returns
    -------
    FeatureConfig
    """
    return FeatureConfig(path)
