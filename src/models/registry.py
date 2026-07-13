"""
Model Registry — central registry with automatic versioning, promotion,
and lifecycle management.

Supports registering models, querying by name/type/version, promoting
models to production, and automatic version increment on re-registration.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for all prediction models.

    Provides automatic versioning — every time a model is registered,
    its version is auto-incremented if a previous version exists.

    Integrates with the experiment tracking database (``experiments``,
    ``runs``, ``best_models``, ``model_artifacts`` tables) and supports
    querying, promoting, and comparing models.

    Parameters
    ----------
    use_db : bool
        If True, persists registrations to the database (default True).
        Set False for unit tests or offline usage.
    auto_version : bool
        If True, auto-increments version on re-registration (default True).
    """

    def __init__(
        self,
        use_db: bool = True,
        auto_version: bool = True,
    ) -> None:
        self.use_db = use_db
        self.auto_version = auto_version

        # In-memory registry (used regardless of DB setting)
        self._models: dict[str, BaseModel] = {}
        self._history: dict[str, list[BaseModel]] = {}
        self._experiment_id: str | None = None

    # ── Properties ─────────────────────────────────────

    @property
    def registered_names(self) -> list[str]:
        """List of registered model names."""
        return sorted(self._models.keys())

    @property
    def registered_models(self) -> list[BaseModel]:
        """List of all registered model instances."""
        return list(self._models.values())

    def count(self, model_type: str | None = None) -> int:
        """Count registered models, optionally filtered by type."""
        if model_type is None:
            return len(self._models)
        return sum(
            1 for m in self._models.values()
            if m.model_type == model_type
        )

    # ── Registration ──────────────────────────────────

    def register(
        self,
        model: BaseModel,
        experiment_name: str | None = None,
        force: bool = False,
    ) -> str:
        """Register a model in the registry.

        If a model with the same name already exists and ``auto_version``
        is true, the version is auto-incremented.

        Parameters
        ----------
        model : BaseModel
            The model instance to register.
        experiment_name : str, optional
            Experiment name for DB integration. Auto-generated if omitted.
        force : bool
            If True, overwrite existing registration without versioning.

        Returns
        -------
        str
            The model's registered version string.
        """
        name = model.model_name
        version = model.model_version

        # Auto-version
        existing = self._models.get(name)
        if existing and self.auto_version and not force:
            new_version = self._bump_version(existing.model_version)
            model.model_version = new_version
            version = new_version
            logger.info(
                "Auto-incremented %s: %s → %s",
                name, existing.model_version, version,
            )

        # Store in-memory
        self._models[name] = model
        self._history.setdefault(name, []).append(model)

        # Persist to DB
        if self.use_db:
            self._persist_registration(model, experiment_name=experiment_name)

        logger.info(
            "Registered model: %s v%s (type=%s)",
            name, version, model.model_type,
        )
        return version

    def unregister(self, name: str) -> bool:
        """Remove a model from the registry.

        Parameters
        ----------
        name : str
            Model name to unregister.

        Returns
        -------
        bool
            True if the model was found and removed.
        """
        if name in self._models:
            del self._models[name]
            logger.info("Unregistered model: %s", name)
            return True
        return False

    # ── Querying ─────────────────────────────────────

    def get(self, name: str, version: str | None = None) -> BaseModel | None:
        """Get a registered model by name and optional version.

        Parameters
        ----------
        name : str
            Model name.
        version : str, optional
            Specific version. If None, returns the latest.

        Returns
        -------
        BaseModel | None
        """
        if version is None:
            return self._models.get(name)

        # History lookup for specific version
        history = self._history.get(name, [])
        for m in reversed(history):
            if m.model_version == version:
                return m
        return None

    def list_by_type(self, model_type: str) -> list[BaseModel]:
        """List all registered models of a given type."""
        return [
            m for m in self._models.values()
            if m.model_type == model_type
        ]

    def list_versions(self, name: str) -> list[str]:
        """List all registered versions of a model."""
        history = self._history.get(name, [])
        return [m.model_version for m in history]

    def get_latest(self, model_type: str | None = None) -> BaseModel | None:
        """Get the most recently registered model.

        Parameters
        ----------
        model_type : str, optional
            If set, returns the latest of this type only.

        Returns
        -------
        BaseModel | None
        """
        if model_type:
            typed = self.list_by_type(model_type)
            return typed[-1] if typed else None
        if self._models:
            # Get the last registered model
            last_key = list(self._models.keys())[-1]
            return self._models[last_key]
        return None

    # ── Promotion ────────────────────────────────────

    def promote(
        self,
        name: str,
        metric_name: str = "val_log_loss",
        rank: int = 1,
        notes: str = "",
    ) -> bool:
        """Promote a model to 'best' status for a given metric.

        Records the model as a ``BestModel`` in the experiment tracking DB.

        Parameters
        ----------
        name : str
            Model name to promote.
        metric_name : str
            Metric to rank by (e.g. ``val_log_loss``).
        rank : int
            Rank (1 = best).
        notes : str
            Optional notes about the promotion.

        Returns
        -------
        bool
            True if promotion succeeded.
        """
        model = self._models.get(name)
        if model is None:
            logger.warning("Cannot promote '%s' — not registered", name)
            return False

        metric_value = model._training_metrics.get(metric_name, 0.0)

        if self.use_db:
            try:
                # Lazy imports to avoid circular deps with experiment_tracking
                from src.database.session import get_session
                from src.experiment_tracking.models import (
                    Experiment, Run, BestModel, ModelArtifact,
                )

                with get_session() as session:
                    # Find or create experiment
                    exp = session.query(Experiment).filter_by(
                        name=f"registry_{model.model_type}"
                    ).first()
                    if exp is None:
                        exp = Experiment(
                            name=f"registry_{model.model_type}",
                            description=f"Auto-tracked {model.model_type} models",
                            model_version=model.model_version,
                        )
                        session.add(exp)
                        session.flush()

                    # Create a run record
                    run = Run.create(
                        experiment_id=exp.id,
                        model_type=model.model_type,
                        hyperparameters=model.metadata,
                        random_seed=model.random_seed,
                    )
                    run.status = "completed"
                    run.metrics = model._training_metrics
                    if model._fit_duration_seconds:
                        run.training_duration_seconds = model._fit_duration_seconds
                    session.add(run)
                    session.flush()

                    # Create BestModel entry
                    best = BestModel(
                        experiment_id=exp.id,
                        run_id=run.id,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        rank=rank,
                        is_promoted=True,
                        promoted_at=datetime.now(timezone.utc),
                        notes=notes or f"Promoted from registry ({model.model_type} v{model.model_version})",
                    )
                    session.add(best)
                    session.flush()

                    # Create artifact reference
                    artifact = ModelArtifact(
                        run_id=run.id,
                        name=f"{model.model_name}_{model.model_version}.joblib",
                        uri=f"models/{model.model_name}_{model.model_version}.joblib",
                        artifact_type="model",
                    )
                    session.add(artifact)

                    logger.info(
                        "Promoted %s v%s to rank %d (metric=%s=%.4f)",
                        name, model.model_version, rank, metric_name, metric_value,
                    )
            except Exception as exc:
                logger.warning("DB promotion failed (non-fatal): %s", exc)

        return True

    # ── Internal ─────────────────────────────────────

    @staticmethod
    def _bump_version(current: str) -> str:
        """Increment the patch version of a semver string.

        Examples
        --------
        ``1.2.3`` → ``1.2.4``
        ``0.1.0`` → ``0.1.1``
        ``1.0.0-beta`` → ``1.0.1``
        """
        try:
            parts = current.split(".")
            patch = int(parts[-1].split("-")[0]) + 1
            return f"{parts[0]}.{parts[1]}.{patch}"
        except (IndexError, ValueError):
            return f"{current}.1"

    def _persist_registration(
        self,
        model: BaseModel,
        experiment_name: str | None = None,
    ) -> None:
        """Persist model registration to the experiment tracking DB."""
        try:
            # Lazy imports to avoid circular deps with experiment_tracking
            from src.database.session import get_session
            from src.experiment_tracking.models import (
                Experiment, Run,
            )

            with get_session() as session:
                exp_name = experiment_name or f"registry_{model.model_type}"
                exp = session.query(Experiment).filter_by(name=exp_name).first()
                if exp is None:
                    exp = Experiment(
                        name=exp_name,
                        description=f"Auto-registered {model.model_type} models",
                        model_version=model.model_version,
                    )
                    session.add(exp)
                    session.flush()

                run = Run(
                    experiment_id=exp.id,
                    model_type=model.model_type,
                    status="registered",
                    hyperparameters=model.metadata,
                    random_seed=model.random_seed,
                )
                session.add(run)
        except Exception as exc:
            logger.debug("DB persistence skipped (non-fatal): %s", exc)

    def __repr__(self) -> str:
        types = {}
        for m in self._models.values():
            t = m.model_type
            types[t] = types.get(t, 0) + 1
        type_summary = ", ".join(f"{t}={n}" for t, n in types.items())
        return f"<ModelRegistry: {len(self._models)} models ({type_summary})>"

    def to_dict(self) -> dict[str, Any]:
        """Export registry state as a dictionary."""
        return {
            "n_models": len(self._models),
            "models": {
                name: {
                    "model_type": m.model_type,
                    "model_version": m.model_version,
                    "fitted": m.fitted,
                    "calibrated": m.calibrated,
                    "fit_duration": m._fit_duration_seconds,
                }
                for name, m in self._models.items()
            },
            "history": {
                name: [v.model_version for v in versions]
                for name, versions in self._history.items()
            },
        }
