"""
Prediction service — orchestrates model inference.

Coordinates feature engineering, model loading, prediction generation,
and storing results to the database or CSV.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import config as _global_config
from src.services import add_target_col, load_and_prepare, resolve_data_path

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for generating and storing match predictions.

    Parameters
    ----------
    model_dir : Path, optional
        Directory where trained models are stored.  Defaults to
        ``config.paths.models``.
    config : Config, optional
        Config instance for dependency injection.  Defaults to the
        global singleton ``config``.
    """

    def __init__(self, model_dir: Path | None = None, config: Any | None = None) -> None:
        self._config = config or _global_config
        self._model_dir = model_dir or self._config.paths.models
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._encoder: Any = None  # SafeTargetEncoder instance loaded from artifact

    # ── Public API ─────────────────────────────────────────

    def predict_upcoming(
        self,
        data_path: str | Path | None = None,
        model_name: str | None = None,
        limit: int = 10,
        output_path: str | Path | None = None,
    ) -> list[dict]:
        """Generate predictions for upcoming (unplayed) matches.

        Loads the most recent trained model and the fixture data,
        builds the feature matrix, runs inference, and returns
        structured predictions.  Optionally saves results to CSV/JSON.

        Parameters
        ----------
        data_path : str | Path, optional
            Path to raw match data CSV.  Auto-resolved if omitted.
        model_name : str, optional
            Model file name within ``models/``.  If omitted, uses the
            most recent ``.joblib`` file.
        limit : int
            Maximum number of upcoming matches to predict (default 10).
            Rows are sorted by date so the *soonest* matches are returned.
        output_path : str | Path, optional
            If provided, save predictions to this file (``.csv`` or
            ``.json``).

        Returns
        -------
        list[dict]
            Prediction results, each with keys: ``match_id``, ``date``,
            ``home_team``, ``away_team``, ``home_win_prob``,
            ``draw_prob``, ``away_win_prob``, ``prediction``,
            ``confidence``.
        """
        cfg = self._config
        logger.info("Predicting upcoming matches (limit=%d)", limit)

        # ── 1. Load model ───────────────────────────────────
        model = self._load_model(model_name)

        # ── 2. Load & prepare data (pipeline) ─────────────
        data_path = resolve_data_path(data_path, config=cfg)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Match data not found at {data_path}. "
                "Run data collection first."
            )

        df = load_and_prepare(data_path, add_temporal=True, config=cfg)

        # Separate upcoming (unplayed) matches
        upcoming_mask = df["result"].isna() | (df["result"] == "")
        upcoming = df[upcoming_mask].copy()

        if len(upcoming) == 0:
            logger.info("No upcoming matches found — all results available.")
            return []

        # Sort by date so `head(limit)` returns the soonest matches
        if "date" in upcoming.columns:
            upcoming = upcoming.sort_values("date").reset_index(drop=True)

        # Apply limit
        if 0 < limit < len(upcoming):
            upcoming = upcoming.head(limit)

        # ── 3. Build features with stable row IDs ──────────
        from src.feature_engineering import build_features

        completed = df[df["result"].notna()].copy()
        if len(completed) == 0:
            logger.warning("No completed matches — cannot build features.")
            return []

        # Add stable row ID before any sorting/concatenation
        completed = completed.reset_index(drop=True)
        upcoming = upcoming.reset_index(drop=True)
        completed["_row_id"] = "completed_" + completed.index.astype(str)
        upcoming["_row_id"] = "upcoming_" + upcoming.index.astype(str)
        row_id_order = list(upcoming["_row_id"])  # preserve order

        # Build features on combined data so rolling stats flow
        # from completed to upcoming matches.
        combined = pd.concat([completed, upcoming], ignore_index=True)
        X_all, _ = build_features(
            combined, is_training=True, config=cfg,
            encoder=self._encoder,
        )

        # Select upcoming rows by stable row ID (not positional offset)
        if "_row_id" in X_all.columns:
            X_upcoming_all = X_all[X_all["_row_id"].str.startswith("upcoming_", na=False)].copy()
            # Reorder to original row_id_order (handles any sorting during feature engineering)
            X_upcoming_all = X_upcoming_all.set_index("_row_id").reindex(row_id_order).reset_index(drop=True)
        else:
            # Legacy fallback (row_id was dropped — use positional)
            logger.warning("_row_id column missing after feature engineering — using positional fallback")
            X_upcoming_all = X_all.iloc[len(completed):].copy()

        # Drop row_id for model input
        X_upcoming = X_upcoming_all.drop(columns=["_row_id"], errors="ignore")

        if len(X_upcoming) == 0:
            logger.warning("No feature rows generated for upcoming matches.")
            return []

        # ── 3b. Align columns if model is a ModelArtifact ──
        from src.models.artifact import ModelArtifact
        if isinstance(model, ModelArtifact):
            X_upcoming = model.select_columns(X_upcoming)

        # ── 4. Run inference ────────────────────────────────
        probs = model.predict_proba(X_upcoming)
        preds = np.argmax(probs, axis=1)
        confidences = probs.max(axis=1)

        # ── 4b. Map probabilities using model.classes_
        classes, target_label_map = _resolve_class_mapping(model)

        # ── 5. Build results ────────────────────────────────
        results = []
        for i, (_, row) in enumerate(upcoming.iterrows()):
            if i >= len(probs):
                break
            pred_class = int(preds[i])
            label = target_label_map.get(pred_class, "?")
            # Map probability columns by model.classes_ order
            probs_i = dict(zip(classes, probs[i]))
            home_win_prob = float(probs_i.get(2, probs[i][2]))
            draw_prob = float(probs_i.get(1, probs[i][1]))
            away_win_prob = float(probs_i.get(0, probs[i][0]))
            results.append({
                "match_id": int(row.get("match_id", i)),
                "date": str(row.get("date", ""))[:10],
                "home_team": str(row.get("home_team", "")),
                "away_team": str(row.get("away_team", "")),
                "home_win_prob": home_win_prob,
                "draw_prob": draw_prob,
                "away_win_prob": away_win_prob,
                "prediction": label,
                "confidence": round(float(confidences[i]), 4),
            })

        # ── 6. Save if requested ────────────────────────────
        if output_path is not None:
            self._save_predictions(results, output_path)

        logger.info("Predicted %d upcoming matches", len(results))
        return results

    def predict_match(self, match_id: int, data_path: str | Path | None = None) -> dict | None:
        """Generate a prediction for a single match.

        Looks up the match by its row index in the raw data, builds
        features on the fly, and returns the predicted outcome.

        Parameters
        ----------
        match_id : int
            Row index (or ``match_id`` column value) of the match to
            predict.
        data_path : str | Path, optional
            Path to the raw match data CSV.

        Returns
        -------
        dict | None
            Prediction result, or ``None`` if the match is not found
            or cannot be predicted.
        """
        cfg = self._config
        logger.info("Predicting single match: id=%d", match_id)

        # ── 1. Load model ───────────────────────────────────
        model = self._load_model()

        # ── 2. Load & prepare data (pipeline) ─────────────
        data_path = resolve_data_path(data_path, config=cfg)
        if not data_path.exists():
            logger.warning("Data not found at %s", data_path)
            return None

        df = load_and_prepare(data_path, add_temporal=False, config=cfg)

        # Try match_id column first; fall back to iloc
        if "match_id" in df.columns:
            mask = df["match_id"] == match_id
        else:
            mask = df.index == match_id

        match_rows = df[mask]
        if len(match_rows) == 0:
            logger.warning("Match id=%d not found in %s", match_id, data_path)
            return None

        match_row = match_rows.iloc[[0]]

        # ── 3. Build features on full data, isolate this row ─
        completed = df[df["result"].notna()].copy()
        upcoming = match_row.copy()

        from src.feature_engineering import build_features
        combined = pd.concat([completed, upcoming], ignore_index=True)
        X_all, _ = build_features(
            combined, is_training=True, config=cfg,
            encoder=self._encoder,
        )
        X_match = X_all.iloc[[len(completed)]].copy()

        if len(X_match) == 0:
            logger.warning("No features generated for match id=%d", match_id)
            return None

        # ── 4. Predict ──────────────────────────────────────
        probs = model.predict_proba(X_match)[0]
        pred_class = int(np.argmax(probs))
        _, target_label_map = _resolve_class_mapping(model)
        label = target_label_map.get(pred_class, "?")
        probs_i = _probs_by_class(model, probs)

        result = {
            "match_id": match_id,
            "date": str(match_row.get("date", "").iloc[0])[:10],
            "home_team": str(match_row.get("home_team", "").iloc[0]),
            "away_team": str(match_row.get("away_team", "").iloc[0]),
            "home_win_prob": round(float(probs_i.get(2, probs[2])), 4),
            "draw_prob": round(float(probs_i.get(1, probs[1])), 4),
            "away_win_prob": round(float(probs_i.get(0, probs[0])), 4),
            "prediction": label,
            "confidence": round(float(probs.max()), 4),
        }

        logger.info(
            "Match %d: %s vs %s → %s (%.1f%%)",
            match_id, result["home_team"], result["away_team"],
            result["prediction"], result["confidence"] * 100,
        )
        return result

    def backfill_predictions(
        self,
        start_date: date,
        end_date: date,
        data_path: str | Path | None = None,
    ) -> list[dict]:
        """Generate predictions for historical matches (semi-walk-forward).

        .. caution::

            **Semi-walk-forward approach.**  The model is loaded once and
            remains fixed for all predictions, which means it may have been
            trained on data *after* the match being predicted (the model
            has seen future matches' patterns).  Features **are** leakage-
            free — only pre-match data is used to build the feature matrix
            for each match.  The backtest accuracy may therefore be
            slightly optimistic vs. a pure walk-forward (which retrains
            the model per match at O(n²) cost).

        For each match in the date range, builds features using only
        information available *before* that match, runs the model, and
        records the prediction alongside the actual result.

        Parameters
        ----------
        start_date : date
            Start of the date range (inclusive).
        end_date : date
            End of the date range (inclusive).
        data_path : str | Path, optional
            Path to the raw match data CSV.

        Returns
        -------
        list[dict]
            Each dict contains match info, predicted probabilities, the
            actual result, and whether the prediction was correct.
        """
        cfg = self._config
        logger.info(
            "Backfilling predictions from %s to %s", start_date, end_date,
        )

        # ── 1. Load model ───────────────────────────────────
        model = self._load_model()

        # ── 2. Load & prepare data (pipeline) ─────────────
        data_path = resolve_data_path(data_path, config=cfg)
        if not data_path.exists():
            raise FileNotFoundError(f"Data not found at {data_path}")

        df = load_and_prepare(data_path, add_temporal=False, config=cfg)

        # Filter to date range
        mask = (df["date"] >= pd.Timestamp(start_date)) & (
            df["date"] <= pd.Timestamp(end_date)
        )
        target_matches = df[mask].copy()
        if len(target_matches) == 0:
            logger.info("No matches found in date range %s — %s", start_date, end_date)
            return []

        # Only completed matches (we need actual results for comparison)
        target_matches = target_matches[target_matches["result"].notna()].copy()
        logger.info("Backfilling %d matches", len(target_matches))

        # ── 3. Iterative walk-forward prediction ────────────
        from src.feature_engineering import build_features

        results: list[dict] = []
        all_sorted = df.sort_values(["date", "home_team"]).reset_index(drop=True)

        label_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

        for idx, (_, match_row) in enumerate(target_matches.iterrows()):
            match_date = match_row["date"]

            # Train on all matches strictly before this date
            train_df = all_sorted[all_sorted["date"] < match_date].copy()

            if len(train_df) < 20:
                logger.debug(
                    "Skipping %s — only %d prior matches",
                    match_date.date(), len(train_df),
                )
                continue

            try:
                # Build features on training data only (no future info)
                X_train, y_train = build_features(
                    train_df, is_training=True,
                    encoder=self._encoder,
                )
                if len(X_train) < 10:
                    continue

                # Build features for this single match appended to training window
                this_match = match_row.to_frame().T
                combined = pd.concat([train_df, this_match], ignore_index=True)
                X_all, _ = build_features(
                    combined, is_training=True, config=cfg,
                    encoder=self._encoder,
                )

                if len(X_all) <= len(train_df):
                    continue
                X_match = X_all.iloc[[-1]].copy()

                probs = model.predict_proba(X_match)[0]
                pred_class = int(np.argmax(probs))
                _, target_label_map = _resolve_class_mapping(model)
                label = target_label_map.get(pred_class, "?")
                probs_i = _probs_by_class(model, probs)

                actual_result = str(match_row.get("result", ""))
                actual_label = label_map.get(actual_result, actual_result)
                is_correct = (
                    (pred_class == 2 and actual_result == "H")
                    or (pred_class == 1 and actual_result == "D")
                    or (pred_class == 0 and actual_result == "A")
                )

                results.append({
                    "match_id": int(idx),
                    "date": str(match_date.date()),
                    "home_team": str(match_row.get("home_team", "")),
                    "away_team": str(match_row.get("away_team", "")),
                    "home_win_prob": round(float(probs_i.get(2, probs[2])), 4),
                    "draw_prob": round(float(probs_i.get(1, probs[1])), 4),
                    "away_win_prob": round(float(probs_i.get(0, probs[0])), 4),
                    "prediction": label,
                    "confidence": round(float(probs.max()), 4),
                    "actual_result": actual_label,
                    "correct": bool(is_correct),
                })

            except Exception as exc:
                logger.debug(
                    "Skipping match %s — feature building failed: %s",
                    match_row.get("date", "?"), exc,
                )
                continue

        correct_count = sum(1 for r in results if r["correct"])
        total = len(results)
        pct = correct_count / max(total, 1) * 100
        logger.info(
            "Backfill complete: %d/%d correct (%.1f%%)", correct_count, total, pct,
        )
        return results

    # ── Internals ──────────────────────────────────────────

    def _load_model(self, model_name: str | None = None) -> Any:
        """Load a trained model from the models directory.

        Supports both ``ModelArtifact`` (new) and raw joblib (legacy).
        If *model_name* is omitted, picks the most recently modified
        ``.joblib`` file.

        Also extracts ``SafeTargetEncoder`` state from the artifact
        and stores it in ``self._encoder`` for use during feature
        engineering.
        """
        from src.models.artifact import ModelArtifact
        from src.features.encoding import SafeTargetEncoder
        import joblib

        self._encoder = None  # Reset from previous load

        if model_name is not None:
            path = self._model_dir / model_name
            if not path.exists():
                raise FileNotFoundError(
                    f"Model '{model_name}' not found in {self._model_dir}. "
                    f"Available: {[p.name for p in self._model_dir.glob('*.joblib')]}"
                )
            obj = joblib.load(path)
            if isinstance(obj, ModelArtifact):
                logger.info(
                    "Loaded artifact: %s (%d features, %s)",
                    model_name, obj.n_features, obj.model_type,
                )
                if obj.target_encoder_state:
                    self._encoder = SafeTargetEncoder.from_state(obj.target_encoder_state)
                    logger.info(
                        "Target encoder restored from artifact (%d cols, prior=%.4f)",
                        len(self._encoder.cols), self._encoder.prior,
                    )
            return obj

        # Auto-pick: most recent .joblib
        candidates = sorted(
            self._model_dir.glob("*.joblib"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No models found in {self._model_dir}. "
                "Train a model first via TrainingService.train()."
            )
        latest = candidates[-1]
        logger.info("Auto-selected model: %s", latest.name)

        obj = joblib.load(latest)
        if isinstance(obj, ModelArtifact):
            logger.info(
                "Loaded artifact: %s (%d features, %s)",
                latest.name, obj.n_features, obj.model_type,
            )
            if obj.target_encoder_state:
                self._encoder = SafeTargetEncoder.from_state(obj.target_encoder_state)
                logger.info(
                    "Target encoder restored from artifact (%d cols, prior=%.4f)",
                    len(self._encoder.cols), self._encoder.prior,
                )
        else:
            logger.info("Loaded legacy model: %s (no target encoder)", latest.name)
        return obj

    def predict_with_odds(
        self,
        data_path: str | Path | None = None,
        model_name: str | None = None,
        limit: int = 10,
        output_path: str | Path | None = None,
    ) -> list[dict]:
        """Generate predictions enriched with live odds and value-bet analysis.

        Wraps :meth:`predict_upcoming` and enriches each prediction with
        multi-bookmaker odds, implied probabilities, edge, and EV via
        the ``OddsCollector``.

        Parameters
        ----------
        data_path : str | Path, optional
            Path to match data CSV.
        model_name : str, optional
            Model to use for predictions.
        limit : int
            Max upcoming matches to predict (default 10).
        output_path : str | Path, optional
            Save enriched results to CSV/JSON.

        Returns
        -------
        list[dict]
            Predictions with a ``"odds_analysis"`` key per match, containing:
            ``home_odds``, ``draw_odds``, ``away_odds``, ``margin_pct``,
            ``source``, ``arbitrage_available``, and per-outcome ``edges``
            with ``odds``, ``fair_prob``, ``edge_pp``, ``ev_pct``, ``is_value``.
        """
        cfg = self._config
        logger.info("Predicting with odds enrichment (limit=%d)", limit)

        predictions = self.predict_upcoming(
            data_path=data_path,
            model_name=model_name,
            limit=limit,
            output_path=None,
        )
        if not predictions:
            return []

        import numpy as np

        # Fetch live odds for each match
        from src.data import OddsCollector
        collector = OddsCollector()

        enriched = []
        for pred in predictions:
            h, a = pred["home_team"], pred["away_team"]
            odds = collector.get_best_odds(h, a)

            if odds and all(odds.get(k, 0) > 0 for k in ["home_odds", "draw_odds", "away_odds"]):
                model_probs = np.array([
                    pred["away_win_prob"],
                    pred["draw_prob"],
                    pred["home_win_prob"],
                ])
                odds_array = np.array([
                    odds["away_odds"],
                    odds["draw_odds"],
                    odds["home_odds"],
                ])

                # Compute fair (margin-adjusted) probabilities
                implied = 1.0 / odds_array
                margin = implied.sum() - 1.0
                fair_probs = implied / (1.0 + margin) if margin > 0 else implied

                # Per-outcome analysis
                outcomes = ["Away Win", "Draw", "Home Win"]
                edges = {}
                for idx, outcome in enumerate(outcomes):
                    mod_prob = model_probs[idx]
                    fair_prob = fair_probs[idx]
                    edge_pp = float((mod_prob - fair_prob) * 100)
                    ev = float((mod_prob * odds_array[idx] - 1.0) * 100)

                    edges[outcome] = {
                        "odds": float(odds_array[idx]),
                        "fair_prob": round(float(fair_prob), 4),
                        "edge_pp": round(edge_pp, 2),
                        "ev_pct": round(ev, 2),
                        "is_value": bool(edge_pp > 5.0 or ev > 5.0),
                    }

                pred["odds_analysis"] = {
                    "home_odds": float(odds["home_odds"]),
                    "draw_odds": float(odds["draw_odds"]),
                    "away_odds": float(odds["away_odds"]),
                    "margin_pct": round(float(margin * 100), 2),
                    "source": str(odds.get("source", "unknown")),
                    "arbitrage_available": bool(
                        odds.get("arbitrage", {}).get("is_arbitrage", False)
                    ),
                    "edges": edges,
                }

            enriched.append(pred)

        # Save if requested
        if output_path is not None:
            self._save_predictions(enriched, output_path)

        n_with_odds = sum(1 for p in enriched if "odds_analysis" in p)
        logger.info(
            "Enriched %d/%d predictions with odds", n_with_odds, len(enriched),
        )
        return enriched

    def _save_predictions(self, results: list[dict], output_path: str | Path) -> None:
        """Save prediction results to CSV or JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(results)
        suffix = path.suffix.lower()

        if suffix == ".json":
            df.to_json(path, orient="records", indent=2, date_format="iso")
        else:
            df.to_csv(path, index=False)

        logger.info("Predictions saved to %s (%d rows)", path, len(df))


# ── Module-level helpers ───────────────────────────────────


def _resolve_class_mapping(model: Any) -> tuple[list[int], dict[int, str]]:
    """Extract class labels and a human-readable label mapping from *model*.

    The mapping is always ``{0: "Away Win", 1: "Draw", 2: "Home Win"}``
    regardless of the order in ``model.classes_``.  The *classes* list
    preserves whatever order the model uses natively so callers can
    build ``dict(zip(classes, probs[i]))``.

    Returns
    -------
    tuple[list[int], dict[int, str]]
        (classes, label_map) where *classes* is the ordered list from
        ``model.classes_`` (or ``[0, 1, 2]`` fallback) and *label_map*
        maps integer class to ``"Away Win"`` / ``"Draw"`` / ``"Home Win"``.
    """
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        classes = [0, 1, 2]
    label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
    return classes, label_map


def _probs_by_class(model: Any, probs: np.ndarray) -> dict[int, float]:
    """Map a probability vector to a dict keyed by ``model.classes_``.

    Parameters
    ----------
    model : Any
        Model with optional ``classes_`` attribute.
    probs : np.ndarray
        1-D probability vector from ``predict_proba``.

    Returns
    -------
    dict[int, float]
        Mapping like ``{0: 0.1, 1: 0.2, 2: 0.7}`` regardless of model
        probability column order.
    """
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        return dict(zip(classes, probs))
    return dict(zip([0, 1, 2], probs))
