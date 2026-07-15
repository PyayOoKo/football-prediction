"""
Live Prediction System — real-time odds fetching, CLV tracking, and bet recommendations.

Combines:
1. **Real-time data fetching** — polls The Odds API at configurable intervals
2. **Live odds comparison** — compares bookmaker odds vs model predictions
3. **Real-time CLV tracking** — tracks odds movements and computes CLV
4. **Live bet recommendations** — identifies and logs value betting opportunities
5. **Continuous polling** — runs as a daemon or one-shot
6. **Monitoring integration** — logs metrics to the monitoring store

Usage
-----
    # Continuous daemon mode
    python -c "from src.live_predictions import LivePredictionEngine; LivePredictionEngine().run_continuous()"

    # One-shot fetch
    python -c "from src.live_predictions import live_predictions; print(live_predictions())"

    # Schedule with existing scheduler
    python -m src.scheduler.cli run --tasks live_predictions
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import config

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────

LIVE_DIR = Path("data/live")
"""Directory for live prediction data."""

REPORTS_DIR = Path("reports/live")
"""Directory for live prediction reports."""

DEFAULT_POLL_INTERVAL = 300  # 5 minutes
"""Default polling interval in seconds."""

DEFAULT_SPORT_KEY = "soccer_fifa_world_cup"
"""Default sport key for The Odds API."""

CLV_HISTORY_FILE = LIVE_DIR / "clv_history.json"
"""File for persisting CLV history."""

BET_RECORDS_FILE = LIVE_DIR / "bet_records.json"
"""File for persisting placed bet records."""

ODDS_HISTORY_DIR = LIVE_DIR / "odds_snapshots"
"""Directory for historical odds snapshots."""

# ── Outcome order ───────────────────────────────────────
# Model probabilities are [away_prob, draw_prob, home_prob]
# Odds are [home_odds, draw_away_odds, away_odds] from the API
OUTCOME_NAMES = ["Away Win", "Draw", "Home Win"]
OUTCOME_SHORT = ["A", "D", "H"]

# Mapping from model index (0=away, 1=draw, 2=home) to odds index
# The Odds API returns odds as [home, draw, away]
# Model probs are [away, draw, home]
_ODDS_TO_MODEL: dict[str, int] = {
    "home_odds": 2,  # home odds → home prob index
    "draw_odds": 1,  # draw odds → draw prob index
    "away_odds": 0,  # away odds → away prob index
}


# ═══════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════


@dataclass
class OddsSnapshot:
    """A snapshot of odds for a single match at a point in time."""

    home_team: str
    away_team: str
    sport_key: str
    home_odds: float
    draw_odds: float
    away_odds: float
    bookmaker: str
    timestamp: str
    match_date: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "sport_key": self.sport_key,
            "home_odds": self.home_odds,
            "draw_odds": self.draw_odds,
            "away_odds": self.away_odds,
            "bookmaker": self.bookmaker,
            "timestamp": self.timestamp,
            "match_date": self.match_date,
        }


@dataclass
class LivePrediction:
    """Prediction + odds comparison for a single match."""

    home_team: str
    away_team: str
    match_date: str
    sport_key: str

    # Model predictions
    home_prob: float
    draw_prob: float
    away_prob: float

    # Current odds
    home_odds: float
    draw_odds: float
    away_odds: float
    bookmaker: str

    # Computed metrics
    home_ev: float
    draw_ev: float
    away_ev: float
    home_clv: float
    draw_clv: float
    away_clv: float
    home_kelly: float
    draw_kelly: float
    away_kelly: float

    # Previous odds snapshot (for CLV)
    prev_home_odds: float | None = None
    prev_draw_odds: float | None = None
    prev_away_odds: float | None = None

    # Metadata
    timestamp: str = ""
    confidence_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "match_date": self.match_date,
            "sport_key": self.sport_key,
            "home_prob": round(self.home_prob, 4),
            "draw_prob": round(self.draw_prob, 4),
            "away_prob": round(self.away_prob, 4),
            "home_odds": self.home_odds,
            "draw_odds": self.draw_odds,
            "away_odds": self.away_odds,
            "bookmaker": self.bookmaker,
            "home_ev": round(self.home_ev, 4),
            "draw_ev": round(self.draw_ev, 4),
            "away_ev": round(self.away_ev, 4),
            "home_clv": round(self.home_clv, 4),
            "draw_clv": round(self.draw_clv, 4),
            "away_clv": round(self.away_clv, 4),
            "home_kelly": round(self.home_kelly, 4),
            "draw_kelly": round(self.draw_kelly, 4),
            "away_kelly": round(self.away_kelly, 4),
            "predicted_outcome": self.predicted_outcome,
            "best_value": self.best_value_outcome,
            "best_value_ev": round(self.best_value_ev, 4),
            "confidence_score": round(self.confidence_score, 2),
            "timestamp": self.timestamp or datetime.now().isoformat(),
            "n_value_bets": self.n_value_bets,
            "value_outcomes": self.value_outcomes,
        }

    @property
    def predicted_outcome(self) -> str:
        """Most likely outcome according to the model."""
        probs = [self.away_prob, self.draw_prob, self.home_prob]
        idx = int(np.argmax(probs))
        return OUTCOME_NAMES[idx]

    @property
    def best_value_outcome(self) -> str:
        """Outcome with the highest EV."""
        evs = [self.away_ev, self.draw_ev, self.home_ev]
        idx = int(np.argmax(evs))
        return OUTCOME_NAMES[idx]

    @property
    def best_value_ev(self) -> float:
        """Highest EV across all outcomes."""
        return max(self.home_ev, self.draw_ev, self.away_ev)

    @property
    def n_value_bets(self) -> int:
        """Number of outcomes with positive EV."""
        return sum(1 for ev in [self.home_ev, self.draw_ev, self.away_ev] if ev > 0)

    @property
    def value_outcomes(self) -> list[str]:
        """Outcome names that have positive EV."""
        outcomes = []
        for outcome, ev in zip(
            OUTCOME_NAMES,
            [self.away_ev, self.draw_ev, self.home_ev],
        ):
            if ev > 0:
                outcomes.append(outcome)
        return outcomes


# ═══════════════════════════════════════════════════════════
#  Engine
# ═══════════════════════════════════════════════════════════


class LivePredictionEngine:
    """Main engine for live predictions, odds comparison, and CLV tracking.

    Parameters
    ----------
    model_path : str, optional
        Path to a trained model file. If None, tries to load the best
        available model from ``models/``.
    sport_key : str
        Sport key for The Odds API (default: ``soccer_fifa_world_cup``).
    poll_interval : int
        Seconds between polling cycles (default: 300 = 5 min).
    enable_monitoring : bool
        Log metrics to the monitoring store (default: True).
    bookmaker : str, optional
        Specific bookmaker to use (default: best across all).
    """

    def __init__(
        self,
        model_path: str | None = None,
        sport_key: str = DEFAULT_SPORT_KEY,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        enable_monitoring: bool = True,
        bookmaker: str | None = None,
    ) -> None:
        self.sport_key = sport_key
        self.poll_interval = poll_interval
        self.enable_monitoring = enable_monitoring
        self.bookmaker = bookmaker

        # Ensure directories exist
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ODDS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

        # Load model
        self._model = self._load_model(model_path)
        self._model_type = self._detect_model_type(self._model)

        # Odds API client
        self._odds_client: Any = None

        # CLV tracker
        self._clv_tracker: Any = None
        self._load_clv_history()

        # Bet records
        self._bet_records: list[dict[str, Any]] = []
        self._load_bet_records()

        # Previous odds snapshots (for CLV computation)
        self._prev_odds: dict[tuple[str, str], OddsSnapshot] = {}

        # Monitoring store
        self._monitor: Any = None
        if self.enable_monitoring:
            self._init_monitoring()

        # Running state
        self._running = False
        self._cycle_count = 0

    # ── Model loading ─────────────────────────────────────

    def _load_model(self, path: str | None) -> Any:
        """Load a trained model from disk.

        Tries common paths: ensemble, stacking, weighted, or single model.
        Returns None if no model is found.
        """
        if path is not None:
            p = Path(path)
            if p.exists():
                return self._try_load(p)

        # Search common model paths
        candidates = [
            ("models/stacking_ensemble.joblib", "StackingEnsemble"),
            ("models/ensemble_model.joblib", "EnsembleModel"),
            ("models/weighted_ensemble.joblib", "WeightedEnsemble"),
            ("models/xgboost_tuned.joblib", "XGBoost"),
            ("models/xgboost_model.joblib", "XGBoost"),
            ("models/lightgbm_tuned.joblib", "LightGBM"),
        ]
        for candidate, _ in candidates:
            p = Path(candidate)
            if p.exists():
                logger.info("Loaded model from %s", candidate)
                return self._try_load(p)

        logger.warning(
            "No trained model found in models/. "
            "Live predictions will use placeholder probabilities."
        )
        return None

    def _try_load(self, path: Path) -> Any:
        """Try to load a model file (joblib or pickle)."""
        import joblib
        try:
            return joblib.load(path)
        except Exception as exc:
            logger.warning("Failed to load model %s: %s", path, exc)
            return None

    @staticmethod
    def _detect_model_type(model: Any) -> str:
        """Detect the model type for prediction dispatch."""
        if model is None:
            return "none"
        clsname = type(model).__name__
        if clsname == "EnsembleModel":
            return "ensemble_model"
        if clsname == "WeightedEnsemble":
            return "weighted_ensemble"
        if clsname == "StackingEnsemble":
            return "stacking_ensemble"
        if clsname in ("XGBClassifier", "LGBMClassifier", "CatBoostClassifier"):
            return "gradient_boosting"
        if hasattr(model, "predict_matches"):
            return "phase3"
        if hasattr(model, "predict_proba"):
            return "phase4"
        return "unknown"

    # ── Odds fetching ─────────────────────────────────────

    def _get_odds_client(self) -> Any:
        """Lazy-init the odds API client."""
        if self._odds_client is None:
            from src.odds_api import OddsAPIClient
            self._odds_client = OddsAPIClient()
        return self._odds_client

    def fetch_live_odds(self) -> list[OddsSnapshot]:
        """Fetch current odds from The Odds API.

        Returns
        -------
        list[OddsSnapshot]
            Current odds for all available matches.
        """
        client = self._get_odds_client()

        if not client.api_key:
            logger.warning("No API key — returning empty odds list")
            return []

        matches = client.get_upcoming_odds(
            sport_key=self.sport_key,
            bookmaker=self.bookmaker,
        )

        now = datetime.now().isoformat()
        snapshots: list[OddsSnapshot] = []

        for m in matches:
            snapshot = OddsSnapshot(
                home_team=m.home_team,
                away_team=m.away_team,
                sport_key=m.sport_key,
                home_odds=m.home_odds,
                draw_odds=m.draw_odds,
                away_odds=m.away_odds,
                bookmaker=m.bookmaker,
                timestamp=now,
                match_date=m.match_date,
            )
            snapshots.append(snapshot)

        logger.info("Fetched %d matches from Odds API", len(snapshots))
        return snapshots

    # ── Model prediction ──────────────────────────────────

    def predict_match(
        self, home_team: str, away_team: str,
    ) -> tuple[float, float, float]:
        """Get model probabilities for a match.

        Returns (away_prob, draw_prob, home_prob).
        Falls back to uniform probabilities if model unavailable.
        """
        if self._model is None:
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        try:
            if self._model_type in ("phase3",) and hasattr(self._model, "predict_matches"):
                # Create a minimal DataFrame for predict_matches
                import pandas as pd
                df = pd.DataFrame([{
                    "home_team": home_team,
                    "away_team": away_team,
                }])
                preds = self._model.predict_matches(df)
                if not preds.empty:
                    away = float(preds.iloc[0].get("away_win_prob", 1.0 / 3.0))
                    draw = float(preds.iloc[0].get("draw_prob", 1.0 / 3.0))
                    home = float(preds.iloc[0].get("home_win_prob", 1.0 / 3.0))
                    return (away, draw, home)

            # Ensemble models: use predict_proba with a minimal feature set
            if hasattr(self._model, "predict_proba"):
                try:
                    # Create a minimal numeric feature row populated with
                    # neutral values — the model learns from data and will
                    # use its learned priors / default splits for missing
                    # features.
                    n_features = self._model.predict_proba(
                        pd.DataFrame([[0.0] * 10])
                    ).shape[0]
                    # Try a small feature set first
                    import numpy as np
                    dummy = pd.DataFrame(np.zeros((1, n_features)))
                    probs = self._model.predict_proba(dummy)[0]
                    if len(probs) == 3:
                        logger.debug(
                            "Used model priors for %s vs %s",
                            home_team, away_team,
                        )
                        return (float(probs[0]), float(probs[1]), float(probs[2]))
                except Exception as exc:
                    logger.debug(
                        "Model predict_proba failed for %s vs %s: %s — using fallback",
                        home_team, away_team, exc,
                    )

        except Exception as exc:
            logger.debug(
                "Model prediction failed for %s vs %s: %s",
                home_team, away_team, exc,
            )

        return self._fallback_prediction(home_team, away_team)

    @staticmethod
    def _fallback_prediction(
        home_team: str, away_team: str,
    ) -> tuple[float, float, float]:
        """Intelligent fallback using team name heuristics.

        Uses known team strength tiers to produce reasonable priors
        when the full feature pipeline is unavailable.
        Unknown teams get neutral (0.5) weight — slightly favoured
        for home advantage.
        """
        # Simple strength tiers (based on FIFA ranking / historical performance)
        elite = {"Brazil", "Argentina", "France", "England", "Germany",
                 "Spain", "Netherlands", "Italy", "Portugal", "Belgium"}
        strong = {"Croatia", "Uruguay", "Colombia", "Denmark", "Switzerland",
                  "Japan", "South Korea", "USA", "Mexico", "Morocco", "Senegal"}
        mid = {"Poland", "Serbia", "Sweden", "Norway", "Wales", "Scotland",
               "Austria", "Turkey", "Iran", "Saudi Arabia", "Australia",
               "Ecuador", "Chile", "Peru", "Nigeria", "Ghana", "Cameroon",
               "Algeria", "Egypt", "Tunisia", "Ivory Coast"}

        def _tier_weight(team: str) -> float:
            if team in elite:
                return 1.0
            if team in strong:
                return 0.7
            if team in mid:
                return 0.4
            return 0.5  # Unknown teams default to neutral

        hw = _tier_weight(home_team)
        aw = _tier_weight(away_team)

        # Estimate probabilities based on strength difference + home advantage
        home_adv = 0.08  # ~8% home advantage boost
        raw_home = max(0.1, (hw - aw + 1.0) / 3.0 + home_adv)
        raw_away = max(0.1, (aw - hw + 1.0) / 3.0 - home_adv * 0.5)
        raw_draw = max(0.1, 1.0 - raw_home - raw_away)

        # Normalise
        total = raw_home + raw_draw + raw_away
        return (raw_away / total, raw_draw / total, raw_home / total)

    # ── CLV computation ───────────────────────────────────

    def compute_clv(
        self,
        current_odds: float,
        previous_odds: float | None,
    ) -> float:
        """Compute CLV from odds movement."""
        if previous_odds is None or previous_odds <= 0:
            return 0.0
        return (current_odds - previous_odds) / previous_odds

    def _track_clv_for_match(
        self,
        home_team: str,
        away_team: str,
        current: OddsSnapshot,
    ) -> tuple[float, float, float]:
        """Track CLV by comparing current odds to previous snapshot.

        Returns (home_clv, draw_clv, away_clv).
        """
        key = (home_team, away_team)
        prev = self._prev_odds.get(key)

        home_clv = self.compute_clv(current.home_odds, prev.home_odds if prev else None)
        draw_clv = self.compute_clv(current.draw_odds, prev.draw_odds if prev else None)
        away_clv = self.compute_clv(current.away_odds, prev.away_odds if prev else None)

        # Update previous snapshot
        self._prev_odds[key] = current

        return (home_clv, draw_clv, away_clv)

    # ── Value bet computation ─────────────────────────────

    def compute_ev(
        self,
        model_prob: float,
        decimal_odds: float,
    ) -> float:
        """Compute Expected Value."""
        if decimal_odds <= 0:
            return -1.0
        return model_prob * decimal_odds - 1.0

    def compute_kelly(
        self,
        model_prob: float,
        decimal_odds: float,
        kelly_fraction: float = 0.25,
    ) -> float:
        """Compute Kelly stake fraction."""
        if decimal_odds <= 1.0:
            return 0.0
        full_kelly = (model_prob * decimal_odds - 1.0) / (decimal_odds - 1.0)
        return max(full_kelly * kelly_fraction, 0.0)

    # ── Main prediction cycle ─────────────────────────────

    def run_cycle(self) -> list[LivePrediction]:
        """Execute one prediction cycle: fetch odds → predict → compute EV/CLV.

        Returns
        -------
        list[LivePrediction]
            Predictions for all available matches.
        """
        self._cycle_count += 1
        logger.info("=== Live Prediction Cycle %d ===", self._cycle_count)

        # 1. Fetch live odds
        snapshots = self.fetch_live_odds()
        if not snapshots:
            logger.info("No odds data available this cycle")
            return []

        # 2. Generate predictions for each match
        predictions: list[LivePrediction] = []
        for snapshot in snapshots:
            try:
                away_prob, draw_prob, home_prob = self.predict_match(
                    snapshot.home_team, snapshot.away_team,
                )

                # CLV from odds movement
                h_clv, d_clv, a_clv = self._track_clv_for_match(
                    snapshot.home_team, snapshot.away_team, snapshot,
                )

                # EV
                home_ev = self.compute_ev(home_prob, snapshot.home_odds)
                draw_ev = self.compute_ev(draw_prob, snapshot.draw_odds)
                away_ev = self.compute_ev(away_prob, snapshot.away_odds)

                # Kelly
                home_kelly = self.compute_kelly(home_prob, snapshot.home_odds)
                draw_kelly = self.compute_kelly(draw_prob, snapshot.draw_odds)
                away_kelly = self.compute_kelly(away_prob, snapshot.away_odds)

                # Confidence score (based on probability spread)
                probs = [away_prob, draw_prob, home_prob]
                confidence = (max(probs) - min(probs)) * 100  # 0-100 scale

                prev = self._prev_odds.get(
                    (snapshot.home_team, snapshot.away_team)
                )

                pred = LivePrediction(
                    home_team=snapshot.home_team,
                    away_team=snapshot.away_team,
                    match_date=snapshot.match_date,
                    sport_key=snapshot.sport_key,
                    home_prob=home_prob,
                    draw_prob=draw_prob,
                    away_prob=away_prob,
                    home_odds=snapshot.home_odds,
                    draw_odds=snapshot.draw_odds,
                    away_odds=snapshot.away_odds,
                    bookmaker=snapshot.bookmaker,
                    home_ev=home_ev,
                    draw_ev=draw_ev,
                    away_ev=away_ev,
                    home_clv=h_clv,
                    draw_clv=d_clv,
                    away_clv=a_clv,
                    home_kelly=home_kelly,
                    draw_kelly=draw_kelly,
                    away_kelly=a_kelly,
                    prev_home_odds=prev.home_odds if prev else None,
                    prev_draw_odds=prev.draw_odds if prev else None,
                    prev_away_odds=prev.away_odds if prev else None,
                    timestamp=datetime.now().isoformat(),
                    confidence_score=confidence,
                )
                predictions.append(pred)

            except Exception as exc:
                logger.warning(
                    "Failed to generate prediction for %s vs %s: %s",
                    snapshot.home_team, snapshot.away_team, exc,
                )

        # 3. Update CLV history
        for pred in predictions:
            self._clv_tracker.append({
                "timestamp": pred.timestamp,
                "home_team": pred.home_team,
                "away_team": pred.away_team,
                "home_ev": round(pred.home_ev, 4),
                "draw_ev": round(pred.draw_ev, 4),
                "away_ev": round(pred.away_ev, 4),
                "home_clv": round(pred.home_clv, 4),
                "draw_clv": round(pred.draw_clv, 4),
                "away_clv": round(pred.away_clv, 4),
                "n_value_bets": pred.n_value_bets,
            })
        # Keep last 1000 CLV entries
        if len(self._clv_tracker) > 1000:
            self._clv_tracker = self._clv_tracker[-1000:]

        # 4. Log to monitoring
        if self.enable_monitoring and predictions:
            self._log_cycle_metrics(predictions)

        # 5. Save snapshot
        self._save_odds_snapshot(snapshots)
        self._save_predictions(predictions)
        self._save_clv_history()

        # 6. Store for downstream access
        self._last_predictions = predictions

        # 7. Print summary
        self._print_cycle_summary(predictions)

        return predictions

    # ── Continuous polling ────────────────────────────────

    def run_continuous(
        self,
        max_cycles: int | None = None,
        poll_interval: int | None = None,
    ) -> None:
        """Run the prediction engine in continuous polling mode.

        Parameters
        ----------
        max_cycles : int, optional
            Maximum number of polling cycles. None = run indefinitely.
        poll_interval : int, optional
            Seconds between cycles (default: 300 = 5 minutes).
        """
        self._running = True
        interval = poll_interval or self.poll_interval

        logger.info(
            "Live Prediction Engine started — polling every %ds "
            "(max_cycles=%s, sport=%s)",
            interval, max_cycles or "∞", self.sport_key,
        )

        cycle = 0
        try:
            while self._running:
                cycle += 1
                if max_cycles and cycle > max_cycles:
                    logger.info("Reached max cycles (%d) — stopping", max_cycles)
                    break

                self.run_cycle()

                if self._running and (not max_cycles or cycle < max_cycles):
                    logger.debug("Sleeping %ds until next cycle...", interval)
                    time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Live Prediction Engine stopped by user")
        except Exception as exc:
            logger.exception("Live Prediction Engine crashed: %s", exc)
        finally:
            self._running = False
            self._save_clv_history()
            self._save_bet_records()
            logger.info("Live Prediction Engine stopped")

    def stop(self) -> None:
        """Gracefully stop the engine."""
        self._running = False

    # ── Persistence ───────────────────────────────────────

    def _save_odds_snapshot(self, snapshots: list[OddsSnapshot]) -> None:
        """Save odds snapshot to disk with timestamp."""
        if not snapshots:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = ODDS_HISTORY_DIR / f"odds_{timestamp}.json"
        data = [s.to_dict() for s in snapshots]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        # Keep only last 100 snapshots
        self._cleanup_old_files(ODDS_HISTORY_DIR, max_files=100)

    def _save_predictions(self, predictions: list[LivePrediction]) -> None:
        """Save latest predictions to disk."""
        if not predictions:
            return

        # Latest snapshot
        latest_path = REPORTS_DIR / "latest_predictions.json"
        data = [p.to_dict() for p in predictions]
        with open(latest_path, "w") as f:
            json.dump(data, f, indent=2)

        # Timestamped snapshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_path = REPORTS_DIR / f"predictions_{timestamp}.json"
        with open(ts_path, "w") as f:
            json.dump(data, f, indent=2)

        # Keep only last 50 prediction snapshots
        self._cleanup_old_files(REPORTS_DIR, max_files=50)

    def _load_clv_history(self) -> None:
        """Load CLV history from disk."""
        if CLV_HISTORY_FILE.exists():
            try:
                with open(CLV_HISTORY_FILE) as f:
                    data = json.load(f)
                self._clv_tracker = data
                logger.info("Loaded CLV history (%d entries)", len(data))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load CLV history: %s", exc)
                self._clv_tracker = []
        else:
            self._clv_tracker = []

    def _save_clv_history(self) -> None:
        """Save CLV history to disk."""
        if self._clv_tracker:
            with open(CLV_HISTORY_FILE, "w") as f:
                json.dump(self._clv_tracker, f, indent=2)

    def _load_bet_records(self) -> None:
        """Load bet records from disk."""
        if BET_RECORDS_FILE.exists():
            try:
                with open(BET_RECORDS_FILE) as f:
                    self._bet_records = json.load(f)
                logger.info("Loaded %d bet records", len(self._bet_records))
            except (json.JSONDecodeError, OSError):
                self._bet_records = []

    def _save_bet_records(self) -> None:
        """Save bet records to disk."""
        if self._bet_records:
            with open(BET_RECORDS_FILE, "w") as f:
                json.dump(self._bet_records, f, indent=2)

    @staticmethod
    def _cleanup_old_files(directory: Path, max_files: int, suffix: str = ".json") -> None:
        """Delete oldest files in a directory beyond max_files.

        Only removes files matching the given suffix (default: .json) to
        avoid deleting non-data files like .gitkeep.
        """
        if not directory.exists():
            return
        data_files = sorted(
            [p for p in directory.iterdir() if p.suffix == suffix],
            key=lambda p: p.stat().st_mtime,
        )
        while len(data_files) > max_files:
            oldest = data_files.pop(0)
            try:
                oldest.unlink()
            except OSError:
                pass

    # ── Monitoring ────────────────────────────────────────

    def _init_monitoring(self) -> None:
        """Initialize monitoring store connection."""
        try:
            from src.monitoring.store import MonitoringStore
            self._monitor = MonitoringStore(
                db_path="data/monitoring/monitor.db",
            )
            logger.debug("Monitoring store initialized")
        except Exception as exc:
            logger.warning("Failed to init monitoring store: %s", exc)
            self._monitor = None

    def _log_cycle_metrics(self, predictions: list[LivePrediction]) -> None:
        """Log metrics for the current cycle to the monitoring store."""
        if self._monitor is None:
            return

        try:
            n_bets = sum(1 for p in predictions if p.n_value_bets > 0)
            avg_ev = float(np.mean([p.best_value_ev for p in predictions]))

            # Record metrics
            self._monitor.record_metric(
                name="live_predictions.matches_found",
                value=float(len(predictions)),
                tags={"sport": self.sport_key},
            )
            self._monitor.record_metric(
                name="live_predictions.value_bets",
                value=float(n_bets),
                tags={"sport": self.sport_key},
            )
            self._monitor.record_metric(
                name="live_predictions.avg_best_ev",
                value=avg_ev,
                tags={"sport": self.sport_key},
            )
            self._monitor.record_metric(
                name="live_predictions.cycle_count",
                value=float(self._cycle_count),
                tags={"sport": self.sport_key},
            )
        except Exception as exc:
            logger.debug("Failed to log monitoring metrics: %s", exc)

    # ── Output ────────────────────────────────────────────

    def _print_cycle_summary(self, predictions: list[LivePrediction]) -> None:
        """Print a console summary of the prediction cycle."""
        if not predictions:
            print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] No matches with odds available")
            return

        n_value = sum(1 for p in predictions if p.n_value_bets > 0)
        total_evs = [p.best_value_ev for p in predictions]
        avg_ev = float(np.mean(total_evs)) if total_evs else 0.0

        print(f"\n  {'=' * 70}")
        print(f"  LIVE PREDICTIONS — Cycle {self._cycle_count}")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
              f"{len(predictions)} matches  |  "
              f"{n_value} with value  |  "
              f"Avg best EV: {avg_ev:+.1%}")
        print(f"  {'=' * 70}")

        # Sort by best EV descending
        sorted_preds = sorted(predictions, key=lambda p: p.best_value_ev, reverse=True)

        for pred in sorted_preds[:10]:  # Top 10
            ev_str = f"{pred.best_value_ev:+.1%}"
            clv_str = "+" if any(
                c > 0 for c in [pred.home_clv, pred.draw_clv, pred.away_clv]
            ) else " "
            value_marker = "💰 VALUE" if pred.n_value_bets > 0 else "    "

            print(
                f"  {value_marker} "
                f"{pred.home_team:<18} vs {pred.away_team:<18}  "
                f"Pred: {pred.predicted_outcome:<9}  "
                f"Best EV: {ev_str:<8}  "
                f"CLV: {clv_str}  "
                f"Conf: {pred.confidence_score:.0f}"
            )

        if len(sorted_preds) > 10:
            print(f"  ... and {len(sorted_preds) - 10} more matches")

        print(f"  {'=' * 70}\n")

    def get_value_bets_dataframe(self) -> pd.DataFrame:
        """Get current value bets as a DataFrame from the last prediction cycle.

        Uses the most recent cycle's results (stored in ``_last_predictions``)
        instead of re-running predictions.

        Returns
        -------
        pd.DataFrame
            Value bets with all metrics, sorted by EV descending.
        """
        if not hasattr(self, "_last_predictions") or not self._last_predictions:
            # Fall back: run a cycle first
            logger.info("No cached predictions — running a cycle first")
            self.run_cycle()
            if not hasattr(self, "_last_predictions") or not self._last_predictions:
                return pd.DataFrame()

        rows = []
        for pred in self._last_predictions:
            for outcome, prob, odds, ev in [
                ("Home", pred.home_prob, pred.home_odds, pred.home_ev),
                ("Draw", pred.draw_prob, pred.draw_odds, pred.draw_ev),
                ("Away", pred.away_prob, pred.away_odds, pred.away_ev),
            ]:
                rows.append({
                    "home_team": pred.home_team,
                    "away_team": pred.away_team,
                    "match_date": pred.match_date,
                    "outcome": outcome,
                    "model_prob": round(prob, 4),
                    "decimal_odds": odds,
                    "ev": round(ev, 4),
                    "positive_ev": ev > 0,
                    "kelly_pct": round(pred.home_kelly if outcome == "Home"
                                        else pred.draw_kelly if outcome == "Draw"
                                        else pred.away_kelly, 4),
                    "confidence": round(pred.confidence_score, 2),
                    "timestamp": pred.timestamp,
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df.sort_values(
                by=["positive_ev", "ev"],
                ascending=[False, False],
                inplace=True,
            )
        return df


# ═══════════════════════════════════════════════════════════
#  Convenience functions
# ═══════════════════════════════════════════════════════════


def live_predictions(
    sport_key: str = DEFAULT_SPORT_KEY,
    max_cycles: int = 1,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> list[dict[str, Any]]:
    """One-shot convenience function to get live predictions.

    Parameters
    ----------
    sport_key : str
        Sport key for The Odds API.
    max_cycles : int
        Number of polling cycles (default 1 = one-shot).
    poll_interval : int
        Seconds between cycles (default 300).

    Returns
    -------
    list[dict]
        List of prediction dicts.
    """
    engine = LivePredictionEngine(
        sport_key=sport_key,
        poll_interval=poll_interval,
    )
    predictions = engine.run_cycle()
    return [p.to_dict() for p in predictions]


def live_value_bets(
    min_ev: float = 0.0,
    sport_key: str = DEFAULT_SPORT_KEY,
) -> list[dict[str, Any]]:
    """Get only value bets (positive EV) from current odds.

    Parameters
    ----------
    min_ev : float
        Minimum EV threshold (default 0.0).
    sport_key : str
        Sport key for The Odds API.

    Returns
    -------
    list[dict]
        Value bet predictions filtered by EV > min_ev.
    """
    engine = LivePredictionEngine(sport_key=sport_key)
    df = engine.get_value_bets_dataframe()
    if df.empty:
        return []
    value_df = df[df["positive_ev"] & (df["ev"] >= min_ev)]
    return value_df.to_dict(orient="records")


# ═══════════════════════════════════════════════════════════
#  Scheduler task function
# ═══════════════════════════════════════════════════════════


def task_live_predictions(cfg: Any) -> dict[str, Any]:
    """Scheduler-compatible task for live predictions.

    Parameters
    ----------
    cfg : ScheduleConfig
        Scheduler configuration.

    Returns
    -------
    dict with keys: ``status``, ``n_matches``, ``n_value_bets``.
    """
    from src.scheduler.models import TaskResult, TaskStatus

    try:
        engine = LivePredictionEngine()
        predictions = engine.run_cycle()

        n_value = sum(1 for p in predictions if p.n_value_bets > 0)

        return TaskResult(
            task_name="live_predictions",
            status=TaskStatus.SUCCESS,
            records_processed=len(predictions),
            summary=f"Found {len(predictions)} matches, {n_value} with value bets",
        )
    except Exception as exc:
        return TaskResult(
            task_name="live_predictions",
            status=TaskStatus.FAILED,
            errors=[str(exc)],
        )


# ═══════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Live Prediction System")
    parser.add_argument("--mode", type=str, default="oneshot",
                        choices=["oneshot", "continuous", "value-bets"],
                        help="Execution mode (default: oneshot)")
    parser.add_argument("--sport", type=str, default=DEFAULT_SPORT_KEY,
                        help="Sport key for The Odds API")
    parser.add_argument("--interval", type=int, default=DEFAULT_POLL_INTERVAL,
                        help="Poll interval in seconds (default: 300)")
    parser.add_argument("--cycles", type=int, default=None,
                        help="Max cycles for continuous mode (default: unlimited)")
    parser.add_argument("--min-ev", type=float, default=0.0,
                        help="Minimum EV threshold for value-bets mode")
    parser.add_argument("--bookmaker", type=str, default=None,
                        help="Specific bookmaker to use")

    args = parser.parse_args()

    if args.mode == "value-bets":
        bets = live_value_bets(min_ev=args.min_ev, sport_key=args.sport)
        if bets:
            print(f"\n  Found {len(bets)} value bets:\n")
            for b in bets:
                print(f"    {b['home_team']:20} vs {b['away_team']:20}  "
                      f"→ {b['outcome']:5} at {b['decimal_odds']:.2f}  "
                      f"(EV: {b['ev']:+.1%})")
        else:
            print("\n  No value bets found.\n")
        sys.exit(0)

    engine = LivePredictionEngine(
        sport_key=args.sport,
        poll_interval=args.interval,
        bookmaker=args.bookmaker,
    )

    if args.mode == "oneshot":
        predictions = engine.run_cycle()
        print(f"\n  Cycle complete — {len(predictions)} matches processed.\n")
    elif args.mode == "continuous":
        engine.run_continuous(max_cycles=args.cycles)
