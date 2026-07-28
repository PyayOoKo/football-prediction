"""LivePredictionEngine — real-time odds fetching, CLV tracking, and bet recommendations."""

from __future__ import annotations

import contextlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.live_predictions.models import LivePrediction, OddsSnapshot

logger = logging.getLogger(__name__)

LIVE_DIR = Path("data/live")
REPORTS_DIR = Path("reports/live")
DEFAULT_POLL_INTERVAL = 300
DEFAULT_SPORT_KEY = "soccer_fifa_world_cup"
CLV_HISTORY_FILE = LIVE_DIR / "clv_history.json"
BET_RECORDS_FILE = LIVE_DIR / "bet_records.json"
ODDS_HISTORY_DIR = LIVE_DIR / "odds_snapshots"


class LivePredictionEngine:
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
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ODDS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        self._model = self._load_model(model_path)
        self._model_type = self._detect_model_type(self._model)
        self._odds_client: Any = None
        self._clv_tracker: Any = None
        self._load_clv_history()
        self._bet_records: list[dict[str, Any]] = []
        self._load_bet_records()
        self._prev_odds: dict[tuple[str, str], OddsSnapshot] = {}
        self._monitor: Any = None
        if self.enable_monitoring:
            self._init_monitoring()
        self._running = False
        self._cycle_count = 0

    def _load_model(self, path: str | None) -> Any:
        if path is not None:
            p = Path(path)
            if p.exists():
                return self._try_load(p)
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
            "No trained model found. Live predictions will use placeholder probabilities."
        )
        return None

    def _try_load(self, path: Path) -> Any:
        import joblib

        try:
            return joblib.load(path)
        except Exception as exc:
            logger.warning("Failed to load model %s: %s", path, exc)
            return None

    @staticmethod
    def _detect_model_type(model: Any) -> str:
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

    def _get_odds_client(self) -> Any:
        if self._odds_client is None:
            from src.odds_api import OddsAPIClient

            self._odds_client = OddsAPIClient()
        return self._odds_client

    def fetch_live_odds(self) -> list[OddsSnapshot]:
        client = self._get_odds_client()
        if not client.api_key:
            logger.warning("No API key — returning empty odds list")
            return []
        matches = client.get_upcoming_odds(
            sport_key=self.sport_key, bookmaker=self.bookmaker
        )
        now = datetime.now().isoformat()
        return [
            OddsSnapshot(
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
            for m in matches
        ]

    def predict_match(
        self, home_team: str, away_team: str
    ) -> tuple[float, float, float]:
        if self._model is None:
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        try:
            if self._model_type in ("phase3",) and hasattr(
                self._model, "predict_matches"
            ):
                import pandas as pd

                df = pd.DataFrame([{"home_team": home_team, "away_team": away_team}])
                preds = self._model.predict_matches(df)
                if not preds.empty:
                    away = float(preds.iloc[0].get("away_win_prob", 1.0 / 3.0))
                    draw = float(preds.iloc[0].get("draw_prob", 1.0 / 3.0))
                    home = float(preds.iloc[0].get("home_win_prob", 1.0 / 3.0))
                    return (away, draw, home)
            if hasattr(self._model, "predict_proba"):
                try:
                    n_features = self._model.predict_proba(
                        pd.DataFrame([[0.0] * 10])
                    ).shape[0]
                    dummy = pd.DataFrame(np.zeros((1, n_features)))
                    probs = self._model.predict_proba(dummy)[0]
                    if len(probs) == 3:
                        return (float(probs[0]), float(probs[1]), float(probs[2]))
                except Exception as exc:
                    logger.debug("Model predict_proba failed: %s", exc)
        except Exception as exc:
            logger.debug(
                "Model prediction failed for %s vs %s: %s", home_team, away_team, exc
            )
        return self._fallback_prediction(home_team, away_team)

    @staticmethod
    def _fallback_prediction(
        home_team: str, away_team: str
    ) -> tuple[float, float, float]:
        elite = {
            "Brazil",
            "Argentina",
            "France",
            "England",
            "Germany",
            "Spain",
            "Netherlands",
            "Italy",
            "Portugal",
            "Belgium",
        }
        strong = {
            "Croatia",
            "Uruguay",
            "Colombia",
            "Denmark",
            "Switzerland",
            "Japan",
            "South Korea",
            "USA",
            "Mexico",
            "Morocco",
            "Senegal",
        }
        mid = {
            "Poland",
            "Serbia",
            "Sweden",
            "Norway",
            "Wales",
            "Scotland",
            "Austria",
            "Turkey",
            "Iran",
            "Saudi Arabia",
            "Australia",
            "Ecuador",
            "Chile",
            "Peru",
            "Nigeria",
            "Ghana",
            "Cameroon",
            "Algeria",
            "Egypt",
            "Tunisia",
            "Ivory Coast",
        }

        def _tier_weight(team: str) -> float:
            if team in elite:
                return 1.0
            if team in strong:
                return 0.7
            if team in mid:
                return 0.4
            return 0.5

        hw = _tier_weight(home_team)
        aw = _tier_weight(away_team)
        home_adv = 0.08
        raw_home = max(0.1, (hw - aw + 1.0) / 3.0 + home_adv)
        raw_away = max(0.1, (aw - hw + 1.0) / 3.0 - home_adv * 0.5)
        raw_draw = max(0.1, 1.0 - raw_home - raw_away)
        total = raw_home + raw_draw + raw_away
        return (raw_away / total, raw_draw / total, raw_home / total)

    def compute_clv(self, current_odds: float, previous_odds: float | None) -> float:
        if previous_odds is None or previous_odds <= 0:
            return 0.0
        return (current_odds - previous_odds) / previous_odds

    def _track_clv_for_match(
        self, home_team: str, away_team: str, current: OddsSnapshot
    ) -> tuple[float, float, float]:
        key = (home_team, away_team)
        prev = self._prev_odds.get(key)
        home_clv = self.compute_clv(current.home_odds, prev.home_odds if prev else None)
        draw_clv = self.compute_clv(current.draw_odds, prev.draw_odds if prev else None)
        away_clv = self.compute_clv(current.away_odds, prev.away_odds if prev else None)
        self._prev_odds[key] = current
        return (home_clv, draw_clv, away_clv)

    def compute_ev(self, model_prob: float, decimal_odds: float) -> float:
        if decimal_odds <= 0:
            return -1.0
        return model_prob * decimal_odds - 1.0

    def compute_kelly(
        self, model_prob: float, decimal_odds: float, kelly_fraction: float = 0.25
    ) -> float:
        if decimal_odds <= 1.0:
            return 0.0
        full_kelly = (model_prob * decimal_odds - 1.0) / (decimal_odds - 1.0)
        return max(full_kelly * kelly_fraction, 0.0)

    def run_cycle(self) -> list[LivePrediction]:
        self._cycle_count += 1
        logger.info("=== Live Prediction Cycle %d ===", self._cycle_count)
        snapshots = self.fetch_live_odds()
        if not snapshots:
            logger.info("No odds data available this cycle")
            return []
        predictions: list[LivePrediction] = []
        for snapshot in snapshots:
            try:
                away_prob, draw_prob, home_prob = self.predict_match(
                    snapshot.home_team, snapshot.away_team
                )
                h_clv, d_clv, a_clv = self._track_clv_for_match(
                    snapshot.home_team, snapshot.away_team, snapshot
                )
                home_ev = self.compute_ev(home_prob, snapshot.home_odds)
                draw_ev = self.compute_ev(draw_prob, snapshot.draw_odds)
                away_ev = self.compute_ev(away_prob, snapshot.away_odds)
                home_kelly = self.compute_kelly(home_prob, snapshot.home_odds)
                draw_kelly = self.compute_kelly(draw_prob, snapshot.draw_odds)
                away_kelly = self.compute_kelly(away_prob, snapshot.away_odds)
                probs = [away_prob, draw_prob, home_prob]
                confidence = (max(probs) - min(probs)) * 100
                prev = self._prev_odds.get((snapshot.home_team, snapshot.away_team))
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
                    away_kelly=away_kelly,
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
                    snapshot.home_team,
                    snapshot.away_team,
                    exc,
                )
        for pred in predictions:
            self._clv_tracker.append(
                {
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
                }
            )
        if len(self._clv_tracker) > 1000:
            self._clv_tracker = self._clv_tracker[-1000:]
        if self.enable_monitoring and predictions:
            self._log_cycle_metrics(predictions)
        self._save_odds_snapshot(snapshots)
        self._save_predictions(predictions)
        self._save_clv_history()
        self._last_predictions = predictions
        self._print_cycle_summary(predictions)
        return predictions

    def run_continuous(
        self, max_cycles: int | None = None, poll_interval: int | None = None
    ) -> None:
        self._running = True
        interval = poll_interval or self.poll_interval
        logger.info(
            "Live Prediction Engine started — polling every %ds (max_cycles=%s, sport=%s)",
            interval,
            max_cycles or "∞",
            self.sport_key,
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
        self._running = False

    def _save_odds_snapshot(self, snapshots: list[OddsSnapshot]) -> None:
        if not snapshots:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = ODDS_HISTORY_DIR / f"odds_{timestamp}.json"
        with open(path, "w") as f:
            json.dump([s.to_dict() for s in snapshots], f, indent=2)
        self._cleanup_old_files(ODDS_HISTORY_DIR, max_files=100)

    def _save_predictions(self, predictions: list[LivePrediction]) -> None:
        if not predictions:
            return
        data = [p.to_dict() for p in predictions]
        with open(REPORTS_DIR / "latest_predictions.json", "w") as f:
            json.dump(data, f, indent=2)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(REPORTS_DIR / f"predictions_{timestamp}.json", "w") as f:
            json.dump(data, f, indent=2)
        self._cleanup_old_files(REPORTS_DIR, max_files=50)

    def _load_clv_history(self) -> None:
        if CLV_HISTORY_FILE.exists():
            try:
                with open(CLV_HISTORY_FILE) as f:
                    self._clv_tracker = json.load(f)
                logger.info("Loaded CLV history (%d entries)", len(self._clv_tracker))
            except (json.JSONDecodeError, OSError):
                self._clv_tracker = []
        else:
            self._clv_tracker = []

    def _save_clv_history(self) -> None:
        if self._clv_tracker:
            with open(CLV_HISTORY_FILE, "w") as f:
                json.dump(self._clv_tracker, f, indent=2)

    def _load_bet_records(self) -> None:
        if BET_RECORDS_FILE.exists():
            try:
                with open(BET_RECORDS_FILE) as f:
                    self._bet_records = json.load(f)
                logger.info("Loaded %d bet records", len(self._bet_records))
            except (json.JSONDecodeError, OSError):
                self._bet_records = []

    def _save_bet_records(self) -> None:
        if self._bet_records:
            with open(BET_RECORDS_FILE, "w") as f:
                json.dump(self._bet_records, f, indent=2)

    @staticmethod
    def _cleanup_old_files(
        directory: Path, max_files: int, suffix: str = ".json"
    ) -> None:
        if not directory.exists():
            return
        data_files = sorted(
            [p for p in directory.iterdir() if p.suffix == suffix],
            key=lambda p: p.stat().st_mtime,
        )
        while len(data_files) > max_files:
            oldest = data_files.pop(0)
            with contextlib.suppress(OSError):
                oldest.unlink()

    def _init_monitoring(self) -> None:
        try:
            from src.monitoring.store import MonitoringStore

            self._monitor = MonitoringStore(db_path="data/monitoring/monitor.db")
        except Exception as exc:
            logger.warning("Failed to init monitoring store: %s", exc)
            self._monitor = None

    def _log_cycle_metrics(self, predictions: list[LivePrediction]) -> None:
        if self._monitor is None:
            return
        try:
            n_bets = sum(1 for p in predictions if p.n_value_bets > 0)
            avg_ev = float(np.mean([p.best_value_ev for p in predictions]))
            self._monitor.record_metric(
                "live_predictions.matches_found",
                float(len(predictions)),
                tags={"sport": self.sport_key},
            )
            self._monitor.record_metric(
                "live_predictions.value_bets",
                float(n_bets),
                tags={"sport": self.sport_key},
            )
            self._monitor.record_metric(
                "live_predictions.avg_best_ev", avg_ev, tags={"sport": self.sport_key}
            )
            self._monitor.record_metric(
                "live_predictions.cycle_count",
                float(self._cycle_count),
                tags={"sport": self.sport_key},
            )
        except Exception as exc:
            logger.debug("Failed to log monitoring metrics: %s", exc)

    def _print_cycle_summary(self, predictions: list[LivePrediction]) -> None:
        if not predictions:
            logger.info(
                "  [%s] No matches with odds available",
                datetime.now().strftime("%H:%M:%S"),
            )
            return
        n_value = sum(1 for p in predictions if p.n_value_bets > 0)
        total_evs = [p.best_value_ev for p in predictions]
        avg_ev = float(np.mean(total_evs)) if total_evs else 0.0
        logger.info("  %s", "=" * 70)
        logger.info("  LIVE PREDICTIONS — Cycle %d", self._cycle_count)
        logger.info(
            "  %s  |  %d matches  |  %d with value  |  Avg best EV: %+.1f%%",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            len(predictions),
            n_value,
            avg_ev * 100,
        )
        logger.info("  %s", "=" * 70)
        sorted_preds = sorted(predictions, key=lambda p: p.best_value_ev, reverse=True)
        for pred in sorted_preds[:10]:
            ev_str = f"{pred.best_value_ev:+.1%}"
            clv_str = (
                "+"
                if any(c > 0 for c in [pred.home_clv, pred.draw_clv, pred.away_clv])
                else " "
            )
            value_marker = "VALUE" if pred.n_value_bets > 0 else "    "
            logger.info(
                "  %s %-18s vs %-18s  Pred: %-9s  Best EV: %-8s  CLV: %s  Conf: %.0f",
                value_marker,
                pred.home_team,
                pred.away_team,
                pred.predicted_outcome,
                ev_str,
                clv_str,
                pred.confidence_score,
            )
        if len(sorted_preds) > 10:
            logger.info("  ... and %d more matches", len(sorted_preds) - 10)
        logger.info("  %s", "=" * 70)

    def get_value_bets_dataframe(self) -> pd.DataFrame:
        if not hasattr(self, "_last_predictions") or not self._last_predictions:
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
                rows.append(
                    {
                        "home_team": pred.home_team,
                        "away_team": pred.away_team,
                        "match_date": pred.match_date,
                        "outcome": outcome,
                        "model_prob": round(prob, 4),
                        "decimal_odds": odds,
                        "ev": round(ev, 4),
                        "positive_ev": ev > 0,
                        "kelly_pct": round(
                            pred.home_kelly
                            if outcome == "Home"
                            else pred.draw_kelly
                            if outcome == "Draw"
                            else pred.away_kelly,
                            4,
                        ),
                        "confidence": round(pred.confidence_score, 2),
                        "timestamp": pred.timestamp,
                    }
                )
        df = pd.DataFrame(rows)
        if not df.empty:
            df.sort_values(
                by=["positive_ev", "ev"], ascending=[False, False], inplace=True
            )
        return df
