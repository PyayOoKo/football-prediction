"""Pipeline functions — live_predictions(), live_value_bets(), task_live_predictions()."""

from __future__ import annotations

from typing import Any, cast

from src.live_predictions.engine import DEFAULT_SPORT_KEY, LivePredictionEngine
from src.scheduler.models import TaskResult, TaskStatus

__all__ = ["live_predictions", "live_value_bets", "task_live_predictions"]


def live_predictions(
    sport_key: str = DEFAULT_SPORT_KEY,
    max_cycles: int = 1,
    poll_interval: int = 300,
) -> list[dict[str, Any]]:
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
    engine = LivePredictionEngine(sport_key=sport_key)
    df = engine.get_value_bets_dataframe()
    if df.empty:
        return []
    value_df = df[df["positive_ev"] & (df["ev"] >= min_ev)]
    return cast("list[dict[str, Any]]", value_df.to_dict(orient="records"))


def task_live_predictions(cfg: Any) -> TaskResult:

    try:
        engine = LivePredictionEngine()
        predictions = engine.run_cycle()
        n_value = sum(1 for p in predictions if p.n_value_bets > 0)
        return TaskResult(
            task_name="live_predictions",
            status=TaskStatus.SUCCESS,
            records_processed=len(predictions),
            output=f"Found {len(predictions)} matches, {n_value} with value bets",
        )
    except Exception as exc:
        return TaskResult(
            task_name="live_predictions",
            status=TaskStatus.FAILED,
            error=str(exc),
        )
