from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.di_container import ConfigProvider, get_container
from src.value_betting import compute_value_bets, print_bets

logger = logging.getLogger(__name__)


class ValueBettingService:
    def __init__(self, config: ConfigProvider | None = None) -> None:
        self._config = config or get_container().resolve(ConfigProvider)

    def find_value_bets(
        self,
        predictions: list[dict[str, Any]],
        min_ev: float | None = None,
        bankroll: float | None = None,
        kelly_fraction: float | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        cfg = self._config

        if not predictions:
            logger.info("No predictions to analyse for value bets.")
            return pd.DataFrame()

        team_matches: list[tuple[str, str]] = []
        odds_list: list[list[float]] = []
        probs_list: list[list[float]] = []

        from src.data import OddsCollector

        collector = OddsCollector()

        for pred in predictions:
            home = str(pred.get("home_team", ""))
            away = str(pred.get("away_team", ""))
            team_matches.append((home, away))

            probs_list.append([
                float(pred.get("away_win_prob", 0)),
                float(pred.get("draw_prob", 0)),
                float(pred.get("home_win_prob", 0)),
            ])

            odds_data = collector.get_best_odds(home, away)
            if odds_data and all(odds_data.get(k, 0) > 0 for k in ["home_odds", "draw_odds", "away_odds"]):
                odds_list.append([
                    float(odds_data["away_odds"]),
                    float(odds_data["draw_odds"]),
                    float(odds_data["home_odds"]),
                ])
            else:
                odds_list.append([0.0, 0.0, 0.0])

        valid_mask = [
            all(o > 0 for o in odds) and all(p >= 0 for p in probs)
            for odds, probs in zip(odds_list, probs_list)
        ]
        valid_odds = [o for o, v in zip(odds_list, valid_mask) if v]
        valid_probs = [p for p, v in zip(probs_list, valid_mask) if v]
        valid_teams = [t for t, v in zip(team_matches, valid_mask) if v]

        if not valid_odds:
            logger.warning("No odds data available for any match.")
            return pd.DataFrame()

        kwargs: dict[str, Any] = {"config": cfg}
        if min_ev is not None:
            kwargs["min_ev"] = min_ev
        if bankroll is not None:
            kwargs["bankroll"] = bankroll
        if kelly_fraction is not None:
            kwargs["kelly_fraction"] = kelly_fraction

        df = compute_value_bets(
            odds=valid_odds,
            model_probs=valid_probs,
            team_matches=valid_teams,
            **kwargs,
        )

        if verbose and len(df) > 0:
            print_bets(df)

        n_positive = int(df["positive_ev"].sum()) if "positive_ev" in df.columns else 0
        logger.info(
            "Found %d value bets out of %d outcomes analysed",
            n_positive, len(df),
        )

        return df
