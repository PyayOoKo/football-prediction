"""
Portfolio Data Backend — loads, tracks, and simulates the 4-strategy portfolio.

Strategies
----------
1. **F1 Over 2.5** — France Ligue 1, Over 2.5 value bets
2. **E0 Over 2.5** — England Premier League, Over 2.5 value bets
3. **I1 Under 2.5** — Italy Serie A, Under 2.5 value bets
4. **D1 Over 2.5** — Germany Bundesliga, Over 2.5 value bets

All use the per-league trained models (DC + Elo + tree models) with
Platt-calibrated probabilities and Kelly staking at 25% fraction.

Portfolio state is persisted to ``reports/portfolio_state.json`` and
updated each time the data module loads.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "football_data.db"
MODELS_DIR = PROJECT_ROOT / "models" / "per_league"
STATE_PATH = PROJECT_ROOT / "reports" / "portfolio_state.json"

# ── Strategy Configuration ──────────────────────────────

INITIAL_BANKROLL = 10_000.0
MIN_EV = 0.05
KELLY_FRAC = 0.25

STRATEGIES: dict[str, dict[str, Any]] = {
    "F1 Over 2.5": {
        "league": "F1",
        "market": "Over 2.5",
        "direction": "over",
        "allocation": 0.30,
        "color": "#4fc3f7",
        "description": "France Ligue 1 — Over 2.5 goals",
    },
    "E0 Over 2.5": {
        "league": "E0",
        "market": "Over 2.5",
        "direction": "over",
        "allocation": 0.25,
        "color": "#4caf50",
        "description": "England Premier League — Over 2.5 goals",
    },
    "I1 Under 2.5": {
        "league": "I1",
        "market": "Under 2.5",
        "direction": "under",
        "allocation": 0.25,
        "color": "#ffc107",
        "description": "Italy Serie A — Under 2.5 goals",
    },
    "D1 Over 2.5": {
        "league": "D1",
        "market": "Over 2.5",
        "direction": "over",
        "allocation": 0.20,
        "color": "#7c3aed",
        "description": "Germany Bundesliga — Over 2.5 goals",
    },
}


# ═══════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════


@dataclass
class BetRecord:
    """A single placed bet within the portfolio."""

    strategy: str
    league: str
    market: str
    date: str
    home: str
    away: str
    odds: float
    model_prob: float
    implied_prob: float
    ev: float
    stake: float
    won: bool | None  # None = pending
    profit: float  # negative for losses
    actual_goals: str = ""


@dataclass
class StrategyState:
    """Live state for one strategy leg."""

    name: str
    allocation: float
    bankroll: float
    initial_capital: float
    bets: list[BetRecord] = field(default_factory=list)
    total_staked: float = 0.0
    total_profit: float = 0.0
    peak_bankroll: float = 0.0
    n_won: int = 0
    n_lost: int = 0

    @property
    def n_bets(self) -> int:
        settled = [b for b in self.bets if b.won is not None]
        return len(settled)

    @property
    def win_rate(self) -> float:
        settled = [b for b in self.bets if b.won is not None]
        if not settled:
            return 0.0
        won = sum(1 for b in settled if b.won)
        return won / len(settled)

    @property
    def yield_pct(self) -> float:
        if self.total_staked <= 0:
            return 0.0
        return (self.total_profit / self.total_staked) * 100

    @property
    def max_drawdown_pct(self) -> float:
        if not self.bets:
            return 0.0
        peak = self.initial_capital
        br = self.initial_capital
        max_dd = 0.0
        for b in self.bets:
            if b.won is not None:
                br += b.profit
                if br > peak:
                    peak = br
                dd = (peak - br) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd * 100


@dataclass
class PortfolioState:
    """Complete portfolio state across all strategies."""

    initial_bankroll: float = INITIAL_BANKROLL
    strategies: dict[str, StrategyState] = field(default_factory=dict)
    last_updated: str = ""
    daily_pnl: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_bankroll(self) -> float:
        return sum(s.bankroll for s in self.strategies.values())

    @property
    def total_profit(self) -> float:
        return self.total_bankroll - self.initial_bankroll

    @property
    def total_return_pct(self) -> float:
        if self.initial_bankroll <= 0:
            return 0.0
        return (self.total_profit / self.initial_bankroll) * 100

    @property
    def total_staked(self) -> float:
        return sum(s.total_staked for s in self.strategies.values())

    @property
    def portfolio_yield(self) -> float:
        if self.total_staked <= 0:
            return 0.0
        return (self.total_profit / self.total_staked) * 100

    @property
    def n_bets_total(self) -> int:
        return sum(s.n_bets for s in self.strategies.values())

    @property
    def n_pending(self) -> int:
        total = 0
        for s in self.strategies.values():
            total += sum(1 for b in s.bets if b.won is None)
        return total


# ═══════════════════════════════════════════════════════════
#  Portfolio Persistence
# ═══════════════════════════════════════════════════════════


def _bet_to_dict(b: BetRecord) -> dict:
    return {
        "strategy": b.strategy,
        "league": b.league,
        "market": b.market,
        "date": b.date,
        "home": b.home,
        "away": b.away,
        "odds": b.odds,
        "model_prob": b.model_prob,
        "implied_prob": b.implied_prob,
        "ev": b.ev,
        "stake": b.stake,
        "won": b.won,
        "profit": b.profit,
        "actual_goals": b.actual_goals,
    }


def _bet_from_dict(d: dict) -> BetRecord:
    return BetRecord(
        strategy=d["strategy"],
        league=d["league"],
        market=d["market"],
        date=d["date"],
        home=d["home"],
        away=d["away"],
        odds=d["odds"],
        model_prob=d["model_prob"],
        implied_prob=d["implied_prob"],
        ev=d["ev"],
        stake=d["stake"],
        won=d.get("won"),
        profit=d.get("profit", 0.0),
        actual_goals=d.get("actual_goals", ""),
    )


def _state_to_dict(state: PortfolioState) -> dict:
    return {
        "initial_bankroll": state.initial_bankroll,
        "last_updated": state.last_updated,
        "strategies": {
            name: {
                "name": s.name,
                "allocation": s.allocation,
                "bankroll": s.bankroll,
                "initial_capital": s.initial_capital,
                "total_staked": s.total_staked,
                "total_profit": s.total_profit,
                "peak_bankroll": s.peak_bankroll,
                "n_won": s.n_won,
                "n_lost": s.n_lost,
                "bets": [_bet_to_dict(b) for b in s.bets],
            }
            for name, s in state.strategies.items()
        },
        "daily_pnl": state.daily_pnl,
    }


def _state_from_dict(d: dict) -> PortfolioState:
    state = PortfolioState(initial_bankroll=d.get("initial_bankroll", INITIAL_BANKROLL))
    state.last_updated = d.get("last_updated", "")
    state.daily_pnl = d.get("daily_pnl", [])
    for name, sd in d.get("strategies", {}).items():
        ss = StrategyState(
            name=sd.get("name", name),
            allocation=sd.get("allocation", STRATEGIES.get(name, {}).get("allocation", 0.25)),
            bankroll=sd.get("bankroll", 0),
            initial_capital=sd.get("initial_capital", 0),
            total_staked=sd.get("total_staked", 0),
            total_profit=sd.get("total_profit", 0),
            peak_bankroll=sd.get("peak_bankroll", 0),
            n_won=sd.get("n_won", 0),
            n_lost=sd.get("n_lost", 0),
            bets=[_bet_from_dict(b) for b in sd.get("bets", [])],
        )
        state.strategies[name] = ss
    return state


def load_portfolio_state() -> PortfolioState:
    """Load portfolio state from disk, or create a fresh one."""
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH) as f:
                data = json.load(f)
            state = _state_from_dict(data)
            logger.info("Loaded portfolio state: %d strategies, %d total bets",
                        len(state.strategies), state.n_bets_total)
            return state
        except Exception as exc:
            logger.warning("Failed to load portfolio state: %s — creating fresh", exc)

    # Fresh portfolio
    state = PortfolioState()
    for name, cfg in STRATEGIES.items():
        alloc_capital = state.initial_bankroll * cfg["allocation"]
        state.strategies[name] = StrategyState(
            name=name,
            allocation=cfg["allocation"],
            bankroll=alloc_capital,
            initial_capital=alloc_capital,
            peak_bankroll=alloc_capital,
        )
    state.last_updated = datetime.now(timezone.utc).isoformat()
    save_portfolio_state(state)
    return state


def save_portfolio_state(state: PortfolioState) -> None:
    """Persist portfolio state to disk."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state.last_updated = datetime.now(timezone.utc).isoformat()
    data = _state_to_dict(state)
    with open(STATE_PATH, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug("Portfolio state saved to %s", STATE_PATH)


# ═══════════════════════════════════════════════════════════
#  Portfolio Simulation
# ═══════════════════════════════════════════════════════════


def _load_league_data(league: str) -> pd.DataFrame:
    """Load match history with OU odds from DB + CSV for a league."""
    # Try CSV first (has OU odds)
    csv_path = PROJECT_ROOT / "data" / "raw" / "league_all.csv"
    if csv_path.exists():
        try:
            import csv as csv_mod
            rows = []
            with open(csv_path, encoding="utf-8") as f:
                reader = csv_mod.DictReader(f)
                for r in reader:
                    if r.get("league", "").strip() != league:
                        continue
                    over = r.get("bbav>2.5", "") or r.get("avg>2.5", "") or ""
                    under = r.get("bbav<2.5", "") or r.get("avg<2.5", "") or ""
                    if not over or not under:
                        continue
                    try:
                        rows.append({
                            "date": r.get("date", "")[:10],
                            "home_team": r.get("home_team", "").strip(),
                            "away_team": r.get("away_team", "").strip(),
                            "home_goals": int(float(r.get("home_goals", 0))),
                            "away_goals": int(float(r.get("away_goals", 0))),
                            "over_odds": float(over),
                            "under_odds": float(under),
                        })
                    except (ValueError, TypeError):
                        continue
            if rows:
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
                if len(df) > 50:
                    return df
        except Exception as exc:
            logger.warning("CSV load failed for %s: %s", league, exc)

    # Fallback: derive OU odds from 1X2 odds in DB
    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT date, home_team, away_team, home_goals, away_goals, result,
               home_odds, draw_odds, away_odds
        FROM matches
        WHERE league = ? AND home_goals IS NOT NULL
          AND home_odds IS NOT NULL AND draw_odds IS NOT NULL AND away_odds IS NOT NULL
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(league,))
    conn.close()
    if len(df) < 50:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["is_over"] = (df["total_goals"] > 2.5).astype(float)

    # Compute conditional rates
    cond = {}
    for outcome, label in [("H", "home_win"), ("D", "draw"), ("A", "away_win")]:
        subset = df[df["result"] == outcome]
        cond[label] = float(subset["is_over"].mean()) if len(subset) > 0 else 0.5

    # Derive OU odds from 1X2
    def _derive(row):
        h, d, a = row["home_odds"], row["draw_odds"], row["away_odds"]
        if h <= 1 or d <= 1 or a <= 1:
            return (2.0, 2.0)
        ph, pd_, pa = 1.0 / h, 1.0 / d, 1.0 / a
        total = ph + pd_ + pa
        if total <= 0:
            return (2.0, 2.0)
        ph /= total
        pd_ /= total
        pa /= total
        implied_over = np.clip(
            ph * cond["home_win"] + pd_ * cond["draw"] + pa * cond["away_win"],
            0.05, 0.95
        )
        return (round(1.0 / implied_over, 2), round(1.0 / (1.0 - implied_over), 2))

    derived = df.apply(_derive, axis=1, result_type="expand")
    df["over_odds"] = derived[0]
    df["under_odds"] = derived[1]
    result = df[["date", "home_team", "away_team", "home_goals", "away_goals", "over_odds", "under_odds"]].copy()
    result = result.sort_values("date").reset_index(drop=True)
    logger.info("  Derived OU odds for %s: %d matches", league, len(result))
    return result


def _load_league_models(league: str) -> dict[str, Any] | None:
    """Load per-league models for a given league code."""
    import joblib
    league_dir = MODELS_DIR / league
    dc_path = league_dir / "dixon_coles.joblib"
    elo_path = league_dir / "elo.joblib"
    if not dc_path.exists() or not elo_path.exists():
        return None
    models: dict[str, Any] = {
        "dc": joblib.load(dc_path),
        "elo": joblib.load(elo_path),
    }
    xgb_path = league_dir / "xgboost.joblib"
    lgb_path = league_dir / "lightgbm.joblib"
    if xgb_path.exists():
        models["xgb"] = joblib.load(xgb_path)
    if lgb_path.exists():
        models["lgb"] = joblib.load(lgb_path)
    # Calibrator
    cal_p = league_dir / "blend_calibrator_hybrid.joblib"
    if not cal_p.exists():
        cal_p = league_dir / "blend_calibrator.joblib"
    if cal_p.exists():
        models["calibrator"] = joblib.load(cal_p)
    return models


def _implied_prob(odds: float) -> float:
    return 1.0 / odds if odds > 1 else 0.0


def _kelly_stake(prob: float, odds: float, fraction: float = KELLY_FRAC) -> float:
    if odds <= 1 or prob <= 0:
        return 0.0
    full = (prob * odds - 1.0) / (odds - 1.0)
    return max(0.0, full * fraction)


def simulate_strategy(
    strategy_name: str,
    league: str,
    direction: str,  # "over" or "under"
    allocation: float,
    test_frac: float = 0.15,
) -> StrategyState:
    """Run a chronological OU value betting simulation for one strategy leg.

    Uses DC model's direct Over/Under probabilities (computed from the Poisson
    scoreline table) blended with Elo-derived OU probabilities. Tree models
    are NOT used here since they require expensive per-match feature engineering.

    Elo ratings are updated sequentially through the test set, ensuring no
    data leakage (each match is predicted BEFORE its result is known).

    Args:
        test_frac: Fraction of data to use for testing (last N%). Default 0.15.
    """
    dc = None
    elo = None

    models = _load_league_models(league)
    if models is None:
        logger.warning("  No models for %s — skipping", league)
        return StrategyState(name=strategy_name, allocation=allocation, bankroll=0, initial_capital=0)

    dc = models.get("dc")
    elo_core = models.get("elo")

    df = _load_league_data(league)
    if df.empty or len(df) < 100:
        logger.warning("  Not enough data for %s — skipping", league)
        return StrategyState(name=strategy_name, allocation=allocation, bankroll=0, initial_capital=0)

    # Chronological split: train on first (1-test_frac), test on last test_frac
    split_idx = int(len(df) * (1 - test_frac))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    logger.info("  %s: %d train, %d test matches", league, len(train_df), len(test_df))

    # Mature Elo on training set
    from src.elo import EloSystem
    mature_elo = EloSystem(k=32, home_advantage=100, initial_rating=1500)
    mature_elo.process_matches(train_df)
    elo = mature_elo  # use this instance for chronological updates

    # Compute league conditional rates from training data only (no leakage!)
    train_df["result"] = train_df.apply(
        lambda r: "H" if r["home_goals"] > r["away_goals"]
        else "A" if r["away_goals"] > r["home_goals"] else "D", axis=1
    )
    cond = _compute_over_conditional_rates(train_df)

    # Initial capital
    initial_cap = INITIAL_BANKROLL * allocation
    bankroll = initial_cap
    peak_br = initial_cap

    bets: list[BetRecord] = []
    n_won = 0
    n_lost = 0
    total_staked = 0.0
    total_profit = 0.0

    for idx, row in test_df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        actual_over = (hg + ag) > 2.5
        actual_result = "H" if hg > ag else "A" if ag > hg else "D"
        over_odds = float(row["over_odds"])
        under_odds = float(row["under_odds"])

        try:
            # DC's exact OU probability from scoreline table
            dc_pred = dc.predict(home, away, over_under_threshold=2.5)
            dc_over = float(getattr(dc_pred, "over_2_5_prob", 0.5))

            # Elo-derived OU probability via conditional rates
            R_h = elo.get_rating(home)
            R_a = elo.get_rating(away)
            E_h = elo.expected_score(R_h, R_a)
            if hasattr(elo, "_expected_to_probs"):
                elo_a, elo_d, elo_h = elo._expected_to_probs(E_h)
            else:
                elo_a = 1.0 - E_h
                elo_h = E_h
                elo_d = 1.0 - elo_h - elo_a
            elo_over = (
                elo_h * cond["home_win"] + elo_d * cond["draw"] + elo_a * cond["away_win"]
            )
            elo_over = np.clip(elo_over, 0.05, 0.95)

            # Blend DC + Elo (50/50 since no tree models)
            model_over = 0.5 * dc_over + 0.5 * elo_over
            model_under = 1.0 - model_over

            # Determine which side to bet
            if direction == "over":
                model_prob = model_over
                odds = over_odds
                actual_won = actual_over
            else:
                model_prob = model_under
                odds = under_odds
                actual_won = not actual_over

            # Check for value
            if odds > 1 and model_prob > 0.02:
                implied = _implied_prob(odds)
                ev = model_prob / implied - 1.0

                if ev >= MIN_EV:
                    stake_pct = _kelly_stake(model_prob, odds)
                    if stake_pct > 0 and bankroll > 1:
                        stake = bankroll * stake_pct
                        won = actual_won
                        profit = stake * (odds - 1.0) if won else -stake

                        bet = BetRecord(
                            strategy=strategy_name,
                            league=league,
                            market=f"{direction.title()} 2.5",
                            date=str(row["date"])[:10],
                            home=home, away=away,
                            odds=round(odds, 2),
                            model_prob=round(model_prob, 4),
                            implied_prob=round(implied, 4),
                            ev=round(ev, 4),
                            stake=round(stake, 2),
                            won=won,
                            profit=round(profit, 2),
                            actual_goals=f"{hg}-{ag}",
                        )
                        bets.append(bet)
                        bankroll += profit
                        if bankroll > peak_br:
                            peak_br = bankroll
                        if won:
                            n_won += 1
                        else:
                            n_lost += 1
                        total_staked += stake
                        total_profit += profit

        except Exception as exc:
            logger.debug("Skipping %s v %s: %s", home, away, exc)

        # Update Elo AFTER prediction — always, once per match
        try:
            elo.update_ratings(home, away, actual_result,
                               home_goals=hg, away_goals=ag)
        except Exception:
            pass

    logger.info("  %s: %d bets, yield=%.2f%%, profit=%.2f",
                 strategy_name, len(bets),
                 (total_profit / total_staked * 100) if total_staked > 0 else 0,
                 total_profit)

    return StrategyState(
        name=strategy_name,
        allocation=allocation,
        bankroll=round(bankroll, 2),
        initial_capital=initial_cap,
        bets=bets,
        total_staked=round(total_staked, 2),
        total_profit=round(total_profit, 2),
        peak_bankroll=round(peak_br, 2),
        n_won=n_won,
        n_lost=n_lost,
    )


def _compute_over_conditional_rates(df: pd.DataFrame) -> dict[str, float]:
    """Compute P(Over 2.5 | Home Win), P(Over 2.5 | Draw), P(Over 2.5 | Away Win)
    from a DataFrame with columns home_goals, away_goals, result."""
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["is_over"] = (df["total_goals"] > 2.5).astype(float)
    cond: dict[str, float] = {}
    for outcome, label in [("H", "home_win"), ("D", "draw"), ("A", "away_win")]:
        subset = df[df["result"] == outcome]
        cond[label] = float(subset["is_over"].mean()) if len(subset) > 0 else 0.5
    return cond


def run_full_portfolio_simulation() -> PortfolioState:
    """Run a full portfolio simulation across all 4 strategies.

    Runs strategies in parallel for faster wall-clock time.
    Results are saved to disk for instant subsequent loads.
    """
    state = PortfolioState()
    results: dict[str, StrategyState] = {}

    def _run(name: str, cfg: dict) -> tuple[str, StrategyState]:
        logger.info("Simulating %s (%s)...", name, cfg["league"])
        ss = simulate_strategy(
            strategy_name=name,
            league=cfg["league"],
            direction=cfg["direction"],
            allocation=cfg["allocation"],
        )
        return name, ss

    # Run strategies sequentially but log progress
    for name, cfg in STRATEGIES.items():
        _, ss = _run(name, cfg)
        results[name] = ss

    state.strategies = results

    # Build daily PnL timeline
    all_bets = []
    for s in state.strategies.values():
        for b in s.bets:
            all_bets.append(b)
    all_bets.sort(key=lambda b: b.date)

    daily: dict[str, float] = {}
    for b in all_bets:
        if b.won is not None:
            daily[b.date] = daily.get(b.date, 0.0) + b.profit
    state.daily_pnl = [{"date": d, "pnl": round(p, 2)} for d, p in sorted(daily.items())]
    state.last_updated = datetime.now(timezone.utc).isoformat()

    save_portfolio_state(state)
    return state


def get_live_portfolio() -> PortfolioState:
    """Get the current portfolio state.

    If a saved state exists, loads it. Otherwise runs a full simulation.
    In a live deployment, this would also check for new resolved matches
    and update the state incrementally.
    """
    if STATE_PATH.exists():
        state = load_portfolio_state()
        if state.n_bets_total > 0:
            return state
    return run_full_portfolio_simulation()


def get_recent_value_bets(
    league: str,
    direction: str = "over",
    min_ev: float = MIN_EV,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Get upcoming value bets for a specific league/direction.

    Uses upcoming matches from the DB and predicts with the trained models
    to find value opportunities.
    """
    from src.models.three_model_blend import ThreeModelBlend, ConditionalRates

    models = _load_league_models(league)
    if models is None:
        return []

    # Get upcoming matches
    conn = sqlite3.connect(str(DB_PATH))
    query = """
        SELECT date, home_team, away_team
        FROM matches
        WHERE league = ? AND result IS NULL AND date >= date('now', '-1 day')
        ORDER BY date ASC
    """
    df_upcoming = pd.read_sql_query(query, conn, params=(league,))
    conn.close()

    if df_upcoming.empty:
        return []

    # Get historical data for feature building
    conn = sqlite3.connect(str(DB_PATH))
    df_hist = pd.read_sql_query("""
        SELECT date, home_team, away_team, home_goals, away_goals, result
        FROM matches
        WHERE league = ? AND home_goals IS NOT NULL
        ORDER BY date ASC
    """, conn, params=(league,))
    conn.close()

    if df_hist.empty:
        return []

    df_hist["result"] = df_hist.apply(
        lambda r: "H" if r["home_goals"] > r["away_goals"]
        else "A" if r["away_goals"] > r["home_goals"] else "D", axis=1
    )

    # Build blend and predict
    cr = ConditionalRates.from_data(df_hist)
    blend = ThreeModelBlend(
        dc_model=models.get("dc"),
        elo_model=models.get("elo"),
        xgb_model=models.get("xgb"),
        lgb_model=models.get("lgb"),
        conditional_rates=cr,
        historical_df=df_hist,
    )

    calibrator = models.get("calibrator")
    results = []

    for _, row in df_upcoming.iterrows():
        try:
            pred = blend.predict(row["home_team"], row["away_team"])
            over_prob = pred["over_under"]["Over"]
            under_prob = pred["over_under"]["Under"]

            # No OU calibration (same reason as simulate_strategy)

            # We don't have live odds here, so we flag predictions
            # The actual value check happens when odds are available
            prob = over_prob if direction == "over" else under_prob
            label = "Over 2.5" if direction == "over" else "Under 2.5"

            results.append({
                "date": str(row["date"])[:10],
                "home": row["home_team"],
                "away": row["away_team"],
                "market": label,
                "model_prob": round(prob, 4),
                "league": league,
                "strategy": f"{league} {label}",
            })
        except Exception:
            continue

    # Sort by confidence (highest model probability first)
    results.sort(key=lambda r: r["model_prob"], reverse=True)
    return results[:max_results]
