"""
Risk Manager — comprehensive risk controls for betting operations.

Enforces four categories of risk limits, each independently togglable:

1. **Daily Loss** — stop betting if losses exceed X% of bankroll in a day
2. **Drawdown** — stop betting if bankroll falls X% from all-time peak
3. **Bet Frequency** — cap bets per hour / day / week
4. **Diversification** — limit exposure per league / team / match

All limits are defined in ``config/risk_management.yaml`` and can be
overridden at construction time.

Usage
-----
::

    from src.betting.risk_management import RiskManager

    rm = RiskManager()

    # Before placing a bet
    allowed, reason = rm.check_bet(slip, bankroll)
    if not allowed:
        print(f"Bet rejected: {reason}")

    # After settling a bet
    rm.record_result(slip, profit=50.0, won=True)

    # Reset for a new day
    rm.reset_daily()

    # Save / load state
    rm.save_state("risk_state.json")
    rm.load_state("risk_state.json")
"""

from __future__ import annotations

import datetime
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.betting.models import Bankroll, BetSlip

logger = logging.getLogger(__name__)

# ── Default config path ─────────────────────────────────
_DEFAULT_CONFIG_PATH = Path("config/risk_management.yaml")


# ═══════════════════════════════════════════════════════════
#  Data classes for internal state
# ═══════════════════════════════════════════════════════════


@dataclass
class _DailyState:
    """Tracks mutable daily counters (reset each day)."""
    date: str = ""                          # ISO date string "2026-07-15"
    starting_bankroll: float = 0.0          # Bankroll at start of day
    daily_loss: float = 0.0                 # Cumulative loss in currency
    daily_profit: float = 0.0               # Cumulative profit in currency
    bets_today: int = 0                     # Number of bets placed today
    drawdown_breached: bool = False         # Drawdown limit was hit today
    consecutive_losses: int = 0             # Current consecutive loss streak


@dataclass
class _FrequencyState:
    """Rolling-window counters for frequency limits."""
    bet_timestamps: deque[datetime.datetime] = field(
        default_factory=lambda: deque(maxlen=200),
    )
    # Weekly: tracked by ISO week number
    bets_this_week: int = 0
    week_number: int = 0


@dataclass
class _ExposureState:
    """Tracks current open-bet exposure."""
    open_bets: list[dict[str, Any]] = field(default_factory=list)
    total_staked_open: float = 0.0
    exposure_by_league: dict[str, float] = field(default_factory=dict)
    exposure_by_team: dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════
#  Risk Manager
# ═══════════════════════════════════════════════════════════


class RiskManager:
    """Comprehensive risk management for betting operations.

    Parameters
    ----------
    config_path : str | Path, optional
        Path to the YAML config file.  Defaults to
        ``config/risk_management.yaml``.
    config_override : dict, optional
        Inline config overrides.  Takes precedence over the file.
        Structure matches the YAML hierarchy.
    auto_load : bool
        If True, load config from *config_path* on ``__init__``
        (default True).
    """

    # ── Constants for state file ─────────────────────────
    _STATE_VERSION = 1

    def __init__(
        self,
        config_path: str | Path | None = None,
        config_override: dict[str, Any] | None = None,
        auto_load: bool = True,
    ) -> None:
        # Resolve config path
        self._config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

        # Internal state
        self._config: dict[str, Any] = self._default_config()
        self._daily = _DailyState()
        self._freq = _FrequencyState()
        self._exposure = _ExposureState()
        self._peak_bankroll: float = 0.0
        self._cooldown_until: datetime.datetime | None = None

        # Load config
        if auto_load:
            self.load_config(self._config_path, config_override)

    # ── Properties (read-only view of current state) ─────

    @property
    def config(self) -> dict[str, Any]:
        """Current risk configuration (read-only)."""
        return dict(self._config)

    @property
    def daily_loss_pct(self) -> float:
        """Current daily loss as % of starting bankroll (0 if no starting bankroll)."""
        if self._daily.starting_bankroll <= 0:
            return 0.0
        return (self._daily.daily_loss / self._daily.starting_bankroll) * 100

    @property
    def drawdown_pct(self, bankroll: Bankroll | None = None) -> float:
        """Current drawdown from peak."""
        peak = bankroll.peak_balance if bankroll else self._peak_bankroll
        current = (
            bankroll.current_balance if bankroll
            else self._daily.starting_bankroll
        )
        if peak <= 0:
            return 0.0
        return max(0.0, (peak - (current or 0)) / peak * 100)

    @property
    def bets_today(self) -> int:
        """Number of bets placed today."""
        return self._daily.bets_today

    @property
    def bets_this_week(self) -> int:
        """Number of bets placed this week."""
        return self._freq.bets_this_week

    @property
    def open_bets_count(self) -> int:
        """Number of currently open (unsettled) bets."""
        return len(self._exposure.open_bets)

    @property
    def total_exposure(self) -> float:
        """Total currency staked on open bets."""
        return self._exposure.total_staked_open

    @property
    def consecutive_losses(self) -> int:
        """Current consecutive loss streak."""
        return self._daily.consecutive_losses

    # ── Config management ────────────────────────────────

    @staticmethod
    def _default_config() -> dict[str, Any]:
        """Return the default configuration as a nested dict."""
        return {
            "risk_manager": {
                "enabled": True,
                "strict_mode": False,
                "daily_loss": {
                    "enabled": True,
                    "max_loss_pct": 15.0,
                    "reset_at_midnight": True,
                    "max_loss_absolute": None,
                },
                "drawdown": {
                    "enabled": True,
                    "max_drawdown_pct": 25.0,
                    "cooldown_on_breach": True,
                    "cooldown_bets": 3,
                },
                "consecutive_losses": {
                    "enabled": True,
                    "max_consecutive": 6,
                    "cooldown_bets": 2,
                },
                "frequency": {
                    "enabled": True,
                    "max_per_day": 10,
                    "max_per_week": 40,
                    "max_per_hour": 3,
                    "cooldown_minutes": 30,
                },
                "stake": {
                    "enabled": True,
                    "max_single_pct": 25.0,
                    "max_total_exposure_pct": 50.0,
                    "min_odds": 1.50,
                    "max_odds": 20.0,
                },
                "diversification": {
                    "enabled": True,
                    "max_per_league": {
                        "pct": 40.0,
                        "absolute": None,
                    },
                    "max_per_team": {
                        "pct": 25.0,
                        "absolute": None,
                    },
                    "max_bets_per_league": 5,
                    "max_bets_per_team": 2,
                    "excluded_leagues": [],
                    "excluded_teams": [],
                    "preferred_leagues": [],
                },
                "exposure": {
                    "enabled": True,
                    "track_open_bets": True,
                    "max_open_bets": 15,
                    "max_open_bets_per_match": 1,
                },
            }
        }

    def load_config(
        self,
        config_path: str | Path | None = None,
        config_override: dict[str, Any] | None = None,
    ) -> None:
        """Load risk parameters from a YAML file (with optional overrides).

        Parameters
        ----------
        config_path : str | Path, optional
            Path to the YAML config file.  Defaults to the path set
            at construction.
        config_override : dict, optional
            Inline overrides.  Merged on top of the file config.
        """
        # Deferred import to avoid crash if PyYAML is not installed
        try:
            import yaml
        except ImportError:
            logger.warning(
                "PyYAML not installed — risk config file will not be loaded. "
                "Install with: pip install pyyaml"
            )
            file_config = {}
            return

        path = Path(config_path) if config_path else self._config_path

        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    file_config = yaml.safe_load(f) or {}
                logger.info("Loaded risk config from %s", path)
                merged = self._default_config()
                self._deep_merge(merged, file_config)
                # Merge overrides on top of file config
                if config_override:
                    self._deep_merge(merged, config_override)
                self._config = merged
            except Exception as exc:
                logger.warning(
                    "Failed to load risk config from %s: %s — using defaults",
                    path, exc,
                )
                self._config = self._default_config()
                if config_override:
                    self._deep_merge(self._config, config_override)
        else:
            logger.info(
                "Risk config %s not found — using defaults",
                path,
            )
            self._config = self._default_config()
            if config_override:
                self._deep_merge(self._config, config_override)
        logger.debug("Risk config loaded: %s", self._summarise_config())

    def save_config(self, path: str | Path | None = None) -> str:
        """Save the current configuration to a YAML file.

        Parameters
        ----------
        path : str | Path, optional
            Output path.  Defaults to the path set at construction.

        Returns
        -------
        str
            Path to the saved file.
        """
        try:
            import yaml
        except ImportError:
            logger.warning(
                "PyYAML not installed — cannot save config. "
                "Install with: pip install pyyaml"
            )
            return ""

        out = Path(path) if path else self._config_path
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        logger.info("Risk config saved to %s", out)
        return str(out)

    @staticmethod
    def _deep_merge(base: dict, overrides: dict) -> None:
        """Recursively merge *overrides* into *base* (mutates base)."""
        for key, value in overrides.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                RiskManager._deep_merge(base[key], value)
            else:
                base[key] = value

    def _summarise_config(self) -> str:
        """Return a short summary of the active config."""
        rm = self._config.get("risk_manager", {})
        parts = []
        dl = rm.get("daily_loss", {})
        if dl.get("enabled"):
            parts.append(f"daily_loss<{dl.get('max_loss_pct')}%")
        dd = rm.get("drawdown", {})
        if dd.get("enabled"):
            parts.append(f"drawdown<{dd.get('max_drawdown_pct')}%")
        fr = rm.get("frequency", {})
        if fr.get("enabled"):
            parts.append(f"freq<{fr.get('max_per_day')}/d,{fr.get('max_per_week')}/w")
        dv = rm.get("diversification", {})
        if dv.get("enabled"):
            parts.append("div_on")
        return " | ".join(parts)

    # ── State persistence ────────────────────────────────

    def save_state(self, path: str | Path | None = None) -> str:
        """Save the current risk manager state to a JSON file.

        Persists daily counters, frequency history, and exposure
        tracking so the manager can resume across restarts.

        Parameters
        ----------
        path : str | Path, optional
            Output path (default ``reports/risk_state.json``).

        Returns
        -------
        str
            Path to the saved file.
        """
        state: dict[str, Any] = {
            "version": self._STATE_VERSION,
            "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "daily": {
                "date": self._daily.date,
                "starting_bankroll": self._daily.starting_bankroll,
                "daily_loss": self._daily.daily_loss,
                "daily_profit": self._daily.daily_profit,
                "bets_today": self._daily.bets_today,
                "drawdown_breached": self._daily.drawdown_breached,
                "consecutive_losses": self._daily.consecutive_losses,
            },
            "frequency": {
                "bet_timestamps": [
                    t.isoformat() for t in self._freq.bet_timestamps
                ],
                "bets_this_week": self._freq.bets_this_week,
                "week_number": self._freq.week_number,
            },
            "exposure": {
                "open_bets": self._exposure.open_bets,
                "total_staked_open": self._exposure.total_staked_open,
                "exposure_by_league": dict(self._exposure.exposure_by_league),
                "exposure_by_team": dict(self._exposure.exposure_by_team),
            },
            "peak_bankroll": self._peak_bankroll,
        }

        out = Path(path) if path else Path("reports") / "risk_state.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info("Risk state saved to %s", out)
        return str(out)

    def load_state(self, path: str | Path | None = None) -> bool:
        """Load a previously saved risk manager state.

        Parameters
        ----------
        path : str | Path
            Path to the JSON state file.

        Returns
        -------
        bool
            True if the state was loaded successfully.
        """
        p = Path(path) if path else Path("reports") / "risk_state.json"
        if not p.exists():
            logger.warning("Risk state file not found: %s", p)
            return False

        try:
            with open(p, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as exc:
            logger.warning("Failed to load risk state: %s", exc)
            return False

        version = state.get("version", 0)
        if version != self._STATE_VERSION:
            logger.warning(
                "Risk state version mismatch: expected %d, got %d — ignoring",
                self._STATE_VERSION, version,
            )
            return False

        # Restore daily state
        d = state.get("daily", {})
        self._daily.date = d.get("date", "")
        self._daily.starting_bankroll = d.get("starting_bankroll", 0.0)
        self._daily.daily_loss = d.get("daily_loss", 0.0)
        self._daily.daily_profit = d.get("daily_profit", 0.0)
        self._daily.bets_today = d.get("bets_today", 0)
        self._daily.drawdown_breached = d.get("drawdown_breached", False)
        self._daily.consecutive_losses = d.get("consecutive_losses", 0)

        # Restore frequency state
        f = state.get("frequency", {})
        timestamps = [
            datetime.datetime.fromisoformat(t)
            for t in f.get("bet_timestamps", [])
        ]
        self._freq.bet_timestamps = deque(timestamps, maxlen=200)
        self._freq.bets_this_week = f.get("bets_this_week", 0)
        self._freq.week_number = f.get("week_number", 0)

        # Restore exposure state
        e = state.get("exposure", {})
        self._exposure.open_bets = e.get("open_bets", [])
        self._exposure.total_staked_open = e.get("total_staked_open", 0.0)
        self._exposure.exposure_by_league = defaultdict(
            float, e.get("exposure_by_league", {}),
        )
        self._exposure.exposure_by_team = defaultdict(
            float, e.get("exposure_by_team", {}),
        )

        self._peak_bankroll = state.get("peak_bankroll", 0.0)

        logger.info("Risk state loaded from %s", path)
        return True

    # ── Core check methods ──────────────────────────────

    def check_bet(
        self,
        slip: BetSlip,
        bankroll: Bankroll,
        **kwargs: Any,
    ) -> tuple[bool, str]:
        """Check if a bet is allowed under all active risk rules.

        Parameters
        ----------
        slip : BetSlip
            The proposed bet.
        bankroll : Bankroll
            Current bankroll state.
        **kwargs
            Additional context (e.g. ``league``, ``team``).

        Returns
        -------
        tuple[bool, str]
            ``(allowed, reason)`` — reason is empty if allowed.
        """
        if not self._is_enabled():
            return True, ""

        # Cooldown check
        allowed, reason = self._check_cooldown()
        if not allowed:
            return False, reason

        # Daily loss check
        allowed, reason = self._check_daily_loss(bankroll)
        if not allowed:
            return False, reason

        # Drawdown check
        allowed, reason = self._check_drawdown(bankroll)
        if not allowed:
            return False, reason

        # Consecutive losses check
        allowed, reason = self._check_consecutive_losses()
        if not allowed:
            return False, reason

        # Frequency checks
        allowed, reason = self._check_frequency()
        if not allowed:
            return False, reason

        # Stake checks
        allowed, reason = self._check_stake(slip, bankroll)
        if not allowed:
            return False, reason

        # Diversification checks
        allowed, reason = self._check_diversification(slip, bankroll, kwargs)
        if not allowed:
            return False, reason

        # Exposure checks
        allowed, reason = self._check_exposure(slip)
        if not allowed:
            return False, reason

        return True, ""

    def check_batch(
        self,
        slips: list[BetSlip],
        bankroll: Bankroll,
        **kwargs: Any,
    ) -> dict[str, tuple[bool, str]]:
        """Check multiple bets.  Returns ``{bet_id: (allowed, reason)}``."""
        league = kwargs.pop("league", None)
        team = kwargs.pop("team", None)

        results: dict[str, tuple[bool, str]] = {}
        for s in slips:
            ctx = {**kwargs}
            if league:
                ctx["league"] = league
            if team:
                ctx["team"] = team
            results[s.bet_id] = self.check_bet(s, bankroll, **ctx)
        return results

    def _is_enabled(self) -> bool:
        """Check if risk management is enabled at all."""
        return self._config.get("risk_manager", {}).get("enabled", True)

    # ── Individual check implementations ─────────────────

    def _check_cooldown(self) -> tuple[bool, str]:
        """Check if we're in a cooldown period after a limit breach."""
        if self._cooldown_until is not None:
            now = datetime.datetime.now(datetime.timezone.utc)
            if now < self._cooldown_until:
                remaining = (self._cooldown_until - now).total_seconds()
                return False, (
                    f"Cooldown active — {remaining:.0f}s remaining "
                    f"(until {self._cooldown_until.strftime('%H:%M:%S')})"
                )
            # Cooldown expired
            self._cooldown_until = None
        return True, ""

    def _check_daily_loss(self, bankroll: Bankroll) -> tuple[bool, str]:
        """Check if daily loss limit has been reached."""
        cfg = self._config.get("risk_manager", {}).get("daily_loss", {})
        if not cfg.get("enabled", True):
            return True, ""

        # Ensure we have a starting bankroll for the day
        self._initialise_daily_state(bankroll)

        # Check % of starting bankroll
        max_loss_pct = cfg.get("max_loss_pct", 15.0)
        loss_pct = self.daily_loss_pct
        if loss_pct >= max_loss_pct:
            return False, (
                f"Daily loss limit reached: {loss_pct:.1f}% "
                f"(max {max_loss_pct:.1f}%)"
            )

        # Check absolute loss cap
        max_abs = cfg.get("max_loss_absolute")
        if max_abs is not None and self._daily.daily_loss >= max_abs:
            return False, (
                f"Daily loss limit reached: "
                f"{self._daily.daily_loss:.2f} (max {max_abs:.2f})"
            )

        return True, ""

    def _check_drawdown(self, bankroll: Bankroll) -> tuple[bool, str]:
        """Check if drawdown limit has been exceeded."""
        cfg = self._config.get("risk_manager", {}).get("drawdown", {})
        if not cfg.get("enabled", True):
            return True, ""

        # Track peak
        if bankroll.peak_balance and bankroll.peak_balance > self._peak_bankroll:
            self._peak_bankroll = bankroll.peak_balance

        dd_pct = self.drawdown_pct(bankroll)
        max_dd = cfg.get("max_drawdown_pct", 25.0)

        if dd_pct >= max_dd:
            # Activate cooldown if configured
            if cfg.get("cooldown_on_breach", True):
                cooldown = cfg.get("cooldown_bets", 3)
                # We don't know when the next N bets happen, so set
                # a time-based cooldown: estimate ~5 min per bet
                self._set_cooldown(minutes=cooldown * 5)

            return False, (
                f"Drawdown limit exceeded: {dd_pct:.1f}% "
                f"(max {max_dd:.1f}%)"
            )

        return True, ""

    def _check_consecutive_losses(self) -> tuple[bool, str]:
        """Check if consecutive loss limit has been reached."""
        cfg = self._config.get("risk_manager", {}).get("consecutive_losses", {})
        if not cfg.get("enabled", True):
            return True, ""

        max_consecutive = cfg.get("max_consecutive", 6)
        if self._daily.consecutive_losses >= max_consecutive:
            cooldown = cfg.get("cooldown_bets", 2)
            self._set_cooldown(minutes=cooldown * 5)

            return False, (
                f"Consecutive loss limit reached: "
                f"{self._daily.consecutive_losses} losses "
                f"(max {max_consecutive})"
            )

        return True, ""

    def _check_frequency(self) -> tuple[bool, str]:
        """Check bet frequency limits (per hour, day, week)."""
        cfg = self._config.get("risk_manager", {}).get("frequency", {})
        if not cfg.get("enabled", True):
            return True, ""

        now = datetime.datetime.now(datetime.timezone.utc)

        # Per hour
        max_per_hour = cfg.get("max_per_hour", 3)
        hour_ago = now - datetime.timedelta(hours=1)
        bets_last_hour = sum(
            1 for t in self._freq.bet_timestamps if t > hour_ago
        )
        if bets_last_hour >= max_per_hour:
            cooldown_min = cfg.get("cooldown_minutes", 30)
            self._set_cooldown(minutes=cooldown_min)
            return False, (
                f"Hourly bet limit reached: {bets_last_hour} bets "
                f"in last hour (max {max_per_hour})"
            )

        # Per day
        max_per_day = cfg.get("max_per_day", 10)
        if self._daily.bets_today >= max_per_day:
            return False, (
                f"Daily bet limit reached: {self._daily.bets_today} "
                f"bets today (max {max_per_day})"
            )

        # Per week
        max_per_week = cfg.get("max_per_week", 40)
        current_week = now.isocalendar()[1]
        if current_week != self._freq.week_number:
            self._freq.bets_this_week = 0
            self._freq.week_number = current_week
        if self._freq.bets_this_week >= max_per_week:
            return False, (
                f"Weekly bet limit reached: {self._freq.bets_this_week} "
                f"bets this week (max {max_per_week})"
            )

        return True, ""

    def _check_stake(
        self, slip: BetSlip, bankroll: Bankroll,
    ) -> tuple[bool, str]:
        """Check stake size and odds limits."""
        cfg = self._config.get("risk_manager", {}).get("stake", {})
        if not cfg.get("enabled", True):
            return True, ""

        current = bankroll.current_balance or 0

        # Max single stake %
        max_single_pct = cfg.get("max_single_pct", 25.0)
        if slip.stake_pct is not None:
            stake_pct = slip.stake_pct * 100  # Convert to percentage
            if stake_pct > max_single_pct:
                return False, (
                    f"Stake {stake_pct:.1f}% exceeds max single "
                    f"stake {max_single_pct:.1f}%"
                )

        # Max total exposure %
        max_exposure_pct = cfg.get("max_total_exposure_pct", 50.0)
        if slip.stake_amount is not None:
            total_exposure = (
                self._exposure.total_staked_open + slip.stake_amount
            )
            exposure_pct = (total_exposure / current * 100) if current > 0 else 0
            if exposure_pct > max_exposure_pct:
                return False, (
                    f"Total exposure {exposure_pct:.1f}% exceeds "
                    f"max {max_exposure_pct:.1f}%"
                )

        # Min odds
        min_odds = cfg.get("min_odds", 1.50)
        if float(slip.decimal_odds) < min_odds:
            return False, (
                f"Odds {float(slip.decimal_odds):.2f} below "
                f"minimum {min_odds:.2f}"
            )

        # Max odds
        max_odds = cfg.get("max_odds", 20.0)
        if float(slip.decimal_odds) > max_odds:
            return False, (
                f"Odds {float(slip.decimal_odds):.2f} above "
                f"maximum {max_odds:.2f}"
            )

        return True, ""

    def _check_diversification(
        self,
        slip: BetSlip,
        bankroll: Bankroll,
        context: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check diversification limits per league and team."""
        cfg = self._config.get("risk_manager", {}).get("diversification", {})
        if not cfg.get("enabled", True):
            return True, ""

        league = context.get("league", "")
        team = context.get("team", "")
        current = bankroll.current_balance or 0

        # Excluded leagues
        excluded_leagues = cfg.get("excluded_leagues", [])
        if league and league in excluded_leagues:
            return False, f"League '{league}' is excluded"

        # Excluded teams
        excluded_teams = cfg.get("excluded_teams", [])
        if team and team in excluded_teams:
            return False, f"Team '{team}' is excluded"

        # Preferred leagues (if set, only these are allowed)
        preferred = cfg.get("preferred_leagues", [])
        if preferred and league and league not in preferred:
            return False, (
                f"League '{league}' not in preferred leagues: {preferred}"
            )

        # Max per league
        max_per_league_cfg = cfg.get("max_per_league", {})
        max_league_pct = max_per_league_cfg.get("pct", 40.0)
        max_league_abs = max_per_league_cfg.get("absolute")

        if league:
            league_exposure = self._exposure.exposure_by_league.get(league, 0.0)
            if slip.stake_amount:
                league_exposure += slip.stake_amount
            if current > 0:
                league_exposure_pct = (league_exposure / current) * 100
                if league_exposure_pct > max_league_pct:
                    return False, (
                        f"League '{league}' exposure {league_exposure_pct:.1f}% "
                        f"exceeds max {max_league_pct:.1f}%"
                    )
            if max_league_abs is not None and league_exposure > max_league_abs:
                return False, (
                    f"League '{league}' exposure {league_exposure:.2f} "
                    f"exceeds max absolute {max_league_abs:.2f}"
                )

        # Max per team
        max_per_team_cfg = cfg.get("max_per_team", {})
        max_team_pct = max_per_team_cfg.get("pct", 25.0)
        max_team_abs = max_per_team_cfg.get("absolute")

        if team:
            team_exposure = self._exposure.exposure_by_team.get(team, 0.0)
            if slip.stake_amount:
                team_exposure += slip.stake_amount
            if current > 0:
                team_exposure_pct = (team_exposure / current) * 100
                if team_exposure_pct > max_team_pct:
                    return False, (
                        f"Team '{team}' exposure {team_exposure_pct:.1f}% "
                        f"exceeds max {max_team_pct:.1f}%"
                    )
            if max_team_abs is not None and team_exposure > max_team_abs:
                return False, (
                    f"Team '{team}' exposure {team_exposure:.2f} "
                    f"exceeds max absolute {max_team_abs:.2f}"
                )

        # Max bets per league
        max_bets_league = cfg.get("max_bets_per_league", 5)
        if league:
            league_bets = sum(
                1 for ob in self._exposure.open_bets
                if ob.get("league") == league
            )
            if league_bets >= max_bets_league:
                return False, (
                    f"Max bets per league reached: {league_bets} "
                    f"(max {max_bets_league}) for '{league}'"
                )

        # Max bets per team
        max_bets_team = cfg.get("max_bets_per_team", 2)
        if team:
            team_bets = sum(
                1 for ob in self._exposure.open_bets
                if ob.get("team") == team
            )
            if team_bets >= max_bets_team:
                return False, (
                    f"Max bets per team reached: {team_bets} "
                    f"(max {max_bets_team}) for '{team}'"
                )

        return True, ""

    def _check_exposure(self, slip: BetSlip) -> tuple[bool, str]:
        """Check open-bet exposure limits."""
        cfg = self._config.get("risk_manager", {}).get("exposure", {})
        if not cfg.get("enabled", True):
            return True, ""

        max_open = cfg.get("max_open_bets", 15)
        if len(self._exposure.open_bets) >= max_open:
            return False, (
                f"Max open bets reached: {len(self._exposure.open_bets)} "
                f"(max {max_open})"
            )

        max_per_match = cfg.get("max_open_bets_per_match", 1)
        match_bets = sum(
            1 for ob in self._exposure.open_bets
            if ob.get("match_id") == slip.match_id
        )
        if match_bets >= max_per_match:
            return False, (
                f"Max bets per match reached: {match_bets} "
                f"(max {max_per_match}) for {slip.match_label}"
            )

        return True, ""

    # ── Result recording ─────────────────────────────────

    def record_result(
        self,
        slip: BetSlip,
        profit: float,
        won: bool,
        **kwargs: Any,
    ) -> None:
        """Record the result of a settled bet.

        Updates daily counters, consecutive loss streaks, and
        exposure tracking.

        Parameters
        ----------
        slip : BetSlip
            The settled bet slip.
        profit : float
            Profit (positive) or loss (negative) amount.
        won : bool
            Whether the bet won.
        **kwargs
            Additional context (e.g. ``league``, ``team``) for
            exposure tracking.
        """
        # Ensure daily state is initialised (in case no check_bet was called today)
        bk = kwargs.get("bankroll")
        if bk is not None:
            self._initialise_daily_state(bk)
            # Track peak bankroll
            if bk.peak_balance and bk.peak_balance > self._peak_bankroll:
                self._peak_bankroll = bk.peak_balance

        # Update daily counters
        self._daily.bets_today += 1
        if profit >= 0:
            self._daily.daily_profit += profit
            self._daily.consecutive_losses = 0
        else:
            self._daily.daily_loss += abs(profit)
            self._daily.consecutive_losses += 1

        # Record timestamp for frequency tracking
        self._freq.bet_timestamps.append(
            datetime.datetime.now(datetime.timezone.utc),
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        current_week = now.isocalendar()[1]
        if current_week != self._freq.week_number:
            self._freq.bets_this_week = 0
            self._freq.week_number = current_week
        self._freq.bets_this_week += 1

        # Remove from open bets if tracked
        self._remove_open_bet(slip.bet_id)

        # Also reset drawdown breach flag on win
        if won:
            self._daily.drawdown_breached = False

    def record_bet_placed(
        self,
        slip: BetSlip,
        **kwargs: Any,
    ) -> None:
        """Record that a bet was placed (tracks open exposure).

        Call this AFTER the bet is placed and confirmed.

        Parameters
        ----------
        slip : BetSlip
            The placed bet slip.
        **kwargs
            Additional context (e.g. ``league``, ``team``).
        """
        if not self._is_enabled():
            return

        league = kwargs.get("league", "")
        team = kwargs.get("team", "")

        open_entry: dict[str, Any] = {
            "bet_id": slip.bet_id,
            "match_id": slip.match_id,
            "stake": slip.stake_amount or 0.0,
            "team": team,
            "league": league,
            "placed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self._exposure.open_bets.append(open_entry)

        stake = slip.stake_amount or 0.0
        self._exposure.total_staked_open += stake

        if league:
            self._exposure.exposure_by_league[league] = (
                self._exposure.exposure_by_league.get(league, 0.0) + stake
            )
        if team:
            self._exposure.exposure_by_team[team] = (
                self._exposure.exposure_by_team.get(team, 0.0) + stake
            )

    # ── State management ─────────────────────────────────

    def reset_daily(self) -> None:
        """Reset all daily counters (for new trading day)."""
        self._daily = _DailyState()
        # Keep the peak bankroll — drawdown is from all-time peak
        logger.debug("Daily risk counters reset")

    def reset_weekly(self) -> None:
        """Reset weekly counters."""
        self._freq.bets_this_week = 0
        self._freq.week_number = 0
        logger.debug("Weekly risk counters reset")

    def reset_all(self) -> None:
        """Reset ALL counters and state (full reset)."""
        self._daily = _DailyState()
        self._freq = _FrequencyState()
        self._exposure = _ExposureState()
        self._peak_bankroll = 0.0
        self._cooldown_until = None
        logger.debug("All risk counters reset")

    # ── Internal helpers ─────────────────────────────────    def _initialise_daily_state(self, bankroll: Bankroll) -> None:
        """Ensure daily counters are initialised for today."""
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        if self._daily.date == today:
            return

        # Reset daily counters
        self._daily.date = today
        self._daily.starting_bankroll = bankroll.current_balance or 0
        self._daily.daily_loss = 0.0
        self._daily.daily_profit = 0.0
        self._daily.bets_today = 0
        # Don't reset drawdown breach — it's all-time
        # Don't reset consecutive losses — it's all-time

        cfg = self._config.get("risk_manager", {}).get("daily_loss", {})
        if cfg.get("reset_at_midnight", True):
            self._daily.drawdown_breached = False
            logger.debug(
                "RiskManager: new day %s — starting_bankroll=%.2f",
                today, self._daily.starting_bankroll,
            )

    def _set_cooldown(self, minutes: int = 30) -> None:
        """Set a cooldown period during which no bets are allowed."""
        self._cooldown_until = (
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(minutes=minutes)
        )
        logger.info(
            "RiskManager: cooldown activated for %d minutes (until %s)",
            minutes,
            self._cooldown_until.strftime("%H:%M:%S UTC"),
        )

    def _remove_open_bet(self, bet_id: str) -> None:
        """Remove a settled/cancelled bet from open exposure tracking."""
        for i, ob in enumerate(self._exposure.open_bets):
            if ob.get("bet_id") == bet_id:
                stake = ob.get("stake", 0.0)
                self._exposure.total_staked_open -= stake

                league = ob.get("league", "")
                if league:
                    self._exposure.exposure_by_league[league] -= stake
                    if self._exposure.exposure_by_league[league] <= 0:
                        del self._exposure.exposure_by_league[league]

                team = ob.get("team", "")
                if team:
                    self._exposure.exposure_by_team[team] -= stake
                    if self._exposure.exposure_by_team[team] <= 0:
                        del self._exposure.exposure_by_team[team]

                self._exposure.open_bets.pop(i)
                break

    def __repr__(self) -> str:
        return (
            f"RiskManager("
            f"dd={self._config.get('risk_manager',{}).get('drawdown',{}).get('max_drawdown_pct','?')}%, "
            f"dl={self.daily_loss_pct:.1f}%, "
            f"bets_today={self._daily.bets_today}, "
            f"open={len(self._exposure.open_bets)}, "
            f"consec_loss={self._daily.consecutive_losses})"
        )
