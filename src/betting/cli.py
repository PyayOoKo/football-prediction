"""
CLI — command-line interface for the betting engine.

Provides commands for evaluating bets, managing bankroll,
checking module status, and running the pipeline.

Usage
-----
::

    # Evaluate a set of matches
    python -m src.betting.cli evaluate --matches '[{"match_id":"m1","home_team":"Arsenal","away_team":"Chelsea","model_probs":[0.52,0.28,0.20],"odds":{"home_odds":2.10,"draw_odds":3.40,"away_odds":3.80}}]'

    # Show bankroll status
    python -m src.betting.cli bankroll

    # List registered modules
    python -m src.betting.cli modules

    # Run a pipeline from a JSON file
    python -m src.betting.cli run --file matches.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.betting.api import BettingAPI, EvaluateBetsRequest
from src.betting.factory import EngineFactory

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  CLI handler
# ═══════════════════════════════════════════════════════════


class BettingCLI:
    """CLI handler for the betting engine."""

    def __init__(self, api: BettingAPI | None = None) -> None:
        self.api = api or BettingAPI()
        self._parser = self._build_parser()

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="betting",
            description="Modular Betting Engine — evaluate, filter, and manage bets.",
        )
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # ── evaluate ─────────────────────────────────
        eval_parser = subparsers.add_parser("evaluate", help="Evaluate a set of matches")
        eval_parser.add_argument(
            "--matches", type=str, default="[]",
            help="JSON string of match dicts",
        )
        eval_parser.add_argument(
            "--staking", type=str, default="fractional_kelly",
            choices=["kelly", "fractional_kelly", "flat_stake"],
            help="Staking method",
        )
        eval_parser.add_argument(
            "--fraction", type=float, default=0.25,
            help="Fraction for fractional Kelly (default 0.25)",
        )
        eval_parser.add_argument(
            "--bankroll", type=float, default=1000.0,
            help="Initial bankroll (default 1000)",
        )
        eval_parser.add_argument(
            "--min-ev", type=float, default=0.0,
            help="Minimum EV threshold (default 0.0)",
        )
        eval_parser.add_argument(
            "--json", action="store_true",
            help="Output raw JSON",
        )
        eval_parser.add_argument(
            "--open", action="store_true",
            help="Open dashboard in browser (wip)",
        )

        # ── bankroll ─────────────────────────────────
        subparsers.add_parser("bankroll", help="Show bankroll status")

        # ── modules ──────────────────────────────────
        subparsers.add_parser("modules", help="List registered modules")

        # ── run ──────────────────────────────────────
        run_parser = subparsers.add_parser("run", help="Run pipeline from file")
        run_parser.add_argument(
            "--file", type=str, required=True,
            help="Path to JSON file with match definitions",
        )
        run_parser.add_argument(
            "--staking", type=str, default="fractional_kelly",
            choices=["kelly", "fractional_kelly", "flat_stake"],
        )

        return parser

    def run(self, args: list[str] | None = None) -> int:
        """Parse arguments and dispatch to the appropriate handler."""
        parsed = self._parser.parse_args(args)

        if parsed.command == "evaluate":
            return self._handle_evaluate(parsed)
        elif parsed.command == "bankroll":
            return self._handle_bankroll()
        elif parsed.command == "modules":
            return self._handle_modules()
        elif parsed.command == "run":
            return self._handle_run(parsed)
        else:
            self._parser.print_help()
            return 1

    # ── Command handlers ─────────────────────────────

    def _handle_evaluate(self, parsed: argparse.Namespace) -> int:
        try:
            matches = json.loads(parsed.matches)
        except json.JSONDecodeError as exc:
            print(f"[ERROR] Invalid --matches JSON: {exc}", file=sys.stderr)
            return 1

        request = EvaluateBetsRequest(
            matches=matches,
            staking_method=parsed.staking,
            staking_params={"fraction": parsed.fraction},
            initial_bankroll=parsed.bankroll,
            min_ev=parsed.min_ev,
        )
        response = self.api.evaluate_bets(request)

        if parsed.json:
            print(json.dumps(response, indent=2, default=str))
        else:
            self._pretty_print_evaluation(response)

        return 0 if response.get("status") == "ok" else 1

    def _handle_bankroll(self) -> int:
        status = self.api.get_bankroll_status()
        print("\n  BANKROLL STATUS")
        print("  " + "=" * 40)
        print(f"  Initial balance:  £{status['initial_balance']:.2f}")
        print(f"  Current balance:  £{status['current_balance']:.2f}")
        print(f"  Total staked:     £{status['total_staked']:.2f}")
        print(f"  Total profit:     £{status['total_profit']:+.2f}")
        print(f"  ROI:              {status['roi_pct']:+.2f}%")
        print(f"  Yield:            {status['yield_pct']:+.2f}%")
        print(f"  Win rate:         {status['win_rate_pct']:.1f}%")
        print(f"  Max drawdown:     {status['max_drawdown_pct']:.2f}%")
        print(f"  Total bets:       {status['total_bets']}")
        return 0

    def _handle_modules(self) -> int:
        modules = self.api.get_engine_config().get("registry", {})
        print("\n  REGISTERED MODULES")
        print("  " + "=" * 50)
        for role, names in modules.items():
            if names:
                print(f"  {role:<25s} {', '.join(names)}")
            else:
                print(f"  {role:<25s} (none)")
        return 0

    def _handle_run(self, parsed: argparse.Namespace) -> int:
        file_path = Path(parsed.file)
        if not file_path.exists():
            print(f"[ERROR] File not found: {file_path}", file=sys.stderr)
            return 1

        try:
            with open(file_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[ERROR] Failed to load file: {exc}", file=sys.stderr)
            return 1

        matches = data if isinstance(data, list) else data.get("matches", [])

        engine = EngineFactory.create(
            staking_method=parsed.staking,
        )
        report = engine.run_pipeline(matches, staking_method=parsed.staking)
        engine.print_summary()
        return 0

    # ── Pretty printing ──────────────────────────────

    def _pretty_print_evaluation(self, response: dict[str, Any]) -> None:
        status = response.get("status", "error")
        if status != "ok":
            print(f"\n  Evaluation failed: {response.get('errors', ['Unknown'])}")
            return

        bets = response.get("bets", [])
        report = response.get("report", {})

        if not bets:
            print("\n  No value bets found.")
            return

        print(f"\n  EVALUATION RESULTS ({len(bets)} bets)")
        print("  " + "=" * 80)
        for bet in bets:
            ev = bet.get("ev", 0)
            stake = bet.get("stake_amount", 0)
            rec = "✅" if bet.get("recommended") else " "
            print(
                f"  {rec} #{bet.get('rank', '?')}: "
                f"{bet.get('match', '?')} — {bet.get('outcome', '?')} "
                f"@{bet.get('decimal_odds', '?'):.2f} "
                f"(EV={ev:+.2%}, stake=£{stake:.2f})"
            )

        r = report
        print(f"\n  REPORT")
        print(f"  Total bets:   {r.get('total_bets', 0)}")
        print(f"  Positive EV:  {r.get('positive_ev_bets', 0)}")
        print(f"  Total staked: £{r.get('total_staked', 0):.2f}")
        print(f"  Total profit: £{r.get('total_profit', 0):+.2f}")
        print(f"  ROI:          {r.get('roi_pct', 0):+.2f}%")
        print(f"  Avg EV:       {r.get('avg_ev', 0):+.4f}")


# ═══════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    cli = BettingCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
