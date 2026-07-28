"""CLI entry point for live predictions."""

import logging
import sys

from src.live_predictions.engine import (
    DEFAULT_POLL_INTERVAL,
    DEFAULT_SPORT_KEY,
    LivePredictionEngine,
)
from src.live_predictions.pipeline import live_value_bets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Live Prediction System")
    parser.add_argument(
        "--mode",
        type=str,
        default="oneshot",
        choices=["oneshot", "continuous", "value-bets"],
        help="Execution mode (default: oneshot)",
    )
    parser.add_argument("--sport", type=str, default=DEFAULT_SPORT_KEY)
    parser.add_argument("--interval", type=int, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--min-ev", type=float, default=0.0)
    parser.add_argument("--bookmaker", type=str, default=None)

    args = parser.parse_args()

    if args.mode == "value-bets":
        bets = live_value_bets(min_ev=args.min_ev, sport_key=args.sport)
        if bets:
            print(f"\n  Found {len(bets)} value bets:\n")
            for b in bets:
                print(
                    f"    {b['home_team']:20} vs {b['away_team']:20}  "
                    f"\u2192 {b['outcome']:5} at {b['decimal_odds']:.2f}  "
                    f"(EV: {b['ev']:+.1%})"
                )
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
        print(f"\n  Cycle complete \u2014 {len(predictions)} matches processed.\n")
    elif args.mode == "continuous":
        engine.run_continuous(max_cycles=args.cycles)


if __name__ == "__main__":
    main()
