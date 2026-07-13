---
tags:
  - python-module
  - backtesting
  - betting
---

# `backtesting.py` — Backtesting Engine

**Path:** `src/backtesting.py`

Simulates value-betting strategies on historical test data. Walks through matches chronologically, places Kelly-sized stakes on positive-EV opportunities, tracks bankroll.

**Key class:** `BacktestEngine` — run, calculate_metrics, print_report, plot_results

**Metrics:** ROI, Yield, Win Rate, Max Drawdown, Profit Factor, Longest Streaks.

**Charts:** 4 publication-quality PNGs (bankroll curve, drawdown, cumulative profit, bet outcomes).

See also: [[value_betting.py]], [[config.py]], [[ensemble.py]], [[Value Betting & Backtesting]], [[Runtime Sequence Diagrams]]
