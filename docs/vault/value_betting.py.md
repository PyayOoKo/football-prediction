---
tags:
  - python-module
  - betting
  - value
  - ev
---

# `value_betting.py` — Value Bet Detection

**Path:** `src/value_betting.py`

Identifies betting opportunities where model probabilities exceed bookmaker-implied probabilities.

**Key function:** `compute_value_bets(odds, model_probs, ...)` — computes implied prob, margin, fair prob, EV, Kelly stake.

**Calculations:** IP = 1/odds, margin = ΣIP - 1, fair = IP/(1+margin), EV = (model×odds)-1, Kelly = EV/(odds-1)×fraction.

See also: [[backtesting.py]], [[config.py]], [[ensemble.py]], [[confidence_scoring.py]], [[Value Betting & Backtesting]], [[Runtime Sequence Diagrams]]
