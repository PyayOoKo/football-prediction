---
tags:
  - python-module
  - betting
  - features
  - odds
  - market
---

# `betting_market.py` — Betting Market Transformer

**Path:** `src/feature_framework/features/betting_market.py`

Computes 33 betting market features per match: opening/closing odds, implied probability, odds movement, CLV, market consensus, favourite/underdog status, volatility, and bookmaker margin.

**Key class:** `BettingMarketTransformer` — 33 output columns, 9 auto-detected bookmakers, multiplicative margin removal

**Factory:** `create_betting_market_transformer()`

**50 tests** in `tests/test_feature_framework/test_betting_market.py`

See also: [[Betting Market Features]], [[Feature Orchestrator]], [[Feature Validation Framework]]
