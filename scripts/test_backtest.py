"""
Quick test for src/betting/backtest.py
"""
import sys
sys.path.insert(0, ".")

from src.betting.backtest import Backtester
from src.betting.staking import StakingFactory
from src.betting.filtering import BetFilter

print("=== BACKTESTER FINAL VERIFICATION ===")
print()

# Generate 50 mixed bets
bets = []
for i in range(50):
    won = i % 3 != 0  # ~67% win rate
    probs = [0.50, 0.50, 0.55, 0.60, 0.45][i % 5]
    odds = [2.00, 2.10, 1.80, 1.70, 2.20][i % 5]
    bets.append({
        "match": f"Match {i+1}",
        "outcome": "Home Win",
        "market": "1X2",
        "model_prob": probs,
        "decimal_odds": odds,
        "actual_result": won,
        "closing_odds": odds + 0.05,
    })

# 1. Default
print("1. Default (FracKelly 25%, def filter)...")
bt = Backtester(initial_bankroll=1000.0)
r = bt.run(bets)
print(f"   Bets: {r.total_bets}/{len(bets)}")
print(f"   P&L:  \u00a3{r.total_profit:.2f}")
print(f"   ROI:  {r.roi_pct:.2f}%")
print(f"   Yield:{r.yield_pct:.2f}%")
print(f"   WinR: {r.win_rate_pct:.1f}%")
print(f"   MaxDD:{r.max_drawdown_pct:.2f}%")
print(f"   Sharp:{r.sharpe_ratio:.2f}")
print(f"   CLV:  {r.avg_clv:.4f}")
print(f"   PF:   {r.profit_factor:.2f}")
print(f"   Str:  {r.longest_win_streak}W/{r.longest_lose_streak}L")
assert r.total_bets > 0

# 2. Full Kelly
print("2. Full Kelly...")
bt2 = Backtester(1000, StakingFactory.create("kelly"),
                 BetFilter(min_ev=0, min_confidence=0.001, min_odds=1.5))
r2 = bt2.run(bets)
print(f"   Bets: {r2.total_bets}, P&L: \u00a3{r2.total_profit:.2f}")
assert r2.total_bets > 0

# 3. Flat
print("3. Flat \u00a325...")
bt3 = Backtester(1000, StakingFactory.create("flat", stake_per_bet=25),
                 BetFilter(min_ev=0, min_confidence=0.001, min_odds=1))
r3 = bt3.run(bets)
print(f"   Bets: {r3.total_bets}, P&L: \u00a3{r3.total_profit:.2f}")
assert r3.total_bets > 0

# 4. Strict filter
print("4. Strict filter (min_ev=0.05, min_conf=0.5)...")
bt4 = Backtester(1000, bet_filter=BetFilter(min_ev=0.05, min_confidence=0.5, min_odds=1.5))
r4 = bt4.run(bets)
print(f"   Bets: {r4.total_bets} (filtered down)")
assert r4.total_bets <= r.total_bets

# 5. Empty
print("5. Empty bets...")
bt5 = Backtester()
r5 = bt5.run([])
assert r5.total_bets == 0 and r5.final_bankroll == 1000.0
print(f"   Bets: 0, bankroll: \u00a3{r5.final_bankroll:.2f}")

# 6. All rejected
print("6. All rejected...")
bt6 = Backtester(bet_filter=BetFilter(min_confidence=0.99))
r6 = bt6.run([{"model_prob": 0.3, "decimal_odds": 3.0, "actual_result": True}])
assert r6.total_bets == 0
print("   Bets: 0")

# 7. Save
print("7. Save results...")
saved = bt.save_results("reports/backtest", "test_model")
print(f"   Saved: {saved}")

# 8. Max bets per match
print("8. Max 1 bet/match...")
bt8 = Backtester(1000, bet_filter=BetFilter(min_ev=0, min_confidence=0.001, min_odds=1), max_bets_per_match=1)
r8 = bt8.run([
    {"match": "A vs B", "model_prob": 0.6, "decimal_odds": 2.0, "actual_result": True, "market": "1X2", "outcome": "H"},
    {"match": "A vs B", "model_prob": 0.6, "decimal_odds": 2.0, "actual_result": True, "market": "1X2", "outcome": "H"},
])
assert r8.total_bets == 1
print(f"   Bets: {r8.total_bets}")

# 9. Print report
print("9. Print report (summary)...")
bt9 = Backtester()
bt9.run(bets[:5])
bt9.print_report()

print()
print("=== ALL TESTS PASSED ===")
