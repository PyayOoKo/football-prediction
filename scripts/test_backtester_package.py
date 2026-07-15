"""Test the new src/backtesting/ package structure."""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

print('=== 1. OLD IMPORTS (__init__.py) ===')
from src.backtesting import BacktestEngine, BacktestMetrics, BetRecord, run_backtest, get_backtest_guide, Backtester
print('  OK: all 6 symbols imported from src.backtesting')

print()
print('=== 2. NEW IMPORTS (backtester.py) ===')
from src.backtesting.backtester import Backtester as BT2, ExtendedBacktestMetrics, BacktestBetRecord
print('  OK: all 3 symbols imported from src.backtesting.backtester')

print()
print('=== 3. OLD BACKTESTENGINE ===')
X = pd.DataFrame({'a': [1, 2, 3]})
y = pd.Series([0, 1, 2])
model = DummyClassifier(strategy='uniform', random_state=42)
model.fit(X, y)
e = BacktestEngine(model, initial_bankroll=1000)
m = e.run(X, y)
print(f'  BacktestEngine: bets={m.total_bets}, bankroll={m.final_bankroll:.2f}')

print()
print('=== 4. NEW BACKTESTER (model + multi-market + void) ===')
class FakeModel:
    def predict_proba(self, X):
        n = len(X)
        return np.array([[0.2, 0.3, 0.5]] * n)

df = pd.DataFrame({
    'home_team': ['TeamA', 'TeamB'],
    'away_team': ['TeamC', 'TeamD'],
    'home_goals': [2, 0],
    'away_goals': [1, 1],
    'feat': [1.0, 2.0],
    'BbAvH': [5.0, 2.2],
    'BbAvD': [3.5, 3.3],
    'BbAvA': [1.8, 3.5],
    'btts_prob': [0.7, 0.4],
    'BbBTTS': [1.8, 2.0],
    'void': [False, True],
})

bt = BT2(FakeModel(), initial_bankroll=1000)
m2 = bt.run(
    df,
    feature_cols=['feat'],
    odds_mapping={
        'home_win': 'BbAvH', 'draw': 'BbAvD', 'away_win': 'BbAvA',
        'btts_yes': 'BbBTTS',
    },
    prob_mapping={'btts_yes': 'btts_prob'},
    void_col='void',
)
print(f'  Bets: {m2.total_bets}, Pushed: {m2.pushed_bets}')
print(f'  P&L: {m2.total_profit:.2f}, ROI: {m2.roi_pct:.2f}%')
print(f'  Markets: {m2.bets_per_market}')
print(f'  Sharpe: {m2.sharpe_ratio:.2f}, Sortino: {m2.sortino_ratio:.2f}')

print()
print('=== 5. SAVE RESULTS ===')
path = bt.save_results(model_name='test_package')
print(f'  Saved to: {path}')

print()
print('=== 6. COMPATIBILITY WRAPPER ===')
bm = bt.to_backtest_metrics()
print(f'  BaseMetrics: bets={bm.total_bets}, roi={bm.roi_pct:.2f}%')

print()
print('=== 7. PRINT REPORT ===')
bt.print_report()

print()
print('=== ALL TESTS PASSED ===')
