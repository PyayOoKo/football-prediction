"""Test the new src/backtesting/metrics.py module."""
import sys
sys.path.insert(0, '.')
from src.backtesting.metrics import (
    MetricsCalculator, BetResult, MetricsResult,
    quick_metrics, metrics_from_dicts
)


def test_basic_metrics():
    """Test basic metrics computation with mixed win/loss."""
    bets = [
        BetResult(stake=50, profit=55, odds=2.10, won=True, bankroll_before=1000, closing_odds=2.05),
        BetResult(stake=25, profit=-25, odds=1.80, won=False, bankroll_before=1055, closing_odds=1.85),
    ]
    calc = MetricsCalculator(initial_bankroll=1000)
    m = calc.compute(bets)

    # Manual verification
    assert m.total_bets == 2, f"Expected 2 bets, got {m.total_bets}"
    assert m.winning_bets == 1
    assert m.losing_bets == 1
    assert m.pushed_bets == 0
    assert m.total_profit == 30.0, f"Expected 30.0, got {m.total_profit}"
    assert abs(m.roi - 30.0/75.0) < 0.0001, f"ROI {m.roi}"
    assert abs(m.yield_per_bet - 15.0) < 0.0001, f"Yield {m.yield_per_bet}"
    assert abs(m.win_rate - 0.5) < 0.0001, f"Win rate {m.win_rate}"
    assert abs(m.profit_factor - (50+55)/(75)) < 0.0001, f"PF {m.profit_factor}"
    assert m.max_drawdown_pct >= 0
    print(f"  PASS: basic_metrics (bets={m.total_bets}, roi={m.roi:.4f})")


def test_no_negative_sortino():
    """Test Sortino capped at 999 when no negative returns."""
    bets = [
        BetResult(stake=50, profit=25, odds=1.50, won=True, bankroll_before=1000),
        BetResult(stake=50, profit=30, odds=1.60, won=True, bankroll_before=1025),
    ]
    calc = MetricsCalculator(initial_bankroll=1000)
    m = calc.compute(bets)
    assert m.sortino_ratio == 999.0, f"Expected 999.0, got {m.sortino_ratio}"
    print(f"  PASS: no_negative_sortino (ratio={m.sortino_ratio})")


def test_missing_odds_raises():
    """Test that missing odds in from_dicts raises ValueError."""
    try:
        metrics_from_dicts([{'stake': 50, 'profit': 55, 'won': True}])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'odds' in str(e)
        print(f"  PASS: missing_odds_raises ({e})")


def test_empty_bets():
    """Test empty bet list returns default metrics."""
    calc = MetricsCalculator(initial_bankroll=1000)
    m = calc.compute([])
    assert m.total_bets == 0
    assert m.total_profit == 0.0
    assert m.profit_factor == 1.0  # default
    print(f"  PASS: empty_bets (pf={m.profit_factor})")


def test_from_dicts():
    """Test metrics_from_dicts convenience function."""
    dicts = [
        {'stake': 50, 'profit': 55, 'odds': 2.10, 'won': True,
         'bankroll_before': 1000, 'closing_odds': 2.05, 'market': '1X2'},
    ]
    m = metrics_from_dicts(dicts, 1000)
    assert m.total_bets == 1
    assert abs(m.roi - 55.0/50.0) < 0.0001
    print(f"  PASS: from_dicts (bets={m.total_bets}, roi={m.roi:.4f})")


def test_save_report():
    """Test save_report exports JSON."""
    bets = [BetResult(stake=50, profit=25, odds=1.50, won=True, bankroll_before=1000)]
    calc = MetricsCalculator(initial_bankroll=1000)
    path = calc.save_report(bets, model_name='test_unit')
    assert path.endswith('.json')
    import os
    assert os.path.exists(path)
    os.remove(path)  # cleanup
    print(f"  PASS: save_report ({path})")


def test_quick_metrics():
    """Test quick_metrics one-liner."""
    bets = [BetResult(stake=50, profit=25, odds=1.50, won=True, bankroll_before=1000)]
    m = quick_metrics(bets, 1000)
    assert m.total_bets == 1
    print(f"  PASS: quick_metrics (bets={m.total_bets})")


def test_pushed_bets():
    """Test pushed bets are excluded from metrics."""
    bets = [
        BetResult(stake=50, profit=55, odds=2.10, won=True, bankroll_before=1000),
        BetResult(stake=25, profit=0, odds=2.00, won=False, pushed=True, bankroll_before=1055),
    ]
    calc = MetricsCalculator(initial_bankroll=1000)
    m = calc.compute(bets)
    assert m.total_bets == 1  # pushed excluded
    assert m.pushed_bets == 1
    assert m.total_profit == 55.0
    print(f"  PASS: pushed_bets (active={m.total_bets}, pushed={m.pushed_bets}, profit={m.total_profit})")


# Run all tests
if __name__ == '__main__':
    print('=== TESTING src/backtesting/metrics.py ===')
    print()
    test_basic_metrics()
    test_no_negative_sortino()
    test_missing_odds_raises()
    test_empty_bets()
    test_from_dicts()
    test_save_report()
    test_quick_metrics()
    test_pushed_bets()
    print()
    print('=== ALL 8 TESTS PASSED ===')
