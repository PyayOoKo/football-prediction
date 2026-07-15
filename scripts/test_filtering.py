"""
Quick test for src/betting/filtering.py
"""
from src.betting.filtering import BetFilter, filter_passed_only, count_passing

print("=== BET FILTER TESTS ===")
print()

# ── Sample bets ──
bets = [
    {
        "match": "Arsenal vs Chelsea",
        "outcome": "Home Win",
        "market": "1X2",
        "model_prob": 0.52,
        "decimal_odds": 2.10,
        "ev": 0.092,
        "bankroll_pct": 0.03,
    },
    {
        "match": "Liverpool vs United",
        "outcome": "Draw",
        "market": "1X2",
        "model_prob": 0.28,
        "decimal_odds": 3.40,
        "ev": -0.048,
        "bankroll_pct": 0.02,
    },
    {
        "match": "City vs Spurs",
        "outcome": "Away Win",
        "market": "1X2",
        "model_prob": 0.35,
        "decimal_odds": 4.50,
        "ev": 0.575,
        "bankroll_pct": 0.08,
    },
    {
        "match": "Barcelona vs Madrid",
        "outcome": "BTTS Yes",
        "market": "BTTS",
        "model_prob": 0.62,
        "decimal_odds": 1.80,
        "ev": 0.116,
        "bankroll_pct": 0.04,
    },
    {
        "match": "Bayern vs Dortmund",
        "outcome": "Over 2.5",
        "market": "Double Chance",
        "model_prob": 0.48,
        "decimal_odds": 2.20,
        "ev": 0.056,
        "bankroll_pct": 0.02,
    },
    {
        "match": "PSG vs Lyon",
        "outcome": "Home Win",
        "market": "1X2",
        "model_prob": 0.68,
        "decimal_odds": 1.40,
        "ev": -0.048,
        "bankroll_pct": 0.03,
    },
    {
        "match": "Milan vs Inter",
        "outcome": "Home Win",
        "market": "1X2",
               "model_prob": None,
        "decimal_odds": 2.50,
        "ev": None,
    },
]

# ── 1. Default filter (EV > 0, min_confidence=0.3, min_odds=1.5) ──
print("1. Default filter (min_ev=0.0, min_conf=0.3, min_odds=1.5)...")
f = BetFilter(min_ev=0.0, min_confidence=0.3, min_odds=1.5)
passed, rejected = f.filter_bets(bets)
print(f"   Passed: {len(passed)}, Rejected: {len(rejected)}")
assert len(passed) <= len(bets)
for b in rejected:
    print(f"   REJECTED: {b['match']:<28s} -> {b['_rejection']}")

# ── 2. Strict filter (EV > 0.05) ──
print()
print("2. Strict filter (min_ev=0.05)...")
f2 = BetFilter(min_ev=0.05, min_confidence=0.3, min_odds=1.5)
p2, r2 = f2.filter_bets(bets)
print(f"   Passed: {len(p2)}, Rejected: {len(r2)}")
assert len(p2) < len(passed)  # should be stricter
for b in r2:
    if "EV too low" in b["_rejection"]:
        print(f"   REJECTED: {b['match']:<28s} -> {b['_rejection']}")

# ── 3. High confidence filter ──
print()
print("3. High confidence filter (min_confidence=0.50)...")
f3 = BetFilter(min_ev=0.0, min_confidence=0.50, min_odds=1.5)
p3, r3 = f3.filter_bets(bets)
print(f"   Passed: {len(p3)}, Rejected: {len(r3)}")
for b in p3:
    assert b.get("model_prob", 0) >= 0.50 or b.get("model_prob") is None, \
        f"Passed bet had prob {b.get('model_prob')}"

# ── 4. Market filter ──
print()
print("4. Market filter (markets=['1X2', 'BTTS'])...")
f4 = BetFilter(min_ev=-1, min_confidence=0, min_odds=1.0,
               markets=("1X2", "BTTS"))
p4, r4 = f4.filter_bets(bets)
print(f"   Passed: {len(p4)}, Rejected: {len(r4)}")
for b in p4:
    assert b["market"] in ("1X2", "BTTS")
for b in r4:
    if "Market" in b["_rejection"]:
        print(f"   REJECTED: {b['match']:<28s} market={b['market']!r}")

# ── 5. Max stake filter ──
print()
print("5. Max stake filter (max_stake=0.05)...")
f5 = BetFilter(min_ev=-1, min_confidence=0, min_odds=1.0,
               max_stake=0.05)
p5, r5 = f5.filter_bets(bets)
print(f"   Passed: {len(p5)}, Rejected: {len(r5)}")
stake_rejected = [b for b in r5 if "Stake too high" in b["_rejection"]]
for b in stake_rejected:
    print(f"   REJECTED: {b['match']:<28s} stake={b['bankroll_pct']:.1%}")

# ── 6. Missing data ──
print()
print("6. Missing data (reject_on_missing=True)...")
f6 = BetFilter(min_ev=0.0, min_confidence=0.3, min_odds=1.5,
               reject_on_missing=True)
p6, r6 = f6.filter_bets(bets)
print(f"   Passed: {len(p6)}, Rejected: {len(r6)}")
missing_rejected = [b for b in r6 if "missing" in b["_rejection"]]
for b in missing_rejected:
    print(f"   REJECTED: {b['match']:<28s} -> {b['_rejection']}")

# ── 7. Missing data allowed ──
print()
print("7. Missing data allowed (reject_on_missing=False)...")
f7 = BetFilter(min_ev=0.0, min_confidence=0.3, min_odds=1.5,
               reject_on_missing=False)
p7, r7 = f7.filter_bets(bets)
print(f"   Passed: {len(p7)}, Rejected: {len(r7)}")

# ── 8. filter_passed_only shortcut ──
print()
print("8. filter_passed_only shortcut...")
p8 = filter_passed_only(bets, min_ev=0.0, min_confidence=0.3, min_odds=1.5)
assert all("_rejection" not in b for b in p8)
print(f"   Passed: {len(p8)}, all clean (no _rejection key)")

# ── 9. count_passing shortcut ──
print()
print("9. count_passing shortcut...")
n = count_passing(bets, min_ev=0.0, min_confidence=0.3, min_odds=1.5)
assert n == len(p8)
print(f"   Count: {n}")

# ── 10. Rejection summary ──
print()
print("10. Rejection summary...")
summary = f.rejection_summary
print(f"    Reasons: {summary}")
assert isinstance(summary, dict)

# ── 11. Edge case: empty list ──
print()
print("11. Edge case: empty bet list...")
p10, r10 = f.filter_bets([])
assert len(p10) == 0 and len(r10) == 0
print("    Passed: 0, Rejected: 0")

# ── 12. Negative EV filter ──
print()
print("12. Negative EV filter (min_ev=0.05)...")
neg_bets = [
    {"match": "Test", "market": "1X2", "model_prob": 0.60,
     "decimal_odds": 1.50, "ev": -0.1},
]
f12 = BetFilter(min_ev=0.05, min_confidence=0, min_odds=1.0)
p12, r12 = f12.filter_bets(neg_bets)
assert len(p12) == 0
print(f"    Negative EV bet correctly rejected: {r12[0]['_rejection']}")

# ── 13. EV auto-computed from prob/odds ──
print()
print("13. EV auto-computed from model_prob + decimal_odds...")
no_ev_bets = [
    {"match": "Auto EV", "market": "1X2", "model_prob": 0.55,
     "decimal_odds": 2.00},
]
f13 = BetFilter(min_ev=0.05, min_confidence=0, min_odds=1.0)
p13, r13 = f13.filter_bets(no_ev_bets)
# EV = 0.55*2.0 - 1 = 0.10 >= 0.05 -> should pass
assert len(p13) == 1
print(f"    Auto-computed EV: bet passed ({len(p13)} passed)")

print()
print("=== ALL TESTS PASSED ===")
