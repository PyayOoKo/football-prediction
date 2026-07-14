"""
analyze_england_norway.py — Comprehensive Analysis: Norway vs England (Quarter-final).

Self-contained script — does not require importing the full src package.
Loads historical World Cup data, computes Poisson predictions, and compares
with the existing XGBoost/ensemble prediction.

Usage:
    python analyze_england_norway.py
"""

from __future__ import annotations

import csv
import sys
from math import exp, factorial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


# ── Poisson helpers ───────────────────────────────────────

def _poisson(k: int, lam: float) -> float:
    """Poisson PMF: P(X=k) = e^{-lam} * lam^k / k!"""
    if lam == 0.0:
        return 1.0 if k == 0 else 0.0
    return exp(-lam) * (lam ** k) / factorial(k)


def load_completed_matches(csv_path: Path) -> list[dict]:
    """Load completed World Cup matches (home_goals, away_goals present)."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    return [r for r in rows if r.get("home_goals") and r["home_goals"].strip()]


def compute_team_stats(completed: list[dict]):
    """Compute league averages and per-team attack/defense stats."""
    h_goals = [float(r["home_goals"]) for r in completed]
    a_goals = [float(r["away_goals"]) for r in completed]
    n = len(completed)
    mu_h = sum(h_goals) / n
    mu_a = sum(a_goals) / n
    mu_all = (mu_h + mu_a) / 2

    scored, conceded, played = {}, {}, {}
    for r in completed:
        h, a = r["home_team"], r["away_team"]
        hg, ag = float(r["home_goals"]), float(r["away_goals"])
        scored[h] = scored.get(h, 0) + hg
        conceded[h] = conceded.get(h, 0) + ag
        played[h] = played.get(h, 0) + 1
        scored[a] = scored.get(a, 0) + ag
        conceded[a] = conceded.get(a, 0) + hg
        played[a] = played.get(a, 0) + 1

    return mu_h, mu_a, mu_all, scored, conceded, played


def team_strength(team: str, scored: dict, conceded: dict, played: dict, mu_all: float):
    """Return (attack_strength, defense_strength) for a team."""
    if team not in played or mu_all == 0:
        return (1.0, 1.0)
    n = played[team]
    gs = scored[team] / n
    gc = conceded[team] / n
    return (gs / mu_all, gc / mu_all)


def poisson_predict(home: str, away: str, mu_h: float, mu_a: float,
                    scored: dict, conceded: dict, played: dict, mu_all: float,
                    max_goals: int = 6):
    """Full Poisson prediction for home vs away."""
    att_home, def_home = team_strength(home, scored, conceded, played, mu_all)
    att_away, def_away = team_strength(away, scored, conceded, played, mu_all)

    lam_h = mu_h * att_home * def_away
    lam_a = mu_a * att_away * def_home

    # Build scoreline table
    hw = dr = aw = 0.0
    probs = []
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = _poisson(i, lam_h) * _poisson(j, lam_a)
            if i > j:
                hw += p
            elif i == j:
                dr += p
            else:
                aw += p
            probs.append((f"{i}-{j}", i, j, p))

    # Normalise
    total = hw + dr + aw
    hw_n, dr_n, aw_n = hw / total, dr / total, aw / total

    # Over/Under
    over_25 = sum(p for _, i, j, p in probs if i + j > 2.5)
    under_25 = 1.0 - over_25
    over_15 = sum(p for _, i, j, p in probs if i + j > 1.5)
    under_15 = 1.0 - over_15

    # BTTS
    p_h0 = _poisson(0, lam_h)
    p_a0 = _poisson(0, lam_a)
    btts = 1.0 - p_h0 - p_a0 + (p_h0 * p_a0)

    # Top scorelines
    probs_sorted = sorted(probs, key=lambda x: -x[3])
    top_12 = [(s, p / total) for s, _, _, p in probs_sorted[:12]]

    return {
        "home_team": home,
        "away_team": away,
        "att_home": att_home,
        "def_home": def_home,
        "att_away": att_away,
        "def_away": def_away,
        "expected_home_goals": lam_h,
        "expected_away_goals": lam_a,
        "expected_total": lam_h + lam_a,
        "home_win_prob": round(hw_n, 4),
        "draw_prob": round(dr_n, 4),
        "away_win_prob": round(aw_n, 4),
        "over_2_5_prob": round(over_25 / total, 4),
        "under_2_5_prob": round(under_25 / total, 4),
        "over_1_5_prob": round(over_15 / total, 4),
        "btts_prob": round(btts, 4),
        "btts_no_prob": round(1.0 - btts, 4),
        "top_scorelines": top_12,
    }


def load_ensemble_prediction() -> dict | None:
    """Load the existing ensemble/XGBoost prediction for Norway vs England."""
    csv_path = PROJECT_ROOT / "reports" / "predictions_worldcup" / "worldcup_predictions.csv"
    if not csv_path.exists():
        return None

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    for r in rows:
        if r["home_team"] == "Norway" and r["away_team"] == "England":
            return {
                "home_team": "Norway",
                "away_team": "England",
                "home_win_prob": float(r["home_win_prob"]),
                "draw_prob": float(r["draw_prob"]),
                "away_win_prob": float(r["away_win_prob"]),
                "confidence": float(r["confidence"]),
            }
    return None


def list_team_matches(completed: list[dict], team: str) -> list[dict]:
    """List all completed matches for a given team, sorted by date."""
    matches = []
    for r in completed:
        if r["home_team"] == team or r["away_team"] == team:
            hg = int(float(r["home_goals"]))
            ag = int(float(r["away_goals"]))
            outcome = "W" if (team == r["home_team"] and hg > ag) or \
                             (team == r["away_team"] and ag > hg) else \
                      "D" if hg == ag else "L"
            matches.append({
                "date": r.get("date", "")[:10],
                "season": r.get("season", ""),
                "round": r.get("round", ""),
                "home": r["home_team"],
                "away": r["away_team"],
                "hg": hg,
                "ag": ag,
                "ground": r.get("ground", ""),
                "outcome": outcome,
            })
    matches.sort(key=lambda m: m["date"])
    return matches


def print_header(title: str, char: str = "#", width: int = 76) -> None:
    """Print a section header compatible with Windows cp1252 encoding."""
    line = char * width
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


# ═══════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════

def main() -> int:
    csv_path = PROJECT_ROOT / "data" / "raw" / "worldcup_all.csv"
    if not csv_path.exists():
        print(f"[X] Data file not found: {csv_path}")
        return 1

    # ── 1. Load data ──
    print_header("2026 WORLD CUP QUARTER-FINAL ANALYSIS")
    print("  Norway vs England")
    print("  Miami (Miami Gardens), July 11, 2026 - Neutral Venue")

    print("\n  Loading World Cup data ...")
    completed = load_completed_matches(csv_path)
    print(f"  {len(completed)} completed matches loaded")

    mu_h, mu_a, mu_all, scored, conceded, played = compute_team_stats(completed)
    print(f"  League averages: u_home={mu_h:.3f}, u_away={mu_a:.3f}, u_overall={mu_all:.3f}")

    # ── 2. Team profiles ──
    print_header("TEAM PROFILES")

    for team in ["England", "Norway"]:
        matches = list_team_matches(completed, team)
        att, defense = team_strength(team, scored, conceded, played, mu_all)
        gs = (scored.get(team, 0) / played.get(team, 1)) if team in played else 0
        gc = (conceded.get(team, 0) / played.get(team, 1)) if team in played else 0
        wins = sum(1 for m in matches if m["outcome"] == "W")
        draws = sum(1 for m in matches if m["outcome"] == "D")
        losses = sum(1 for m in matches if m["outcome"] == "L")
        pct = wins / len(matches) * 100 if matches else 0

        print(f"\n  {team} ({len(matches)} World Cup matches)")
        print(f"  {'Record:':<20} {wins}W {draws}D {losses}L ({pct:.0f}% win rate)")
        print(f"  {'Goals For/Game:':<20} {gs:.3f}")
        print(f"  {'Goals Against/Game:':<20} {gc:.3f}")
        a_str = 'strong' if att > 1.1 else ('avg' if att > 0.9 else 'weak')
        d_str = 'strong' if defense < 0.9 else ('avg' if defense < 1.1 else 'weak')
        print(f"  {'Attack (a):':<20} {att:.3f} ({a_str})")
        print(f"  {'Defense (b):':<20} {defense:.3f} ({d_str})")
        print(f"  {'Overall Rating (a/b):':<20} {att/defense:.2f}")

        print(f"\n  {team}'s 2026 World Cup run:")
        for m in [mm for mm in matches if mm["season"] == "2026"]:
            icon = "[W]" if m["outcome"] == "W" else ("[D]" if m["outcome"] == "D" else "[L]")
            print(f"    {icon} {m['date']} [{m['round']:16s}] {m['home']} {m['hg']}-{m['ag']} {m['away']}")

    # ── 3. Poisson Prediction ──
    print_header("POISSON MODEL PREDICTION")

    # Note: neutral venue -- Norway is nominally home (from the bracket)
    pred = poisson_predict("Norway", "England", mu_h, mu_a, scored, conceded, played, mu_all)

    print(f"  Match: {pred['home_team']} vs {pred['away_team']}")
    print(f"  Expected goals: Norway {pred['expected_home_goals']:.3f} -- England {pred['expected_away_goals']:.3f}")
    print(f"  Expected total: {pred['expected_total']:.3f}")
    print()
    print("  Team strengths in this matchup:")
    print(f"    Norway  -> attack a={pred['att_home']:.3f} x England defense b={pred['def_away']:.3f}")
    print(f"    England -> attack a={pred['att_away']:.3f} x Norway defense b={pred['def_home']:.3f}")
    print()
    print("  Outcome probabilities:")
    print(f"    Norway win:  {pred['home_win_prob']:.1%}")
    print(f"    Draw:        {pred['draw_prob']:.1%}")
    print(f"    England win: {pred['away_win_prob']:.1%}")
    print()
    print("  Goal markets:")
    print(f"    Over 2.5 goals:  {pred['over_2_5_prob']:.1%}")
    print(f"    Under 2.5 goals: {pred['under_2_5_prob']:.1%}")
    print(f"    Over 1.5 goals:  {pred['over_1_5_prob']:.1%}")
    print(f"    BTTS (Both):     {pred['btts_prob']:.1%}")
    print(f"    BTTS No:         {pred['btts_no_prob']:.1%}")

    # ── 4. Top Scorelines ──
    print_header("MOST LIKELY SCORELINES (Poisson)")

    print(f"  {'Score':>8}  {'Probability':>12}  {'Cumulative':>12}  {'Outcome'}")
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 20}")
    cumulative = 0.0
    for score, prob in pred["top_scorelines"]:
        cumulative += prob
        i, j = int(score[0]), int(score[2])
        outcome = "Norway win" if i > j else ("Draw" if i == j else "England win")
        print(f"  {score:>8}  {prob:>10.2%}  {cumulative:>10.2%}  {outcome}")
    print(f"\n  Remaining scorelines: {1 - cumulative:.2%}")

    # ── 5. Ensemble Comparison ──
    ensemble = load_ensemble_prediction()
    if ensemble:
        print_header("MODEL COMPARISON: Ensemble (XGBoost) vs Poisson")

        print(f"\n  {'Metric':<30} {'Ensemble':>14} {'Poisson':>14} {'Diff':>10}")
        print(f"  {'-' * 30} {'-' * 14} {'-' * 14} {'-' * 10}")

        metrics = [
            ("Norway win prob", ensemble["home_win_prob"], pred["home_win_prob"]),
            ("Draw prob", ensemble["draw_prob"], pred["draw_prob"]),
            ("England win prob", ensemble["away_win_prob"], pred["away_win_prob"]),
        ]

        for label, x_val, p_val in metrics:
            diff = (x_val - p_val) * 100
            print(f"  {label:<30} {x_val:>10.1%}   {p_val:>10.1%}   {diff:>+7.1f}pp")

        print(f"\n  {'Metric':<30} {'Ensemble':>14} {'Poisson':>14}")
        print(f"  {'-' * 30} {'-' * 14} {'-' * 14}")
        print(f"  {'Expected Norway goals':<30} {'-':>14} {pred['expected_home_goals']:>10.3f}")
        print(f"  {'Expected England goals':<30} {'-':>14} {pred['expected_away_goals']:>10.3f}")
        print(f"  {'Expected total goals':<30} {'-':>14} {pred['expected_total']:>10.3f}")

        # Analysis
        print("\n  --- ANALYSIS ---")

        nor_diff = (ensemble["home_win_prob"] - pred["home_win_prob"]) * 100
        eng_diff = (ensemble["away_win_prob"] - pred["away_win_prob"]) * 100
        draw_diff = (ensemble["draw_prob"] - pred["draw_prob"]) * 100

        print(f"\n  Norway: Ensemble ({ensemble['home_win_prob']:.1%}) vs Poisson ({pred['home_win_prob']:.1%})")
        if abs(nor_diff) > 3:
            direction = "more" if nor_diff > 0 else "less"
            print(f"     -> Ensemble is {abs(nor_diff):.1f}pp {direction} confident in Norway.")
        else:
            print(f"     -> Both models agree (within {abs(nor_diff):.1f}pp).")

        print(f"\n  England: Ensemble ({ensemble['away_win_prob']:.1%}) vs Poisson ({pred['away_win_prob']:.1%})")
        if abs(eng_diff) > 3:
            direction = "more" if eng_diff > 0 else "less"
            print(f"     -> Ensemble is {abs(eng_diff):.1f}pp {direction} confident in England.")
        else:
            print(f"     -> Both models agree (within {abs(eng_diff):.1f}pp).")

        print(f"\n  Draw: Ensemble ({ensemble['draw_prob']:.1%}) vs Poisson ({pred['draw_prob']:.1%})")
        if abs(draw_diff) > 3:
            direction = "higher" if draw_diff > 0 else "lower"
            print(f"     -> Ensemble sees {direction} draw probability by {abs(draw_diff):.1f}pp.")
        else:
            print(f"     -> Both models agree on draw probability (within {abs(draw_diff):.1f}pp).")

    # ── 6. Value Betting Analysis ──
    print_header("VALUE BETTING PERSPECTIVE")

    nor_odds, draw_odds, eng_odds = 4.20, 3.60, 1.85

    # Use the average of both models for value assessment
    model_nor = (ensemble["home_win_prob"] + pred["home_win_prob"]) / 2 if ensemble else pred["home_win_prob"]
    model_draw = (ensemble["draw_prob"] + pred["draw_prob"]) / 2 if ensemble else pred["draw_prob"]
    model_eng = (ensemble["away_win_prob"] + pred["away_win_prob"]) / 2 if ensemble else pred["away_win_prob"]

    def value_bet(model_prob: float, odds: float) -> tuple[float, str]:
        implied = 1 / odds
        edge = model_prob - implied
        if edge > 0.05:
            return (edge, "VALUE BET (strong)")
        elif edge > 0.02:
            return (edge, "Slight value")
        elif edge > -0.02:
            return (edge, "Fair value")
        else:
            return (edge, "No value (overpriced)")

    print(f"\n  {'Outcome':<18} {'Odds':>8} {'Implied':>10} {'Model Prob':>12} {'Edge':>8}  {'Verdict'}")
    print(f"  {'-' * 18} {'-' * 8} {'-' * 10} {'-' * 12} {'-' * 8}  {'-' * 28}")
    for outcome, odds, model_p in [
        ("Norway win", nor_odds, model_nor),
        ("Draw", draw_odds, model_draw),
        ("England win", eng_odds, model_eng),
    ]:
        edge, verdict = value_bet(model_p, odds)
        implied = 1 / odds
        print(f"  {outcome:<18} {odds:>8.2f} {implied:>9.1%} {model_p:>10.1%}  {edge:>+6.1%}  {verdict}")

    # ── 7. Verdict ──
    print_header("VERDICT")

    ensemble_winner = "England" if ensemble and ensemble["away_win_prob"] > ensemble["home_win_prob"] else "Norway"
    poisson_winner = "England" if pred["away_win_prob"] > pred["home_win_prob"] else "Norway"

    print(f"  Ensemble (XGBoost) favours:  {ensemble_winner}")
    print(f"  Poisson (historical) favours: {poisson_winner}")

    print("\n  Key Takeaways:")
    print("  * This is the closest quarterfinal on paper -- neither team is a clear favourite.")
    if ensemble:
        sep = abs(ensemble["home_win_prob"] - ensemble["away_win_prob"]) * 100
        print(f"  * The ensemble gives only ~{sep:.0f}pp separation between England and Norway.")
    print("  * Norway has been on an incredible run: beat Brazil in R16 plus Iraq and Ivory Coast.")
    print(f"  * England has historically strong defense (b={pred['def_away']:.3f}) -- key vs Norway's elite attack (a={pred['att_home']:.3f}).")
    print(f"  * High-scoring match expected: ~{pred['over_2_5_prob']:.0%} chance of Over 2.5 goals, ~{pred['btts_prob']:.0%} BTTS.")
    if pred["expected_total"] > 2.5:
        print(f"  * Expected total of {pred['expected_total']:.2f} goals suggests an open, entertaining game.")

    print("\n  Best Value Bets (from model vs market):")
    if model_nor > 1/nor_odds + 0.02:
        print(f"     Norway @ {nor_odds} -- model sees edge here (giant-killing momentum)")
    if model_eng < 1/eng_odds - 0.02:
        print(f"     England @ {eng_odds} -- slightly overpriced, bookies overrating big name")
    if pred["over_2_5_prob"] > 0.65:
        print("     Over 2.5 goals -- strongly supported by Poisson model")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
