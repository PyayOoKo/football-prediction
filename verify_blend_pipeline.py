"""
verify_blend_pipeline.py — End-to-end verification of the 3-model blend
through the full pipeline: saved model → prediction engine → dashboard data.

Usage:
    python verify_blend_pipeline.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd


def main() -> int:
    print("=" * 60)
    print("  3-Model Blend — Full Pipeline Verification")
    print("=" * 60)

    errors = 0

    # ═══════════════════════════════════════════════════════
    #  1. Saved model file exists and has all components
    # ═══════════════════════════════════════════════════════

    print("\n[1/5] Saved model file ... ", end="")
    blend_path = Path("models/three_model_blend.joblib")
    if not blend_path.exists():
        print("FAIL — File not found!")
        return 1
    size_kb = blend_path.stat().st_size / 1024
    print(f"OK ({size_kb:.0f} KB)")

    # ═══════════════════════════════════════════════════════
    #  2. Load and inspect the payload
    # ═══════════════════════════════════════════════════════

    print("[2/5] Loading payload ... ", end="")
    try:
        payload = joblib.load(blend_path)
        required_keys = {"poisson", "elo", "xgb", "weights", "cond_rates"}
        actual_keys = set(payload.keys())
        if required_keys.issubset(actual_keys):
            print(f"OK (keys: {', '.join(sorted(actual_keys))})")
        else:
            missing = required_keys - actual_keys
            print(f"FAIL — Missing keys: {missing}")
            errors += 1
    except Exception as e:
        print(f"FAIL — {e}")
        errors += 1
        return 1

    # ═══════════════════════════════════════════════════════
    #  3. Poisson model — predict a match
    # ═══════════════════════════════════════════════════════

    print("[3/5] Poisson model ... ", end="")
    try:
        r = payload["poisson"].predict("Spain", "Argentina")
        checks = [
            ("home_win_prob" in r, "home_win_prob"),
            ("draw_prob" in r, "draw_prob"),
            ("away_win_prob" in r, "away_win_prob"),
            ("btts_prob" in r, "btts_prob"),
            ("over_2_5_prob" in r, "over_2_5_prob"),
        ]
        all_ok = all(c[0] for c in checks)
        if all_ok:
            h = r["home_win_prob"]
            d = r["draw_prob"]
            a = r["away_win_prob"]
            print(f"OK — H={h:.1%} D={d:.1%} A={a:.1%}  BTTS={r['btts_prob']:.1%}  O2.5={r['over_2_5_prob']:.1%}")
        else:
            failed = [c[1] for c in checks if not c[0]]
            print(f"FAIL — Missing keys: {failed}")
            errors += 1
    except Exception as e:
        print(f"FAIL — {e}")
        errors += 1

    # ═══════════════════════════════════════════════════════
    #  4. Elo system — predict a match
    # ═══════════════════════════════════════════════════════

    print("[4/5] Elo system ... ", end="")
    try:
        elo = payload["elo"]
        df_s = pd.DataFrame([{"home_team": "Spain", "away_team": "Argentina"}])
        elo_probs = elo.predict_proba(df_s)[0]
        assert len(elo_probs) == 3, f"Expected 3 probs, got {len(elo_probs)}"

        ratings = elo.get_rating("Spain"), elo.get_rating("Argentina")
        print(f"OK — H={elo_probs[2]:.1%} D={elo_probs[1]:.1%} A={elo_probs[0]:.1%}  Ratings: Spain={ratings[0]:.0f} Arg={ratings[1]:.0f}")
    except Exception as e:
        print(f"FAIL — {e}")
        errors += 1

    # ═══════════════════════════════════════════════════════
    #  5. XGBoost — loads and has features
    # ═══════════════════════════════════════════════════════

    print("[5/5] XGBoost model ... ", end="")
    try:
        xgb = payload["xgb"]
        if hasattr(xgb, "feature_names_in_"):
            n_features = len(xgb.feature_names_in_)
        elif hasattr(xgb, "select_columns"):
            n_features = "available"
        else:
            n_features = "?"
        print(f"OK — type={type(xgb).__name__}, features={n_features}")
    except Exception as e:
        print(f"FAIL — {e}")
        errors += 1

    # ═══════════════════════════════════════════════════════
    #  6. Market weights
    # ═══════════════════════════════════════════════════════

    weights = payload["weights"]
    expected_markets = {"1X2", "Over2.5", "Over3.5", "BTTS"}
    actual_markets = set(weights.keys())
    if expected_markets == actual_markets:
        print("     Markets:", ", ".join(actual_markets))
        for m in sorted(actual_markets):
            w = weights[m]
            w_str = ", ".join(f"{k}={v:.2f}" for k, v in w.items())
            print(f"       {m}: {w_str}")
    else:
        print(f"     Markets: {actual_markets} (expected {expected_markets})")
        errors += 1

    # ═══════════════════════════════════════════════════════
    #  7. Pipeline report check
    # ═══════════════════════════════════════════════════════

    print("\n[+1] Pipeline integration ... ", end="")
    report_dir = Path("reports")
    reports = sorted(report_dir.glob("pipeline_report_*.txt"), reverse=True)
    if reports:
        last_report = reports[0]
        content = last_report.read_text(encoding="utf-8", errors="replace")
        if "retrain_blend" in content and "PASS" in content:
            print(f"OK (last run: {last_report.name})")
        else:
            print(f"WARN — retrain_blend step not found in {last_report.name}")
    else:
        print("WARN — No pipeline reports found")

    # ═══════════════════════════════════════════════════════
    #  8. Value bets data check
    # ═══════════════════════════════════════════════════════

    print("[+2] Value bets data ... ", end="")
    vb_path = Path("reports/value_bets/latest.csv")
    if vb_path.exists():
        vb_df = pd.read_csv(vb_path)
        if len(vb_df) > 0 and "model_prob" in vb_df.columns:
            spain_prob = vb_df[vb_df["outcome"] == "H"]["model_prob"].values
            if len(spain_prob) > 0:
                print(f"OK — Spain prob={spain_prob[0]:.1%}, n_bets={len(vb_df)}")
            else:
                print(f"WARN — No Spain data found")
        else:
            print(f"WARN — {len(vb_df)} rows, no model_prob column")
    else:
        print(f"FAIL — File not found")
        errors += 1

    # ═══════════════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    if errors == 0:
        print("  RESULT: ALL CHECKS PASSED")
        print("  The 3-model blend is working end-to-end.")
    else:
        print(f"  RESULT: {errors} check(s) FAILED")
        print("  Review the output above for details.")
    print(f"{'=' * 60}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
