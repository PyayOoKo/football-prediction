"""
predict_league.py — Predict upcoming fixtures for any league using per-league models.

Usage
-----
    python predict_league.py SE1          # Swedish Superettan
    python predict_league.py E0           # Premier League
    python predict_league.py I1           # Serie A
    python predict_league.py SP1 D1 F1    # La Liga + Bundesliga + Ligue 1

If per-league models don't exist, they'll be auto-trained first.
"""

from __future__ import annotations

import argparse
import logging
import sys
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("predict_league")

# ── Paths ────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "football_data.db"
MODELS_DIR = PROJECT_ROOT / "models" / "per_league"
REPORTS_DIR = PROJECT_ROOT / "reports" / "predictions"

# League codes and names
LEAGUE_NAMES: dict[str, str] = {
    "E0":  "Premier League",
    "SP1": "La Liga",
    "D1":  "Bundesliga",
    "I1":  "Serie A",
    "F1":  "Ligue 1",
    "SE1": "Sweden Superettan",
    "SWE": "Sweden Allsvenskan",
    "NOR": "Norway Eliteserien",
    "FI":  "Finland Veikkausliiga",
    "DN1": "Denmark 1st Division",
    "IRL": "Ireland Premier Division",
    "POL": "Poland I Liga",
}

# Equal 4-model blend weights
EQUAL_WEIGHTS = {
    # Research-validated per-market model selection (2026-07-25):
    # 1X2: 4-model blend (DC + Elo + XGB + LGB)
    # Binary markets: DC-only (trees degrade performance on O/U and BTTS)
    "1X2":     {"dc": 0.25, "elo": 0.25, "xgb": 0.25, "lgb": 0.25},
    "Over2.5": {"dc": 1.00},
    "Over3.5": {"dc": 1.00},
    "BTTS":    {"dc": 1.00},
}


# ═══════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════


def load_league_models(league: str) -> dict | None:
    """Load saved per-league models. Returns None if DC/Elo not found."""
    import joblib

    league_dir = MODELS_DIR / league
    dc_path = league_dir / "dixon_coles.joblib"
    elo_path = league_dir / "elo.joblib"
    if not dc_path.exists() or not elo_path.exists():
        return None

    dc = joblib.load(dc_path)
    elo = joblib.load(elo_path)

    xgb_path = league_dir / "xgboost.joblib"
    xgb = joblib.load(xgb_path) if xgb_path.exists() else None

    lgb_path = league_dir / "lightgbm.joblib"
    lgb = joblib.load(lgb_path) if lgb_path.exists() else None

    # Load calibrator (try hybrid, fall back to platt, then generic)
    calibrator = None
    for name in ["hybrid", "platt", "isotonic", ""]:
        cal_path = league_dir / f"blend_calibrator_{name}.joblib" if name else league_dir / "blend_calibrator.joblib"
        if cal_path.exists():
            try:
                calibrator = joblib.load(cal_path)
                logger.info("  Loaded calibrator: %s", cal_path.name)
                break
            except Exception:
                continue

    return {
        "dc": dc,
        "elo": elo,
        "xgb": xgb,
        "lgb": lgb,
        "calibrator": calibrator,
    }


def train_league_auto(league: str) -> dict | None:
    """Auto-train per-league models by calling train_league_models.py."""
    import subprocess
    import sys as _sys

    logger.info("  No saved models found. Auto-training %s...", league)
    python_exe = _sys.executable
    script = PROJECT_ROOT / "train_league_models.py"
    if not script.exists():
        logger.error("  train_league_models.py not found! Can't auto-train.")
        return None

    result = subprocess.run(
        [python_exe, str(script), "--leagues", league],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error("  Auto-training failed: %s", result.stderr[-300:])
        return None
    logger.info("  Auto-training complete.")
    return load_league_models(league)


# ═══════════════════════════════════════════════════════════
#  Fixture Loading
# ═══════════════════════════════════════════════════════════


def load_historical_data(league: str) -> pd.DataFrame:
    """Load completed matches for a league (for blend construction)."""
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        """SELECT * FROM matches
           WHERE league = ?
             AND home_goals IS NOT NULL AND away_goals IS NOT NULL
           ORDER BY date ASC""",
        conn,
        params=(league,),
    )
    conn.close()
    return df


def load_fixtures(league: str) -> pd.DataFrame:
    """Load upcoming fixtures (no result yet) for a league."""
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        """SELECT date, home_team, away_team
           FROM matches
           WHERE league = ?
             AND home_goals IS NULL
           ORDER BY date ASC""",
        conn,
        params=(league,),
    )
    conn.close()
    return df


# ═══════════════════════════════════════════════════════════
#  Prediction & Display
# ═══════════════════════════════════════════════════════════


def predict_league(league: str, models: dict) -> pd.DataFrame | None:
    """Generate and display predictions for a league's upcoming fixtures."""
    league_name = LEAGUE_NAMES.get(league, league)

    # Load data
    logger.info("Loading %s (%s) historical data...", league, league_name)
    train_df = load_historical_data(league)
    logger.info("  %d historical matches loaded", len(train_df))

    logger.info("Loading upcoming fixtures...")
    fixtures_df = load_fixtures(league)
    logger.info("  %d upcoming fixtures found", len(fixtures_df))

    if fixtures_df.empty:
        print(f"\n  [i] No upcoming fixtures for {league} ({league_name})")
        return None

    # Build blend
    from src.models.three_model_blend import ThreeModelBlend, ConditionalRates

    logger.info("Building blend model...")
    cond_rates = ConditionalRates.from_data(train_df)
    blend = ThreeModelBlend(
        dc_model=models["dc"],
        elo_model=models["elo"],
        xgb_model=models["xgb"],
        lgb_model=models["lgb"],
        weights=EQUAL_WEIGHTS,
        conditional_rates=cond_rates,
        historical_df=train_df,
    )

    # Attach form adjuster
    try:
        from src.form_adjuster import RecentFormAdjuster
        adjuster = RecentFormAdjuster(n_matches=6, form_weight=50.0)
        adjuster.fit(train_df)
        blend.form_adjuster = adjuster
    except Exception as exc:
        logger.warning("  Form adjuster skipped: %s", exc)
        adjuster = None

    # Show form report
    teams = set(fixtures_df["home_team"].tolist() + fixtures_df["away_team"].tolist())
    if adjuster is not None:
        form_report = adjuster.get_form_report(list(teams))
        print(f"\n  RECENT FORM (last 6 matches, weight={adjuster.form_weight:.0f} Elo pts)")
        print(f"  {'-' * 55}")
        for _, r in form_report.iterrows():
            adj = r["elo_adjustment"]
            sign = "+" if adj >= 0 else ""
            print(f"  {r['team']:30s}  score={r['form_score']:+.3f}  "
                  f"Elo adj={sign}{adj:.1f}  [{r['form_label']}]")
        print(f"  {'-' * 55}")

    # Predict
    logger.info("Generating predictions for %d matches...", len(fixtures_df))
    raw_preds = blend.predict_matches(fixtures_df)

    # Apply calibration
    calibrator = models.get("calibrator")
    if calibrator is not None:
        raw_probs = np.column_stack([
            raw_preds["away_win_prob"].values,
            raw_preds["draw_prob"].values,
            raw_preds["home_win_prob"].values,
        ])
        cal_probs = calibrator.transform(raw_probs)
        raw_preds["away_win_prob"] = cal_probs[:, 0]
        raw_preds["draw_prob"] = cal_probs[:, 1]
        raw_preds["home_win_prob"] = cal_probs[:, 2]
        for i in range(len(raw_preds)):
            hw, dr, aw = cal_probs[i, 2], cal_probs[i, 1], cal_probs[i, 0]
            if hw >= dr and hw >= aw:
                raw_preds.at[raw_preds.index[i], "predicted_outcome"] = "Home Win"
            elif dr >= aw:
                raw_preds.at[raw_preds.index[i], "predicted_outcome"] = "Draw"
            else:
                raw_preds.at[raw_preds.index[i], "predicted_outcome"] = "Away Win"
            raw_preds.at[raw_preds.index[i], "confidence"] = round(max(hw, dr, aw), 4)

    preds = raw_preds

    # ── Display ──
    cal_tag = "  |  Calibrated" if calibrator is not None else ""
    SEP = "=" * 95
    print(f"\n{SEP}")
    print(f"  >> {league} ({league_name}) PREDICTIONS  --  Multi-Model Blend{cal_tag}")
    print(SEP)

    for _, row in preds.iterrows():
        ht, at = row["home_team"], row["away_team"]
        hw, dr, aw = row["home_win_prob"], row["draw_prob"], row["away_win_prob"]

        if hw >= dr and hw >= aw:
            outcome = "HOME WIN"
        elif dr >= aw:
            outcome = "DRAW"
        else:
            outcome = "AWAY WIN"

        print(f"\n  {ht:30s} vs {at:30s}  [{row.get('date', '')}]")
        print(f"  1X2:     H {hw*100:5.1f}%  D {dr*100:5.1f}%  A {aw*100:5.1f}%")
        print(f"  O/U 2.5: Over {row['over_2_5_prob']*100:.1f}%  "
              f"Under {row['under_2_5_prob']*100:.1f}%")
        print(f"  O/U 3.5: Over {row['over_3_5_prob']*100:.1f}%  "
              f"Under {row['under_3_5_prob']*100:.1f}%")
        print(f"  BTTS:    Yes {row['btts_prob']*100:.1f}%  "
              f"No {row['btts_no_prob']*100:.1f}%")
        print(f"  >> Predicted: {outcome}  (confidence: {row['confidence']*100:.0f}%)")

    print(f"\n{SEP}")

    # Fair odds
    print(f"\n  FAIR ODDS (1 / probability)")
    print(f"  {'-' * 95}")
    for _, row in preds.iterrows():
        hw, dr, aw = row["home_win_prob"], row["draw_prob"], row["away_win_prob"]
        ho = 1.0 / hw if hw > 0 else 999
        do = 1.0 / dr if dr > 0 else 999
        ao = 1.0 / aw if aw > 0 else 999
        print(f"  {row['home_team']:30s} vs {row['away_team']:30s}")
        print(f"    Fair odds:  {ho:5.2f}  {do:5.2f}  {ao:5.2f}  (H / D / A)")
    print()

    # Save
    out_path = REPORTS_DIR / f"{league}_predictions.csv"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)
    logger.info("Predictions saved to %s", out_path)

    return preds


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Predict upcoming fixtures for any league using per-league models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "leagues", nargs="+",
        help="League code(s) to predict (e.g. E0 SE1 I1). "
             f"Available: {', '.join(sorted(LEAGUE_NAMES.keys()))}",
    )
    parser.add_argument(
        "--no-train", action="store_true",
        help="Don't auto-train if models are missing (just show error)",
    )
    args = parser.parse_args()

    for league in args.leagues:
        league_upper = league.upper()
        league_name = LEAGUE_NAMES.get(league_upper, league_upper)

        print()
        print("=" * 60)
        print(f"  {league_upper} - {league_name}")
        print("=" * 60)

        # Load models
        models = load_league_models(league_upper)
        if models is None:
            if args.no_train:
                logger.error("No saved models for %s. Run: python train_league_models.py --leagues %s",
                             league_upper, league_upper)
                continue
            models = train_league_auto(league_upper)
            if models is None:
                logger.error("Could not get models for %s", league_upper)
                continue

        # Show what models are loaded
        loaded = []
        if models.get("dc"): loaded.append("DC")
        if models.get("elo"): loaded.append("Elo")
        if models.get("xgb"): loaded.append("XGB")
        if models.get("lgb"): loaded.append("LGB")
        cal = " + calibrator" if models.get("calibrator") else ""
        logger.info("Models loaded: %s%s", ", ".join(loaded), cal)

        # Predict
        predict_league(league_upper, models)

    print("\nDone.")


if __name__ == "__main__":
    main()
