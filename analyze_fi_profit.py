import sqlite3, sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
DB_PATH = Path("data/football_data.db")
MODELS_DIR = Path("models/per_league")
INITIAL_BANKROLL = 10000.0

def load_league_data(league):
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query("""
        SELECT date, home_team, away_team, home_goals, away_goals, result,
               home_odds, draw_odds, away_odds, season
        FROM matches WHERE league = ? AND home_goals IS NOT NULL
          AND home_odds IS NOT NULL AND draw_odds IS NOT NULL AND away_odds IS NOT NULL
        ORDER BY date ASC
    """, conn, params=(league,))
    conn.close()
    return df

def get_blend_predictions(df, league):
    import joblib
    from src.models.three_model_blend import ThreeModelBlend
    
    models_dir = MODELS_DIR / league
    dc = joblib.load(models_dir / "dixon_coles.joblib")
    elo = joblib.load(models_dir / "elo.joblib")
    xgb = joblib.load(models_dir / "xgboost.joblib") if (models_dir / "xgboost.joblib").exists() else None
    lgb = joblib.load(models_dir / "lightgbm.joblib") if (models_dir / "lightgbm.joblib").exists() else None
    cal = joblib.load(models_dir / "blend_calibrator.joblib") if (models_dir / "blend_calibrator.joblib").exists() else None

    blend = ThreeModelBlend(dc_model=dc, elo_model=elo, xgb_model=xgb, lgb_model=lgb, historical_df=df)
    
    # Build predictions rolling chronologically
    all_probs = []
    for i in range(len(df)):
        # Use all data before this match for training features
        train_df = df.iloc[:i].copy() if i > 0 else df.iloc[:0].copy()
        match = df.iloc[i]
        
        # Get DC+Elo
        dc_pred = dc.predict(match["home_team"], match["away_team"])
        dc_probs = np.array([dc_pred.away_win_prob, dc_pred.draw_prob, dc_pred.home_win_prob])
        
        R_home = elo.get_rating(match["home_team"])
        R_away = elo.get_rating(match["away_team"])
        E_home = elo.expected_score(R_home, R_away)
        elo_away, elo_draw, elo_home = elo._expected_to_probs(E_home)
        elo_probs = np.array([elo_away, elo_draw, elo_home])
        
        blend_probs = (dc_probs + elo_probs) / 2.0
        
        # Apply calibration
        if cal is not None:
            blend_probs = cal.transform(blend_probs.reshape(1, -1))[0]
        
        # Update Elo
        elo.update_ratings(match["home_team"], match["away_team"], match["result"],
                          home_goals=match["home_goals"], away_goals=match["away_goals"])
        
        all_probs.append(blend_probs)
    
    return np.array(all_probs)

def analyze(league):
    df = load_league_data(league)
    print(f"\n{'='*60}")
    print(f"  {league} - {len(df)} matches total")
    print(f"{'='*60}")
    
    # Split 85/15 chronological
    split = int(len(df) * 0.85)
    backtest_df = df.iloc[split:].copy()
    print(f"  Backtest period: {backtest_df.iloc[0]['date']} to {backtest_df.iloc[-1]['date']} ({len(backtest_df)} matches)")
    
    # Get DC+Elo predictions (simple, fast, reliable)
    import joblib
    dc = joblib.load(MODELS_DIR / league / "dixon_coles.joblib")
    elo = joblib.load(MODELS_DIR / league / "elo.joblib")
    cal = joblib.load(MODELS_DIR / league / "blend_calibrator.joblib") if (MODELS_DIR / league / "blend_calibrator.joblib").exists() else None
    
    bets = []
    bankroll = INITIAL_BANKROLL
    
    for idx, row in backtest_df.iterrows():
        try:
            dc_pred = dc.predict(row["home_team"], row["away_team"])
            dc_probs = np.array([dc_pred.away_win_prob, dc_pred.draw_prob, dc_pred.home_win_prob])
            
            R_home = elo.get_rating(row["home_team"])
            R_away = elo.get_rating(row["away_team"])
            E_home = elo.expected_score(R_home, R_away)
            elo_away, elo_draw, elo_home = elo._expected_to_probs(E_home)
            elo_probs = np.array([elo_away, elo_draw, elo_home])
            
            blend_probs = (dc_probs + elo_probs) / 2.0
            if cal is not None:
                blend_probs = cal.transform(blend_probs.reshape(1, -1))[0]
            
            elo.update_ratings(row["home_team"], row["away_team"], row["result"],
                              home_goals=row["home_goals"], away_goals=row["away_goals"])
        except Exception:
            continue
        
        outcomes = [
            ("H", 2, row["home_odds"], blend_probs[2]),
            ("D", 1, row["draw_odds"], blend_probs[1]),
            ("A", 0, row["away_odds"], blend_probs[0]),
        ]
        actual_idx = {"A": 0, "D": 1, "H": 2}.get(row["result"], -1)
        
        for label, idx2, odds, prob in outcomes:
            if odds <= 1 or prob <= 0:
                continue
            implied = 1.0 / odds
            ev = prob / implied - 1.0
            if ev < 0.05:
                continue
            
            full_kelly = (prob * odds - 1.0) / (odds - 1.0)
            stake_pct = max(0.0, full_kelly * 0.25)
            if stake_pct <= 0 or bankroll <= 1:
                continue
            
            stake = bankroll * stake_pct
            won = idx2 == actual_idx
            profit = stake * (odds - 1.0) if won else -stake
            bankroll += profit
            
            bets.append({
                "date": row["date"],
                "season": row.get("season", ""),
                "home": row["home_team"],
                "away": row["away_team"],
                "bet": label,
                "odds": odds,
                "model_prob": prob,
                "implied": implied,
                "ev": ev,
                "stake": stake,
                "won": won,
                "profit": profit,
                "odds_bucket": "favorite" if odds < 2.5 else ("mid" if odds < 5.0 else "underdog"),
            })
    
    if not bets:
        print("  No bets placed!")
        return
    
    bets_df = pd.DataFrame(bets)
    total_profit = bets_df["profit"].sum()
    total_staked = bets_df["stake"].sum()
    
    print(f"\n  Total bets: {len(bets_df)}")
    print(f"  Total profit: {total_profit:+.2f}")
    print(f"  Yield: {total_profit/total_staked*100:+.2f}%")
    print(f"  Win rate: {bets_df['won'].mean()*100:.1f}%")
    print(f"  Avg odds: {bets_df['odds'].mean():.2f}")
    
    # Per-season
    print(f"\n  --- Per Season ---")
    for season, grp in bets_df.groupby("season"):
        p = grp["profit"].sum()
        s = grp["stake"].sum()
        w = grp["won"].mean()
        print(f"  {season}: bets={len(grp):>3} profit={p:>+8.0f} yield={p/s*100:>+6.1f}% wr={w*100:.0f}% avg_odds={grp['odds'].mean():.2f}")
    
    # Per odds bucket
    print(f"\n  --- Per Odds Range ---")
    for bucket, grp in bets_df.groupby("odds_bucket"):
        p = grp["profit"].sum()
        s = grp["stake"].sum()
        w = grp["won"].mean()
        print(f"  {bucket}: bets={len(grp):>3} profit={p:>+8.0f} yield={p/s*100:>+6.1f}% wr={w*100:.0f}% avg_odds={grp['odds'].mean():.2f}")
    
    # Per bet type
    print(f"\n  --- Per Bet Type ---")
    for label, grp in bets_df.groupby("bet"):
        p = grp["profit"].sum()
        s = grp["stake"].sum()
        w = grp["won"].mean()
        print(f"  {label}: bets={len(grp):>3} profit={p:>+8.0f} yield={p/s*100:>+6.1f}% wr={w*100:.0f}% avg_odds={grp['odds'].mean():.2f}")
    
    # Per home/away
    print(f"\n  --- Home vs Away ---")
    for is_home, grp in bets_df.groupby(bets_df["bet"] == "H"):
        label = "Home" if is_home else "Away/Draw"
        p = grp["profit"].sum()
        s = grp["stake"].sum()
        w = grp["won"].mean()
        print(f"  {label}: bets={len(grp):>3} profit={p:>+8.0f} yield={p/s*100:>+6.1f}% wr={w*100:.0f}% avg_odds={grp['odds'].mean():.2f}")
    
    return bets_df

# Run for FI and a comparison league
bets_fi = analyze("FI")
bets_nor = analyze("NOR")
bets_swe = analyze("SWE")
