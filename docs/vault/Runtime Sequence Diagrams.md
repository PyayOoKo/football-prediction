---
tags:
  - football-prediction
  - runtime
  - sequences
  - mermaid
created: 2026-07-12
---

# ⏱ Runtime Sequence Diagrams

> Dynamic views of how modules interact during execution. Read top-to-bottom — each arrow is a method call or data transfer.

See also: [[Architecture Overview]], [[Ensemble Model]], [[Feature Engineering Pipeline]], [[Value Betting & Backtesting]]

---

## 1. Full Pipeline Execution

Shows `run_pipeline.py` orchestrating all 5 steps:

```mermaid
sequenceDiagram
    participant CLI as CLI Args
    participant PIP as run_pipeline.py
    participant DC as src/data_collection/collector.py
    participant PRE as src/preprocessing.py
    participant FE as src/feature_engineering.py
    participant ENS as src/ensemble.py (EnsembleModel)
    participant REP as Report Writer
    
    Note over CLI,PIP: python run_pipeline.py
    CLI->>PIP: parse_args()
    
    alt --skip-download or --lightweight
        PIP-->>PIP: Skip Step 1
    else
        PIP->>DC: step_download()
        DC-->>PIP: {new_rows, total_rows}
    end
    
    alt --lightweight
        PIP-->>PIP: Skip Step 2
    else
        PIP->>PRE: step_preprocess()
        PRE-->>PIP: {rows, columns, saved_to}
    end
    
    alt --skip-train or --lightweight
        PIP-->>PIP: Skip Step 3
    else
        PIP->>FE: build_features(df)
        FE-->>PIP: X, y
        PIP->>FE: train_val_test_split(X, y)
        FE-->>PIP: splits
        PIP->>ENS: EnsembleModel()
        PIP->>ENS: fit(X_train, y_train, X_val, y_val, df_train, df_val)
        Note over ENS: Trains XGBoost + LR + Poisson<br/>Optimises weights via grid search
        ENS-->>PIP: {val_log_loss, weights}
        PIP->>ENS: save(model_path)
    end
    
    PIP->>FE: build_features(df, is_training=True)
    Note over FE: Uses is_training=True (y is discarded via _)
    FE-->>PIP: X_pred, _
    PIP->>ENS: predict_proba(X_pred, df_raw)
    ENS-->>PIP: probs [n, 3]
    PIP->>ENS: predict(X_pred, df_raw)
    ENS-->>PIP: preds [n,]
    PIP->>REP: step_report(results)
    REP-->>PIP: {report_path}
    
    Note over PIP: Pipeline complete ✓
```

---

## 2. Ensemble Training (fit)

Details the internal flow of `EnsembleModel.fit()`:

```mermaid
sequenceDiagram
    participant CALLER as Pipeline / Caller
    participant ENS as EnsembleModel
    participant XGB as XGBoost
    participant LR as LogisticRegression
    participant POI as PoissonModel
    participant GRID as Weight Grid Search
    
    CALLER->>ENS: fit(X_train, y_train, X_val, y_val, df_train, df_val)
    
    Note over ENS: Step 1: Train ML sub-models
    ENS->>XGB: XGBClassifier.fit(X_train, y_train, eval_set)
    Note over XGB: 80 trees, depth 5, multi:softprob
    XGB-->>ENS: trained xgboost model
    
    ENS->>LR: LogisticRegression.fit(X_train_clean, y_train)
    Note over LR: lbfgs, balanced, C=1.0
    LR-->>ENS: trained logistic_regression model
    
    Note over ENS: Step 2: Train Poisson (on raw data)
    ENS->>POI: PoissonModel.fit(df_train)
    Note over POI: Computes league averages + team strengths
    POI-->>ENS: fitted poisson model
    
    Note over ENS: Step 3: Get validation predictions
    ENS->>XGB: predict_proba(X_val)
    XGB-->>ENS: xgb_val_probs [n, 3]
    ENS->>LR: predict_proba(X_val_clean)
    LR-->>ENS: lr_val_probs [n, 3]
    ENS->>POI: predict_matches(df_val)
    POI-->>ENS: poisson_val_probs [n, 3]
    
    Note over ENS: Step 4: Optimise weights
    ENS->>GRID: Enumerate weight combinations (step=0.10)
    Note over GRID: ~66 combinations for 3 models
    GRID-->>ENS: best_weights {name: weight}
    
    Note over ENS: Step 5: Apply constraints & evaluate
    ENS->>ENS: _apply_weight_constraints()
    ENS->>ENS: compute weighted val log-loss
    
    ENS-->>CALLER: {train_log_loss, val_log_loss, weights, individual_log_losses}
```

---

## 3. Feature Engineering Sequence

Shows `build_features()` calling each sub-module:

```mermaid
sequenceDiagram
    participant CALLER as Pipeline / Caller
    participant BF as build_features()
    participant ELO as src/elo.py
    participant ODD as src/odds_processing.py
    participant PLAY as src/player_info.py
    participant XG as src/xg_features.py
    participant POI as src/poisson_model.py
    participant DCX as src/dixon_coles.py
    participant ROLL as _add_rolling_features()
    participant H2H as _add_h2h_features()
    
    CALLER->>BF: build_features(df, is_training=True)
    BF->>BF: df.sort_values(["date", "home_team"])
    
    BF->>ELO: add_elo_features(df, k, home_advantage, ...)
    Note over ELO: Pre-match ratings per team, no leakage
    ELO-->>BF: df + [Home_Elo, Away_Elo, Elo_Difference]
    
    BF->>ODD: add_odds_features(df, opening_cols, closing_cols)
    Note over ODD: Auto-detects BbMx/BbAv/B365 columns
    ODD-->>BF: df + [odds_*, fair_prob_*, clv_*, ...]
    
    alt player_info.enabled
        BF->>PLAY: add_player_features(df, players_df, lineups_df)
        PLAY-->>BF: df + [h_injured_count, h_rotation_index, ...]
    end
    
    BF->>XG: add_xg_features(df)
    Note over XG: Auto-detects home_xg or creates placeholders
    XG-->>BF: df + [h_xg_avg5, a_xg_avg5, h_xpts, ...]
    
    BF->>POI: add_poisson_features(df)
    Note over POI: Expanding window per match: α, β, λ
    POI-->>BF: df + [Expected_Home_Goals, ...]
    
    alt dixon_coles.enabled
        BF->>DCX: add_features(df, refit_every=500)
        Note over DCX: MLE with warm-start refits
        DCX-->>BF: df + [DC_Expected_*, DC_Rho, ...]
    end
    
    BF->>BF: _add_competition_importance(df)
    BF->>ROLL: _add_rolling_features(df, windows=(5,10,20))
    Note over ROLL: All rolling features use .shift(1) for leakage prevention
    ROLL-->>BF: df + [h_points_avg5, a_goals_scored_avg10, ...]
    
    BF->>H2H: _add_h2h_features(df, window=6)
    H2H-->>BF: df + [h2h_home_points_avg, ...]
    
    BF->>BF: _add_league_position_features(df)
    BF->>BF: _encode_categoricals(df)
    BF->>BF: _add_attack_defence_ratios(df)
    BF->>BF: Drop target, keep numerics, sanitise names
    
    BF-->>CALLER: X (feature matrix), y (target)
```

---

## 4. Prediction Flow

Shows `EnsembleModel.predict()`:

```mermaid
sequenceDiagram
    participant CALLER as Pipeline / Caller
    participant ENS as EnsembleModel
    participant XGB as XGBoost (loaded)
    participant LR as LogisticRegression (loaded)
    participant POI as PoissonModel (loaded)
    
    CALLER->>ENS: predict_proba(X, df_raw)
    
    Note over ENS: Get predictions from each sub-model
    ENS->>XGB: predict_proba(X_filled)
    Note over XGB: NaN-filled with column means
    XGB-->>ENS: xgb_probs [n, 3]
    
    ENS->>LR: predict_proba(X_filled)
    LR-->>ENS: lr_probs [n, 3]
    
    alt df_raw provided and poisson present
        ENS->>POI: predict_matches(df_raw)
        POI-->>ENS: poisson_preds_df
        ENS->>ENS: extract [away, draw, home]
    else
        Note over ENS: Equal probs fallback
    end
    
    ENS->>ENS: _apply_weights({xgb, lr, poisson}, weights)
    Note over ENS: weighted = Σ w_i × probs_i, renormalise
    ENS-->>CALLER: probs [n, 3] — [away, draw, home]
```

---

## 5. Value Bet Detection

```mermaid
sequenceDiagram
    participant CALLER as User / Script
    participant VB as compute_value_bets()
    participant LOOP as Per-Match Loop
    
    CALLER->>VB: compute_value_bets(odds, model_probs, team_matches, ...)
    VB->>VB: Validate shapes
    
    loop For each match i
        Note over LOOP: implied = 1.0 / decimal_odds
        Note over LOOP: margin = sum(implied) - 1.0
        Note over LOOP: fair = implied / (1 + margin)
        Note over LOOP: ev = (model_prob × decimal_odds) - 1.0
        
        alt odds > 1.0 and ev > 0
            LOOP->>LOOP: kelly = ev / (odds - 1.0) × kelly_fraction
        else
            LOOP->>LOOP: kelly = 0.0
        end
        
        alt ev >= min_ev AND model > fair AND kelly > 0
            LOOP-->>VB: Append VALUE BET row
        else
            LOOP-->>VB: Append No Value row
        end
    end
    
    VB->>VB: df.sort_values(["positive_ev", "ev"], ascending=[False, False])
    VB-->>CALLER: DataFrame with [match, outcome, ev, kelly_pct, ...]
```

---

## 6. Backtesting Loop

```mermaid
sequenceDiagram
    participant CALLER as User / Script
    participant BT as BacktestEngine
    participant MODEL as Trained Model
    participant METRICS as Metrics Calculator
    
    CALLER->>BT: BacktestEngine(model, bankroll=1000, kelly=0.25, min_ev=0.0)
    CALLER->>BT: run(X_test, y_test, odds_df)
    
    BT->>MODEL: predict_proba(X_test)
    MODEL-->>BT: y_proba [n_test, 3]
    BT->>BT: bankroll = initial, bets = []
    
    loop For i = 0 to n_test-1
        alt odds not available or NaN
            BT-->>BT: skip match
        else
            BT->>BT: implied = 1.0 / odds[i]
            BT->>BT: margin = sum(implied) - 1.0
            BT->>BT: fair = implied / (1 + margin)
            
            loop For each outcome j (Away, Draw, Home)
                BT->>BT: ev = (model_prob × decimal_odds) - 1.0
                alt ev > 0 and odds > 1.0
                    BT->>BT: kelly_pct = ev / (odds - 1.0) × kelly_fraction
                else
                    BT->>BT: kelly_pct = 0.0
                end
                
                alt ev >= min_ev AND model > fair AND kelly > 0
                    Note over BT: PLACE BET
                    BT->>BT: stake = bankroll × kelly_pct
                    alt bet won
                        BT->>BT: profit = stake × (odds - 1.0)
                    else
                        BT->>BT: profit = -stake
                    end
                    BT->>BT: bankroll += profit, record BetRecord
                end
            end
        end
    end
    
    BT->>METRICS: calculate_metrics()
    METRICS-->>BT: BacktestMetrics (ROI, yield, drawdown, ...)
    BT-->>CALLER: metrics
```
