---
tags:
  - football-prediction
  - performance
  - optimization
created: 2026-07-12
---

# ⚡ Performance Optimization

> Tips and techniques for improving prediction pipeline speed. The full guide is in the parent [performance_optimization.md](../performance_optimization.md) document.

---

## Quick Tips

| Issue | Solution |
|-------|----------|
| **Slow first run** | Use `--skip-download` or `--lightweight` flags |
| **Hyper-parameter tuning** | Disable in config: `tune_base_models = False` |
| **Player/lineup collection** | Use `--skip-lineups` flag |
| **Dixon-Coles MLE** | Disabled by default — slow on large datasets |
| **Data download** | Use `--skip-download` after first collection |

## Expected Run Times

| Command | Time |
|---------|------|
| `python train_worldcup.py` | 20-30s |
| `python run_pipeline.py --skip-download` | 30-60s |
| `python run_pipeline.py --lightweight` | 5-10s |
| `python today_value_bets_live.py` | 10-20s |
| `python run_dashboard.py` | < 5s |

---

> **Full reference:** See [performance_optimization.md](../performance_optimization.md) for the complete guide including profiling, caching, and database indexing tips.
