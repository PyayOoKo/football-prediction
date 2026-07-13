---
tags:
  - football-prediction
  - setup
  - quickstart
created: 2026-07-12
---

# ⚡ Quick Start Guide

> The fastest way to go from zero to predictions.

Related notes: [[Architecture Overview]], [[Scripts Reference]]

---

## Setup

```bash
# 1. Clone / navigate to the project
cd football-prediction

# 2. Create environment
cp .env.example .env
python -m venv .venv

# 3. Activate (choose one):
#    Command Prompt: .venv\Scripts\activate
#    PowerShell:     .venv\Scripts\Activate.ps1
#    Git Bash:       source .venv/Scripts/activate

# 4. Install dependencies
pip install -r requirements.txt
```

> **💡 Tip:** If you get SSL errors, try `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt`

---

## Fastest Path to Predictions

```bash
# 1. Collect World Cup data (2002-2026)
python collect_all_worldcups.py

# 2. Train and predict
python train_worldcup.py          # ~20-30 seconds

# 3. Or use the daily pipeline (lightweight mode)
python run_pipeline.py --lightweight   # ~5-10 seconds
```

---

## Common Commands

| Command | What It Does | Est. Time |
|---------|-------------|-----------|
| `python collect_all_worldcups.py` | Download 2002–2026 World Cup data | ~10s |
| `python train_worldcup.py` | Train XGBoost + predict World Cup | ~20-30s |
| `python run_pipeline.py` | Full pipeline (download → train → predict) | ~30-60s |
| `python run_pipeline.py --lightweight` | Predict only (skip download + retrain) | ~5-10s |
| `python run_dashboard.py` | Launch Streamlit dashboard | ~3s |
| `python today_value_bets_live.py` | Live value bets from The Odds API | ~10-20s |
| `python run_backtest.py` | Historical backtest simulation | ~10-30s |

---

## Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| `No module named pandas` | Activate virtual environment → `pip install -r requirements.txt` |
| `File not found: data/raw/worldcup_all.csv` | Run `python collect_all_worldcups.py` |
| SSL certificate errors | Add `--trusted-host pypi.org --trusted-host files.pythonhosted.org` |
| MinGW Python (Inkscape bundled) | Use Microsoft Store Python instead: `\"/c/Users/dell/AppData/Local/Microsoft/WindowsApps/python3.exe\" -m venv .venv` |
| Slow performance | Use `--lightweight` or `--skip-train` flags |

---

## 🧩 Obsidian Plugins for Developer Workflows

> Enhance your vault experience with these community plugins. Install via **Settings → Community Plugins → Browse** (turn off Restricted Mode first).

| Plugin | ID (search this) | Why It's Useful for This Vault |
|--------|-----------------|--------------------------------|
| **Code Styler** | `code-styler` | Enhanced code blocks with line numbers, language badges, file references, and custom styling per language. Makes the Python snippets in this vault much more readable. |
| **Code Link** | `code-link` | **The key plugin for this vault.** Makes `[[wikilinks]]` resolve to actual `.py` source files instead of just companion notes. Supports function/class-level links like `[[ensemble.py#EnsembleModel]]`. See full guide: [[Code Link Plugin Setup]] |
| **Execute Code** | `execute-code` | Run Python/Shell code directly inside notes. Great for testing small ML snippets or plotting quick charts without leaving Obsidian. ⚠ Sandboxes execution — safe for exploratory work. |
| **Obsidian Git** | `obsidian-git` | Automatic version control for the vault. Commits, pushes, and pulls your vault notes to a Git repo — keeps documentation in sync with the codebase. |
| **Obsidian Linter** | `obsidian-linter` | Auto-formats YAML frontmatter, tags, and wikilinks across the vault. Keeps the `tags:`, `created:`, and `See also:` metadata consistent across all 39+ notes in this vault. |
| **Tag Wrangler** | `tag-wrangler` | Bulk-rename, manage, and search tags across all vault notes. Useful for the `python-module` and `football-prediction` tags used throughout. |

### Quick Install

```markdown
1. Open Obsidian → Settings (Ctrl+,)
2. Go to Community Plugins → Turn off Restricted Mode
3. Click Browse → search the ID above (e.g. "code-styler")
4. Click Install → Enable
```

> **💡 Tip:** After installing **Code Styler**, configure it under its settings tab to enable line numbers and file references. For **Obsidian Git**, set an auto-commit interval (e.g. every 30 minutes) under its settings.

> **⚠ Security:** Community plugins are third-party software. Review each plugin's GitHub page before installing, especially ones that execute code (`execute-code`) or access the file system (`obsidian-git`).

---

## Next Steps

- [[Architecture Overview]] — understand how the system fits together
- [[Ensemble Model]] — the default prediction model
- [[Scripts Reference]] — complete list of all scripts
