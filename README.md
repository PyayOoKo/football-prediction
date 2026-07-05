# ⚽ Football Match Outcome Prediction

A modular, production-oriented machine learning pipeline that predicts the outcome of football (soccer) matches — **home win**, **draw**, or **away win**.

---

## 📁 Project Structure

```
football_prediction/
├── data/               # Raw, processed & external datasets
│   ├── raw/            #   Original CSV / API dumps (gitignored)
│   ├── processed/      #   Cleaned, feature-engineered data  (gitignored)
│   └── external/       #   Third-party reference data        (gitignored)
├── models/             # Serialised trained models           (gitignored)
├── notebooks/          # Jupyter notebooks for EDA & prototyping
├── src/                # Source package
│   ├── data_loader.py          # Data ingestion (CSV / API / DB)
│   ├── feature_engineering.py  # Rolling averages, H2H, encodings
│   ├── train.py                # Model training & cross-validation
│   ├── predict.py              # Match-outcome prediction
│   └── evaluate.py             # Metrics, plots, reports
├── app/                # Future web / CLI interface
├── config.py           # Centralised configuration (dataclasses)
├── setup.py            # Package installer
├── requirements.txt    # Pinned dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & enter the project

```bash
git clone https://github.com/yourusername/football-prediction.git
cd football_prediction
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For development extras (testing, linting):

```bash
pip install -e ".[dev]"
```

For deep learning support (PyTorch):

```bash
pip install -e ".[deep]"
```

---

## 🔧 Configuration

All settings live in **`config.py`** under typed dataclasses:

| Config       | Controls                                  |
|--------------|-------------------------------------------|
| `Paths`      | Directory structure                       |
| `DataConfig` | Data source, file names, split ratios     |
| `FeatureConfig` | Rolling windows, H2H, encoding        |
| `TrainConfig`   | Model type, hyper-parameters, CV folds |
| `PredictConfig` | Threshold, output format               |
| `EvalConfig`    | Metrics, plots to generate             |

Import the pre-built singleton:

```python
from config import config
config.train.learning_rate = 0.03   # override
```

---

## 📊 Pipeline Overview

```
┌──────────┐    ┌──────────────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐
│  Load    │───▶│  Feature         │───▶│  Train   │───▶│  Predict  │───▶│ Evaluate │
│  Data    │    │  Engineering     │    │  Model   │    │  Fixtures │    │  Report  │
└──────────┘    └──────────────────┘    └──────────┘    └───────────┘    └──────────┘
```

1. **Data Loader** — reads match results & fixtures from CSV, API, or database.
2. **Feature Engineering** — creates rolling averages, head-to-head stats, league-position features, and encodes categoricals.
3. **Training** — trains a configurable model (Random Forest, XGBoost, LightGBM, or Neural Network) with cross-validation.
4. **Prediction** — generates win/draw/loss probabilities for upcoming fixtures.
5. **Evaluation** — computes accuracy, precision, recall, F1, ROC-AUC, and saves visualisation plots.

---

## 🧪 Example Usage

```python
from src.data_loader import load_results, load_fixtures
from src.feature_engineering import build_features, train_val_test_split
from src.train import train_model
from src.predict import predict_fixtures
from src.evaluate import evaluate_model

# 1. Load
results = load_results()
fixtures = load_fixtures()

# 2. Feature engineering
X, y = build_features(results)
splits = train_val_test_split(X, y)

# 3. Train
model, history = train_model(splits["X_train"], splits["y_train"],
                              splits["X_val"], splits["y_val"])

# 4. Evaluate
report = evaluate_model(model, splits["X_test"], splits["y_test"])
print(report["classification_report"])

# 5. Predict
predictions = predict_fixtures(model, fixtures)
```

---

## 🧹 Development

### Code quality

```bash
# Format
black src/ app/ config.py

# Lint
ruff check src/ app/ config.py

# Type-check
mypy src/ app/ config.py

# Tests
pytest -v --cov=src
```

---

## 📈 Roadmap

- [ ] Add a real dataset (e.g. [Football-Data.org](https://www.football-data.org/) API)
- [ ] Implement rolling-average & H2H feature logic
- [ ] Add hyper-parameter tuning (Optuna / GridSearchCV)
- [ ] Build a Streamlit / FastAPI web dashboard
- [ ] Support multi-league & live odds comparison

---

## 📄 License

MIT — feel free to use, modify, and share.
