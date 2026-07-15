# Installation Guide — Football Prediction System

This guide covers installation methods for the Football Prediction System.
Choose the method that best suits your environment.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Install (pip)](#2-quick-install-pip)
3. [Developer Install (editable)](#3-developer-install-editable)
4. [Docker Install](#4-docker-install)
5. [Virtual Environment Setup](#5-virtual-environment-setup)
6. [Configuration](#6-configuration)
7. [Verifying the Installation](#7-verifying-the-installation)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.12+ | 3.13 also supported |
| **pip** | 24+ | Included with Python |
| **PostgreSQL** | 16+ | Optional — only needed for database features |
| **Docker** | 24+ | Optional — for containerized deployment |

**Optional ML dependencies:**
- **PyTorch** (for neural network model): `pip install torch`
- **FastAPI + uvicorn** (for REST API): `pip install fastapi uvicorn`
- **LightGBM** — installed automatically with the package

---

## 2. Quick Install (pip)

### From PyPI (when published)

```bash
pip install football-prediction
```

### From source

```bash
# Clone the repository
git clone https://github.com/yourusername/football-prediction.git
cd football-prediction

# Install core package
pip install .

# Or install with all extras
pip install .[all]
```

### Install with extras

```bash
# Core only (prediction, training, betting, dashboard)
pip install .

# With REST API support
pip install .[api]

# With deep learning (PyTorch)
pip install .[deep]

# With development tools
pip install .[dev]

# All features
pip install .[all]
```

---

## 3. Developer Install (editable)

For development work, install in editable mode so changes take effect immediately:

```bash
# Clone and enter the project directory
git clone https://github.com/yourusername/football-prediction.git
cd football-prediction

# Create virtual environment (recommended)
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify installation
football-predict --version
```

---

## 4. Docker Install

### Build the image

```bash
docker build -t football-prediction .
```

### Run containers

```bash
# Show help
docker run --rm football-prediction --help

# Train a model
docker run --rm -v "$(pwd)/models:/app/models" football-prediction train

# Launch the API server
docker run --rm -p 8000:8000 football-prediction api

# Launch the dashboard
docker run --rm -p 8501:8501 football-prediction dashboard

# Run the full prediction pipeline
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" football-prediction pipeline
```

### Using Docker Compose

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down
```

---

## 5. Virtual Environment Setup

### Windows (Command Prompt)

```batch
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 6. Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `THE_ODDS_API_KEY` | No | — | API key for The Odds API (live odds) |
| `FOOTBALL_DATA_API_KEY` | No | — | API key for football-data.org |
| `PREDICTION_API_KEY` | No | — | API key for the REST API endpoint |
| `DATABASE_URL` | No | — | PostgreSQL connection string |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `API_HOST` | No | `0.0.0.0` | Host to bind the API server |
| `API_PORT` | No | `8000` | Port for the API server |

### Configuration File

The system uses `config.py` at the project root. Key settings can be
overridden via environment variables or by creating a `.env` file:

```bash
# .env file
THE_ODDS_API_KEY=your_odds_api_key_here
DATABASE_URL=postgresql://user:pass@localhost:5432/football_db
LOG_LEVEL=DEBUG
```

---

## 7. Verifying the Installation

### Check the CLI

```bash
football-predict --version
# Expected output: football-predict v2.0.0

football-predict --help
# Shows available commands
```

### Run a quick test

```bash
# Download World Cup data
python collect_all_worldcups.py

# Train a model
python train_worldcup.py

# Generate predictions
python predict_worldcup.py
```

### Launch the dashboard

```bash
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

### Run the test suite

```bash
pytest tests/ -v
```

---

## 8. Troubleshooting

### "ModuleNotFoundError: No module named 'customtkinter'"

**Problem:** CustomTkinter is not installed.

**Solution:**
```bash
pip install customtkinter
```

### "ModuleNotFoundError: No module named 'dotenv'"

**Problem:** python-dotenv is not installed.

**Solution:**
```bash
pip install python-dotenv
```

### "No trained model found"

**Problem:** You tried to run predictions without training a model first.

**Solution:**
```bash
python collect_all_worldcups.py   # Download match data
python train_worldcup.py          # Train the model
```

### "PostgreSQL connection refused"

**Problem:** The database is not running or credentials are wrong.

**Solution:**
```bash
# Start PostgreSQL via Docker
docker compose up -d db

# Or check your DATABASE_URL environment variable
echo $DATABASE_URL
```

### "streamlit: command not found"

**Problem:** Streamlit is not on your PATH.

**Solution:**
```bash
# Activate your virtual environment first
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS/Linux

# Or run directly
python -m streamlit run dashboard/app.py
```

### "pip install fails with build errors"

**Problem:** Missing system build dependencies.

**Solution:**
```bash
# Debian/Ubuntu
sudo apt-get install python3-dev build-essential libpq-dev

# macOS (with Homebrew)
brew install postgresql

# Windows
# Use Microsoft Store Python 3.12+ (pre-compiled wheels available)
```

---

*For additional help, open an issue on GitHub or check the `docs/` directory.*
