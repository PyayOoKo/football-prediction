# Football Data Collection System

Production-grade data pipeline for collecting, cleaning, and storing football match data for machine learning models.

## Coverage

| League | Code | football-data.co.uk | FBref | Notes |
|---|---|---|---|---|
| 🇸🇪 Sweden Superettan | `SE1` | ✅ Auto | ✅ Manual | 5 seasons |
| 🇳🇴 Norway OBOS-ligaen | `NO2` | ✅ Auto | ✅ Manual | 5 seasons |
| 🇫🇮 Finland Ykkösliiga | `FI2` | ✅ Auto | ✅ Manual | 5 seasons |
| 🇮🇪 Ireland Premier Division | `IRL` | ✅ Auto | ✅ Manual | 5 seasons |
| 🇩🇰 Denmark 1st Division | `D2` | ✅ Auto | ✅ Manual | 5 seasons |
| 🇵🇱 Poland I Liga | `P1` | ✅ Auto | ✅ Manual | 5 seasons |

**Legend:** Auto = automatic download, Manual = browser-save required

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Collect all leagues (automatic — football-data.co.uk)
python -c "
import asyncio
from football_data.scheduler import DailyUpdater
asyncio.run(DailyUpdater.run_cli())
"

# Collect specific leagues only
python -c "
import asyncio
from football_data.scheduler import DailyUpdater
asyncio.run(DailyUpdater.run_cli(['SE1', 'NO2', 'FI2']))
"
```

## Data Structure

```
football_data/
├── collectors/
│   ├── football_data.py     # football-data.co.uk (auto CSV download)
│   ├── fbref.py             # FBref (browser-save parser)
│   ├── transfermarkt.py     # Transfermarkt (manual CSV import)
│   └── weather.py           # Open-Meteo weather (free, no API key)
├── processors/
│   ├── clean.py             # Data cleaning & team normalisation
│   └── validation.py        # Data validation & quality checks
├── database/
│   └── sqlite.py            # SQLite with auto-schema & WAL mode
├── scheduler/
│   └── update_daily.py      # Daily collection orchestrator
├── config.py                # All configuration
├── requirements.txt
└── README.md
```

## Data Sources

### 1. Football-Data.co.uk (AUTO — recommended)
Clean CSV files, updated weekly. No API key needed.
- Historical fixtures and results
- Closing odds (BetBrain average, Pinnacle, Bet365)
- Match statistics (shots, corners, cards)
- Up to 10 seasons per league

### 2. FBref (MANUAL — browser save)
For additional team/player statistics:
1. Visit FBref.com in Chrome
2. Navigate to your league's "Scores & Fixtures" page
3. Right-click → Save As → "Webpage, HTML only"
4. Save to: `football_data/data/fbref_pages/`
5. Run the parser

### 3. Open-Meteo Weather (AUTO — free, no API key)
Free historical weather data from 1940 onward.
- Temperature, humidity, precipitation
- Wind speed, pressure
- 10,000 requests/day free tier

### 4. Transfermarkt (MANUAL — CSV import)
For squad values, ages, and injury data:
1. Visit Transfermarkt.com
2. Copy table data into CSV
3. Save to: `football_data/data/transfermarkt_imports/`
4. Run the importer

## Database Schema

The SQLite database (`football_data.db`) has these tables:

| Table | Contents |
|---|---|
| `matches` | Core match results with odds and statistics |
| `team_stats` | Per-match team statistics |
| `player_stats` | Per-match player statistics |
| `injuries` | Player injury records |
| `weather` | Match weather conditions |
| `collection_log` | Pipeline run history |

## Scheduling

### Windows Task Scheduler
```cmd
schtasks /create /tn "FootballDataUpdate" /tr "python -m football_data.scheduler.update_daily" /sc daily /st 08:00
```

### Cron (Linux/Mac)
```bash
0 8 * * * cd /path/to/project && python -m football_data.scheduler.update_daily
```

## Validation

Every collected record is validated before insertion:

- Required fields check (date, teams, league)
- Date validity and range checks
- Goals consistency (result matches scoreline)
- Odds sanity checks (> 1.0 and < 100.0)
- Duplicate detection via (source, league, date, home, away)

## Adding New Leagues

1. Check if the league code exists on [football-data.co.uk](https://www.football-data.co.uk/data.php)
2. Add the code to `config.py` → `LEAGUE_NAMES` and `DEFAULT_LEAGUES`
3. Add FBref URL to `config.py` → `FBrefConfig.league_urls`
4. Add default coordinates to `weather.py` → `LEAGUE_DEFAULT_COORDS`
5. Run the pipeline

## Why Not Scrape FBref/Transfermarkt?

FBref, Transfermarkt, and Sofascore all block automated scraping with Cloudflare or prohibit it in their ToS. Rather than implementing fragile scrapers that violate ToS, this system uses:

- **football-data.co.uk** — Legitimate CSV downloads (designed for analysis)
- **Open-Meteo** — Legitimate free API (no API key needed)
- **Browser-save** — You download the page, we parse it (legal grey area avoided)

For a fully automated solution with these leagues, consider the paid **API-Football** subscription on RapidAPI (~$25/month for 500 requests/day).
