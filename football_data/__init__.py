"""
football_data — Production-grade football data collection system.

Collects, cleans, validates, and stores football match data from
free and legal sources for machine learning and betting models.

Target leagues
-------------
- Sweden Superettan (SE1)
- Norway OBOS-ligaen (NO2)
- Finland Ykkösliiga (FI2)
- Ireland Premier Division (IRL)
- Denmark 1st Division (D2)
- Poland I Liga (P1)

Data sources
-----------
- football-data.co.uk — Historical fixtures, results, closing odds
- FBref — Team & player statistics (manual browser-save mode)
- Open-Meteo — Free weather data (no API key required)
"""

__version__ = "1.0.0"
__all__ = ["config"]
