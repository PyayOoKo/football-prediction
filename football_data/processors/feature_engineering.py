"""
Feature engineering — out of scope for the data collection layer.

This module exists as a placeholder for ML feature engineering
that operates on the collected data AFTER it's been stored in
the database.

Features to implement downstream:
- Rolling team form (last 5, 10 matches)
- Head-to-head records
- Home/away performance splits
- Elo ratings
- League strength indicators
- Schedule congestion metrics
- Weather-adjusted expected goals

The data collection system's job is to provide clean, validated
match data in the SQLite database. Feature engineering operates
on that data and is handled by the main pipeline (see src/).
"""
