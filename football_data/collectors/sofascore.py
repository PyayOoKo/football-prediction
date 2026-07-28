"""
Sofascore data collector — NOT IMPLEMENTED.

Sofascore (https://www.sofascore.com) provides detailed match events,
starting lineups, formations, player ratings, and live statistics.

Why this collector is not included:
-----------------------------------
1. Sofascore does not provide an official public API.
2. Their Terms of Service prohibit automated scraping.
3. Their internal APIs are undocumented and change frequently.

For Sofascore-level data (lineups, formations, player ratings),
consider these alternatives:

- **FBref** (via browser-save mode) — Provides player ratings,
  match events, and detailed statistics for many leagues.
- **API-Football** (paid tier) — Official API with lineups,
  formations, player ratings, and match events for 100+ leagues.
- **Understat** (if available for the league) — xG data and match
  statistics.

The existing collectors already provide:
- football-data.co.uk → Match results, odds, basic stats (shots, cards)
- FBref → Extended team/player stats (via saved HTML pages)
- Open-Meteo → Weather data (free, no API key)
"""
