"""
Data Collection — download, clean, and store football match data.

Exposes the top-level orchestrator::

    from src.data_collection import collect_all, update
    collect_all()   # Full historical download
    update()        # Incremental update of new matches
"""

from __future__ import annotations

from src.data_collection.collector import (
    collect_all,
    collect_league,
    collect_worldcup,
    list_worldcup_teams,
    list_worldcup_groups,
    update,
)

__all__ = [
    "collect_all",
    "collect_league",
    "collect_worldcup",
    "list_worldcup_teams",
    "list_worldcup_groups",
    "update",
]
