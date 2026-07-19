"""
Data Collection Service — orchestrates data download and ingestion.

Handles downloading match data from various sources (World Cup, leagues, players),
normalizing schemas, and storing in the database or CSV files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.di_container import ConfigProvider, get_container

logger = logging.getLogger(__name__)


class DataCollectionService:
    """Service for collecting and ingesting football data.

    Parameters
    ----------
    config : ConfigProvider, optional
        Config provider for dependency injection. Defaults to the
        global container's ConfigProvider.
    """

    def __init__(self, config: ConfigProvider | None = None) -> None:
        self._config = config or get_container().resolve(ConfigProvider)
        self._data_dir = self._config.paths.raw
        self._data_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────

    def collect_worldcup(self, save: bool = True) -> pd.DataFrame:
        """Collect World Cup match data from 2002-2026.

        Downloads data from openfootball/worldcup.json repository,
        normalizes to standard schema, and optionally saves to CSV.

        Parameters
        ----------
        save : bool
            Whether to save the combined dataset to CSV (default True).

        Returns
        -------
        pd.DataFrame
            Combined World Cup match data.
        """
        logger.info("Collecting World Cup data...")
        
        try:
            from collect_all_worldcups import main as wc_main
            # We need to refactor this to not rely on script-level main
            # For now, we'll call the script's main function
            wc_main()
            logger.info("World Cup data collection completed")
            
            # Load the saved data
            output_path = self._data_dir / "worldcup_all.csv"
            if output_path.exists():
                df = pd.read_csv(output_path, low_memory=False)
                logger.info(f"Loaded {len(df)} World Cup matches")
                return df
            else:
                logger.warning("World Cup data file not found after collection")
                return pd.DataFrame()
                
        except Exception as exc:
            logger.error(f"World Cup data collection failed: {exc}")
            raise

    def collect_leagues(self, save: bool = True) -> pd.DataFrame:
        """Collect league match data.

        Downloads data from various league sources and normalizes to standard schema.

        Parameters
        ----------
        save : bool
            Whether to save the dataset to CSV (default True).

        Returns
        -------
        pd.DataFrame
            League match data.
        """
        logger.info("Collecting league data...")
        
        try:
            from collect_leagues import main as league_main
            league_main()
            logger.info("League data collection completed")
            
            # Try to load the most recent league data
            candidates = [
                self._data_dir / "leagues.csv",
                self._data_dir / "league_data.csv",
            ]
            for path in candidates:
                if path.exists():
                    df = pd.read_csv(path, low_memory=False)
                    logger.info(f"Loaded {len(df)} league matches")
                    return df
                    
            logger.warning("League data file not found after collection")
            return pd.DataFrame()
            
        except Exception as exc:
            logger.error(f"League data collection failed: {exc}")
            raise

    def collect_players(self, save: bool = True) -> pd.DataFrame:
        """Collect player statistics data.

        Downloads player information and match statistics.

        Parameters
        ----------
        save : bool
            Whether to save the dataset to CSV (default True).

        Returns
        -------
        pd.DataFrame
            Player data.
        """
        logger.info("Collecting player data...")
        
        try:
            from collect_player_data import main as player_main
            player_main()
            logger.info("Player data collection completed")
            
            # Try to load player data
            candidates = [
                self._data_dir / "players.csv",
                self._data_dir / "player_data.csv",
            ]
            for path in candidates:
                if path.exists():
                    df = pd.read_csv(path, low_memory=False)
                    logger.info(f"Loaded {len(df)} player records")
                    return df
                    
            logger.warning("Player data file not found after collection")
            return pd.DataFrame()
            
        except Exception as exc:
            logger.error(f"Player data collection failed: {exc}")
            raise

    def collect_all(self, sources: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """Collect data from multiple sources.

        Parameters
        ----------
        sources : list[str], optional
            List of sources to collect. Options: 'worldcup', 'leagues', 'players'.
            If None, collects all available sources.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping source name to collected DataFrame.
        """
        if sources is None:
            sources = ['worldcup', 'leagues', 'players']
        
        results = {}
        
        for source in sources:
            try:
                if source == 'worldcup':
                    results['worldcup'] = self.collect_worldcup()
                elif source == 'leagues':
                    results['leagues'] = self.collect_leagues()
                elif source == 'players':
                    results['players'] = self.collect_players()
                else:
                    logger.warning(f"Unknown source: {source}")
            except Exception as exc:
                logger.error(f"Failed to collect {source}: {exc}")
                results[source] = pd.DataFrame()
        
        return results
