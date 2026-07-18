#!/usr/bin/env python3
"""
Football Prediction Desktop App — CustomTkinter interface for match prediction,
fixture importing, value betting analysis, and bet tracking.

Usage:
    python -m app.main
    python app/main.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import customtkinter as ctk
import pandas as pd

# Ensure project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import config

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# ── Theme configuration ───────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ── Window dimensions ─────────────────────────────────────
WINDOW_TITLE = "⚽ Football Prediction System"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 820
MIN_WIDTH = 960
MIN_HEIGHT = 640


class FootballApp(ctk.CTk):
    """Main application window for the Football Prediction Desktop App."""

    def __init__(self) -> None:
        super().__init__()

        # ── Window setup ──────────────────────────────────────
        self.title(WINDOW_TITLE)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(MIN_WIDTH, MIN_HEIGHT)

        # Center on screen
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - WINDOW_WIDTH) // 2
        y = (sh - WINDOW_HEIGHT) // 2
        self.geometry(f"+{x}+{y}")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)

        # Bind close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Build UI ──────────────────────────────────────────
        self._build_header()
        self._build_tab_view()
        self._build_status_bar()

        # ── Keyboard shortcuts ────────────────────────────────
        self.bind("<Control-p>", lambda e: self._switch_tab("Predictions"))
        self.bind("<Control-i>", lambda e: self._switch_tab("Import"))
        self.bind("<Control-h>", lambda e: self._switch_tab("History"))
        self.bind("<Control-q>", lambda e: self._on_close())

        # Initialise data
        self.shared_data: pd.DataFrame | None = None
        self.shared_data_source: str = ""

        # ── Auto-load default dataset via pipeline ───────────
        self._auto_load_default_data()

    # ── Startup data loading ────────────────────────────

    def _auto_load_default_data(self) -> None:
        """Pre-load the default match dataset on startup.

        Uses ``load_and_prepare()`` from the services layer to run
        the full data pipeline (loader → cleaner → preprocessor).
        If no data is found, the app starts with an empty state
        and the user can import data via the Import tab.
        """
        try:
            from src.services import load_and_prepare

            df = load_and_prepare(add_temporal=True)
            if df is not None and len(df) > 0:
                self.shared_data = df
                self.shared_data_source = "Default dataset (auto-loaded)"
                # Use _on_data_loaded so PredictionTab.get teams populated too
                self._on_data_loaded(df, "Default dataset (auto-loaded)")
                logger.info(
                    "Auto-loaded default dataset: %d rows",
                    len(df),
                )
            else:
                logger.info("No default dataset found — app starts empty")
        except Exception as exc:
            logger.info(
                "Default data auto-load skipped: %s — use Import tab to load data manually",
                exc,
            )

    # ── UI Components ────────────────────────────────────────

    def _build_header(self) -> None:
        """Build the app header with logo and title."""
        header = ctk.CTkFrame(self, fg_color="#1a1d27", height=60, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        header.grid_propagate(False)

        # Logo / title
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=0, padx=(20, 0), pady=8)

        ctk.CTkLabel(
            title_frame, text="⚽",
            font=ctk.CTkFont(size=28),
        ).pack(side="left", padx=(0, 8))

        ctk.CTkLabel(
            title_frame, text="Football Predictor",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(side="left")

        # Version badge
        version_badge = ctk.CTkFrame(header, fg_color="#2d2d4e", corner_radius=8)
        version_badge.grid(row=0, column=2, padx=(0, 20), pady=8)
        ctk.CTkLabel(
            version_badge, text="v2.0",
            font=ctk.CTkFont(size=11),
            text_color="#7c7cae",
        ).pack(padx=8, pady=2)

    def _build_tab_view(self) -> None:
        """Build the tabbed interface."""
        self.tab_view = ctk.CTkTabview(
            self,
            fg_color="#121218",
            segmented_button_fg_color="#1a1d27",
            segmented_button_selected_color="#7c3aed",
            segmented_button_selected_hover_color="#6d28d9",
            segmented_button_unselected_color="#1a1d27",
            segmented_button_unselected_hover_color="#2a2d3e",
            text_color="#ffffff",
            corner_radius=8,
        )
        self.tab_view.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        # ── Tab 1: Predictions ────────────────────────────────
        self.tab_view.add("🔮 Predictions")
        self.prediction_tab = self._create_prediction_tab()

        # ── Tab 2: Import ─────────────────────────────────────
        self.tab_view.add("📂 Import")
        self.import_tab = self._create_import_tab()

        # ── Tab 3: History ────────────────────────────────────
        self.tab_view.add("📋 History")
        self.history_tab = self._create_history_tab()

    def _create_prediction_tab(self) -> Any:
        """Create and return the Prediction tab."""
        from app.prediction_tab import PredictionTab

        tab = self.tab_view.tab("🔮 Predictions")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        predictor = PredictionTab(tab)
        predictor.grid(row=0, column=0, sticky="nsew")
        return predictor

    def _create_import_tab(self) -> Any:
        """Create and return the Import tab."""
        from app.import_tab import ImportTab

        tab = self.tab_view.tab("📂 Import")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        importer = ImportTab(tab)
        importer.grid(row=0, column=0, sticky="nsew")
        importer.on_data_loaded = self._on_data_loaded
        return importer

    def _create_history_tab(self) -> Any:
        """Create and return the History tab."""
        from app.history_tab import HistoryTab

        tab = self.tab_view.tab("📋 History")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        hist = HistoryTab(tab)
        hist.grid(row=0, column=0, sticky="nsew")
        return hist

    # ── Status Bar ──────────────────────────────────────────

    def _build_status_bar(self) -> None:
        """Build the bottom status bar."""
        self.status_bar = ctk.CTkFrame(self, fg_color="#1a1d27", height=32, corner_radius=0)
        self.status_bar.grid(row=2, column=0, sticky="ew")
        self.status_bar.grid_columnconfigure(1, weight=1)
        self.status_bar.grid_propagate(False)

        self.status_label = ctk.CTkLabel(
            self.status_bar, text="✓ Ready",
            font=ctk.CTkFont(size=11), text_color="gray",
        )
        self.status_label.grid(row=0, column=0, padx=(15, 0), pady=4)

        self.data_status = ctk.CTkLabel(
            self.status_bar, text="📁 No data loaded",
            font=ctk.CTkFont(size=11), text_color="gray",
        )
        self.data_status.grid(row=0, column=1, padx=10, pady=4, sticky="w")

        # Hours of operation
        self.time_status = ctk.CTkLabel(
            self.status_bar, text="",
            font=ctk.CTkFont(size=10), text_color="#555",
        )
        self.time_status.grid(row=0, column=2, padx=(0, 15), pady=4)

    # ── Data Sharing ────────────────────────────────────────

    def _on_data_loaded(self, df: pd.DataFrame, source: str) -> None:
        """Callback when data is loaded from the Import tab."""
        self.shared_data = df
        self.shared_data_source = source
        self.data_status.configure(text=f"📁 {source} — {len(df)} rows")
        self.set_status(f"Loaded {len(df)} rows from {source}")

        # Update prediction tab with teams
        if hasattr(self, "prediction_tab") and self.prediction_tab:
            self.prediction_tab.set_data(df)

    def set_status(self, message: str) -> None:
        """Update the status bar message."""
        self.status_label.configure(text=message)

    # ── Tab Switching ───────────────────────────────────────

    _TAB_MAP: dict[str, str] = {
        "predictions": "🔮 Predictions",
        "import": "📂 Import",
        "history": "📋 History",
    }

    def _switch_tab(self, tab_name: str) -> None:
        """Switch to a tab by name."""
        tab_key = self._TAB_MAP.get(tab_name.lower())
        if tab_key:
            try:
                self.tab_view.set(tab_key)
            except Exception:
                pass

    # ── Event Handlers ─────────────────────────────────────

    def _on_close(self) -> None:
        """Clean up and close the application."""
        logger.info("Shutting down Football Prediction Desktop App")
        self.destroy()

    def start(self) -> None:
        """Start the application main loop."""
        logger.info("Starting Football Prediction Desktop App")
        self.set_status("✓ Ready — select a tab to begin")
        self.mainloop()


# ── Entry Point ─────────────────────────────────────────────

def main() -> None:
    """Launch the desktop application."""
    app = FootballApp()
    app.start()


if __name__ == "__main__":
    main()
