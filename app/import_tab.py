"""
Import Tab — import fixtures from CSV, JSON, manual entry, or API.
"""

from __future__ import annotations

import json
import logging
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk
import pandas as pd

from config import config

logger = logging.getLogger(__name__)


class ImportTab(ctk.CTkFrame):
    """Tab for importing fixture data from multiple sources."""

    def __init__(self, master: Any, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.imported_data: pd.DataFrame | None = None
        self.on_data_loaded: Any = None  # Callback(dataframe, source_label)
        self._build_ui()

    # ── UI Build ─────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        # ── Header ──────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        ctk.CTkLabel(
            header, text="📂 Import Fixtures",
            font=ctk.CTkFont(size=28, weight="bold"),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header, text="Load match data from CSV, JSON, manual entry, or the live API",
            font=ctk.CTkFont(size=13), text_color="gray",
        ).grid(row=1, column=0, sticky="w")

        # ── Scrollable content ──────────────────────────────
        self.content = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.content.grid_columnconfigure(0, weight=1)

        self._build_csv_section()
        self._build_json_section()
        self._build_manual_section()
        self._build_api_section()
        self._build_preview_section()

    # ── CSV Import ───────────────────────────────────────────

    def _build_csv_section(self) -> None:
        card = ctk.CTkFrame(self.content, fg_color="#1e1e2e", corner_radius=12)
        card.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            card, text="📄 Import from CSV",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=20, pady=(15, 5))

        ctk.CTkLabel(
            card, text="Expected columns: date, home_team, away_team, home_goals, away_goals",
            font=ctk.CTkFont(size=12), text_color="gray",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=20, pady=(0, 10))

        self.csv_path_var = ctk.StringVar(value="data/raw/worldcup_all.csv")
        ctk.CTkEntry(
            card, textvariable=self.csv_path_var,
            placeholder_text="Path to CSV file...",
            font=ctk.CTkFont(size=13),
        ).grid(row=2, column=0, columnspan=2, sticky="ew", padx=(20, 10), pady=(0, 15))

        btn_frame = ctk.CTkFrame(card, fg_color="transparent")
        btn_frame.grid(row=2, column=2, sticky="ew", padx=(0, 20), pady=(0, 15))

        ctk.CTkButton(
            btn_frame, text="📁 Browse", width=100,
            font=ctk.CTkFont(size=12), fg_color="#2a2a3e", hover_color="#3a3a5e",
            command=self._browse_csv,
        ).grid(row=0, column=0, padx=(0, 8))

        ctk.CTkButton(
            btn_frame, text="⬇ Load CSV", width=100,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#7c3aed", hover_color="#6d28d9",
            command=self._load_csv,
        ).grid(row=0, column=1)

    def _browse_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(config.paths.raw),
        )
        if path:
            self.csv_path_var.set(path)

    def _load_csv(self) -> None:
        path = self.csv_path_var.get().strip()
        if not path:
            messagebox.showerror("Error", "Please select a CSV file first.")
            return
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            self.imported_data = df
            self._show_preview(df, f"CSV: {Path(path).name}")
            if self.on_data_loaded:
                self.on_data_loaded(df, f"CSV: {Path(path).name}")
            logger.info("Loaded CSV with %d rows × %d cols from %s", *df.shape, path)
        except Exception as exc:
            messagebox.showerror("CSV Load Error", str(exc))
            logger.error("CSV load failed: %s", exc)

    # ── JSON Import ──────────────────────────────────────────

    def _build_json_section(self) -> None:
        card = ctk.CTkFrame(self.content, fg_color="#1e1e2e", corner_radius=12)
        card.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            card, text="📋 Import from JSON",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=20, pady=(15, 5))

        ctk.CTkLabel(
            card, text="Supports JSON arrays of match objects or keyed objects",
            font=ctk.CTkFont(size=12), text_color="gray",
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=20, pady=(0, 10))

        self.json_path_var = ctk.StringVar(value="reports/predictions_worldcup/worldcup_predictions.json")
        ctk.CTkEntry(
            card, textvariable=self.json_path_var,
            placeholder_text="Path to JSON file...",
            font=ctk.CTkFont(size=13),
        ).grid(row=2, column=0, columnspan=2, sticky="ew", padx=(20, 10), pady=(0, 15))

        jbtn_frame = ctk.CTkFrame(card, fg_color="transparent")
        jbtn_frame.grid(row=2, column=2, sticky="ew", padx=(0, 20), pady=(0, 15))

        ctk.CTkButton(
            jbtn_frame, text="📁 Browse", width=100,
            font=ctk.CTkFont(size=12), fg_color="#2a2a3e", hover_color="#3a3a5e",
            command=self._browse_json,
        ).grid(row=0, column=0, padx=(0, 8))

        ctk.CTkButton(
            jbtn_frame, text="⬇ Load JSON", width=100,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#7c3aed", hover_color="#6d28d9",
            command=self._load_json,
        ).grid(row=0, column=1)

    def _browse_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Select JSON File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.json_path_var.set(path)

    def _load_json(self) -> None:
        path = self.json_path_var.get().strip()
        if not path:
            messagebox.showerror("Error", "Please select a JSON file first.")
            return
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Try common keys: "matches", "fixtures", "predictions", "data"
                for key in ("matches", "fixtures", "predictions", "data", "results"):
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break
                else:
                    data = [data]
            df = pd.DataFrame(data)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            self.imported_data = df
            self._show_preview(df, f"JSON: {Path(path).name}")
            if self.on_data_loaded:
                self.on_data_loaded(df, f"JSON: {Path(path).name}")
            logger.info("Loaded JSON with %d rows × %d cols from %s", *df.shape, path)
        except Exception as exc:
            messagebox.showerror("JSON Load Error", str(exc))
            logger.error("JSON load failed: %s", exc)

    # ── Manual Entry ─────────────────────────────────────────

    def _build_manual_section(self) -> None:
        card = ctk.CTkFrame(self.content, fg_color="#1e1e2e", corner_radius=12)
        card.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        card.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(
            card, text="✏️ Manual Fixture Entry",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=20, pady=(15, 10))

        # Row 1: Home team, Away team
        ctk.CTkLabel(card, text="🏠 Home Team", font=ctk.CTkFont(size=12),
                     text_color="gray").grid(row=1, column=0, sticky="w", padx=(20, 5))
        ctk.CTkLabel(card, text="✈️ Away Team", font=ctk.CTkFont(size=12),
                     text_color="gray").grid(row=1, column=1, sticky="w", padx=(5, 5))

        self.manual_home = ctk.CTkEntry(
            card, placeholder_text="e.g. Brazil", font=ctk.CTkFont(size=13),
        )
        self.manual_home.grid(row=2, column=0, sticky="ew", padx=(20, 5), pady=(0, 10))

        self.manual_away = ctk.CTkEntry(
            card, placeholder_text="e.g. Argentina", font=ctk.CTkFont(size=13),
        )
        self.manual_away.grid(row=2, column=1, sticky="ew", padx=(5, 5), pady=(0, 10))

        # Row 2: Date, Competition
        ctk.CTkLabel(card, text="📅 Date", font=ctk.CTkFont(size=12),
                     text_color="gray").grid(row=3, column=0, sticky="w", padx=(20, 5))
        ctk.CTkLabel(card, text="🏆 Competition", font=ctk.CTkFont(size=12),
                     text_color="gray").grid(row=3, column=1, sticky="w", padx=(5, 5))

        self.manual_date = ctk.CTkEntry(
            card, placeholder_text="2026-07-15", font=ctk.CTkFont(size=13),
        )
        self.manual_date.grid(row=4, column=0, sticky="ew", padx=(20, 5), pady=(0, 15))

        self.manual_comp = ctk.CTkEntry(
            card, placeholder_text="World Cup 2026", font=ctk.CTkFont(size=13),
        )
        self.manual_comp.grid(row=4, column=1, sticky="ew", padx=(5, 5), pady=(0, 15))

        ctk.CTkButton(
            card, text="➕ Add Fixture",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#7c3aed", hover_color="#6d28d9",
            command=self._add_manual_fixture,
        ).grid(row=4, column=2, columnspan=2, sticky="ew", padx=(5, 20), pady=(0, 15))

        # Manual fixture list
        self.manual_list_frame = ctk.CTkFrame(card, fg_color="#2a2a3e", corner_radius=8)
        self.manual_list_frame.grid(row=5, column=0, columnspan=4, sticky="ew",
                                    padx=20, pady=(0, 15))
        self.manual_fixtures: list[dict[str, str]] = []

        ctk.CTkLabel(
            self.manual_list_frame, text="No fixtures added yet",
            font=ctk.CTkFont(size=12), text_color="gray",
        ).pack(pady=10)

        # Load button
        btn_frame = ctk.CTkFrame(card, fg_color="transparent")
        btn_frame.grid(row=6, column=0, columnspan=4, pady=(0, 15))
        ctk.CTkButton(
            btn_frame, text="✅ Load Manual Fixtures",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2d6a4f", hover_color="#1b4332",
            command=self._load_manual,
        ).pack()

    def _add_manual_fixture(self) -> None:
        home = self.manual_home.get().strip()
        away = self.manual_away.get().strip()
        if not home or not away:
            messagebox.showwarning("Warning", "Please enter both home and away teams.")
            return
        if home == away:
            messagebox.showwarning("Warning", "Teams must be different.")
            return

        fixture = {
            "home_team": home,
            "away_team": away,
            "date": self.manual_date.get().strip() or "TBD",
            "competition": self.manual_comp.get().strip() or "Unknown",
        }
        self.manual_fixtures.append(fixture)

        # Refresh list display
        for w in self.manual_list_frame.winfo_children():
            w.destroy()

        for i, fix in enumerate(self.manual_fixtures):
            f = ctk.CTkFrame(self.manual_list_frame, fg_color="transparent")
            f.pack(fill="x", padx=10, pady=2)
            ctk.CTkLabel(
                f, text=f"  {fix['home_team']} vs {fix['away_team']}  —  {fix['date']}",
                font=ctk.CTkFont(size=12),
            ).pack(side="left")
            ctk.CTkButton(
                f, text="✕", width=30,
                font=ctk.CTkFont(size=10),
                fg_color="#6b2020", hover_color="#8b3030",
                command=lambda i=i: self._remove_manual_fixture(i),
            ).pack(side="right", padx=5)

        self.manual_home.delete(0, "end")
        self.manual_away.delete(0, "end")

    def _remove_manual_fixture(self, index: int) -> None:
        if 0 <= index < len(self.manual_fixtures):
            self.manual_fixtures.pop(index)
            for w in self.manual_list_frame.winfo_children():
                w.destroy()
            if not self.manual_fixtures:
                ctk.CTkLabel(
                    self.manual_list_frame, text="No fixtures added yet",
                    font=ctk.CTkFont(size=12), text_color="gray",
                ).pack(pady=10)
            else:
                for i, fix in enumerate(self.manual_fixtures):
                    f = ctk.CTkFrame(self.manual_list_frame, fg_color="transparent")
                    f.pack(fill="x", padx=10, pady=2)
                    ctk.CTkLabel(
                        f, text=f"  {fix['home_team']} vs {fix['away_team']}  —  {fix['date']}",
                        font=ctk.CTkFont(size=12),
                    ).pack(side="left")
                    ctk.CTkButton(
                        f, text="✕", width=30,
                        font=ctk.CTkFont(size=10),
                        fg_color="#6b2020", hover_color="#8b3030",
                        command=lambda i=i: self._remove_manual_fixture(i),
                    ).pack(side="right", padx=5)

    def _load_manual(self) -> None:
        if not self.manual_fixtures:
            messagebox.showwarning("Warning", "Add at least one fixture first.")
            return
        df = pd.DataFrame(self.manual_fixtures)
        # Add placeholder columns
        for col in ["home_goals", "away_goals", "result"]:
            if col not in df.columns:
                df[col] = 0
        self.imported_data = df
        self._show_preview(df, f"Manual ({len(df)} fixtures)")
        if self.on_data_loaded:
            self.on_data_loaded(df, f"Manual ({len(df)} fixtures)")
        logger.info("Loaded %d manual fixtures", len(df))

    # ── API Configuration ────────────────────────────────────

    def _build_api_section(self) -> None:
        card = ctk.CTkFrame(self.content, fg_color="#1e1e2e", corner_radius=12)
        card.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        card.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(
            card, text="🌐 The Odds API",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=20, pady=(15, 5))

        ctk.CTkLabel(
            card, text="Fetch live odds for upcoming matches. Requires an API key from the-odds-api.com.",
            font=ctk.CTkFont(size=12), text_color="gray",
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=20, pady=(0, 10))

        ctk.CTkLabel(card, text="API Key:", font=ctk.CTkFont(size=12)).grid(
            row=2, column=0, sticky="w", padx=(20, 5))
        self.api_key_var = ctk.StringVar(value=os.environ.get("THE_ODDS_API_KEY", ""))
        ctk.CTkEntry(
            card, textvariable=self.api_key_var,
            placeholder_text="Enter your API key",
            font=ctk.CTkFont(size=13), show="*",
        ).grid(row=2, column=1, sticky="ew", padx=(5, 20), pady=(0, 10))

        ctk.CTkLabel(card, text="Sport:", font=ctk.CTkFont(size=12)).grid(
            row=3, column=0, sticky="w", padx=(20, 5))
        self.api_sport_var = ctk.StringVar(value="soccer_fifa_world_cup")
        ctk.CTkOptionMenu(
            card, variable=self.api_sport_var,
            values=["soccer_fifa_world_cup", "soccer_epl", "soccer_uefa_champs_league",
                    "soccer_la_liga", "soccer_serie_a", "soccer_bundesliga", "soccer_ligue_1"],
            font=ctk.CTkFont(size=12),
        ).grid(row=3, column=1, sticky="ew", padx=(5, 20), pady=(0, 15))

        ctk.CTkButton(
            card, text="🌐 Fetch Live Odds",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#7c3aed", hover_color="#6d28d9",
            command=self._fetch_api_odds,
        ).grid(row=4, column=0, columnspan=2, padx=20, pady=(0, 15))

    def _fetch_api_odds(self) -> None:
        """Fetch odds from The Odds API."""
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showerror("API Error", "Please enter your API key first.\nGet one at: the-odds-api.com")
            return
        try:
            import requests
            sport = self.api_sport_var.get()
            url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
            params = {
                "apiKey": api_key,
                "regions": "us,uk,eu",
                "markets": "h2h",
                "oddsFormat": "decimal",
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                messagebox.showinfo("No Data", "No upcoming matches found for this sport.")
                return

            rows = []
            for match in data:
                home_team = match.get("home_team", "Unknown")
                away_team = match.get("away_team", "Unknown")
                commence = match.get("commence_time", "")[:10]
                for bookmaker in match.get("bookmakers", []):
                    for market in bookmaker.get("markets", []):
                        outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                        rows.append({
                            "home_team": home_team,
                            "away_team": away_team,
                            "date": commence,
                            "odds_home": outcomes.get(home_team, 0),
                            "odds_draw": outcomes.get("Draw", 0),
                            "odds_away": outcomes.get(away_team, 0),
                            "sport": sport,
                            "source": "The Odds API",
                        })
            df = pd.DataFrame(rows)
            self.imported_data = df
            self._show_preview(df, f"API: {sport}")
            if self.on_data_loaded:
                self.on_data_loaded(df, f"API: {sport}")
            logger.info("Fetched %d matches from API", len(df))
        except ImportError:
            messagebox.showerror("Error", "requests library not installed. Run: pip install requests")
        except Exception as exc:
            messagebox.showerror("API Error", str(exc))
            logger.error("API fetch failed: %s", exc)

    # ── Preview ──────────────────────────────────────────────

    def _build_preview_section(self) -> None:
        card = ctk.CTkFrame(self.content, fg_color="#1e1e2e", corner_radius=12)
        card.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            card, text="👁 Preview",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(15, 5))

        self.preview_label = ctk.CTkLabel(
            card, text="No data loaded yet", font=ctk.CTkFont(size=12), text_color="gray",
        )
        self.preview_label.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 5))

        self.preview_frame = ctk.CTkFrame(card, fg_color="#2a2a3e", corner_radius=8)
        self.preview_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 15))

        ctk.CTkLabel(self.preview_frame, text="Load data to see preview",
                     font=ctk.CTkFont(size=12), text_color="gray").pack(pady=20)

    def _show_preview(self, df: pd.DataFrame, source: str) -> None:
        """Show a preview of the imported data."""
        self.preview_label.configure(
            text=f"📊 {source} — {len(df)} rows × {len(df.columns)} cols"
        )
        for w in self.preview_frame.winfo_children():
            w.destroy()

        cols = [c for c in df.columns if c in (
            "date", "home_team", "away_team", "home_goals", "away_goals",
            "result", "odds_home", "odds_draw", "odds_away", "competition",
        )]
        display = df[cols].head(20) if cols else df.head(20)

        text = ctk.CTkTextbox(self.preview_frame, height=200, font=ctk.CTkFont(size=11))
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert("0.0", display.to_string(index=False, max_rows=20))
        text.configure(state="disabled")

        if len(df) > 20:
            ctk.CTkLabel(
                self.preview_frame,
                text=f"... and {len(df) - 20} more rows",
                font=ctk.CTkFont(size=11, slant="italic"),
                text_color="gray",
            ).pack(pady=(0, 10))

        # Load into prediction tab
        ctk.CTkButton(
            self.preview_frame, text="🚀 Use This Data for Predictions",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2d6a4f", hover_color="#1b4332",
            command=lambda: self._notify_data_loaded(df, source),
        ).pack(pady=(0, 10))

    def _notify_data_loaded(self, df: pd.DataFrame, source: str) -> None:
        """Notify listeners that data was loaded."""
        if self.on_data_loaded:
            self.on_data_loaded(df, source)
        messagebox.showinfo("Data Loaded", f"{len(df)} rows loaded from {source}")
