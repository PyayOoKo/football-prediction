"""
History Tab — track bets, record results, and view historical performance.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from typing import Any

import customtkinter as ctk
import pandas as pd

from config import config

logger = logging.getLogger(__name__)

HISTORY_FILE = config.paths.data / "bet_history.json"


class HistoryTab(ctk.CTkFrame):
    """Tab for tracking betting history and performance."""

    def __init__(self, master: Any, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.bets: list[dict[str, Any]] = []
        self._load_history()
        self._build_ui()

    # ── Persistence ──────────────────────────────────────────

    def _history_path(self) -> Path:
        return HISTORY_FILE

    def _load_history(self) -> None:
        path = self._history_path()
        if path.exists():
            try:
                with open(path) as f:
                    self.bets = json.load(f)
                logger.info("Loaded %d bets from history", len(self.bets))
            except Exception as exc:
                logger.warning("Failed to load bet history: %s", exc)
                self.bets = []

    def _save_history(self) -> None:
        path = self._history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.bets, f, indent=2, default=str)
        logger.info("Saved %d bets to history", len(self.bets))

    # ── UI Build ─────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        # ── Header ──────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        ctk.CTkLabel(
            header, text="📋 Bet History & Tracking",
            font=ctk.CTkFont(size=28, weight="bold"),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header, text="Record bets, track results, and monitor your P&L",
            font=ctk.CTkFont(size=13), text_color="gray",
        ).grid(row=1, column=0, sticky="w")

        # Refresh button
        ctk.CTkButton(
            header, text="🔄 Refresh", width=100,
            font=ctk.CTkFont(size=12), fg_color="#2a2a3e", hover_color="#3a3a5e",
            command=self._refresh_display,
        ).grid(row=0, column=1, sticky="e")

        # ── Scrollable content ──────────────────────────────
        self.content = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.content.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.content.grid_columnconfigure(0, weight=1)

        self._build_summary()
        self._build_new_bet_form()
        self._build_history_table()
        self._refresh_display()

    def _build_summary(self) -> None:
        """Build the P&L summary card."""
        card = ctk.CTkFrame(self.content, fg_color="#1e1e2e", corner_radius=12)
        card.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for i in range(4):
            card.grid_columnconfigure(i, weight=1)

        ctk.CTkLabel(
            card, text="📊 Performance Summary",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=20, pady=(15, 10))

        self.summary_labels: dict[str, ctk.CTkLabel] = {}
        metrics = ["Total Bets", "Won", "Lost", "Win Rate", "P&L", "ROI"]
        for i, metric in enumerate(metrics):
            row = 1 + i // 3
            col = i % 3
            f = ctk.CTkFrame(card, fg_color="#2a2a3e", corner_radius=8)
            f.grid(row=row, column=col, sticky="ew", padx=10, pady=(0, 15))
            f.grid_columnconfigure(0, weight=1)

            label = ctk.CTkLabel(f, text="—", font=ctk.CTkFont(size=22, weight="bold"))
            label.grid(row=0, column=0, pady=(10, 0))
            self.summary_labels[metric] = label

            ctk.CTkLabel(
                f, text=metric, font=ctk.CTkFont(size=11),
                text_color="gray",
            ).grid(row=1, column=0, pady=(0, 10))

    def _build_new_bet_form(self) -> None:
        """Build the new bet entry form."""
        card = ctk.CTkFrame(self.content, fg_color="#1e1e2e", corner_radius=12)
        card.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        card.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        ctk.CTkLabel(
            card, text="➕ Record a New Bet",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, columnspan=5, sticky="w", padx=20, pady=(15, 10))

        # Row 1: Labels
        for i, label in enumerate(["🏠 Home", "✈️ Away", "📊 Odds", "💰 Stake", "📅 Date"]):
            ctk.CTkLabel(card, text=label, font=ctk.CTkFont(size=11),
                         text_color="gray").grid(row=1, column=i, sticky="w", padx=8)

        # Row 2: Entries
        self.bet_home = ctk.CTkEntry(card, placeholder_text="Brazil",
                                     font=ctk.CTkFont(size=12))
        self.bet_home.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))

        self.bet_away = ctk.CTkEntry(card, placeholder_text="Argentina",
                                     font=ctk.CTkFont(size=12))
        self.bet_away.grid(row=2, column=1, sticky="ew", padx=8, pady=(0, 8))

        self.bet_odds = ctk.CTkEntry(card, placeholder_text="2.10",
                                     font=ctk.CTkFont(size=12))
        self.bet_odds.grid(row=2, column=2, sticky="ew", padx=8, pady=(0, 8))

        self.bet_stake = ctk.CTkEntry(card, placeholder_text="25.00",
                                      font=ctk.CTkFont(size=12))
        self.bet_stake.grid(row=2, column=3, sticky="ew", padx=8, pady=(0, 8))

        self.bet_date = ctk.CTkEntry(card, placeholder_text=datetime.now().strftime("%Y-%m-%d"),
                                     font=ctk.CTkFont(size=12))
        self.bet_date.grid(row=2, column=4, sticky="ew", padx=8, pady=(0, 8))

        # Row 3: Outcome + Button
        ctk.CTkLabel(card, text="Bet On:", font=ctk.CTkFont(size=11),
                     text_color="gray").grid(row=3, column=0, sticky="w", padx=8)
        self.bet_outcome = ctk.CTkOptionMenu(
            card, values=["Home Win", "Draw", "Away Win"],
            font=ctk.CTkFont(size=12),
        )
        self.bet_outcome.grid(row=3, column=1, sticky="ew", padx=8, pady=(0, 8))

        ctk.CTkLabel(card, text="Result:", font=ctk.CTkFont(size=11),
                     text_color="gray").grid(row=3, column=2, sticky="w", padx=8)
        self.bet_result = ctk.CTkOptionMenu(
            card, values=["Pending", "Won", "Lost", "Push"],
            font=ctk.CTkFont(size=12),
        )
        self.bet_result.grid(row=3, column=3, sticky="ew", padx=8, pady=(0, 8))

        ctk.CTkButton(
            card, text="✅ Add Bet",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#7c3aed", hover_color="#6d28d9",
            command=self._add_bet,
        ).grid(row=3, column=4, sticky="ew", padx=8, pady=(0, 15))

    def _build_history_table(self) -> None:
        """Build the scrollable bet history table."""
        card = ctk.CTkFrame(self.content, fg_color="#1e1e2e", corner_radius=12)
        card.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            card, text="📜 Bet History",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(15, 10))

        # Filter bar
        filter_frame = ctk.CTkFrame(card, fg_color="transparent")
        filter_frame.grid(row=0, column=0, sticky="e", padx=20, pady=(15, 10))

        self.filter_var = ctk.StringVar(value="All")
        ctk.CTkOptionMenu(
            filter_frame, variable=self.filter_var,
            values=["All", "Won", "Lost", "Pending", "Push"],
            font=ctk.CTkFont(size=11), width=100,
            command=lambda _: self._refresh_display(),
        ).grid(row=0, column=0, padx=(0, 8))

        ctk.CTkButton(
            filter_frame, text="🗑 Clear All", width=80,
            font=ctk.CTkFont(size=11), fg_color="#6b2020", hover_color="#8b3030",
            command=self._clear_history,
        ).grid(row=0, column=1)

        # Scrollable table
        self.table_frame = ctk.CTkScrollableFrame(card, fg_color="#2a2a3e", corner_radius=8)
        self.table_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 15))

    # ── Bet Operations ───────────────────────────────────────

    def _add_bet(self) -> None:
        """Add a new bet to history."""
        home = self.bet_home.get().strip()
        away = self.bet_away.get().strip()
        odds_str = self.bet_odds.get().strip()
        stake_str = self.bet_stake.get().strip()

        if not home or not away:
            messagebox.showwarning("Warning", "Please enter both teams.")
            return

        try:
            odds = float(odds_str)
            stake = float(stake_str)
        except ValueError:
            messagebox.showwarning("Warning", "Odds and Stake must be valid numbers.")
            return

        if odds < 1.0:
            messagebox.showwarning("Warning", "Odds must be at least 1.0.")
            return
        if stake <= 0:
            messagebox.showwarning("Warning", "Stake must be positive.")
            return

        date = self.bet_date.get().strip() or datetime.now().strftime("%Y-%m-%d")
        outcome = self.bet_outcome.get()
        result = self.bet_result.get()

        bet: dict[str, Any] = {
            "id": len(self.bets) + 1,
            "date": date,
            "home_team": home,
            "away_team": away,
            "bet_on": outcome,
            "odds": odds,
            "stake": stake,
            "result": result,
            "created_at": datetime.now().isoformat(),
        }

        # Calculate P&L
        if result == "Won":
            bet["profit"] = round(stake * (odds - 1), 2)
        elif result == "Lost":
            bet["profit"] = -stake
        elif result == "Push":
            bet["profit"] = 0.0
        else:
            bet["profit"] = 0.0  # Pending

        self.bets.append(bet)
        self._save_history()
        self._refresh_display()

        # Clear form
        self.bet_home.delete(0, "end")
        self.bet_away.delete(0, "end")
        self.bet_odds.delete(0, "end")
        self.bet_stake.delete(0, "end")

    def _delete_bet(self, bet_id: int) -> None:
        """Delete a bet by ID."""
        self.bets = [b for b in self.bets if b.get("id") != bet_id]
        self._save_history()
        self._refresh_display()

    def _clear_history(self) -> None:
        """Clear all bet history."""
        if messagebox.askyesno("Clear All", "Delete all bet history?\nThis cannot be undone."):
            self.bets = []
            self._save_history()
            self._refresh_display()

    # ── Display ──────────────────────────────────────────────

    def _refresh_display(self) -> None:
        """Refresh the summary and history table."""
        self._update_summary()
        self._render_history_table()

    def _update_summary(self) -> None:
        """Update the P&L summary metrics."""
        total = len(self.bets)
        won = sum(1 for b in self.bets if b.get("result") == "Won")
        lost = sum(1 for b in self.bets if b.get("result") == "Lost")
        pushed = sum(1 for b in self.bets if b.get("result") == "Push")
        settled = won + lost + pushed
        total_pnl = sum(b.get("profit", 0.0) for b in self.bets)
        total_stake = sum(b.get("stake", 0.0) for b in self.bets if b.get("result") != "Pending")

        win_rate = won / settled if settled > 0 else 0.0
        roi = (total_pnl / total_stake * 100) if total_stake > 0 else 0.0

        self.summary_labels["Total Bets"].configure(text=str(total))
        self.summary_labels["Won"].configure(text=str(won))
        self.summary_labels["Lost"].configure(text=str(lost))

        wr_color = "#4caf50" if win_rate >= 0.5 else "#f44336"
        self.summary_labels["Win Rate"].configure(
            text=f"{win_rate:.1%}", text_color=wr_color,
        )

        pnl_color = "#4caf50" if total_pnl >= 0 else "#f44336"
        self.summary_labels["P&L"].configure(
            text=f"£{total_pnl:+.2f}" if total_pnl >= 0 else f"-£{abs(total_pnl):.2f}",
            text_color=pnl_color,
        )

        roi_color = "#4caf50" if roi >= 0 else "#f44336"
        self.summary_labels["ROI"].configure(
            text=f"{roi:+.1f}%" if roi >= 0 else f"{roi:.1f}%",
            text_color=roi_color,
        )

    def _render_history_table(self) -> None:
        """Render the bet history table."""
        for w in self.table_frame.winfo_children():
            w.destroy()

        filtered = self._get_filtered_bets()

        if not filtered:
            ctk.CTkLabel(
                self.table_frame, text="No bets recorded yet. Add your first bet above!",
                font=ctk.CTkFont(size=13), text_color="gray",
            ).pack(pady=30)
            return

        # Table header
        cols = ["Date", "Home", "Away", "Bet On", "Odds", "Stake", "Result", "P&L", ""]
        widths = [100, 130, 130, 90, 70, 80, 80, 80, 40]

        hdr = ctk.CTkFrame(self.table_frame, fg_color="#1a1a2e", corner_radius=4)
        hdr.pack(fill="x", padx=5, pady=(5, 2))
        for j, (col, w) in enumerate(zip(cols, widths)):
            ctk.CTkLabel(
                hdr, text=col, font=ctk.CTkFont(size=10, weight="bold"),
                text_color="gray", width=w,
            ).pack(side="left", padx=3, pady=4)

        # Table rows
        for bet in reversed(filtered):
            result = bet.get("result", "Pending")
            profit = bet.get("profit", 0.0)

            bg = {"Won": "#1a3a1a", "Lost": "#3a1a1a", "Push": "#2a2a1a", "Pending": "transparent"}
            row_frame = ctk.CTkFrame(
                self.table_frame, fg_color=bg.get(result, "transparent"), corner_radius=2,
            )
            row_frame.pack(fill="x", padx=5, pady=1)

            values = [
                bet.get("date", "")[:10],
                bet.get("home_team", ""),
                bet.get("away_team", ""),
                bet.get("bet_on", ""),
                f"{bet.get('odds', 0):.2f}",
                f"£{bet.get('stake', 0):.2f}",
                result,
                f"+£{profit:.2f}" if profit > 0 else f"-£{abs(profit):.2f}" if profit < 0 else "£0.00",
            ]

            for j, (val, w) in enumerate(zip(values, widths)):
                c = None
                if j == 6:  # Result column
                    c = {"Won": "#4caf50", "Lost": "#f44336", "Push": "#ffc107"}.get(result)
                if j == 7:  # P&L column
                    c = "#4caf50" if profit > 0 else "#f44336" if profit < 0 else None

                ctk.CTkLabel(
                    row_frame, text=val, font=ctk.CTkFont(size=11),
                    text_color=c, width=w,
                ).pack(side="left", padx=3, pady=3)

            # Delete button
            bid = bet.get("id", 0)
            ctk.CTkButton(
                row_frame, text="✕", width=30,
                font=ctk.CTkFont(size=9),
                fg_color="#6b2020", hover_color="#8b3030",
                command=lambda b=bid: self._delete_bet(b),
            ).pack(side="right", padx=3, pady=2)

    def _get_filtered_bets(self) -> list[dict[str, Any]]:
        """Get bets filtered by the current filter selection."""
        filt = self.filter_var.get()
        if filt == "All":
            return self.bets
        return [b for b in self.bets if b.get("result") == filt]
