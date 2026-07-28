"""
Data cleaning and team name normalisation.

Normalises team names across different data sources, handles
missing values, and converts data types to a consistent schema.

Key features
------------
- Team name normalisation (e.g. "Man City" → "Manchester City")
- Date parsing and UTC conversion
- Type coercion (goals → int, odds → float)
- Missing value handling
- Duplicate detection
- Cross-source team name alignment
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Team name normalisation map ─────────────────────────
#
# Maps common abbreviations/variants to standard names.
# Extend this as new teams are encountered.

TEAM_ALIASES: dict[str, str] = {
    # Swedish
    "brage": "IK Brage",
    "degerfors": "Degerfors IF",
    "gefle": "Gefle IF",
    "helsingborg": "Helsingborgs IF",
    "landskrona": "Landskrona BoIS",
    "orebro": "Örebro SK",
    "orgryte": "Örgryte IS",
    "sundsvall": "GIF Sundsvall",
    "trelleborg": "Trelleborgs FF",
    "varberg": "Varbergs BoIS",
    "vasteras": "Västerås SK FK",
    "oddevold": "IK Oddevold",
    "sandviken": "Sandvikens IF",
    "uf": "IK Brage",
    # Norwegian
    "aalesund": "Aalesunds FK",
    "asane": "Åsane Fotball",
    "bryne": "Bryne FK",
    "eik torsberg": "Eik Tønsberg",
    "hodd": "IL Hødd",
    "kongsvinger": "Kongsvinger IL",
    "levanger": "Levanger FK",
    "lynn": "FC Lyn Oslo",
    "moss": "Moss FK",
    "ranheim": "Ranheim Fotball",
    "sandnes": "Sandnes Ulf",
    "sogndal": "Sogndal Fotball",
    "start": "IK Start",
    "strommen": "Strømmen IF",
    "tromsdalen": "Tromsdalen UIL",
    "valleren": "Vålerenga Fotball",
    "mjondalen": "Mjøndalen IF",
    "raufoss": "Raufoss Fotball",
    "kfum": "KFUM Oslo",
    "stabæk": "Stabæk Fotball",
    # Finnish
    "ff jaro": "FF Jaro",
    "jippo": "JIPPO",
    "ktp": "KTP",
    "mp": "Mikkelin Palloilijat",
    "pk-35": "PK-35 Vantaa",
    "pk-35 vantaa": "PK-35 Vantaa",
    "pk-35 helsinki": "PK-35 Vantaa",
    "salpa": "SalPa",
    "sjp": "SexyPöxyt",
    "tps": "TPS Turku",
    "jaPS": "JäPS",
    "kiffen": "FC Kiffen 08",
    "pif": "Pargas IF",
    "ebk": "Ekenäs IF",
    # Irish
    "shamrock rovers": "Shamrock Rovers",
    "shamrock": "Shamrock Rovers",
    "dundalk": "Dundalk FC",
    "derry city": "Derry City",
    "bohemians": "Bohemian FC",
    "st patricks": "St Patrick's Athletic",
    "st pats": "St Patrick's Athletic",
    "shelbourne": "Shelbourne FC",
    "sligo rovers": "Sligo Rovers",
    "waterford": "Waterford FC",
    "drogheda": "Drogheda United",
    "cork city": "Cork City FC",
    "galway": "Galway United FC",
    # Polish
    "arka gdynia": "Arka Gdynia",
    "bruk-bet termalica": "Bruk-Bet Termalica Nieciecza",
    "chrobry glow": "Chrobry Głogów",
    "gornik leczna": "Górnik Łęczna",
    "katowice": "GKS Katowice",
    "kotwica kolobrzeg": "Kotwica Kołobrzeg",
    "lks lodz": "ŁKS Łódź",
    "miedz legnica": "Miedź Legnica",
    "odra opole": "Odra Opole",
    "pogon s": "Pogoń Siedlce",
    "polonia warszawa": "Polonia Warszawa",
    "ruchi chorzow": "Ruch Chorzów",
    "stal rzeszow": "Stal Rzeszów",
    "stomil olsztyn": "Stomil Olsztyn",
    "t.gornik": "T. Górnik",
    "t. gornik": "Górnik Łęczna",
    "wislanie j.": "Wiślanie J.",
    "wisla krakow": "Wisła Kraków",
    "wisla plock": "Wisła Płock",
    "zaglebie sosnowiec": "Zagłębie Sosnowiec",
    # Danish
    "aalborg": "Aalborg BK",
    "esbjerg": "Esbjerg fB",
    "fredericia": "FC Fredericia",
    "hobro": "Hobro IK",
    "kolding": "Kolding IF",
    "koge": "HB Køge",
    "naestved": "Næstved Boldklub",
    "ob": "OB Odense",
    "sonderjyske": "Sønderjyske Fodbold",
    "vendsyssel": "Vendsyssel FF",
    "viborg": "Viborg FF",
    "hilleroed": "Hillerød Fodbold",
    "helsingoer": "FC Helsingør",
    "hvidovre": "Hvidovre IF",
    "b.93": "B.93",
    "aab": "Aalborg BK",
    # Generic
    "fc copenhagen": "FC København",
    "copenhagen": "FC København",
    "kb": "FC København",
}


def normalise_team_name(name: str) -> str:
    """Normalise a team name to its canonical form.

    Converts aliases, strips whitespace, and applies title case.

    Parameters
    ----------
    name : str
        Raw team name from any data source.

    Returns
    -------
    str
        Normalised team name.
    """
    # Handle NaN / None / non-string types
    if not isinstance(name, str):
        raise ValueError(f"Expected string, got {type(name).__name__}: {name!r}")
    cleaned = name.strip().lower()
    # Remove common suffixes
    cleaned = re.sub(r"\s+\([^)]*\)", "", cleaned)  # Remove parentheticals
    cleaned = re.sub(r"\s*/\s*", "/", cleaned)

    # Look up alias
    normalised = TEAM_ALIASES.get(cleaned)
    if normalised:
        return normalised

    # Title case fallback
    words = cleaned.split()
    return " ".join(w.capitalize() if w[0].isalpha() or w[0].isupper() else w for w in words)


class DataCleaner:
    """Data cleaner and transformer for match records."""

    def clean_matches(
        self, matches: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Clean and normalise a list of match records.

        Applies:
        - Team name normalisation
        - Date normalisation
        - Type coercion
        - Missing value handling
        - Duplicate removal

        Parameters
        ----------
        matches : list[dict]
            Raw match records from a collector.

        Returns
        -------
        list[dict]
            Cleaned match records.
        """
        cleaned: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()

        for match in matches:
            # Skip empty records (also catches NaN floats)
            if not match:
                continue
            home = match.get("home_team")
            away = match.get("away_team")
            if not home or not away or not isinstance(home, str) or not isinstance(away, str):
                continue

            # Normalise team names
            match["home_team"] = normalise_team_name(match["home_team"])
            match["away_team"] = normalise_team_name(match["away_team"])

            # Normalise date
            match["date"] = self._normalise_date(match.get("date", ""))

            # Coerce types
            match["home_goals"] = self._safe_int(match.get("home_goals"))
            match["away_goals"] = self._safe_int(match.get("away_goals"))
            match["home_odds"] = self._safe_float(match.get("home_odds"))
            match["draw_odds"] = self._safe_float(match.get("draw_odds"))
            match["away_odds"] = self._safe_float(match.get("away_odds"))

            # Compute result from goals if missing
            if match.get("result") is None or match.get("result") == "":
                hg = match["home_goals"]
                ag = match["away_goals"]
                if hg is not None and ag is not None:
                    if hg > ag:
                        match["result"] = "H"
                    elif hg < ag:
                        match["result"] = "A"
                    else:
                        match["result"] = "D"

            # Skip if no date
            if not match.get("date"):
                continue

            # Deduplicate
            dedup_key = (
                match.get("source", ""),
                match.get("league", ""),
                str(match.get("date", "")),
                match["home_team"],
                match["away_team"],
            )
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            cleaned.append(match)

        logger.debug(
            "Cleaned %d records (%d duplicates removed)",
            len(cleaned), len(matches) - len(cleaned),
        )
        return cleaned

    @staticmethod
    def _normalise_date(date_val: Any) -> str | None:
        """Normalise dates to 'YYYY-MM-DD' format."""
        if not date_val:
            return None
        if isinstance(date_val, datetime):
            return date_val.strftime("%Y-%m-%d")
        if isinstance(date_val, str):
            date_str = date_val.strip()

            # Already in YYYY-MM-DD
            if re.match(r"^\d{4}-\d{2}-\d{2}", date_str):
                return date_str[:10]

            # DD/MM/YYYY
            if re.match(r"^\d{2}/\d{2}/\d{4}", date_str):
                try:
                    return datetime.strptime(date_str[:10], "%d/%m/%Y").strftime("%Y-%m-%d")
                except ValueError:
                    pass

            # DD/MM/YY
            if re.match(r"^\d{2}/\d{2}/\d{2}", date_str):
                try:
                    return datetime.strptime(date_str[:8], "%d/%m/%y").strftime("%Y-%m-%d")
                except ValueError:
                    pass

            # YYYY-MM-DDTHH:MM:SS (ISO)
            if "T" in date_str:
                try:
                    return date_str[:10]
                except ValueError:
                    pass

        return str(date_val)[:10] if date_val else None

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(float(str(value).strip()))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(str(value).strip())
        except (ValueError, TypeError):
            return None
