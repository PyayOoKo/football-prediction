"""
Football Data Validation Framework.

Detects data quality issues in football match datasets and
generates professional reports (HTML, CSV, JSON).

Detection rules
---------------
1. Duplicate matches   — same teams + same date = duplicate
2. Invalid dates       — non-existent dates, future dates, null dates
3. Invalid odds        — odds <= 1.0, negative odds, null odds for finished matches
4. Missing goals       — finished matches without goal data
5. Missing teams       — null or empty team names
6. Incorrect leagues   — unrecognised competition names/codes
7. Invalid statistics  — impossible possession (>100%), negative corners, etc.
8. Duplicate IDs       — non-unique match/row identifiers
9. Impossible scores   — negative goals, excessively high scores (>20)

Usage
-----
::

    from src.validation import ValidationEngine

    data = [
        {"match_id": 1, "date": "2024-01-07", "home_team": "Arsenal", ...},
        ...
    ]

    engine = ValidationEngine()
    results = engine.run(data)

    # Generate reports
    results.to_html("reports/validation.html")
    results.to_csv("reports/validation.csv")
    results.to_json("reports/validation.json")
"""

from src.validation.engine import ValidationEngine
from src.validation.models import ValidationResult, CheckResult, Severity

__all__ = [
    "ValidationEngine",
    "ValidationResult",
    "CheckResult",
    "Severity",
]
