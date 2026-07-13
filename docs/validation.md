# Validation Framework

> 9 domain-specific validation checks for football data with HTML/CSV/JSON reporting.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       ValidationEngine                          │
│                                                                  │
│  Checks (default 9):                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 1. Duplicate Matches   │ 2. Invalid Dates              │    │
│  │ 3. Invalid Odds        │ 4. Missing Goals              │    │
│  │ 5. Missing Teams       │ 6. Incorrect Leagues          │    │
│  │ 7. Invalid Statistics  │ 8. Duplicate IDs              │    │
│  │ 9. Impossible Scores   │                               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Output Formats                       │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │    │
│  │  │  HTML     │  │  JSON    │  │  CSV     │              │    │
│  │  │(reporter) │  │(raw)     │  │(flat)    │              │    │
│  │  └──────────┘  └──────────┘  └──────────┘              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from src.validation import ValidationEngine

data = [
    {"home_team": "Arsenal", "away_team": "Chelsea", "date": "2025-03-15",
     "home_goals": 2, "away_goals": 1, "result": "H", "league": "E0"},
    # ... more rows
]

engine = ValidationEngine()
result = engine.run(data, source_name="my_dataset")

# Check results
print(f"Passed: {result.passed_checks}/{result.total_checks}")
print(f"Violations: {result.total_violations}")

# Export reports
result.to_html("reports/validation_report.html")  # Self-contained HTML
result.to_json("reports/validation_report.json")  # Machine-readable
result.to_csv("reports/validation_report.csv")    # Flat table

# Run only selected checks
result = engine.run_selected(data, ["Duplicate Matches", "Invalid Dates"])
```

## The 9 Checks

| # | Check | Severity | What It Detects |
|---|-------|----------|-----------------|
| 1 | **Duplicate Matches** | ERROR | Same teams, same date listed twice |
| 2 | **Invalid Dates** | ERROR | Null, malformed, or >3yr-future dates |
| 3 | **Invalid Odds** | WARNING | Odds ≤ 1.0, non-numeric odds |
| 4 | **Missing Goals** | ERROR | Finished matches without goal data |
| 5 | **Missing Teams** | ERROR | Null, empty, or identical team names |
| 6 | **Incorrect Leagues** | WARNING | Unrecognised competition names |
| 7 | **Invalid Statistics** | WARNING | Negatives, unrealistic attendance |
| 8 | **Duplicate IDs** | ERROR | Non-unique row/match identifiers |
| 9 | **Impossible Scores** | ERROR | Negative goals, scores >20, result mismatch |

## HTML Report

The `HTMLReporter` generates a self-contained HTML dashboard with:

```
┌─────────────────────────────────────────────────────────────┐
│  ⚽ Football Data Validation                                 │
│  📁 Source: my_dataset  📅 2025-03-15  📊 15,234 rows       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │ ✅ 9/9  │ 15,234 rows │ 0 violations │ 0 failed      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─ Duplicate Matches ────────────────── ✅ PASS ─────┐    │
│  │ 0 violations / 15,234 rows                          │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─ Invalid Odds ──────────────────── ⚠️ WARN ───────┐    │
│  │ 12 violations / 15,234 rows                         │    │
│  │ ┌──────┬────────┬────────┬──────────────────────┐   │    │
│  │ │ Row  │ Field  │ Value  │ Issue                 │   │    │
│  │ ├──────┼────────┼────────┼──────────────────────┤   │    │
│  │ │ 142  │ BbAvH  │ 1.00   │ Odds must be > 1.0   │   │    │
│  │ │ 891  │ BbAvA  │ 0.95   │ Odds must be > 1.0   │   │    │
│  │ └──────┴────────┴────────┴──────────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Custom Checks

Add your own checks — each is a stateless function:

```python
from src.validation.models import CheckResult, Severity

def check_negative_attendance(data: list[dict]) -> CheckResult:
    violations = []
    for i, row in enumerate(data):
        att = row.get("attendance")
        if att is not None and float(att) < 0:
            violations.append({
                "row_index": i,
                "field": "attendance",
                "value": str(att),
                "message": "Negative attendance",
            })
    return CheckResult(
        check_name="Negative Attendance",
        description="Attendance values below zero",
        severity=Severity.ERROR,
        passed=len(violations) == 0,
        total_rows=len(data),
        violation_count=len(violations),
        violations=violations,
    )

# Register with engine
from src.validation import ValidationEngine
engine = ValidationEngine(
    checks=[("Negative Attendance", check_negative_attendance, {})]
)
```

## Integration with Scheduler

The scheduler runs validation as part of the daily pipeline:

```yaml
# scheduler_config.yaml
tasks:
  - name: validate_data
    description: Run all 9 validation checks
    timeout_seconds: 120
    dependencies: ["download_fixtures"]
    retry_count: 1
```

## Known League Names

The `Incorrect Leagues` check validates against a built-in list of 50+ known leagues:

```
England:    E0, E1, E2, E3, EC, "Premier League", "Championship", ...
Germany:    D1, D2, "Bundesliga", "2. Bundesliga", ...
Spain:      SP1, SP2, "La Liga", "La Liga 2", ...
Italy:      I1, I2, "Serie A", "Serie B", ...
France:     F1, F2, "Ligue 1", "Ligue 2", ...
Netherlands: N1, "Eredivisie", ...
International: "World Cup", "Champions League", "Europa League", ...
```

Extend with kwargs: `engine.run(data, known_leagues=["New League", ...])`

## Error Handling

Each check is wrapped in a try/except so a single failing check never crashes the engine:

```
  [FAIL (1 violations)] Invalid Statistics — 9 fields checked in 0.12s
  [PASS]                Invalid Dates — 15,234 rows checked in 0.08s
  [FAIL]                Duplicate IDs — Check raised an exception: ...
```

## Data Models

```python
@dataclass
class ValidationResult:
    source_name: str
    timestamp: str
    total_rows: int
    checks: list[CheckResult]

    @property
    def passed(self) -> bool        # all checks passed
    @property
    def total_violations(self) -> int  # sum of all violations
    def to_html(self, path) -> None
    def to_json(self, path) -> None
    def to_csv(self, path) -> None

@dataclass
class CheckResult:
    check_name: str
    severity: Severity    # ERROR | WARNING | INFO
    passed: bool
    violation_count: int
    violations: list[dict]
```
