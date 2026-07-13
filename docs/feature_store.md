# Feature Store

> Production feature registry with versioning, dependency tracking, batch computation, and lineage provenance.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FeatureRegistry                             │
│                                                                     │
│  ┌─────────────────┐  ┌────────────────┐  ┌──────────────────────┐ │
│  │  Definitions    │  │  Dependencies  │  │  Batch Management    │ │
│  │  Register       │  │  DAG Traversal │  │  Schedule Computes   │ │
│  │  Version        │  │  Hard/Soft     │  │  Track Batches       │ │
│  └─────────────────┘  └────────────────┘  └──────────────────────┘ │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────┐
│                        Database Tables                              │
│  ┌──────────────────┐  ┌────────────────┐  ┌──────────────────────┐ │
│  │ feature_         │  │ feature_       │  │ feature_             │ │
│  │ definitions      │  │ values         │  │ dependencies         │ │
│  ├──────────────────┤  ├────────────────┤  ├──────────────────────┤ │
│  │ feature_         │  │ feature_       │  │ 5 tables total       │ │
│  │ versions         │  │ computation_   │  │                      │ │
│  │                  │  │ batches        │  │                      │ │
│  └──────────────────┘  └────────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Tables

### `feature_definitions`
Registry of all features — what exists and how it behaves.

| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | Unique identifier |
| `name` | VARCHAR(255) | Feature name (e.g. `home_attack_strength_10`) |
| `version` | INT | Feature version (incremented on change) |
| `feature_type` | VARCHAR(100) | Semantic type (e.g. `rolling_stat`, `elo`) |
| `category` | ENUM | Thematic category (16 categories) |
| `entity_type` | ENUM | What entity: `match`, `team`, `league`, `player`, `global` |
| `computation_params` | JSON | Parameters (window size, K-factor, etc.) |
| `validation_rules` | JSON | Bounds, nullability, cardinality rules |
| `dependencies` | JSON | Feature names this depends on |
| `status` | ENUM | `draft`, `active`, `deprecated`, `retired` |
| `is_active` | BOOLEAN | Quick filter for active features |
| `created_at` | TIMESTAMPTZ | Creation time |
| `updated_at` | TIMESTAMPTZ | Last update |

**Unique constraint:** `(name, version)`

### `feature_values`
Computed feature values for specific entities.

| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | Unique identifier |
| `feature_definition_id` | FK | → `feature_definitions` |
| `match_id` | INT (nullable) | FK to matches |
| `team_id` | INT (nullable) | FK to teams |
| `league_id` | INT (nullable) | FK to competitions |
| `numeric_value` | FLOAT (nullable) | Scalar feature value |
| `text_value` | TEXT (nullable) | Categorical feature value |
| `json_value` | JSON (nullable) | Complex feature value |
| `computed_at` | TIMESTAMPTZ | When computed |
| `computed_by` | VARCHAR(255) | Computer identifier |
| `batch_id` | FK (nullable) | → `feature_computation_batches` |

**Unique constraint:** `(feature_definition_id, match_id, team_id)`

### `feature_dependencies`
Directed edges in the feature dependency DAG.

| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | Unique identifier |
| `dependent_feature_id` | FK | Feature that depends on another |
| `dependency_feature_id` | FK | Prerequisite feature |
| `is_hard` | BOOLEAN | `true` = required, `false` = soft |
| `created_at` | TIMESTAMPTZ | |

### `feature_versions`
Version history for feature definitions.

| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | |
| `feature_definition_id` | FK | → `feature_definitions` |
| `version` | INT | Version number |
| `is_current` | BOOLEAN | Active version |
| `changelog` | TEXT | What changed |
| `snapshot` | JSON | Full definition snapshot |

### `feature_computation_batches`
Audit trail for batch computation runs.

| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | |
| `batch_label` | VARCHAR(255) | Human-readable (e.g. `daily-2026-07-13`) |
| `trigger` | VARCHAR(50) | `manual`, `scheduled`, `pipeline` |
| `features_computed` | JSON | Feature names computed |
| `entity_count` | INT | Entities processed |
| `started_at` | TIMESTAMPTZ | |
| `completed_at` | TIMESTAMPTZ (nullable) | |
| `duration_seconds` | FLOAT (nullable) | |
| `success` | BOOLEAN | |
| `error_message` | TEXT (nullable) | |

## Feature Categories

| Category | Description | Examples |
|---|---|---|
| `rolling_stat` | Rolling averages over N matches | Goals avg (last 5, 10, 20) |
| `team_form` | Recent form indicators | Points last 5, win streak |
| `elo_rating` | ELO-based strength ratings | Home ELO, away ELO |
| `attack_strength` | Attacking capability | Goals scored per game |
| `defense_strength` | Defensive capability | Goals conceded per game |
| `home_advantage` | Home field advantage | Home win rate |
| `away_advantage` | Away performance | Away points per game |
| `rest_days` | Days since last match | Fatigue indicator |
| `fixture_congestion` | Match density | Matches in last 7 days |
| `league_strength` | League quality rating | UEFA coefficient |
| `team_momentum` | Recent trend | Last 5 results trend |
| `market_movement` | Odds movement | Odds change over 24h |
| `h2h_stat` | Head-to-head stats | H2H goal diff |
| `xg_feature` | Expected goals features | xG for/against |
| `odds_feature` | Odds-derived features | Implied probability |
| `composite` | Aggregated composite | Combined strength score |

## Usage

```python
from src.feature_store import FeatureRegistry
from src.database.session import get_session

with get_session() as session:
    registry = FeatureRegistry(session)

    # Register a new feature
    feat = registry.register(
        name="home_attack_strength_10",
        feature_type="rolling_stat",
        category="attack_strength",
        entity_type="team",
        computation_params={"window": 10, "metric": "goals_scored"},
        description="Average goals scored in last 10 home matches",
    )

    # Compute feature values
    values = registry.compute(
        feature_name="home_attack_strength_10",
        entity_ids=[1, 2, 3, ...],  # team IDs
    )

    # Get a feature definition
    defn = registry.get_definition("home_attack_strength_10")

    # List all features in a category
    attack_features = registry.list_by_category("attack_strength")
```

## Lineage Tracking

Each feature value is traceable back to its computation batch:

```python
from src.feature_store.lineage import FeatureLineageEntry

# Query lineage for a specific match prediction
lineage = session.query(FeatureLineageEntry).filter_by(
    prediction_id="some_prediction_id",
).all()

for entry in lineage:
    print(f"{entry.feature_name} v{entry.feature_version} "
          f"→ computed at {entry.computed_at}")
```

## Dependency DAG Example

```
home_attack_strength_10
    ├── goals_scored_home (raw stat)
    ├── minutes_played (raw stat)
    └── opponent_defense_strength (composite)
        └── goals_conceded_away (raw stat)

team_momentum_5
    ├── result_last_5 (rolling stat)
    │   └── result (raw)
    ├── goals_scored_last_5 (rolling stat)
    └── goals_conceded_last_5 (rolling stat)
```

## Versioning

Feature definitions support independent versioning:

1. Create a new version of a feature → new row in `feature_definitions`
2. Old values remain linked to old definition version
3. `is_active` flag controls which version new computations use
4. `feature_versions` table records full snapshot for rollback

## Batch Computation

```python
from datetime import datetime, timezone

batch = registry.create_batch(
    label=f"daily-{datetime.now().strftime('%Y-%m-%d')}",
    trigger="scheduled",
    features=["home_attack_strength_10", "team_momentum_5"],
)

try:
    registry.compute_batch(batch.id)
    batch.complete(success=True)
except Exception as e:
    batch.complete(success=False, error=str(e))
```
