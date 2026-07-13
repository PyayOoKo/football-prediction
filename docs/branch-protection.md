# 🔒 Branch Protection Recommendations

This document describes the recommended branch protection rules for the `master`/`main` branch. These rules ensure that all code merged to the primary branch has passed the full CI/CD pipeline, been reviewed by a maintainer, and meets the project's quality standards.

---

## 1. Rule Summary

| Setting | Value | Notes |
|---------|-------|-------|
| **Branch** | `master`, `main` | Protect both if both exist; or alias `main` → `master` |
| **Require PR** | ✅ Yes | Prevents direct pushes |
| **Required approvals** | 1 | Single reviewer for most changes; 2 for `src/` core logic |
| **Dismiss stale reviews** | ✅ Yes | Re-request review when new commits are pushed |
| **Require status checks** | ✅ Yes | See section 2 |
| **Require branches up to date** | ✅ Yes | Prevents merge skew |
| **Require signed commits** | ✅ Yes | GPG or SSH signatures |
| **Require linear history** | ✅ Yes | Squash merge only — keeps `git log` clean |
| **Allow force pushes** | ❌ No | Except for repository admins in emergencies |
| **Allow deletions** | ❌ No | Never delete `master`/`main` |

---

## 2. Required Status Checks

Configure the following status checks as **required** in the branch protection rule:

| Check Name | Workflow | Purpose | Estimated Time |
|------------|----------|---------|---------------|
| `CI Status` | `ci.yml` | Aggregate gate — passes only if lint + test + docker all succeed | ~15 min |
| `Lint & Format` | `ci.yml` | Black formatting + Ruff linting + MyPy type checking + config validation | ~3 min |
| `Tests (Python 3.12)` | `ci.yml` | Pytest with coverage + Alembic migrations against PostgreSQL | ~8 min |
| `Docker Build` | `ci.yml` | Builds the production Docker image | ~4 min |
| `PR Metadata` | `pr-quality.yml` | Validates title format, description, labels | ~30 s |
| `Dependency Check` | `pr-quality.yml` | Flags changes to `requirements.txt` or `pyproject.toml` | ~30 s |
| `File Size` | `pr-quality.yml` | Flags oversized PRs or large committed files | ~30 s |
| `Merge Conflict` | `pr-quality.yml` | Auto-labels PRs with merge conflicts | ~15 s |

### How to Configure in GitHub UI

```
Settings → Branches → Add rule (for master/main)

☑ Require pull request reviews before merging
    ☑ Require approvals: 1
    ☑ Dismiss stale pull request approvals when new commits are pushed
    ☑ Require review from Code Owners

☑ Require status checks to pass before merging
    ☑ Require branches to be up to date before merging
    ☑ CI Status
    ☑ Tests (Python 3.12)
    ☑ PR Metadata

☑ Require signed commits

☑ Require linear history

☑ Include administrators
```

---

## 3. Code Owners

Create a `CODEOWNERS` file at `.github/CODEOWNERS` to automatically request reviews from the right people based on file changes:

```gitignore
# Global owners — review everything
* @your-org/maintainers

# Core ML pipeline — needs ML team review
/src/feature_engineering.py    @your-org/ml-team
/src/ensemble.py               @your-org/ml-team
/src/train.py                  @your-org/ml-team
/src/elo.py                    @your-org/ml-team
/src/poisson_model.py          @your-org/ml-team
/src/dixon_coles.py            @your-org/ml-team
/src/evaluate.py               @your-org/ml-team
/src/hyperparameter_tuning.py  @your-org/ml-team
/src/value_betting.py          @your-org/ml-team
/src/backtesting.py            @your-org/ml-team
/src/calibration.py            @your-org/ml-team
/src/confidence_scoring.py     @your-org/ml-team
/src/predict.py                @your-org/ml-team

# Feature Store & Experiment Tracking — ML platform team
/src/feature_store/            @your-org/ml-platform-team
/src/experiment_tracking/      @your-org/ml-platform-team

# Database & ETL — data engineering team
/src/database/                 @your-org/data-eng-team
/src/etl/                      @your-org/data-eng-team
/src/importers/                @your-org/data-eng-team
/src/data_collection/          @your-org/data-eng-team
/src/data_loader.py            @your-org/data-eng-team
/src/preprocessing.py          @your-org/data-eng-team

# Data quality & validation
/src/validation/               @your-org/data-eng-team
/src/monitoring/               @your-org/data-eng-team
/src/data_profiling/           @your-org/data-eng-team

# Scheduler & deployment
/src/scheduler/                @your-org/devops-team
/Dockerfile                    @your-org/devops-team
/docker-compose.yml            @your-org/devops-team
/.github/workflows/            @your-org/devops-team

# Configuration
/config.py                     @your-org/maintainers
/pyproject.toml                @your-org/maintainers
/.env.example                  @your-org/maintainers

# Tests — reviewed by the same team as the module being tested
/tests/test_etl/               @your-org/data-eng-team
/tests/test_database/          @your-org/data-eng-team
/tests/test_feature_store/     @your-org/ml-platform-team
/tests/test_experiment_tracking/ @your-org/ml-platform-team
/tests/test_validation/        @your-org/data-eng-team

# Documentation
/docs/                         @your-org/maintainers
/README.md                     @your-org/maintainers
/CONTRIBUTING.md               @your-org/maintainers
```

---

## 4. Recommended Branch Strategy

```
main ──── feat-a ── feat-a ── feat-a ── (squash) ────────────────────
              \                                         \
               └─── feat-b ── feat-b ── feat-b ── (squash merge) ────
                                                               \
                                                                └── hotfix ── (squash) ──
```

| Branch | Source | Merges Into | Lifespan | Purpose |
|--------|--------|-------------|----------|---------|
| `main` | — | — | Permanent | Production-ready code |
| `develop` | `main` | `main` | Permanent | Integration branch |
| `feat/*` | `develop` | `develop` | Short-lived | New features |
| `fix/*` | `develop` | `develop` | Short-lived | Bug fixes |
| `hotfix/*` | `main` | `main`, `develop` | Emergency | Production hotfixes |
| `release/*` | `develop` | `main`, `develop` | Temporary | Release candidates |

### Merge Strategy: Squash Merge

**Always use squash merge** when merging feature branches to `develop` or `main`.

**Why squash merge?**
- Maintains a linear, readable `git log`
- Each commit on `main` represents one complete, reviewed change
- Avoids the messy "merge commit + 35 WIP commits" pattern
- Makes `git bisect` reliable — every commit is a known-good state

---

## 5. Security Recommendations

| Setting | Recommended | Why |
|---------|-------------|-----|
| **Require signed commits** | ✅ Yes | Ensures commits are from verified identities |
| **Dismiss stale reviews** | ✅ Yes | Prevents bypassing review with cosmetic push |
| **Restrict push access** | ✅ Yes | Only CI + admins can push to `main` |
| **Secrets scanning** | ✅ Enable | GitHub Advanced Security — prevent accidental credential leaks |
| **Dependabot alerts** | ✅ Enable | Automated vulnerability detection in dependencies |
| **Dependabot auto-merge** | ⚠️ Patch only | Auto-merge non-breaking dependency updates after CI passes |
| **Code scanning** | ✅ Enable | GitHub CodeQL — static analysis for security vulnerabilities |
| **Secret scanning push protection** | ✅ Enable | Blocks pushes containing known secret patterns |

---

## 6. Recommended GitHub Actions Secrets

The current CI pipeline does **not** push Docker images to any registry (it only performs a build verification). If you add image publishing in the future, you will need:

| Secret | Used By | Description |
|--------|---------|-------------|
| `DOCKER_USERNAME` | `ci.yml` (future) | Docker Hub username for image push |
| `DOCKER_PASSWORD` | `ci.yml` (future) | Docker Hub token or password |
| `CODECOV_TOKEN` | `ci.yml` (optional) | Codecov upload token for coverage reports |

### Currently required secrets
- `GITHUB_TOKEN` — automatically provided by GitHub Actions, used for artifact uploads and status checks. No manual setup needed.

---

## 7. Summary Checklist for Repository Admins

- [ ] Go to **Settings → Branches → Add rule**
- [ ] Set **Branch name pattern** to `master` (and `main`)
- [ ] ✅ **Require pull request reviews before merging** — 1 approval
- [ ] ✅ **Dismiss stale pull request approvals** when new commits are pushed
- [ ] ✅ **Require review from Code Owners**
- [ ] ✅ **Require status checks to pass before merging**
  - [ ] `CI Status` (aggregate gate)
  - [ ] `Tests (Python 3.12)`
  - [ ] `PR Metadata`
- [ ] ✅ **Require branches to be up to date** before merging
- [ ] ✅ **Require signed commits**
- [ ] ✅ **Require linear history**
- [ ] ✅ **Include administrators**
- [ ] ❌ **Allow force pushes** — disabled (admins only, on emergency)
- [ ] ❌ **Allow deletions** — disabled
- [ ] ✅ Create `.github/CODEOWNERS` with team assignments
- [ ] ✅ Enable Dependabot alerts + security updates
- [ ] ✅ Enable GitHub CodeQL scanning
- [ ] ✅ Enable secret scanning + push protection
