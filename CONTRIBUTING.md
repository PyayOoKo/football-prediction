# Contributing

Thank you for your interest in contributing to the Football Prediction project!

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behaviour to the project maintainers.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:

   ```bash
   git clone https://github.com/your-username/football-prediction.git
   cd football-prediction
   ```

3. **Set up the development environment:**

   ```bash
   make dev-install
   ```

4. **Create a branch** for your changes:

   ```bash
   git checkout -b feature/my-feature
   ```

## Development Workflow

### Code style

- **Formatting:** [Black](https://black.readthedocs.io/) with 88-character line length.
- **Linting:** [Ruff](https://docs.astral.sh/ruff/) for fast Python linting.
- **Type checking:** [mypy](https://mypy.readthedocs.io/) in strict mode.
- **Pre-commit hooks:** Run `pre-commit install` to enable automatic checks.

### Running tests

```bash
make test          # Run all tests
make test-cov      # Run with coverage report
```

### Database migrations

If your changes involve database schema modifications:

1. Modify the ORM model in `src/database/models/`.
2. Auto-generate a migration:

   ```bash
   make db-autogen
   ```

3. Review and edit the generated file in `alembic/versions/`.
4. Apply the migration:

   ```bash
   make db-migrate
   ```

## Pull Request Checklist

- [ ] Code follows the project's style guidelines (black, ruff, mypy).
- [ ] Tests pass (`make test`).
- [ ] New code includes appropriate test coverage.
- [ ] Database migrations are included (if applicable).
- [ ] Documentation is updated (docstrings, README, etc.).
- [ ] Changes are rebased onto the latest `main` branch.
- [ ] Commit messages are clear and descriptive.

## Project Structure

```
football-prediction/
├── src/                  # Application source code
│   ├── config/           # Configuration and settings
│   ├── data/             # Data processing and feature engineering
│   ├── database/         # SQLAlchemy ORM and repositories
│   ├── models/           # ML model definitions
│   ├── scrapers/         # Data collection from external sources
│   ├── services/         # Business logic orchestration
│   └── utils/            # Cross-cutting helpers
├── tests/                # Test suite (mirrors src/)
├── alembic/              # Database migrations
├── docker/               # Docker configuration
└── data/                 # Data files (gitignored)
```

## Reporting Issues

Use the GitHub issue tracker:

- **Bug report:** Use the "Bug report" template.
- **Feature request:** Use the "Feature request" template.
