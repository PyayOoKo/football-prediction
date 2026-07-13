"""Entry point for ``python -m src.monitoring``.

Delegates to the CLI module so all commands work via either::

    python -m src.monitoring summary
    python -m src.monitoring.cli summary
"""

from __future__ import annotations

import sys

from src.monitoring.cli import main

if __name__ == "__main__":
    sys.exit(main())
