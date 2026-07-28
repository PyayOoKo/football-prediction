"""Configure PostgreSQL server-side connection timeout settings.

Complement the application-level ``pool_recycle=3600`` (in
``src/database/session.py``) with database-side guards that prevent
stale or orphaned connections from accumulating.

Settings applied via ``ALTER DATABASE``
---------------------------------------

=========================  =======  =========================================
Parameter                  Value    Purpose
=========================  =======  =========================================
``idle_in_transaction_     ``60000``  Kill queries stuck in an open transaction
session_timeout``                   for > 60 s (rolls back, frees the conn)
``statement_timeout``      ``300000`` Max execution time per statement: 5 min
                                     (prevents runaway analytical queries)
``tcp_keepalives_idle``    ``300``   Seconds before first TCP keepalive probe
                                     (default = 7200 on Linux; 300 is safer)
``tcp_keepalives_interval`` ``60``   Seconds between keepalive retries
``tcp_keepalives_count``   ``5``     Probes before dropping connection
                                     (5 × 60 s = 5 min total wait)
=========================  =======  =========================================

How these work together with pool_recycle
-----------------------------------------
- ``pool_recycle=3600`` in the app: the SQLAlchemy pool drops connections
  older than 1 hour and creates fresh ones.
- ``tcp_keepalives`` (DB-side): the PostgreSQL server notices dead
  connections within ~5 minutes and terminates them, so the pool doesn't
  accumulate stale sockets.
- ``idle_in_transaction_session_timeout``: catches the case where a
  session starts a transaction and then crashes — the server cleans it up
  within 60 s instead of holding locks forever.

Safety
------
``ALTER DATABASE SET`` is idempotent — running the migration multiple times
is harmless.  The downgrade resets each parameter to its PostgreSQL default
(``0`` / ``0`` / ``7200`` / ``0`` / ``0``).

.. note::
   ``ALTER DATABASE`` requires **superuser** (or ``rds_superuser`` on RDS)
   privileges.  If the migration user cannot run ``ALTER DATABASE``, skip
   this migration or apply the settings manually via the database console.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Get the current database name from the connection ──
    conn = op.get_bind()
    result = conn.execute("SELECT current_database()")
    db_name = result.scalar()

    # ── Idle transaction timeout: 60 s ──────────────────
    # Prevents transactions from holding locks forever when
    # the application crashes mid-transaction.
    conn.execute(
        f"ALTER DATABASE {db_name} SET idle_in_transaction_session_timeout = '60000'"
    )

    # ── Statement timeout: 5 min ────────────────────────
    # Catches runaway queries from ad-hoc analytics or
    # unoptimised feature computations.
    conn.execute(
        f"ALTER DATABASE {db_name} SET statement_timeout = '300000'"
    )

    # ── TCP keepalives: detect dead connections faster ──
    # Default Linux TCP keepalive waits 7200 s (2 hours)
    # before probing.  Bring this down to 300 s (5 min).
    conn.execute(
        f"ALTER DATABASE {db_name} SET tcp_keepalives_idle = '300'"
    )
    conn.execute(
        f"ALTER DATABASE {db_name} SET tcp_keepalives_interval = '60'"
    )
    conn.execute(
        f"ALTER DATABASE {db_name} SET tcp_keepalives_count = '5'"
    )


def downgrade() -> None:
    conn = op.get_bind()
    result = conn.execute("SELECT current_database()")
    db_name = result.scalar()

    # Reset to PostgreSQL defaults
    conn.execute(
        f"ALTER DATABASE {db_name} RESET idle_in_transaction_session_timeout"
    )
    conn.execute(
        f"ALTER DATABASE {db_name} RESET statement_timeout"
    )
    conn.execute(
        f"ALTER DATABASE {db_name} RESET tcp_keepalives_idle"
    )
    conn.execute(
        f"ALTER DATABASE {db_name} RESET tcp_keepalives_interval"
    )
    conn.execute(
        f"ALTER DATABASE {db_name} RESET tcp_keepalives_count"
    )
