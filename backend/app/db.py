import atexit
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config import REPO_ROOT, settings

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    """One pool per process, opened on first use so scripts and the API share the code."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=settings.demo_max_concurrent + 2,
            configure=register_vector,
            open=True,
        )
        atexit.register(close_pool)
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    """Borrow a pooled connection. Commits on clean exit, rolls back on exception."""
    with get_pool().connection() as conn:
        yield conn


def apply_schema() -> None:
    schema = (REPO_ROOT / "backend" / "schema.sql").read_text()
    with get_conn() as conn:
        conn.execute(schema)  # type: ignore[arg-type]  # a multi-statement script, not a parametrized query
