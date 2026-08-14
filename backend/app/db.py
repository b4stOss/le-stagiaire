from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from app.config import REPO_ROOT, settings


@contextmanager
def get_conn():
    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        yield conn


def apply_schema() -> None:
    schema = (REPO_ROOT / "backend" / "schema.sql").read_text()
    with psycopg.connect(settings.database_url) as conn:
        conn.execute(schema)
        conn.commit()
