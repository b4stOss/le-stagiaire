"""Apply schema.sql to the database. Idempotent."""

import logging

from app.db import apply_schema
from app.logs import setup_logging

if __name__ == "__main__":
    setup_logging()
    apply_schema()
    logging.getLogger(__name__).info("schema applied")
