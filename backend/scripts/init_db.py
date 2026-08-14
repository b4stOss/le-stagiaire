"""Apply schema.sql to the database. Idempotent."""

from app.db import apply_schema

if __name__ == "__main__":
    apply_schema()
    print("schema applied")
