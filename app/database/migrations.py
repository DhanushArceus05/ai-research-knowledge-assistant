"""
Additive-only schema migration utility.

We didn't pull in Alembic for this since every schema change so far has been
additive: new nullable columns, new tables. SQLAlchemy's create_all() already
handles new tables (users, image_assets, extracted_tables) fine on an
existing database, but it won't add new columns to a table that already
exists -- e.g. adding user_id to a documents table created before multi-user
support existed. That's what this module handles: it inspects each existing
table via SQLite's PRAGMA table_info and adds any missing columns as
nullable ALTER TABLE ADD COLUMN statements. It never drops, renames, or
mutates existing data, and is safe to run on every startup.
"""
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.core.logging import get_logger

logger = get_logger(__name__)

# Map of table_name -> list of (column_name, sqlite_column_type) that may need
# to be added to a pre-existing database created before these columns existed.
_ADDITIVE_COLUMNS = {
    "documents": [("user_id", "VARCHAR")],
    "chunks": [("user_id", "VARCHAR"), ("extraction_method", "VARCHAR")],
    "conversation_sessions": [("user_id", "VARCHAR")],
    "query_logs": [("user_id", "VARCHAR")],
}


def run_additive_migrations(engine: Engine) -> None:
    """Adds any missing nullable columns to existing SQLite tables. No-op for fresh databases."""
    if engine.dialect.name != "sqlite":
        logger.info("Skipping additive column migration: not a SQLite database (dialect=%s).", engine.dialect.name)
        return

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table_name, columns in _ADDITIVE_COLUMNS.items():
            if table_name not in existing_tables:
                # Table doesn't exist yet at all; create_all() will create it
                # fresh (with every current column already present).
                continue

            existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
            for column_name, column_type in columns:
                if column_name in existing_columns:
                    continue
                logger.info("Migrating schema: adding column '%s.%s' (%s).", table_name, column_name, column_type)
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"))
