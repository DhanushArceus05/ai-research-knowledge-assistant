"""
SQLAlchemy declarative base and engine/session factory.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import get_settings

settings = get_settings()

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db() -> None:
    """Create all database tables if they do not already exist, and apply additive migrations."""
    # Import models here so they are registered on Base.metadata before create_all is called.
    from app.database import models  # noqa: F401
    from app.database.migrations import run_additive_migrations

    # Run additive column migrations first so pre-existing tables (created by
    # an older version of the app) gain any new nullable columns before
    # create_all() runs (create_all() only creates missing tables, it never
    # alters existing ones).
    run_additive_migrations(engine)
    Base.metadata.create_all(bind=engine)
