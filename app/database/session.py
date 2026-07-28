"""
Request-scoped database session dependency for FastAPI.
"""
from typing import Generator

from app.database.base import SessionLocal


def get_db() -> Generator:
    """Yield a SQLAlchemy session and ensure it is always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
