"""
Password hashing (bcrypt) and JWT access token creation/verification.

Uses the `bcrypt` library directly (rather than passlib) to avoid known
version-compatibility issues between passlib and bcrypt>=4.1 on some
platforms; bcrypt itself is a small, stable, well-maintained dependency.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import bcrypt
import jwt

from app.core.config import get_settings

TOKEN_TYPE = "access"


def _truncate_for_bcrypt(password: str) -> bytes:
    """bcrypt only uses the first 72 bytes of a password; truncate defensively to avoid errors."""
    return password.encode("utf-8")[:72]


def hash_password(plain_password: str) -> str:
    """Hashes a plaintext password with bcrypt. Never log or persist the plaintext."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(_truncate_for_bcrypt(plain_password), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(_truncate_for_bcrypt(plain_password), hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(subject: str, role: str, expires_minutes: Optional[int] = None) -> str:
    """Creates a signed JWT access token. `subject` is the user_id."""
    settings = get_settings()
    minutes = expires_minutes if expires_minutes is not None else settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "role": role,
        "type": TOKEN_TYPE,
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decodes and validates a JWT access token. Raises jwt exceptions on failure."""
    settings = get_settings()
    payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    if payload.get("type") != TOKEN_TYPE:
        raise jwt.InvalidTokenError("Unexpected token type.")
    return payload
