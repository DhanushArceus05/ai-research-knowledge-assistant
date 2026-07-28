"""
Shared FastAPI dependencies for the API routes.
"""
import jwt
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.database.session import get_db  # re-exported for convenience
from app.core.security import decode_access_token
from app.core.exceptions import InvalidTokenError, ForbiddenError
from app.database.repositories import UserRepository
from app.database.models import User, UserRole

_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Resolves the authenticated user from the `Authorization: Bearer <token>` header."""
    if credentials is None or not credentials.credentials:
        raise InvalidTokenError("Missing authentication token. Include an 'Authorization: Bearer <token>' header.")

    try:
        payload = decode_access_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise InvalidTokenError("Authentication token has expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise InvalidTokenError("Invalid authentication token.")

    user_id = payload.get("sub")
    user = UserRepository(db).get_by_id(user_id) if user_id else None
    if not user or not user.is_active:
        raise InvalidTokenError("The user associated with this token no longer exists or is inactive.")

    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Dependency that additionally requires the current user to have the ADMIN role."""
    if current_user.role != UserRole.ADMIN:
        raise ForbiddenError("This endpoint requires administrator privileges.")
    return current_user


__all__ = ["get_db", "get_current_user", "require_admin"]
