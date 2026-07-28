from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_current_user
from app.services.auth_service import AuthService
from app.core.config import get_settings
from app.database.models import User
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    """Registers a new user account and returns an access token (the first registered user becomes an admin)."""
    settings = get_settings()
    service = AuthService(db)
    user = service.register(payload.email, payload.password, payload.display_name)
    token = service.issue_token(user)

    return TokenResponse(
        access_token=token,
        expires_in_minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
        user=UserResponse(**user.to_dict()),
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    """Authenticates a user and returns a fresh JWT access token."""
    settings = get_settings()
    service = AuthService(db)
    user = service.authenticate(payload.email, payload.password)
    token = service.issue_token(user)

    return TokenResponse(
        access_token=token,
        expires_in_minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
        user=UserResponse(**user.to_dict()),
    )


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Returns the profile of the currently authenticated user."""
    return UserResponse(**current_user.to_dict())
