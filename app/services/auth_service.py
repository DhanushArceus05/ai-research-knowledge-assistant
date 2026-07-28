"""
Authentication service: registration, login, and access-token issuance.
"""
from typing import Optional
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import hash_password, verify_password, create_access_token
from app.core.exceptions import EmailAlreadyRegisteredError, InvalidCredentialsError
from app.database.repositories import UserRepository
from app.database.models import User, UserRole


class AuthService:
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.user_repo = UserRepository(db)

    def register(self, email: str, password: str, display_name: Optional[str] = None) -> User:
        if self.user_repo.get_by_email(email):
            raise EmailAlreadyRegisteredError()

        # The very first registered user becomes an admin automatically so the
        # system always has at least one admin account without requiring a
        # separate manual bootstrap step.
        is_first_user = self.user_repo.get_by_email(email) is None and self._is_empty_user_table()
        role = UserRole.ADMIN if is_first_user else UserRole.USER

        hashed = hash_password(password)
        return self.user_repo.create(email=email, hashed_password=hashed, display_name=display_name, role=role)

    def _is_empty_user_table(self) -> bool:
        return self.db.query(User).count() == 0

    def authenticate(self, email: str, password: str) -> User:
        user = self.user_repo.get_by_email(email)
        if not user or not user.is_active or not verify_password(password, user.hashed_password):
            raise InvalidCredentialsError()
        return user

    def issue_token(self, user: User) -> str:
        return create_access_token(subject=user.id, role=user.role.value)
