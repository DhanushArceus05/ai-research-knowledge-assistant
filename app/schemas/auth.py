import re
from typing import Optional
from pydantic import BaseModel, field_validator

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: Optional[str] = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        normalized = v.strip().lower()
        if not _EMAIL_RE.match(normalized):
            raise ValueError("A valid email address is required.")
        return normalized

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        if v.isalpha() or v.isdigit():
            raise ValueError("Password must contain a mix of letters and numbers/symbols.")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("password")
    @classmethod
    def password_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("Password must not be empty.")
        return v


class UserResponse(BaseModel):
    user_id: str
    email: str
    display_name: Optional[str] = None
    role: str
    is_active: bool
    created_at: Optional[str] = None


class TokenResponse(BaseModel):
    success: bool = True
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    user: UserResponse
