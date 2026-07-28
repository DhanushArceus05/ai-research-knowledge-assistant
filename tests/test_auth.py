def test_register_new_user_succeeds(client):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "newperson@example.com", "password": "StrongPass1!", "display_name": "New Person"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["access_token"]
    assert body["user"]["email"] == "newperson@example.com"
    assert "password" not in body["user"]
    assert "hashed_password" not in body["user"]


def test_register_duplicate_email_is_rejected(client):
    payload = {"email": "duplicate@example.com", "password": "StrongPass1!"}
    first = client.post("/api/v1/auth/register", json=payload)
    assert first.status_code == 200

    second = client.post("/api/v1/auth/register", json=payload)
    assert second.status_code == 400


def test_register_rejects_weak_password(client):
    response = client.post(
        "/api/v1/auth/register", json={"email": "weakpass@example.com", "password": "onlyletters"}
    )
    assert response.status_code == 422


def test_register_rejects_invalid_email(client):
    response = client.post(
        "/api/v1/auth/register", json={"email": "not-an-email", "password": "StrongPass1!"}
    )
    assert response.status_code == 422


def test_register_normalizes_email_case(client):
    response = client.post(
        "/api/v1/auth/register", json={"email": "MixedCase@Example.COM", "password": "StrongPass1!"}
    )
    assert response.status_code == 200
    assert response.json()["user"]["email"] == "mixedcase@example.com"


def test_login_success(client):
    client.post("/api/v1/auth/register", json={"email": "loginuser@example.com", "password": "StrongPass1!"})
    response = client.post("/api/v1/auth/login", json={"email": "loginuser@example.com", "password": "StrongPass1!"})
    assert response.status_code == 200
    assert response.json()["access_token"]


def test_login_failure_wrong_password(client):
    client.post("/api/v1/auth/register", json={"email": "wrongpass@example.com", "password": "StrongPass1!"})
    response = client.post("/api/v1/auth/login", json={"email": "wrongpass@example.com", "password": "WrongPassword1!"})
    assert response.status_code == 401


def test_login_failure_nonexistent_user(client):
    response = client.post("/api/v1/auth/login", json={"email": "doesnotexist@example.com", "password": "StrongPass1!"})
    assert response.status_code == 401


def test_me_endpoint_requires_token(client):
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401


def test_me_endpoint_rejects_invalid_token(client):
    client.headers.update({"Authorization": "Bearer not-a-real-token"})
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401


def test_me_endpoint_returns_current_user(auth_client):
    response = auth_client.get("/api/v1/auth/me")
    assert response.status_code == 200
    assert response.json()["email"] == auth_client.current_user_email


def test_protected_endpoint_without_token_returns_401(client):
    response = client.get("/api/v1/documents")
    assert response.status_code == 401


def test_expired_token_is_rejected(client):
    import jwt
    from datetime import datetime, timedelta, timezone
    from app.core.config import get_settings

    settings = get_settings()
    expired_payload = {
        "sub": "some-user-id",
        "role": "user",
        "type": "access",
        "iat": datetime.now(timezone.utc) - timedelta(hours=2),
        "exp": datetime.now(timezone.utc) - timedelta(hours=1),
    }
    expired_token = jwt.encode(expired_payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    client.headers.update({"Authorization": f"Bearer {expired_token}"})
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401


def test_admin_endpoint_rejects_regular_user(auth_client):
    response = auth_client.get("/api/v1/admin/analytics")
    assert response.status_code == 403


def test_first_registered_user_in_a_fresh_database_becomes_admin():
    """The very first user ever registered in a brand-new database becomes an
    admin automatically. Uses a dedicated in-memory SQLite database (rather
    than the shared test-session database, which already has users registered
    by other tests) so this test is order-independent and deterministic."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database.base import Base
    from app.services.auth_service import AuthService
    from app.database.models import UserRole

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        service = AuthService(db)
        first_user = service.register("first@example.com", "StrongPass1!")
        assert first_user.role == UserRole.ADMIN

        second_user = service.register("second@example.com", "StrongPass1!")
        assert second_user.role == UserRole.USER
    finally:
        db.close()


def test_admin_cache_diagnostics_requires_admin_role(auth_client):
    response = auth_client.get("/api/v1/admin/cache")
    assert response.status_code == 403


def test_admin_endpoint_accessible_after_role_promotion(auth_client):
    """Directly promotes the authenticated test user to admin (simulating an
    out-of-band admin grant) and confirms the admin-only endpoint then succeeds."""
    from app.database.session import get_db
    from app.database.repositories import UserRepository
    from app.database.models import UserRole

    db = next(get_db())
    try:
        user = UserRepository(db).get_by_id(auth_client.current_user_id)
        user.role = UserRole.ADMIN
        db.commit()
    finally:
        db.close()

    response = auth_client.get("/api/v1/admin/analytics")
    assert response.status_code == 200
