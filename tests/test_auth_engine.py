"""Unit tests for omega.auth.engine using SQLite in-memory database."""
from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from omega.auth.engine import AuthEngine, _hash_password, _verify_password
from omega.db.models import Base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine():
    """Create an in-memory SQLite engine with all tables."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture()
def auth(engine):
    """Return an AuthEngine wired to the in-memory database."""
    sf = sessionmaker(bind=engine)
    return AuthEngine(sf)


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

class TestPasswordHashing:
    def test_roundtrip(self):
        hashed = _hash_password("hunter2")
        assert _verify_password("hunter2", hashed) is True

    def test_wrong_password(self):
        hashed = _hash_password("hunter2")
        assert _verify_password("wrong", hashed) is False

    def test_different_salts(self):
        h1 = _hash_password("same")
        h2 = _hash_password("same")
        # Different salts produce different hashes
        assert h1 != h2
        # But both verify correctly
        assert _verify_password("same", h1)
        assert _verify_password("same", h2)


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

class TestUserManagement:
    def test_create_user(self, auth: AuthEngine):
        user = auth.create_user("alice@example.com", "secret")
        assert user.email == "alice@example.com"
        assert user.id is not None
        assert user.is_active == 1

    def test_verify_password_success(self, auth: AuthEngine):
        auth.create_user("alice@example.com", "secret")
        user = auth.verify_password("alice@example.com", "secret")
        assert user is not None
        assert user.email == "alice@example.com"

    def test_verify_password_wrong(self, auth: AuthEngine):
        auth.create_user("alice@example.com", "secret")
        assert auth.verify_password("alice@example.com", "bad") is None

    def test_verify_password_no_such_user(self, auth: AuthEngine):
        assert auth.verify_password("nobody@example.com", "x") is None


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

class TestSessionManagement:
    def test_create_and_validate_session(self, auth: AuthEngine):
        user = auth.create_user("bob@example.com", "pw")
        token = auth.create_session(user.id)
        assert isinstance(token, str)
        assert len(token) > 20

        validated = auth.validate_session(token)
        assert validated is not None
        assert validated.id == user.id

    def test_invalid_token_returns_none(self, auth: AuthEngine):
        assert auth.validate_session("bogus-token") is None

    def test_revoke_session(self, auth: AuthEngine):
        user = auth.create_user("carol@example.com", "pw")
        token = auth.create_session(user.id)
        assert auth.validate_session(token) is not None

        auth.revoke_session(token)
        assert auth.validate_session(token) is None

    def test_session_with_device_info(self, auth: AuthEngine):
        user = auth.create_user("dave@example.com", "pw")
        token = auth.create_session(user.id, device_info="Firefox on macOS")
        assert auth.validate_session(token) is not None


# ---------------------------------------------------------------------------
# API tokens
# ---------------------------------------------------------------------------

class TestAPITokens:
    def test_create_and_validate(self, auth: AuthEngine):
        user = auth.create_user("eve@example.com", "pw")
        raw = auth.create_api_token(user.id, "my-token")
        assert isinstance(raw, str)

        validated = auth.validate_api_token(raw)
        assert validated is not None
        assert validated.id == user.id

    def test_invalid_api_token(self, auth: AuthEngine):
        assert auth.validate_api_token("not-a-real-token") is None

    def test_scoped_token(self, auth: AuthEngine):
        user = auth.create_user("frank@example.com", "pw")
        raw = auth.create_api_token(user.id, "read-only", scopes=["read"])
        assert auth.validate_api_token(raw) is not None

    def test_expired_token(self, auth: AuthEngine):
        user = auth.create_user("grace@example.com", "pw")
        # Create a token that expires in the past
        raw = auth.create_api_token(
            user.id, "expired", expires_in=timedelta(seconds=-1),
        )
        assert auth.validate_api_token(raw) is None

    def test_future_expiry_token(self, auth: AuthEngine):
        user = auth.create_user("heidi@example.com", "pw")
        raw = auth.create_api_token(
            user.id, "long-lived", expires_in=timedelta(days=365),
        )
        assert auth.validate_api_token(raw) is not None


# ---------------------------------------------------------------------------
# Expired sessions
# ---------------------------------------------------------------------------

class TestExpiredSessions:
    def test_expired_session(self, engine):
        sf = sessionmaker(bind=engine)
        # Very short TTL
        auth = AuthEngine(sf, session_ttl_hours=0)
        user = auth.create_user("ivan@example.com", "pw")
        token = auth.create_session(user.id)
        # Session created with 0-hour TTL is already expired
        assert auth.validate_session(token) is None
