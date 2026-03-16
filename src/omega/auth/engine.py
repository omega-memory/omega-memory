"""Core authentication engine for OMEGA multi-tenant auth.

Uses hashlib.scrypt for password hashing, secrets.token_urlsafe for token
generation, and hashlib.sha256 for token storage hashes.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from sqlalchemy import select

from omega.db.models import APIToken, User, UserSession

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as SASession, sessionmaker


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: bytes | None = None) -> str:
    """Hash *password* with scrypt, returning ``salt_hex:hash_hex``."""
    if salt is None:
        salt = os.urandom(16)
    derived = hashlib.scrypt(
        password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32,
    )
    return salt.hex() + ":" + derived.hex()


def _verify_password(password: str, stored: str) -> bool:
    """Verify *password* against a ``salt_hex:hash_hex`` string."""
    salt_hex, hash_hex = stored.split(":", 1)
    salt = bytes.fromhex(salt_hex)
    derived = hashlib.scrypt(
        password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32,
    )
    return secrets.compare_digest(derived.hex(), hash_hex)


def _hash_token(token: str) -> str:
    """SHA-256 hex digest of a raw bearer / session token."""
    return hashlib.sha256(token.encode()).hexdigest()


# ---------------------------------------------------------------------------
# AuthEngine
# ---------------------------------------------------------------------------

class AuthEngine:
    """Stateless authentication engine backed by SQLAlchemy sessions.

    Parameters
    ----------
    session_factory:
        A ``sessionmaker`` (or compatible callable) that returns a
        ``sqlalchemy.orm.Session``.
    session_ttl_hours:
        Lifetime of browser / device sessions in hours.  Defaults to 72.
    """

    def __init__(
        self,
        session_factory: sessionmaker,
        session_ttl_hours: int = 72,
    ) -> None:
        self._sf = session_factory
        self._session_ttl = timedelta(hours=session_ttl_hours)

    # -- helpers ------------------------------------------------------------

    def _now(self) -> str:  # noqa: PLR6301
        return datetime.now(timezone.utc).isoformat()

    # -- user management ----------------------------------------------------

    def create_user(self, email: str, password: str) -> User:
        """Create a new user and return the ORM instance."""
        now = self._now()
        user = User(
            id=str(uuid4()),
            email=email,
            password_hash=_hash_password(password),
            created_at=now,
            updated_at=now,
            is_active=1,
        )
        with self._sf() as session:
            session.add(user)
            session.commit()
            session.refresh(user)
            # Expunge so caller can use object outside session
            session.expunge(user)
        return user

    def verify_password(self, email: str, password: str) -> Optional[User]:
        """Return the ``User`` if *email* and *password* match, else ``None``."""
        with self._sf() as session:
            user = session.execute(
                select(User).where(User.email == email, User.is_active == 1),
            ).scalar_one_or_none()
            if user is None or user.password_hash is None:
                return None
            if not _verify_password(password, user.password_hash):
                return None
            session.expunge(user)
            return user

    # -- session management -------------------------------------------------

    def create_session(
        self,
        user_id: str,
        device_info: str | None = None,
    ) -> str:
        """Create a new user session and return the raw token."""
        raw_token = secrets.token_urlsafe(48)
        now = self._now()
        expires = (
            datetime.now(timezone.utc) + self._session_ttl
        ).isoformat()
        us = UserSession(
            id=str(uuid4()),
            user_id=user_id,
            token_hash=_hash_token(raw_token),
            device_info=device_info,
            created_at=now,
            expires_at=expires,
            is_revoked=0,
        )
        with self._sf() as session:
            session.add(us)
            session.commit()
        return raw_token

    def validate_session(self, token: str) -> Optional[User]:
        """Return the ``User`` for a valid, non-expired session token."""
        hashed = _hash_token(token)
        now = self._now()
        with self._sf() as session:
            us = session.execute(
                select(UserSession).where(
                    UserSession.token_hash == hashed,
                    UserSession.is_revoked == 0,
                ),
            ).scalar_one_or_none()
            if us is None or us.expires_at < now:
                return None
            user = session.execute(
                select(User).where(User.id == us.user_id, User.is_active == 1),
            ).scalar_one_or_none()
            if user is not None:
                session.expunge(user)
            return user

    def revoke_session(self, token: str) -> None:
        """Revoke a session token so it can no longer be used."""
        hashed = _hash_token(token)
        with self._sf() as session:
            us = session.execute(
                select(UserSession).where(UserSession.token_hash == hashed),
            ).scalar_one_or_none()
            if us is not None:
                us.is_revoked = 1
                session.commit()

    # -- API tokens ---------------------------------------------------------

    def create_api_token(
        self,
        user_id: str,
        name: str,
        scopes: list[str] | None = None,
        expires_in: timedelta | None = None,
    ) -> str:
        """Create an API token and return the raw bearer value."""
        raw_token = secrets.token_urlsafe(48)
        now = self._now()
        expires_at: str | None = None
        if expires_in is not None:
            expires_at = (
                datetime.now(timezone.utc) + expires_in
            ).isoformat()

        api_tok = APIToken(
            id=str(uuid4()),
            user_id=user_id,
            token_hash=_hash_token(raw_token),
            name=name,
            scopes=json.dumps(scopes) if scopes else None,
            expires_at=expires_at,
            created_at=now,
            is_revoked=0,
        )
        with self._sf() as session:
            session.add(api_tok)
            session.commit()
        return raw_token

    def validate_api_token(self, token: str) -> Optional[User]:
        """Return the ``User`` associated with a valid API token."""
        hashed = _hash_token(token)
        now = self._now()
        with self._sf() as session:
            api_tok = session.execute(
                select(APIToken).where(
                    APIToken.token_hash == hashed,
                    APIToken.is_revoked == 0,
                ),
            ).scalar_one_or_none()
            if api_tok is None:
                return None
            if api_tok.expires_at is not None and api_tok.expires_at < now:
                return None
            # Update last_used_at
            api_tok.last_used_at = now
            session.commit()

            user = session.execute(
                select(User).where(
                    User.id == api_tok.user_id, User.is_active == 1,
                ),
            ).scalar_one_or_none()
            if user is not None:
                session.expunge(user)
            return user
