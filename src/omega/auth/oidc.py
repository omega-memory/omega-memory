"""OpenID Connect provider integration for OMEGA.

Uses only stdlib (urllib) for HTTP.  PyJWT is an optional dependency
used for local ID-token validation; if missing, ``validate_id_token``
raises ``RuntimeError``.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

try:
    import jwt as _jwt  # PyJWT
except ImportError:  # pragma: no cover
    _jwt = None  # type: ignore[assignment]


class OIDCProvider:
    """Lightweight OIDC relying-party client.

    Parameters
    ----------
    discovery_url:
        The ``/.well-known/openid-configuration`` endpoint URL.
    client_id:
        OAuth 2 client identifier.
    client_secret:
        OAuth 2 client secret (optional for public clients / PKCE).
    """

    def __init__(
        self,
        discovery_url: str,
        client_id: str,
        client_secret: str | None = None,
    ) -> None:
        self.discovery_url = discovery_url
        self.client_id = client_id
        self.client_secret = client_secret

        # Cached discovery document
        self._discovery: dict[str, Any] | None = None
        self._discovery_ts: float = 0.0
        self._cache_ttl: float = 3600.0  # 1 hour

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _fetch_discovery(self) -> dict[str, Any]:
        now = time.monotonic()
        if self._discovery is not None and (now - self._discovery_ts) < self._cache_ttl:
            return self._discovery
        req = urllib.request.Request(self.discovery_url)
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        self._discovery = data
        self._discovery_ts = now
        return data

    @property
    def discovery(self) -> dict[str, Any]:
        return self._fetch_discovery()

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------

    def get_authorization_url(
        self,
        redirect_uri: str,
        state: str,
        scope: str = "openid email profile",
        nonce: str | None = None,
    ) -> str:
        """Build the authorization endpoint redirect URL."""
        disc = self.discovery
        params: dict[str, str] = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "state": state,
        }
        if nonce is not None:
            params["nonce"] = nonce
        return disc["authorization_endpoint"] + "?" + urllib.parse.urlencode(params)

    # ------------------------------------------------------------------
    # Token exchange
    # ------------------------------------------------------------------

    def exchange_code(
        self,
        code: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """Exchange an authorization *code* for tokens."""
        disc = self.discovery
        payload: dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self.client_id,
        }
        if self.client_secret is not None:
            payload["client_secret"] = self.client_secret

        data = urllib.parse.urlencode(payload).encode()
        req = urllib.request.Request(
            disc["token_endpoint"],
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            return json.loads(resp.read().decode())  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # User info
    # ------------------------------------------------------------------

    def get_userinfo(self, access_token: str) -> dict[str, Any]:
        """Call the UserInfo endpoint with *access_token*."""
        disc = self.discovery
        req = urllib.request.Request(
            disc["userinfo_endpoint"],
            headers={"Authorization": f"Bearer {access_token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            return json.loads(resp.read().decode())  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # ID token validation (requires PyJWT + cryptography)
    # ------------------------------------------------------------------

    def validate_id_token(
        self,
        id_token: str,
        nonce: Optional[str] = None,
    ) -> dict[str, Any]:
        """Decode and validate an ID token.

        Requires ``PyJWT[crypto]``.  Raises ``RuntimeError`` if the
        library is not installed.
        """
        if _jwt is None:
            raise RuntimeError(
                "PyJWT is required for ID-token validation.  "
                "Install it with: pip install PyJWT[crypto]"
            )
        disc = self.discovery
        jwks_uri = disc["jwks_uri"]
        # Fetch JWKS
        req = urllib.request.Request(jwks_uri)
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            jwks = json.loads(resp.read().decode())

        # Build signing keys
        signing_keys = _jwt.PyJWKSet.from_dict(jwks)
        header = _jwt.get_unverified_header(id_token)
        key = signing_keys[header["kid"]]

        options: dict[str, Any] = {}
        if nonce is not None:
            options["nonce"] = nonce

        return _jwt.decode(  # type: ignore[no-any-return]
            id_token,
            key=key.key,
            algorithms=[header.get("alg", "RS256")],
            audience=self.client_id,
            issuer=disc.get("issuer"),
            options=options,
        )
