"""Multi-strategy ASGI authentication middleware for OMEGA.

Chain of trust (checked in order):
1. mTLS client certificate (agent-to-agent)
2. Bearer token (API token or session token via AuthEngine)
3. Auth0 / OIDC JWT (external IdP)
4. Legacy x-api-key header (backwards compat)

If none succeed and auth is required, returns 401.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("omega.server.auth_middleware")

_PUBLIC_PATHS = frozenset({
    "/health",
    "/healthz",
    "/.well-known/mcp.json",
    "/.well-known/mcp/server-card.json",
    "/.well-known/oauth-protected-resource",
    "/.well-known/openid-configuration",
})


class AuthMiddleware:
    """Pluggable ASGI middleware: mTLS → Bearer → OIDC JWT → API key."""

    def __init__(
        self,
        app,
        auth_engine=None,
        mtls_ca=None,
        oidc_provider=None,
        api_key: Optional[str] = None,
    ):
        self.app = app
        self.auth_engine = auth_engine
        self.mtls_ca = mtls_ca
        self.oidc_provider = oidc_provider
        self.api_key = api_key

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        if path in _PUBLIC_PATHS:
            return await self.app(scope, receive, send)

        # No auth configured at all → pass through (dev mode)
        if not self.auth_engine and not self.api_key and not _auth0_enabled():
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers", []))
        user = None

        # 1. mTLS
        if self.mtls_ca and not user:
            cert_pem = headers.get(b"ssl-client-cert", b"").decode("utf-8")
            if not cert_pem:
                # Check ASGI TLS extensions
                tls = scope.get("extensions", {}).get("tls", {})
                cert_pem = tls.get("client_cert_pem", "")
            if cert_pem:
                try:
                    info = self.mtls_ca.verify_client_cert(cert_pem)
                    if self.auth_engine:
                        user = self.auth_engine.get_user_by_id(info.get("user_id"))
                    if not user:
                        user = {"id": info.get("user_id"), "agent_name": info.get("agent_name")}
                except Exception as exc:
                    logger.debug("mTLS verification failed: %s", exc)

        # 2. Bearer token (AuthEngine: API token or session)
        auth_header = headers.get(b"authorization", b"").decode("utf-8")
        if self.auth_engine and not user and auth_header.lower().startswith("bearer "):
            token = auth_header.split(None, 1)[1]
            user = self.auth_engine.validate_api_token(token)
            if not user:
                user = self.auth_engine.validate_session(token)

        # 3. Auth0 / OIDC JWT
        if not user and auth_header.lower().startswith("bearer ") and _auth0_enabled():
            try:
                from omega.server.auth import validate_token
                payload = validate_token(auth_header)
                user = {"id": payload.get("sub"), "email": payload.get("email"), "jwt": True}
            except Exception as exc:
                logger.debug("JWT validation failed: %s", exc)

        # 4. Legacy x-api-key
        if not user and self.api_key:
            provided = headers.get(b"x-api-key", b"").decode("utf-8")
            if not provided:
                # Check query params
                from urllib.parse import parse_qs
                qs = scope.get("query_string", b"").decode("utf-8")
                params = parse_qs(qs)
                provided = params.get("api_key", [""])[0]
            if provided == self.api_key:
                user = {"id": "api_key_user", "api_key": True}

        if not user:
            return await self._send_401(scope, send)

        scope["user"] = user
        return await self.app(scope, receive, send)

    @staticmethod
    async def _send_401(scope, send):
        body = b'{"error": "Unauthorized"}'
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                [b"content-type", b"application/json"],
                [b"www-authenticate", b'Bearer realm="omega"'],
                [b"content-length", str(len(body)).encode()],
            ],
        })
        await send({"type": "http.response.body", "body": body})


def _auth0_enabled() -> bool:
    return bool(os.environ.get("AUTH0_DOMAIN") and os.environ.get("AUTH0_AUDIENCE"))
