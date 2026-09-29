"""Access control for the OMEGA MCP HTTP daemon (``omega serve --daemon`` / ``omega serve install``).

The daemon listens on loopback, but loopback is reachable from any web page
the user opens: a page can POST to ``http://127.0.0.1:<port>/mcp`` directly,
or reach it by DNS rebinding. Two checks close that:

- Host and Origin validation (the MCP SDK's ``TransportSecuritySettings``),
  which defeats DNS rebinding.
- A bearer key, on by default, which a page cannot know. It lives in
  ``$OMEGA_HOME/mcp_api_key`` (mode 0600) and is created on first use by
  either the daemon or ``omega serve migrate-config``, which writes it into
  the Claude Code entry it configures. ``OMEGA_MCP_API_KEY`` overrides it.

This module imports nothing heavy so the CLI can share it without loading
the server.
"""

from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path

API_KEY_ENV = "OMEGA_MCP_API_KEY"
_KEY_FILE_NAME = "mcp_api_key"
# Paths that answer without the key: liveness only, still Host-checked.
_UNAUTHENTICATED_PATHS = frozenset({"/health"})


def api_key_path() -> Path:
    """Where the daemon's bearer key is kept: ``$OMEGA_HOME/mcp_api_key``."""
    return Path(os.environ.get("OMEGA_HOME", str(Path.home() / ".omega"))) / _KEY_FILE_NAME


def resolve_api_key(*, create: bool = True) -> str | None:
    """The bearer key the daemon requires.

    ``OMEGA_MCP_API_KEY`` wins, then the key file. With ``create`` a missing
    key file is generated (0600); without it, None is returned instead, for
    read-only callers such as ``omega doctor``.
    """
    env_key = os.environ.get(API_KEY_ENV, "").strip()
    if env_key:
        return env_key
    path = api_key_path()
    if path.exists():
        file_key = path.read_text(encoding="utf-8").strip()
        if file_key:
            return file_key
    if not create:
        return None
    key = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(key + "\n")
    return key


def allowed_hosts(host: str, port: int) -> list[str]:
    """Host header values a local client may send to the daemon."""
    return list(dict.fromkeys([f"{host}:{port}", f"127.0.0.1:{port}", f"localhost:{port}"]))


def allowed_origins(host: str, port: int) -> list[str]:
    """Origin header values accepted when a client sends one (browsers always do)."""
    return [f"http://{value}" for value in allowed_hosts(host, port)]


def authorization_header(api_key: str) -> dict[str, str]:
    """The header block a client configuration needs, e.g. Claude Code's ``headers``."""
    return {"Authorization": f"Bearer {api_key}"}


class BearerKeyMiddleware:
    """ASGI middleware: every HTTP request except ``/health`` must carry the bearer key."""

    def __init__(self, app, api_key: str):
        self.app = app
        self._expected = f"Bearer {api_key}".encode("utf-8")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") in _UNAUTHENTICATED_PATHS:
            await self.app(scope, receive, send)
            return
        presented = dict(scope.get("headers") or []).get(b"authorization", b"")
        if hmac.compare_digest(presented, self._expected):
            await self.app(scope, receive, send)
            return
        body = b'{"error":"unauthorized: send Authorization: Bearer <key from omega serve migrate-config>"}'
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b'Bearer realm="omega-mcp"'),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body})
