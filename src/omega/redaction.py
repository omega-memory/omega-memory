"""Remove credential-shaped values from text captured without an explicit ask.

Hooks capture prompts and transcripts as they pass by. A user who pastes a
key into a prompt has not asked for it to be remembered, and a memory store
is synced, exported and surfaced into later sessions. Explicit ``omega_store``
calls are not redacted: storing is then the caller's deliberate choice.
"""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"

# Provider formats: the whole match is the secret.
_SECRET_SHAPES = tuple(
    re.compile(pattern)
    for pattern in (
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)",
        r"\bAKIA[0-9A-Z]{16}\b",  # AWS access key ID
        r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",  # GitHub token
        r"\bgithub_pat_[A-Za-z0-9_]{22,}\b",
        r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b",  # Stripe
        r"\bsk-ant-[A-Za-z0-9_-]{20,}",  # Anthropic
        r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}",  # OpenAI
        r"\bxox[abprs]-[A-Za-z0-9-]{10,}",  # Slack
        r"\bAIza[0-9A-Za-z_-]{35}\b",  # Google API key
        r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",  # JWT
    )
)

# user:password@ in a connection URL: only the password is redacted.
_URL_PASSWORD = re.compile(r"(\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)([^\s@]+)(@)")

# "password is X", "token: X", "API_KEY=X": only the value is redacted, and
# only when it looks like a credential (a digit or symbol, or 16+ characters),
# so "the token is valid" survives.
_ASSIGNED_SECRET = re.compile(
    r"(\b(?:password|passwd|pwd|secret|token|api[ _-]?key|access[ _-]?key|auth[ _-]?key)"
    r"\s*(?:is|=|:)\s*)"
    r"(['\"]?)((?=[^\s'\"]*[0-9_\-+/=!@#$%^&*])[^\s'\"]{6,}|[^\s'\"]{16,})\2",
    re.IGNORECASE,
)

# Long base62/base64 runs mixing upper case, lower case and digits. Git SHAs
# and UUIDs are lower-case hex and are left alone.
_OPAQUE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_+/=-])(?=[A-Za-z0-9_+/=-]*[a-z])(?=[A-Za-z0-9_+/=-]*[A-Z])"
    r"(?=[A-Za-z0-9_+/=-]*[0-9])[A-Za-z0-9_+/=-]{24,}"
)


def redact_secrets(text: str) -> tuple[str, int]:
    """Replace credential-shaped values in ``text`` with ``[REDACTED]``.

    Returns the redacted text and the number of values replaced.
    """
    count = 0
    for shape in _SECRET_SHAPES:
        text, n = shape.subn(REDACTED, text)
        count += n
    text, n = _URL_PASSWORD.subn(lambda m: f"{m.group(1)}{REDACTED}{m.group(3)}", text)
    count += n
    text, n = _ASSIGNED_SECRET.subn(lambda m: f"{m.group(1)}{REDACTED}", text)
    count += n
    text, n = _OPAQUE_TOKEN.subn(REDACTED, text)
    count += n
    return text, count
