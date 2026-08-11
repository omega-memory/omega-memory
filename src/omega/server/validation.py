"""Core MCP-handler input validators.

These helpers guard agent-supplied identifiers (session_id, entity_id)
against path traversal and injection before they reach the storage
layer. They live in Core — not Pro — because every MCP handler that
takes a session_id or entity_id needs them, and Core-only installs
must not import Pro modules at handler-load time.

The Pro package re-exports these functions from
``omega_platform.server.validation`` so existing Pro-side imports
continue to resolve. Path-level validators (validate_safe_path,
is_url_scheme) remain Pro-only.
"""

import logging
import re

logger = logging.getLogger("omega.server.validation")

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


def validate_session_id(session_id: str | None) -> str | None:
    """Validate session_id to prevent path traversal.

    Returns the cleaned session_id, or None if it contains traversal
    characters (``..``, ``/``, ``\\``) or characters outside the safe
    alphanumeric / dot / underscore / hyphen set.
    """
    if not session_id:
        return session_id
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        logger.warning("Rejected session_id with path traversal: %s", session_id[:50])
        return None
    if not _SAFE_ID_RE.match(session_id):
        logger.warning("Rejected session_id with invalid chars: %s", session_id[:50])
        return None
    return session_id


def validate_entity_id(entity_id: str | None) -> str | None:
    """Validate entity_id format.

    Allows alphanumerics, dots, underscores, and hyphens. Returns the
    cleaned entity_id, or None if any other character is present.
    """
    if not entity_id:
        return entity_id
    if not _SAFE_ID_RE.match(entity_id):
        logger.warning("Rejected entity_id with invalid chars: %s", entity_id[:50])
        return None
    return entity_id
