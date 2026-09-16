"""Helpers shared by the hook daemon's handlers and server core."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("omega.hook_server")


def omega_home() -> Path:
    """The OMEGA data directory, honouring ``OMEGA_HOME`` like the bridge does."""
    return Path(os.environ.get("OMEGA_HOME", str(Path.home() / ".omega")))


_MAX_LOG_BYTES = 5 * 1024 * 1024


def _rotate_if_needed(log_path: Path) -> None:
    """Keep hooks.log bounded: one previous generation, same cap as the hook scripts."""
    try:
        if log_path.exists() and log_path.stat().st_size > _MAX_LOG_BYTES:
            log_path.replace(log_path.with_suffix(".log.1"))
    except OSError:
        logger.debug("hooks.log rotation failed", exc_info=True)


def _secure_append(log_path: Path, data: str) -> None:
    """Append to a file that only the owner can read (0o600)."""
    log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _rotate_if_needed(log_path)
    fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data.encode("utf-8"))
    finally:
        os.close(fd)


def _log_hook_error(hook_name: str, error: Exception, *, session_id: str = "") -> None:
    """Record a handler failure in ``hooks.log`` and the server log."""
    logger.warning("hook %s failed: %s", hook_name, error, exc_info=True)
    try:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        sid_tag = f" sid={session_id[:12]}" if session_id else ""
        _secure_append(
            omega_home() / "hooks.log",
            f"[{timestamp}] hook_server/{hook_name} [{type(error).__name__}]{sid_tag}: {error}\n"
            f"{traceback.format_exc()}\n",
        )
    except OSError:
        logger.debug("could not write hooks.log", exc_info=True)


def _log_timing(hook_name: str, elapsed_ms: float) -> None:
    """Record one handler's wall time in ``hooks.log``."""
    try:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _secure_append(omega_home() / "hooks.log", f"[{timestamp}] hook_server/{hook_name}: OK ({elapsed_ms:.0f}ms)\n")
    except OSError:
        logger.debug("could not write hooks.log", exc_info=True)


_debounce_lock = threading.Lock()


def _debounce_check(cache: OrderedDict, key, debounce_s: float, max_entries: int) -> bool:
    """Return True when ``key`` may proceed, recording it; False when seen within ``debounce_s``.

    Keeps the cache bounded at ``max_entries`` by evicting the least recently
    used key. Handlers run on a thread pool, so the check-then-move sequence
    is locked; without it a concurrent eviction between the membership test
    and ``move_to_end`` raises KeyError.
    """
    now = time.monotonic()
    with _debounce_lock:
        if key in cache and now - cache[key] < debounce_s:
            cache.move_to_end(key)
            return False
        cache[key] = now
        cache.move_to_end(key)
        if len(cache) > max_entries:
            cache.popitem(last=False)
    return True


def _parse_tool_input(payload: dict) -> dict:
    """Parse ``tool_input`` (a JSON string on the wire) into a dict; ``{}`` on failure."""
    raw = payload.get("tool_input", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _get_file_path_from_input(input_data: dict) -> str:
    """The file a tool call targets: ``file_path`` for Edit/Write/Read, ``notebook_path`` for NotebookEdit."""
    return input_data.get("file_path", input_data.get("notebook_path", ""))


# Deterministic human-readable names for session ids. Kept for API parity with
# the Pro coordination layer, which imports it from this package.
_AGENT_NAMES = [
    "Alder", "Aspen", "Birch", "Briar", "Brook", "Cedar", "Cliff", "Cloud",
    "Coral", "Cove", "Crane", "Creek", "Dale", "Dawn", "Dune", "Echo",
    "Elm", "Ember", "Fern", "Finch", "Flame", "Flint", "Flora", "Fox",
    "Frost", "Glen", "Grove", "Hare", "Haven", "Hawk", "Hazel", "Heath",
    "Heron", "Holly", "Iris", "Ivy", "Jade", "Jay", "Juniper", "Lake",
    "Lark", "Laurel", "Leaf", "Lily", "Maple", "Marsh", "Meadow", "Moss",
    "Myrtle", "Oak", "Olive", "Onyx", "Opal", "Orca", "Osprey", "Otter",
    "Pearl", "Pebble", "Pine", "Plum", "Quail", "Rain", "Raven", "Reed",
    "Ridge", "River", "Robin", "Rook", "Rose", "Rowan", "Rush", "Sage",
    "Shore", "Sky", "Slate", "Sparrow", "Stone", "Storm", "Swift", "Teal",
    "Thorn", "Thyme", "Tide", "Vale", "Vine", "Violet", "Willow", "Wren",
]


def _agent_nickname(session_id: str) -> str:
    """Deterministic human-readable nickname from a session id."""
    if not session_id:
        return "Unknown"
    digest = int(hashlib.md5(session_id.encode()).hexdigest()[:8], 16)
    return _AGENT_NAMES[digest % len(_AGENT_NAMES)]
