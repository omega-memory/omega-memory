"""OMEGA hook server: an in-process daemon for fast Claude Code hook dispatch.

Runs inside the MCP server process and reuses its warm bridge singletons.
``fast_hook.py`` connects over ``~/.omega/hook.sock`` (TCP loopback on
Windows), sends one JSON request, and reads one JSON response. That replaces
a ~750ms interpreter cold start per hook with a few milliseconds.

This package is the community edition: it serves the five hooks that
``omega setup`` registers. omega-pro extends it through
:func:`register_hook_handler` and the state registry below.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger("omega.hook_server")

# Windows uses TCP loopback; everything else uses a Unix domain socket.
# fast_hook.py hardcodes the same locations, so these stay under ~/.omega
# regardless of OMEGA_HOME.
if sys.platform == "win32":
    SOCK_PATH = None
    HOOK_HOST = "127.0.0.1"
    HOOK_PORT = 19876
else:
    SOCK_PATH = Path.home() / ".omega" / "hook.sock"
    HOOK_HOST = None
    HOOK_PORT = None

# ---------------------------------------------------------------------------
# In-memory state (reset on server restart)
# ---------------------------------------------------------------------------

# file_path -> monotonic timestamp of the last surfacing for that file
_last_surface: OrderedDict[str, float] = OrderedDict()
SURFACE_DEBOUNCE_S = 15.0
_MAX_SURFACE_ENTRIES = 500

# session_id -> names of the protocol tools (omega_welcome, omega_protocol)
# the session has called. Written by the MCP tool handlers.
_protocol_calls: dict[str, set[str]] = {}

# Counters the Pro coordination handlers populate. Empty on core installs;
# defined here so omega_protocol(section="gate_status") can read them.
_heartbeat_count: dict[str, int] = {}
_gate_call_count: dict[str, int] = {}
_session_peer_count: dict[str, int] = {}
_session_peer_count_time: dict[str, float] = {}
_last_deadlock_push: OrderedDict[str, float] = OrderedDict()
DEADLOCK_PUSH_DEBOUNCE_S = 600.0

from .state import DebouncedState  # noqa: E402

_debounce_state = DebouncedState()
_debounce_state.register_time_keyed("last_surface", _last_surface)
_debounce_state.register_time_keyed("last_deadlock_push", _last_deadlock_push)
for _name, _mapping in (
    ("protocol_calls", _protocol_calls),
    ("heartbeat_count", _heartbeat_count),
    ("gate_call_count", _gate_call_count),
    ("session_peer_count", _session_peer_count),
    ("session_peer_count_time", _session_peer_count_time),
):
    _debounce_state.register_session_keyed(_name, _mapping)

# The hook modules keep their own per-session caps; release them with the session.
from omega.hooks import assistant_capture as _assistant_capture  # noqa: E402
from omega.hooks import auto_capture as _auto_capture  # noqa: E402
from omega.hooks import surface_memories as _surface_memories  # noqa: E402

_debounce_state.register_session_keyed("auto_capture_count", _auto_capture._captures_by_session)
_debounce_state.register_session_keyed("assistant_capture_count", _assistant_capture._captures_by_session)
_debounce_state.register_session_keyed("error_hashes", _surface_memories._error_hashes_by_session)
_debounce_state.register_session_keyed("error_count", _surface_memories._error_count_by_session)


def mark_protocol_call(session_id: str, tool_name: str) -> None:
    """Record that ``session_id`` called a required protocol tool.

    Also drops a marker file under ``~/.omega/gates/`` so standalone hook
    scripts can check protocol compliance when the daemon is unavailable.
    An empty session id records nothing: guessing a session would attribute
    the call to the wrong agent in multi-session setups.
    """
    if not session_id:
        return
    _protocol_calls.setdefault(session_id, set()).add(tool_name)
    try:
        gate_dir = Path.home() / ".omega" / "gates"
        gate_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        (gate_dir / f"{session_id}.{tool_name}").write_text(str(time.time()))
    except OSError:
        logger.debug("protocol gate marker not written", exc_info=True)


from .utils import _agent_nickname, _log_hook_error, _log_timing  # noqa: E402
from .handlers import (  # noqa: E402
    handle_assistant_capture,
    handle_auto_capture,
    handle_session_start,
    handle_session_stop,
    handle_surface_memories,
)
from .core import (  # noqa: E402
    _CORE_HOOK_HANDLERS,
    HOOK_HANDLERS,
    handle_connection,
    register_hook_handler,
    start_hook_server,
    stop_hook_server,
)

__all__ = [
    "DEADLOCK_PUSH_DEBOUNCE_S",
    "DebouncedState",
    "HOOK_HANDLERS",
    "HOOK_HOST",
    "HOOK_PORT",
    "SOCK_PATH",
    "SURFACE_DEBOUNCE_S",
    "_CORE_HOOK_HANDLERS",
    "_agent_nickname",
    "_debounce_state",
    "_gate_call_count",
    "_heartbeat_count",
    "_last_deadlock_push",
    "_last_surface",
    "_log_hook_error",
    "_log_timing",
    "_protocol_calls",
    "_session_peer_count",
    "_session_peer_count_time",
    "handle_assistant_capture",
    "handle_auto_capture",
    "handle_connection",
    "handle_session_start",
    "handle_session_stop",
    "handle_surface_memories",
    "logger",
    "mark_protocol_call",
    "register_hook_handler",
    "start_hook_server",
    "stop_hook_server",
]
