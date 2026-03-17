"""OMEGA Hook Server — Unix Domain Socket daemon for fast hook dispatch.

Runs inside the MCP server process, reusing warm bridge/coordination singletons.
Hooks connect via ~/.omega/hook.sock, send a JSON request, and get a JSON response.
This eliminates ~750ms of cold-start overhead per hook invocation.

This package was split from a monolithic hook_server.py for maintainability.
All public names are re-exported here for backward compatibility.
"""

import logging
import sys
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger("omega.hook_server")

# Windows uses TCP loopback; Unix uses domain socket
if sys.platform == "win32":
    SOCK_PATH = None
    HOOK_HOST = "127.0.0.1"
    HOOK_PORT = 19876
else:
    SOCK_PATH = Path.home() / ".omega" / "hook.sock"
    HOOK_HOST = None
    HOOK_PORT = None

# Debounce state (in-memory, reset on server restart)
_last_heartbeat: OrderedDict[str, float] = OrderedDict()  # session_id -> monotonic timestamp
_last_claim: OrderedDict[tuple[str, str], float] = OrderedDict()  # (session_id, file_path) -> timestamp
_last_surface: OrderedDict[str, float] = OrderedDict()  # file_path -> timestamp (capped at 500)
_MAX_SURFACE_ENTRIES = 500
_last_overlap_notify: OrderedDict[tuple[str, str, str], float] = OrderedDict()  # (sid, other_sid, file) -> timestamp
OVERLAP_NOTIFY_DEBOUNCE_S = 300.0  # 5 minutes
_last_block_notify: OrderedDict[tuple[str, str, str], float] = OrderedDict()  # (blocked_sid, owner_sid, file) -> timestamp
BLOCK_NOTIFY_DEBOUNCE_S = 300.0  # 5 minutes
_last_peer_dir_check: OrderedDict[str, float] = OrderedDict()  # dir_path -> timestamp
PEER_DIR_CHECK_DEBOUNCE_S = 300.0  # 5 minutes
_last_coord_query: OrderedDict[str, float] = OrderedDict()  # session_id -> monotonic timestamp
COORD_QUERY_DEBOUNCE_S = 120.0  # 2 minutes
_last_reminder_check: float = 0.0
REMINDER_CHECK_DEBOUNCE_S = 300.0  # 5 minutes

_last_deadlock_push: OrderedDict[str, float] = OrderedDict()  # cycle_hash -> monotonic timestamp
DEADLOCK_PUSH_DEBOUNCE_S = 600.0  # 10 minutes — don't spam same cycle

_last_file_read: OrderedDict[tuple[str, str], float] = OrderedDict()  # (session_id, file_path) -> timestamp
FILE_READ_DEBOUNCE_S = 60.0  # Same (session, file) at most once per minute
_MAX_READ_ENTRIES = 1000

HEARTBEAT_DEBOUNCE_S = 30.0
CLAIM_DEBOUNCE_S = 30.0
SURFACE_DEBOUNCE_S = 15.0  # Was 5s — increased to reduce embedding pressure with 8-10 concurrent sessions

_heartbeat_count: dict[str, int] = {}  # session_id -> call count (for inbox surfacing)
_gate_call_count: dict[str, int] = {}  # session_id -> PreToolUse invocation count (independent of heartbeat debounce)
_drift_check_counter: dict[str, int] = {}  # session_id -> PostToolUse call count (for periodic drift check)
DRIFT_CHECK_INTERVAL = 20  # Check drift every N PostToolUse calls
_session_peer_count_time: dict[str, float] = {}  # session_id -> monotonic timestamp of last peer count check
_peer_snapshot: dict[str, set] = {}  # session_id -> set of known peer session_ids
_session_intent: dict[str, str] = {}  # session_id -> latest classified intent
_last_progress_beat: dict[str, int] = {}  # session_id -> heartbeat count at last progress call
_last_pulse_state: dict[str, str] = {}  # session_id -> "completed:in_prog:agents" for COORD-PULSE debounce
_last_project: dict[str, str] = {}  # session_id -> last known project path (for switch detection)

# Urgent message queue — push notifications from send_message to recipient's next heartbeat
_pending_urgent: dict[str, list[dict]] = {}  # session_id -> urgent message summaries
_MAX_URGENT_PER_SESSION = 10


# Approach-first tracking — domains for which a session has stated an approach (solo mode)
_session_approach_domains: dict[str, set[str]] = {}  # session_id -> set of domains with registered decisions
_session_approach_warned: dict[str, set[str]] = {}  # session_id -> set of domains already warned about

# Pre-irreversible advisory — track which action types have been warned per session
_session_external_actions: dict[str, set[str]] = {}  # session_id -> set of action_types warned

# Error dedup state (mirrors standalone surface_memories.py)
_error_hashes: set = set()  # capped at 200 per session
_MAX_ERROR_HASHES = 200
_error_counts: dict[str, int] = {}  # session_id -> error count
_MAX_ERRORS_PER_SESSION = 5

# Protocol compliance tracking — which required MCP tools each session has called
_protocol_calls: dict[str, set] = {}  # session_id -> set of tool names called

# Assistant capture tracking — how many captures per session (capped at 10)
_assistant_capture_count: dict[str, int] = {}  # session_id -> capture count
_session_peer_count: dict[str, int] = {}  # session_id -> cached peer count (0 or 1+)
SESSION_PEER_COUNT_TTL_S = 30.0  # refresh peer count cache after this many seconds

# Hook error rate tracking — errors per handler per session
_hook_error_counts: dict[str, int] = {}  # "handler:session_id" -> count

# Intelligence Card trackers — per-session complexity + surfacing state
_card_trackers: dict = {}  # session_id -> SessionCardTracker


def get_card_tracker(session_id: str) -> "SessionCardTracker":
    """Get or create a SessionCardTracker for the given session."""
    if session_id not in _card_trackers:
        from .card_tracker import SessionCardTracker

        _card_trackers[session_id] = SessionCardTracker(session_id)
    return _card_trackers[session_id]


def mark_protocol_call(session_id: str, tool_name: str) -> None:
    """Record that a session called a required protocol tool."""
    if session_id:
        _protocol_calls.setdefault(session_id, set()).add(tool_name)


# ---------------------------------------------------------------------------
# Agent nicknames — deterministic human-readable names from session IDs
# SYNC: This list must match ~/.claude/omega-statusline.sh _NAMES. Do not edit independently.
# ---------------------------------------------------------------------------

_AGENT_NAMES = [
    "Alder",
    "Aspen",
    "Birch",
    "Briar",
    "Brook",
    "Cedar",
    "Cliff",
    "Cloud",
    "Coral",
    "Cove",
    "Crane",
    "Creek",
    "Dale",
    "Dawn",
    "Dune",
    "Echo",
    "Elm",
    "Ember",
    "Fern",
    "Finch",
    "Flame",
    "Flint",
    "Flora",
    "Fox",
    "Frost",
    "Glen",
    "Grove",
    "Hare",
    "Haven",
    "Hawk",
    "Hazel",
    "Heath",
    "Heron",
    "Holly",
    "Iris",
    "Ivy",
    "Jade",
    "Jay",
    "Juniper",
    "Lake",
    "Lark",
    "Laurel",
    "Leaf",
    "Lily",
    "Maple",
    "Marsh",
    "Meadow",
    "Moss",
    "Myrtle",
    "Oak",
    "Olive",
    "Onyx",
    "Opal",
    "Orca",
    "Osprey",
    "Otter",
    "Pearl",
    "Pebble",
    "Pine",
    "Plum",
    "Quail",
    "Rain",
    "Raven",
    "Reed",
    "Ridge",
    "River",
    "Robin",
    "Rook",
    "Rose",
    "Rowan",
    "Rush",
    "Sage",
    "Shore",
    "Sky",
    "Slate",
    "Sparrow",
    "Stone",
    "Storm",
    "Swift",
    "Teal",
    "Thorn",
    "Thyme",
    "Tide",
    "Vale",
    "Vine",
    "Violet",
    "Willow",
    "Wren",
]



# ---------------------------------------------------------------------------
# Debounce state manager — wraps the dicts above with lifecycle methods
# ---------------------------------------------------------------------------

from .state import DebouncedState  # noqa: E402

_debounce_state = DebouncedState(
    last_heartbeat=_last_heartbeat,
    heartbeat_count=_heartbeat_count,
    gate_call_count=_gate_call_count,
    last_coord_query=_last_coord_query,
    last_progress_beat=_last_progress_beat,
    last_pulse_state=_last_pulse_state,
    session_intent=_session_intent,
    pending_urgent=_pending_urgent,
    error_counts=_error_counts,
    last_claim=_last_claim,
    last_overlap_notify=_last_overlap_notify,
    last_block_notify=_last_block_notify,
    last_file_read=_last_file_read,
    last_surface=_last_surface,
    last_peer_dir_check=_last_peer_dir_check,
    last_deadlock_push=_last_deadlock_push,
    error_hashes=_error_hashes,
    hook_error_counts=_hook_error_counts,
    peer_snapshot=_peer_snapshot,
    protocol_calls=_protocol_calls,
    session_peer_count=_session_peer_count,
    session_peer_count_time=_session_peer_count_time,
    assistant_capture_count=_assistant_capture_count,
    card_trackers=_card_trackers,
    session_approach_domains=_session_approach_domains,
    session_approach_warned=_session_approach_warned,
    last_project=_last_project,
    session_external_actions=_session_external_actions,
    drift_check_counter=_drift_check_counter,
)



# ── Re-exports from sub-modules ────────────────────────────────────────
# Import order matters: utils first (no cross-deps), then handlers, then core

from .utils import (  # noqa: E402
    _agent_nickname,
    notify_session,
    _drain_urgent_queue,
    _secure_append,
    _log_hook_error,
    _log_timing,
    _auto_cloud_sync,
    _resolve_entity,
    _relative_time_from_iso,
    _check_milestone,
    _try_acquire_periodic,
    _rollback_marker,
    _should_run_periodic,
    _update_marker,
    _parse_tool_input,
    _get_file_path_from_input,
    _debounce_check,
    _format_age,
    _append_output,
    _get_user_name,
    _agent_name_only,
    _get_last_session_info,
    _get_current_branch,
    _parse_checkout_target,
)

from .session import (  # noqa: E402
    handle_session_start,
    _auto_feedback_on_surfaced,
    handle_session_stop,
)

from .coordination import (  # noqa: E402
    handle_coord_session_start,
    _check_git_sync,
    _session_resume,
    handle_coord_session_stop,
)

from .memory import (  # noqa: E402
    handle_surface_memories,
    _auto_drift_check,
    _track_surfaced_ids,
    _ext_to_tags,
    _apply_confidence_boost,
    _surface_for_edit,
    _surface_lessons,
    _extract_error_summary,
    _capture_error,
    _track_git_commit,
    _auto_claim_branch,
)

from .heartbeat import (  # noqa: E402
    handle_coord_heartbeat,
)

from .assistant import (  # noqa: E402
    handle_assistant_capture,
    _clean_assistant_message,
    _extract_best_sentence,
)

from .insights import (  # noqa: E402
    handle_pre_insight_surface,
)

from .guards import (  # noqa: E402
    _suggest_alternative_dir,
    _infer_domain_from_path,
    _mark_approach_registered,
    classify_action_risk,
    handle_auto_claim_file,
    _file_guard_block_msg,
    handle_pre_file_guard,
    handle_pre_task_guard,
    _clean_task_text,
    handle_auto_capture,
    handle_pre_push_guard,
    _handle_push_divergence,
    _handle_branch_claims,
    _block_if_branch_claimed,
    _handle_auto_claim_branch,
    _log_push_event,
    handle_pre_commit_guard,
    handle_pre_deploy_guard,
    handle_pre_alignment_gate,
    handle_pre_protocol_gate,
    handle_pre_irreversible_advisor,
    handle_pre_agent_memory,
)

from .core import (  # noqa: E402, F401
    HOOK_HANDLERS,
    _CORE_HOOK_HANDLERS,
    _COMMERCIAL_HOOK_HANDLERS,
    register_hook_handler,
    handle_connection,
    start_hook_server,
    stop_hook_server,
)

__all__ = [
    "DebouncedState",
    "HOOK_HANDLERS",
    "HOOK_HOST",
    "HOOK_PORT",
    "SOCK_PATH",
    "_debounce_state",
    "_agent_name_only",
    "_agent_nickname",
    "_append_output",
    "_apply_confidence_boost",
    "_auto_claim_branch",
    "_assistant_capture_count",
    "_auto_cloud_sync",
    "_auto_drift_check",
    "_drift_check_counter",
    "DRIFT_CHECK_INTERVAL",
    "_auto_feedback_on_surfaced",
    "_session_approach_domains",
    "_session_approach_warned",
    "_block_if_branch_claimed",
    "_card_trackers",
    "_capture_error",
    "_clean_assistant_message",
    "_check_git_sync",
    "_check_milestone",
    "_clean_task_text",
    "_last_file_read",
    "FILE_READ_DEBOUNCE_S",
    "_MAX_READ_ENTRIES",
    "_debounce_check",
    "_drain_urgent_queue",
    "_ext_to_tags",
    "_extract_best_sentence",
    "_extract_error_summary",
    "_file_guard_block_msg",
    "_format_age",
    "_infer_domain_from_path",
    "_get_current_branch",
    "_get_file_path_from_input",
    "_get_last_session_info",
    "_get_user_name",
    "_handle_auto_claim_branch",
    "_handle_branch_claims",
    "_handle_push_divergence",
    "_log_hook_error",
    "_log_push_event",
    "_log_timing",
    "_parse_checkout_target",
    "_parse_tool_input",
    "_relative_time_from_iso",
    "_resolve_entity",
    "_rollback_marker",
    "_secure_append",
    "_session_resume",
    "_should_run_periodic",
    "_suggest_alternative_dir",
    "_surface_for_edit",
    "_surface_lessons",
    "_track_git_commit",
    "_track_surfaced_ids",
    "_try_acquire_periodic",
    "_update_marker",
    "classify_action_risk",
    "handle_assistant_capture",
    "handle_auto_capture",
    "handle_auto_claim_file",
    "handle_connection",
    "handle_coord_heartbeat",
    "handle_coord_session_start",
    "handle_coord_session_stop",
    "handle_pre_commit_guard",
    "_mark_approach_registered",
    "handle_pre_alignment_gate",
    "handle_pre_deploy_guard",
    "handle_pre_file_guard",
    "handle_pre_insight_surface",
    "handle_pre_irreversible_advisor",
    "handle_pre_agent_memory",
    "handle_pre_protocol_gate",
    "handle_pre_push_guard",
    "handle_pre_task_guard",
    "mark_protocol_call",
    "get_card_tracker",
    "handle_session_start",
    "handle_session_stop",
    "handle_surface_memories",
    "logger",
    "notify_session",
    "register_hook_handler",
    "start_hook_server",
    "stop_hook_server",
]

