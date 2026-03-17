"""Safety guards — file, task, push, commit, and deploy."""

import logging
import json
import os
import re
import subprocess
import time

logger = logging.getLogger("omega.hook_server.guards")

from . import (
    BLOCK_NOTIFY_DEBOUNCE_S,
    CLAIM_DEBOUNCE_S,
    COORD_QUERY_DEBOUNCE_S,
    OVERLAP_NOTIFY_DEBOUNCE_S,
    SESSION_PEER_COUNT_TTL_S,
    _MAX_SURFACE_ENTRIES,
    _gate_call_count,
    _last_block_notify,
    _last_claim,
    _last_coord_query,
    _last_overlap_notify,
    _protocol_calls,
    _session_approach_domains,
    _session_approach_warned,
    _session_external_actions,
    _session_intent,
    _session_peer_count,
    _session_peer_count_time,
)

from .utils import (
    _agent_nickname,
    _debounce_check,
    _get_current_branch,
    _get_file_path_from_input,
    _log_hook_error,
    _omega_dir,
    _parse_checkout_target,
    _parse_tool_input,
    _resolve_entity,
)



_PLAN_DIRS = [
    "/.claude/plans/",     # Claude Code
    "/.cursor/plans/",     # Cursor
]


def _is_plan_file(file_path: str) -> bool:
    """Check if file is a plan file from any supported client (session-private, no coordination needed)."""
    if not file_path:
        return False
    return any(d in file_path for d in _PLAN_DIRS)


_DOMAIN_MAP = {
    "src/auth": "auth",
    "src/omega/auth": "auth",
    "src/omega/coordination": "coordination",
    "src/omega/knowledge": "knowledge",
    "src/omega/server": "server",
    "src/omega/cloud": "cloud",
    "src/omega/entity": "entity",
    "tests": "testing",
    "hooks": "hooks",
    "docs": "docs",
    "website": "website",
    ".github": "ci",
}


def _infer_domain_from_path(file_path: str) -> str | None:
    """Infer a decision domain from a file path.

    Walks path components to find the best matching domain key.
    Returns None if no domain can be inferred.
    """
    if not file_path:
        return None
    # Normalize to forward slashes, strip leading /
    norm = file_path.replace("\\", "/")
    # Try longest prefix matches first
    for prefix, domain in sorted(_DOMAIN_MAP.items(), key=lambda x: -len(x[0])):
        if f"/{prefix}/" in norm or norm.endswith(f"/{prefix}"):
            return domain
    # Fallback: use first meaningful directory after src/
    parts = norm.split("/")
    for i, p in enumerate(parts):
        if p == "src" and i + 1 < len(parts):
            return parts[i + 1]
    return None


# Debounce state for alignment gate
from collections import OrderedDict as _OrderedDict
_last_alignment_check: _OrderedDict[tuple[str, str], float] = _OrderedDict()
ALIGNMENT_DEBOUNCE_S = 60.0


def _mark_approach_registered(session_id: str, domain: str) -> None:
    """Mark that a session has registered a decision/approach for a domain.

    Called when omega_decision_register is invoked, so the approach-first
    reminder won't fire for that domain again.
    """
    _session_approach_domains.setdefault(session_id, set()).add(domain)


def handle_pre_alignment_gate(payload: dict) -> dict:
    """Check planned action against active decisions.

    Multi-agent: Surfaces [ALIGNMENT] info for file edits in domains with active decisions.
    Solo: Surfaces [APPROACH-FIRST] reminder on first edit in a new domain if no
          approach has been registered via omega_decision_register. Non-blocking.
    Fail-open: errors never block.
    """
    tool_name = payload.get("tool_name", "")
    if tool_name not in ("Edit", "Write", "NotebookEdit", "Bash"):
        return {"output": "", "error": None}

    session_id = payload.get("session_id", "")
    if not session_id:
        return {"output": "", "error": None}

    # Extract domain from file path or Bash command
    input_data = _parse_tool_input(payload)
    domain = None

    if tool_name in ("Edit", "Write", "NotebookEdit"):
        file_path = _get_file_path_from_input(input_data)
        if file_path:
            domain = _infer_domain_from_path(file_path)
    elif tool_name == "Bash":
        command = input_data.get("command", "")
        for token in command.split():
            if "/" in token and not token.startswith("-"):
                domain = _infer_domain_from_path(token)
                if domain:
                    break

    if not domain:
        return {"output": "", "error": None}

    # Check single-agent vs multi-agent
    is_solo = True
    mgr = None
    try:
        from omega.coordination import get_manager

        mgr = get_manager()
        if mgr.active_session_count() > 1:
            is_solo = False
    except Exception as e:
        logger.debug("alignment gate agent count check failed: %s", e)

    if is_solo:
        # Solo mode: approach-first reminder (non-blocking, once per domain)
        approached = _session_approach_domains.get(session_id, set())
        warned = _session_approach_warned.get(session_id, set())

        if domain in approached or domain in warned:
            return {"output": "", "error": None}

        # Mark as warned so we don't nag again
        _session_approach_warned.setdefault(session_id, set()).add(domain)

        return {
            "output": (
                f"[APPROACH-FIRST] First edit in domain '{domain}'. "
                f"State your approach before cross-domain or multi-module changes. "
                f"Register with omega_decision_register to suppress this reminder."
            ),
            "error": None,
        }

    # Multi-agent path: debounce per (session, domain)
    if not _debounce_check(
        _last_alignment_check, (session_id, domain), ALIGNMENT_DEBOUNCE_S, _MAX_SURFACE_ENTRIES
    ):
        return {"output": "", "error": None}

    project = payload.get("project", "")
    if not project:
        return {"output": "", "error": None}

    try:
        decisions = mgr.query_decisions(project=project, domain=domain, status="active", limit=3)
        if not decisions:
            return {"output": "", "error": None}

        lines = [f"[ALIGNMENT] {len(decisions)} active decision(s) in domain '{domain}':"]
        for d in decisions:
            lines.append(f"  #{d['id']} [{d['domain']}]: {d['decision'][:120]}")
        lines.append("  Comply with these decisions or supersede with omega_decision_register.")

        return {"output": "\n".join(lines), "error": None}

    except Exception as e:
        _log_hook_error("pre_alignment_gate", e)
        return {"output": "", "error": None}


_PROTOCOL_GATE_TOOLS = {"Edit", "Write", "NotebookEdit", "Bash"}
_PROTOCOL_GATE_MAX_CALLS = 20  # stop enforcing after this many tool calls (~10 min)
_PROTOCOL_GATE_EARLY = 8  # blocking window for inbox check


def handle_pre_protocol_gate(payload: dict) -> dict:
    """Enforce protocol compliance before code-modifying tools.

    Multi-agent: warn once if omega_inbox not called within first 8 gate calls.
    Solo: warn once if omega_welcome not called within first 8 gate calls.
    Multi-agent: warn once if coord_status not checked (full window).
    All modes: nudge once if 15+ edits without omega_store.
    After 20 gate calls: stop enforcement entirely.
    Each warning fires only once per session to avoid nagging agents that
    lack coordination pro tools.
    Uses _gate_call_count (incremented per PreToolUse) instead of _heartbeat_count
    (which only increments on 30s debounce pass).
    Peer count has 30s TTL to detect peer departures mid-session.
    Fail-open: any error returns allow.
    """
    try:
        tool_name = payload.get("tool_name", "")
        if tool_name not in _PROTOCOL_GATE_TOOLS:
            return {"output": "", "error": None}

        session_id = payload.get("session_id", "")
        if not session_id:
            return {"output": "", "error": None}

        # Increment gate-specific counter (independent of heartbeat debounce)
        call_count = _gate_call_count.get(session_id, 0) + 1
        _gate_call_count[session_id] = call_count

        # Enforcement window closed
        if call_count > _PROTOCOL_GATE_MAX_CALLS:
            return {"output": "", "error": None}

        calls_made = _protocol_calls.get(session_id, set())

        # Detect multi-agent with TTL-based cache (refreshes every 30s)
        multi_agent = False
        now = time.monotonic()
        cached_time = _session_peer_count_time.get(session_id, 0)
        if now - cached_time < SESSION_PEER_COUNT_TTL_S and session_id in _session_peer_count:
            multi_agent = _session_peer_count[session_id] > 0
        else:
            try:
                from omega.coordination import get_manager
                mgr = get_manager()
                count = mgr.active_session_count()
                _session_peer_count[session_id] = max(0, count - 1)
                _session_peer_count_time[session_id] = now
                multi_agent = _session_peer_count[session_id] > 0
            except Exception:
                _session_peer_count[session_id] = 0
                _session_peer_count_time[session_id] = now

        # Multi-agent: warn once if inbox not checked (early window only)
        if multi_agent and call_count <= _PROTOCOL_GATE_EARLY and "omega_inbox" not in calls_made:
            if "_gate_inbox_warned" not in calls_made:
                _protocol_calls.setdefault(session_id, set()).add("_gate_inbox_warned")
                return {
                    "output": "[PROTOCOL-GATE] You have active peers. Call omega_inbox() to check for messages before editing files.",
                    "error": None,
                }
            return {"output": "", "error": None}

        # Solo: warn once if welcome not called (early window)
        if not multi_agent and call_count <= _PROTOCOL_GATE_EARLY and "omega_welcome" not in calls_made:
            if "_gate_welcome_warned" not in calls_made:
                _protocol_calls.setdefault(session_id, set()).add("_gate_welcome_warned")
                return {
                    "output": "[PROTOCOL-REMINDER] Call omega_welcome() for memory context before starting work.",
                    "error": None,
                }
            return {"output": "", "error": None}

        # Solo: warn once if protocol not called after welcome (early window)
        if not multi_agent and call_count <= _PROTOCOL_GATE_EARLY and "omega_welcome" in calls_made and "omega_protocol" not in calls_made:
            if "_gate_protocol_warned" not in calls_made:
                _protocol_calls.setdefault(session_id, set()).add("_gate_protocol_warned")
                return {
                    "output": "[PROTOCOL-REMINDER] Call omega_protocol() to load your operating rules before editing files.",
                    "error": None,
                }
            return {"output": "", "error": None}

        # Multi-agent: warn once if intent not announced (full window)
        if multi_agent and call_count <= _PROTOCOL_GATE_MAX_CALLS and "omega_intent_announce" not in calls_made:
            if "_gate_intent_warned" not in calls_made:
                _protocol_calls.setdefault(session_id, set()).add("_gate_intent_warned")
                return {
                    "output": "[PROTOCOL-REMINDER] Announce your intent with omega_intent_announce(description=...) so peers know your plan.",
                    "error": None,
                }
            return {"output": "", "error": None}

        # Multi-agent: warn once if coord_status not checked
        if multi_agent and call_count <= _PROTOCOL_GATE_MAX_CALLS and "omega_coord_status" not in calls_made:
            if "_gate_coord_warned" not in calls_made:
                _protocol_calls.setdefault(session_id, set()).add("_gate_coord_warned")
                return {
                    "output": "[PROTOCOL-REMINDER] Check team status with omega_coord_status before modifying shared code.",
                    "error": None,
                }
            return {"output": "", "error": None}

        # Nudge after 15+ gate calls without any omega_store
        if call_count >= 15 and "omega_store" not in calls_made:
            if "_gate_store_warned" not in calls_made:
                _protocol_calls.setdefault(session_id, set()).add("_gate_store_warned")
                return {
                    "output": "[PROTOCOL-REMINDER] Many edits without omega_store(). Store key decisions to persist them.",
                    "error": None,
                }
            return {"output": "", "error": None}

        return {"output": "", "error": None}

    except Exception as e:
        # Fail-open: never block when OMEGA is broken
        _log_hook_error("pre_protocol_gate", e)
        return {"output": "", "error": None}


# Patterns for irreversible/external commands (advisory, never blocks)
_IRREVERSIBLE_PATTERNS = [
    (re.compile(r"\bgh\s+repo\s+create\b"), "repo creation"),
    (re.compile(r"\bgh\s+repo\s+delete\b"), "repo deletion"),
    (re.compile(r"\bgh\s+api\b.*\b-X\s+(?:PUT|DELETE|POST)\b"), "GitHub API mutation"),
    (re.compile(r"\brm\s+-rf\b"), "recursive deletion"),
    (re.compile(r"\bgit\s+push\s+.*--force\b"), "force push"),
    (re.compile(r"\bvercel\s+(?:deploy|--prod)\b"), "production deploy"),
    (re.compile(r"\bnpm\s+publish\b"), "package publish"),
]


def handle_pre_irreversible_advisor(payload: dict) -> dict:
    """Advisory nudge before irreversible/external Bash commands. Always exit 0 (never blocks)."""
    try:
        tool_input = payload.get("tool_input", {})
        if isinstance(tool_input, str):
            import json as _json
            try:
                tool_input = _json.loads(tool_input)
            except (ValueError, TypeError):
                tool_input = {}
        command = tool_input.get("command", "")
        session_id = payload.get("session_id", "")
        if not command or not session_id:
            return {"output": "", "error": None}

        warned = _session_external_actions.setdefault(session_id, set())

        for pattern, action_type in _IRREVERSIBLE_PATTERNS:
            if pattern.search(command) and action_type not in warned:
                warned.add(action_type)
                return {
                    "output": (
                        f"[IRREVERSIBLE-ACTION] Detected: {action_type}. Before proceeding:\n"
                        "1. omega_checkpoint() if you haven't recently\n"
                        "2. State your recovery plan if this fails\n"
                        "3. After completion: omega_store() the outcome"
                    ),
                    "error": None,
                }
        return {"output": "", "error": None}
    except Exception:
        return {"output": "", "error": None}


def _suggest_alternative_dir(file_path: str) -> str | None:
    """Suggest sibling directories as alternative work areas."""
    parent = os.path.dirname(file_path)
    grandparent = os.path.dirname(parent)
    if not grandparent or grandparent == parent:
        return None
    try:
        siblings = [
            d
            for d in sorted(os.listdir(grandparent))
            if os.path.isdir(os.path.join(grandparent, d))
            and d != os.path.basename(parent)
            and not d.startswith((".", "__"))
        ]
        if siblings:
            return f"nearby dirs: {', '.join(siblings[:3])}"
    except OSError as e:
        logger.debug("suggest alternative dir failed: %s", e)
    return None


# Risk classification patterns for coordination gate
_HIGH_RISK_PATTERNS = [
    r"\bvercel\s+(?:deploy|link|project\s+add|domains?\s+add)",
    r"\bvercel\s+--prod\b",
    r"\bfly\s+deploy\b",
    r"\bgit\s+push\s+.*--force\b",
    r"\bgit\s+push\s+-f\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\brm\s+-rf\b",
    r"\bgit\s+branch\s+-[dD]\b",
]
_MEDIUM_RISK_PATTERNS = [
    r"\bgit\s+commit\b",
    r"\bgit\s+push\b",
    r"\bgit\s+checkout\s+-b\b",
    r"\bgit\s+switch\s+-c\b",
    r"\bnpm\s+install\b",
    r"\bpip\s+install\b",
]




def classify_action_risk(tool_name: str, command: str = "") -> str:
    """Classify a tool invocation into LOW, MEDIUM, or HIGH risk.

    HIGH: deploy, force-push, destructive git ops, rm -rf
    MEDIUM: git commit, git push, branch creation, package install
    LOW: everything else (edits, reads, tests, status)
    """
    if tool_name != "Bash":
        return "LOW"
    for pattern in _HIGH_RISK_PATTERNS:
        if re.search(pattern, command):
            return "HIGH"
    for pattern in _MEDIUM_RISK_PATTERNS:
        if re.search(pattern, command):
            return "MEDIUM"
    return "LOW"




def handle_auto_claim_file(payload: dict) -> dict:
    """Auto-claim files on Edit/Write/NotebookEdit; record reads in multi-agent mode."""
    tool_name = payload.get("tool_name", "")
    if tool_name not in ("Edit", "Write", "NotebookEdit", "Read"):
        return {"output": "", "error": None}

    session_id = payload.get("session_id", "")
    if not session_id:
        return {"output": "", "error": None}

    input_data = _parse_tool_input(payload)
    file_path = _get_file_path_from_input(input_data)
    if not file_path or _is_plan_file(file_path):
        return {"output": "", "error": None}

    # Read tool: record file read (not a claim) in multi-agent mode only
    if tool_name == "Read":
        try:
            from omega.coordination import get_manager

            mgr = get_manager()
            if mgr.active_session_count() > 1:
                from . import FILE_READ_DEBOUNCE_S, _MAX_READ_ENTRIES, _last_file_read

                read_key = (session_id, file_path)
                if _debounce_check(_last_file_read, read_key, FILE_READ_DEBOUNCE_S, _MAX_READ_ENTRIES):
                    mgr.record_file_read(session_id, file_path)
        except Exception as e:
            logger.debug("auto file read tracking failed: %s", e)
        return {"output": "", "error": None}

    # Debounce: skip full claim if same (session, file) claimed recently,
    # but still refresh last_activity to prevent TTL expiry during active editing
    claim_key = (session_id, file_path)
    if not _debounce_check(_last_claim, claim_key, CLAIM_DEBOUNCE_S, _MAX_SURFACE_ENTRIES):
        try:
            from omega.coordination import get_manager

            mgr = get_manager()
            mgr.refresh_file_activity(session_id, file_path)
        except Exception as e:
            logger.debug("file activity refresh failed: %s", e)
        return {"output": "", "error": None}

    output = ""
    try:
        from omega.coordination import get_manager

        mgr = get_manager()
        result = mgr.claim_file(session_id, file_path, task="auto-claimed on edit")
        if result.get("conflict"):
            owner_name = _agent_nickname(result["claimed_by"])
            owner_task = result.get("task") or "unknown task"
            output = (
                f"[CONFLICT] {os.path.basename(file_path)} is claimed by "
                f"{owner_name} ({owner_task}). Coordinate before editing."
            )
        elif result.get("success"):
            # Single-agent fast path: skip intent + overlap checks when alone
            try:
                multi_agent = mgr.active_session_count() > 1
            except Exception as e:
                logger.debug("multi-agent count check failed: %s", e)
                multi_agent = True  # Assume multi-agent when check fails

            if multi_agent:
                # Auto-announce intent for coordination visibility
                try:
                    mgr.announce_intent(
                        session_id=session_id,
                        description=f"Editing {os.path.basename(file_path)}",
                        intent_type="edit",
                        target_files=[file_path],
                        ttl_minutes=5,
                    )
                except Exception as e:
                    logger.debug("intent announcement failed: %s", e)

                # Check for intent overlaps with other agents (best-effort, max 2 warnings)
                try:
                    overlap_result = mgr.check_intents(session_id)
                    if overlap_result.get("has_overlaps"):
                        overlaps = overlap_result["overlaps"][:2]

                        # Batch: deduplicate file paths across all overlaps, check each once
                        all_ov_files: set[str] = set()
                        for ov in overlaps:
                            for f in ov.get("overlapping_files", [])[:3]:
                                all_ov_files.add(f)
                        file_claim_cache: dict[str, dict] = {}
                        try:
                            for of in all_ov_files:
                                file_claim_cache[of] = mgr.check_file(of)
                        except Exception as e:
                            logger.debug("overlap escalation batch check failed: %s", e)

                        for ov in overlaps:
                            ov_sid_full = ov["session_id"]
                            ov_name = _agent_nickname(ov_sid_full)
                            ov_desc = ov.get("description", "")[:60]
                            ov_file_paths = ov.get("overlapping_files", [])[:3]
                            ov_files = ", ".join(os.path.basename(f) for f in ov_file_paths)

                            # Escalate if overlapping files are already CLAIMED by the other agent
                            escalated = False
                            for of in ov_file_paths:
                                check = file_claim_cache.get(of, {})
                                if check.get("claimed") and check.get("session_id") == ov_sid_full:
                                    escalated = True
                                    break

                            if escalated:
                                warning = (
                                    f"[CONFLICT] {ov_name} owns "
                                    f"{', '.join(os.path.basename(f) for f in ov_file_paths)}"
                                    f" — consider working in a different area"
                                )
                                alt = _suggest_alternative_dir(file_path)
                                if alt:
                                    warning += f"\n  Suggestion: {alt}"
                            else:
                                warning = f"[INTENT-OVERLAP] {ov_name}: {ov_desc}"
                                if ov_files:
                                    warning += f" (files: {ov_files})"

                            output += ("\n" if output else "") + warning

                            # Notify the other agent about the overlap (debounced)
                            try:
                                notify_key = (session_id, ov_sid_full, file_path)
                                if _debounce_check(_last_overlap_notify, notify_key, OVERLAP_NOTIFY_DEBOUNCE_S, _MAX_SURFACE_ENTRIES):
                                    filename = os.path.basename(file_path)
                                    mgr.send_message(
                                        from_session=session_id,
                                        to_session=ov_sid_full,
                                        msg_type="inform",
                                        subject=f"Overlap: both editing {filename}",
                                        body=(
                                            f"I'm editing {file_path}. You announced intent "
                                            f"to work on overlapping files. Let's coordinate."
                                        ),
                                        ttl_minutes=30,
                                    )
                            except (KeyError, TypeError, ValueError) as e:
                                logger.debug("intent overlap notification failed: %s", e)
                except Exception as e:
                    logger.debug("intent overlap check failed: %s", e)
    except Exception as e:
        error_str = str(e)
        if "database is locked" in error_str:
            logger.debug("auto_claim_file skipped: database locked")
        elif "already claimed" not in error_str.lower():
            _log_hook_error("auto_claim_file", e)

    return {"output": output, "error": None}




def _file_guard_block_msg(file_path: str, owner_sid: str, owner_task: str, blocked_sid: str = "") -> dict:
    """Return a block response for the file guard.

    Also sends a [WAITING] message to the file owner so they know
    someone is blocked (debounced to once per 5 min per tuple).
    """
    filename = os.path.basename(file_path)
    owner_name = _agent_nickname(owner_sid)
    msg = (
        f"\n[FILE-GUARD] BLOCKED: {filename} is claimed by {owner_name} ({owner_task}).\n"
        f"  Options:\n"
        f"    1. Wait for the other agent to finish and release\n"
        f"    2. Ask other agent to call omega_file_release\n"
        f"    3. The claim expires automatically after 10 minutes of inactivity\n"
        f"    4. Ask the human to decide if a force-override is safe"
    )

    # Record metric for guard-level blocks (closes metric gap — blocks at
    # check-time aren't captured by claim_file's conflict_detected metric)
    try:
        from omega.coordination import get_manager

        mgr = get_manager()
        mgr.record_metric(
            "conflict_blocked_by_guard",
            session_id=blocked_sid or None,
            metadata={"file": filename, "owner": owner_sid[:20]},
        )
    except Exception:
        pass  # Metrics are best-effort

    # Notify the owner that someone is waiting
    if blocked_sid and owner_sid:
        key = (blocked_sid, owner_sid, file_path)
        if _debounce_check(_last_block_notify, key, BLOCK_NOTIFY_DEBOUNCE_S, _MAX_SURFACE_ENTRIES):
            try:
                from omega.coordination import get_manager

                mgr = get_manager()
                blocked_name = _agent_nickname(blocked_sid)
                mgr.send_message(
                    from_session=blocked_sid,
                    subject=f"[WAITING] {blocked_name} wants to edit {filename} — consider releasing if done",
                    msg_type="inform",
                    to_session=owner_sid,
                )
            except Exception as e:
                logger.debug("file guard block notification failed: %s", e)

    return {"output": msg, "error": None, "exit_code": 2}




def handle_pre_file_guard(payload: dict) -> dict:
    """Check file claims BEFORE editing — blocks if claimed by another agent.

    Returns exit_code=2 in the response dict when blocking.
    Fail-open: any error returns allow (exit_code=0).
    """
    tool_name = payload.get("tool_name", "")
    if tool_name not in ("Edit", "Write", "NotebookEdit"):
        return {"output": "", "error": None}

    session_id = payload.get("session_id", "")
    input_data = _parse_tool_input(payload)
    file_path = _get_file_path_from_input(input_data)
    if not file_path or _is_plan_file(file_path):
        return {"output": "", "error": None}

    try:
        from omega.coordination import get_manager

        mgr = get_manager()
        info = mgr.check_file(file_path)

        if info.get("claimed"):
            if session_id and info.get("session_id") == session_id:
                return {"output": "", "error": None}

            # Claimed by different session (or no session_id to prove identity) — BLOCK
            owner_sid = info.get("session_id", "unknown")
            owner_task = info.get("task") or "unknown task"
            return _file_guard_block_msg(file_path, owner_sid, owner_task, blocked_sid=session_id)

        # Unclaimed — if we have session_id, claim atomically to prevent TOCTOU race
        if session_id:
            result = mgr.claim_file(session_id, file_path, task="pre-edit guard claim")
            if result.get("conflict"):
                owner_sid = result["claimed_by"]
                owner_task = result.get("task") or "unknown task"
                return _file_guard_block_msg(file_path, owner_sid, owner_task, blocked_sid=session_id)

        # --- Constraint rules check ---
        try:
            from omega.bridge import check_constraints
            project = payload.get("project") or payload.get("session_meta", {}).get("project")
            violations = check_constraints(file_path, project)
            if violations:
                blockers = [v for v in violations if v.get("severity") == "block"]
                warnings = [v for v in violations if v.get("severity") != "block"]
                if blockers:
                    msg = f"⛔ BLOCKED by constraint rules for {file_path}:\n"
                    for b in blockers:
                        msg += f"  - [{b['pattern']}] {b['constraint']}\n"
                    return {"output": msg, "error": None, "exit_code": 2}
                if warnings:
                    msg = f"⚠️ Constraint warnings for {file_path}:\n"
                    for w in warnings:
                        msg += f"  - [{w['pattern']}] {w['constraint']}\n"
                    return {"output": msg, "error": None}
        except Exception:
            pass  # Fail-open: constraint check failure never blocks

        # No session_id + unclaimed → allow (true single-agent)
        return {"output": "", "error": None}

    except Exception as e:
        # Fail-open: never block when OMEGA is unavailable
        _log_hook_error("pre_file_guard", e)
        return {"output": "", "error": None}




def handle_pre_task_guard(payload: dict) -> dict:
    """Check task declaration BEFORE editing — blocks if no active task.

    Opt-in: only enforces when the project has non-terminal tasks.
    Returns exit_code=2 in the response dict when blocking.
    Fail-open: any error returns allow (exit_code absent).
    """
    tool_name = payload.get("tool_name", "")
    if tool_name not in ("Edit", "Write", "NotebookEdit"):
        return {"output": "", "error": None}

    session_id = payload.get("session_id", "")
    if not session_id:
        # Single-agent mode — no enforcement
        return {"output": "", "error": None}

    input_data = _parse_tool_input(payload)
    file_path = _get_file_path_from_input(input_data)
    if not file_path or _is_plan_file(file_path):
        return {"output": "", "error": None}

    project = payload.get("project", "")
    if not project:
        return {"output": "", "error": None}

    # Skip if file is outside the project directory
    try:
        if not os.path.abspath(file_path).startswith(os.path.abspath(project)):
            return {"output": "", "error": None}
    except Exception as e:
        logger.debug("path validation failed: %s", e)
        return {"output": "", "error": None}

    try:
        from omega.coordination import get_manager

        mgr = get_manager()

        # Opt-in: only enforce if project has active tasks
        if not mgr.project_has_active_tasks(project):
            return {"output": "", "error": None}

        # Check if session has an in_progress task
        result = mgr.has_active_task(session_id)
        if result.get("has_task"):
            return {"output": "", "error": None}

        # No active task — BLOCK
        filename = os.path.basename(file_path)
        project_name = os.path.basename(project)
        msg = (
            f"\n[TASK-GUARD] BLOCKED: No active task for this session on {project_name}.\n"
            f"  Create and claim a task before editing {filename}:\n"
            f'    1. omega_task_create(title="Your task", project="{project}")\n'
            f'    2. omega_task_claim(task_id=<id>, session_id="{session_id}")\n'
            f"  Or complete/cancel existing tasks to disable enforcement."
        )
        return {"output": msg, "error": None, "exit_code": 2}

    except Exception as e:
        # Fail-open: never block when OMEGA is unavailable
        _log_hook_error("pre_task_guard", e)
        return {"output": "", "error": None}




def _clean_task_text(prompt: str) -> str:
    """Delegate to shared implementation in omega.task_utils."""
    from omega.task_utils import clean_task_text

    return clean_task_text(prompt)


def _summarize_task_text(prompt: str) -> str:
    """Summarize task via Haiku, falling back to clean_task_text."""
    from omega.task_utils import summarize_task_text

    return summarize_task_text(prompt)




def handle_auto_capture(payload: dict) -> dict:
    """Auto-capture decisions and lessons from user prompts (UserPromptSubmit)."""
    # Prefer top-level keys (set by fast_hook.py from parsed stdin JSON).
    # Fall back to re-parsing payload["stdin"] for legacy/direct callers.
    prompt = payload.get("prompt", "")
    stdin_parsed = {}
    if not prompt:
        stdin_data = payload.get("stdin", "")
        if not stdin_data:
            return {"output": "", "error": None}
        try:
            stdin_parsed = json.loads(stdin_data)
        except (json.JSONDecodeError, TypeError):
            return {"output": "", "error": None}
        prompt = stdin_parsed.get("prompt", "")

    # Extract session_id/cwd from top-level first, fall back to parsed stdin
    session_id = payload.get("session_id") or stdin_parsed.get("session_id", "")
    cwd = payload.get("cwd") or payload.get("project") or stdin_parsed.get("cwd", "")
    entity_id = _resolve_entity(cwd) if cwd else None

    if not prompt:
        return {"output": "", "error": None}

    # Auto-set session task from first prompt (DB as source of truth)
    # Runs before the 20-char guard — short prompts are valid tasks
    if session_id:
        try:
            from omega.coordination import get_manager as _get_mgr_task

            _mgr = _get_mgr_task()
            row = _mgr._conn.execute(
                "SELECT task FROM coord_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is not None and not row[0]:
                task_text = _summarize_task_text(prompt)
                if task_text:
                    with _mgr._lock:
                        _mgr._conn.execute(
                            "UPDATE coord_sessions SET task = ? WHERE session_id = ?",
                            (task_text, session_id),
                        )
                        _mgr._conn.commit()
        except Exception as e:
            logger.debug("auto-task set failed: %s", e)

    # Short prompts: task is set above, but skip decision/lesson/preference capture
    if len(prompt) < 20:
        return {"output": "", "error": None}

    # --- Surface peer work on planning prompts (coordination awareness) ---
    coord_output = ""
    _planning_patterns = [
        r"\bwhat.?s?\s+next\b",
        r"\bwhat\s+should\s+(?:i|we)\b",
        r"\bwhat\s+(?:to|can)\s+(?:do|work)\b",
        r"\bnext\s+(?:step|task|priority)\b",
        r"\bpriorities?\b",
        r"\broadmap\b",
        r"\bwhat\s+(?:remains?|is\s+left)\b",
    ]
    prompt_lower_early = prompt.lower()
    is_planning = any(re.search(pat, prompt_lower_early) for pat in _planning_patterns)
    if is_planning and session_id and cwd:
        now = time.monotonic()
        if session_id not in _last_coord_query or now - _last_coord_query[session_id] >= COORD_QUERY_DEBOUNCE_S:
            try:
                from omega.coordination import get_manager as _get_mgr_coord

                mgr = _get_mgr_coord()
                sessions = mgr.list_sessions(auto_clean=False)
                peers = [s for s in sessions if s.get("session_id") != session_id and s.get("project") == cwd][:4]
                if peers:
                    # Fetch in-progress coord tasks for this project
                    in_progress_tasks = []
                    try:
                        in_progress_tasks = mgr.list_tasks(project=cwd, status="in_progress")
                    except Exception as e:
                        _log_hook_error("handle_auto_capture", e)
                    task_by_session: dict[str, dict] = {}
                    for t in in_progress_tasks:
                        sid = t.get("session_id")
                        if sid and sid not in task_by_session:
                            task_by_session[sid] = t

                    coord_lines = [f"[COORD] {len(peers)} peer{'s' if len(peers) != 1 else ''} active on this project:"]
                    for p in peers:
                        p_sid = p["session_id"]
                        p_name = _agent_nickname(p_sid)
                        # Prefer coord_task over session.task
                        ct = task_by_session.get(p_sid)
                        if ct:
                            pct = f" [{ct['progress']}%]" if ct.get("progress") else ""
                            p_task = f"#{ct['id']} {ct['title'][:40]}{pct}"
                        else:
                            p_task = (p.get("task") or "idle")[:50]
                        # File claims
                        p_files = ""
                        try:
                            claims = mgr.get_session_claims(p_sid)
                            file_claims = claims.get("file_claims", [])
                            if file_claims:
                                fnames = [os.path.basename(f) for f in file_claims[:3]]
                                if len(file_claims) > 3:
                                    fnames.append(f"+{len(file_claims) - 3}")
                                p_files = f" [{', '.join(fnames)}]"
                        except Exception as e:
                            _log_hook_error("handle_auto_capture", e)
                        coord_lines.append(f"  {p_name}: {p_task}{p_files}")
                    coord_output = "\n".join(coord_lines)
                _last_coord_query[session_id] = now
            except Exception as e:
                logger.debug("coordination query failed: %s", e)

    # Auto-classify intent (router integration)
    router_output = ""
    classified_intent = None
    try:
        from omega.router.classifier import classify_intent

        intent, confidence = classify_intent(prompt)
        if confidence >= 0.6:
            classified_intent = intent
            router_output = f"[ROUTER] Intent: {intent} ({confidence:.0%})"
            if session_id:
                _session_intent[session_id] = intent
                # Enrich session task for dashboard visibility (only if empty)
                try:
                    from omega.coordination import get_manager
                    get_manager().update_session_task_if_empty(session_id, intent)
                except Exception:
                    pass
    except ImportError as e:
        logger.debug("router import not available: %s", e)
    except Exception as e:
        logger.debug("intent classification failed: %s", e)

    # Preference pattern matching (checked first — highest priority)
    preference_patterns = [
        r"\bi\s+(?:prefer|like|love|enjoy|favor|favour)\s+\w",
        r"\bmy\s+(?:preference|favorite|favourite|default)\b",
        r"\balways\s+use\b",
        r"\bi\s+(?:want|need)\s+(?:it|things?|everything)\s+(?:in|with|to\s+be)\b",
        r"\bremember\s+(?:that\s+)?i\s+(?:prefer|like|want|use|need)\b",
        r"\bdon'?t\s+(?:ever\s+)?(?:use|suggest|recommend)\b",
        r"\bi\s+(?:hate|dislike|avoid)\b",
        r"\bremember\s+(?:to|that\s+i|my|i\s+)\b",  # Narrowed: "remember to/that I/my/I ..."
    ]

    # Decision pattern matching (tightened: require commitment language + 2 words after trigger)
    decision_patterns = [
        r"\blet'?s?\s+(?:go\s+with|switch\s+to|move\s+to|adopt|implement)\s+\S+\s+\S+",
        r"\bi\s+(?:decided|chose)\s+\S+\s+\S+",
        r"\bwe\s+(?:will|are\s+going\s+to)\s+(?:use|go\s+with|switch|adopt|implement)\s+\S+\s+\S+",
        r"\b(?:decision|approach|strategy):\s+\w+\s+\w+",
        r"\binstead\s+of\s+\S+\s+(?:use|adopt|switch\s+to)\s+\S+",
    ]

    # Fact/context pattern matching (new: captures project details and factual statements)
    fact_patterns = [
        r"\bthe\s+(?:database|db|server|api|endpoint|url|path|port|config)\s+is\s+\S",
        r"\b(?:this|the|our)\s+(?:project|repo|codebase|app|service)\s+(?:uses?|runs?|is|has)\b",
        r"\b\S+\s+(?:lives?|runs?|is\s+(?:at|on|in))\s+(?:/|https?://)\S",
        r"\b(?:configured|set\s+up|deployed)\s+(?:with|on|at|to)\b",
        r"\b(?:credentials?|password|token|key)\s+(?:is|are)\s+(?:in|at|stored)\b",
        r"\bversion\s+\d",
        r"\b(?:runs?|listens?|deployed)\s+on\s+port\s+\d",
    ]

    # Lesson pattern matching
    lesson_patterns = [
        r"\bi\s+learned\s+that\b",
        r"\bturns?\s+out\b",
        r"\bthe\s+trick\s+is\b",
        r"\bnote\s+to\s+self\b",
        r"\btil\b|\btoday\s+i\s+learned\b",
        r"\bthe\s+fix\s+was\b",
        r"\bthe\s+problem\s+was\b",
        r"\bdon'?t\s+forget\b",
        r"\bimportant:\s*\S",
        r"\bkey\s+(?:insight|takeaway|learning)\b",
        r"\bnever\s+(?:again|do|use)\b",
        r"\balways\s+(?:make\s+sure|remember|check)\b",
    ]

    prompt_lower = prompt.lower()
    is_preference = any(re.search(pat, prompt_lower) for pat in preference_patterns)
    is_decision = any(re.search(pat, prompt_lower) for pat in decision_patterns)
    is_fact = any(re.search(pat, prompt_lower) for pat in fact_patterns)
    is_lesson = any(re.search(pat, prompt_lower) for pat in lesson_patterns)

    if not is_preference and not is_decision and not is_fact and not is_lesson:
        # Pass through any accumulated coord/router output even if no capture
        passthrough = "\n".join(filter(None, [coord_output, router_output]))
        return {"output": passthrough, "error": None}

    # Preference > Decision > Fact > Lesson priority
    if is_preference:
        event_type = "user_preference"
        content_prefix = "Preference"
    elif is_decision:
        event_type = "decision"
        content_prefix = "Decision"
    elif is_fact:
        event_type = "user_fact"
        content_prefix = "Fact"
    else:
        event_type = "lesson_learned"
        content_prefix = "Lesson"

    # Decision quality gate: min 80 chars, >= 12 words (filter trivial "let's use X")
    if event_type == "decision":
        if len(prompt) < 80 or len(prompt.split()) < 12:
            return {"output": "", "error": None}
        # Blocklist: reject system marker content that accidentally triggers decision patterns
        _decision_blocklist_prefixes = (
            "[CONTEXT", "[GIT", "[HANDOFF", "[MEMORY", "[COORD",
            "[BROADCAST", "[CAPTURED", "[ROUTER",
        )
        prompt_head = prompt.lstrip()[:200]
        if any(prefix in prompt_head for prefix in _decision_blocklist_prefixes):
            return {"output": "", "error": None}

    # Lesson quality gate: min 30 chars, >= 5 words (relaxed from 50/7)
    # Pattern match already signals intent — no secondary tech signal required
    if event_type == "lesson_learned":
        if len(prompt) < 30 or len(prompt.split()) < 5:
            return {"output": "", "error": None}

    try:
        from omega.bridge import auto_capture

        meta = {"source": "auto_capture_hook", "project": cwd}
        if classified_intent:
            meta["intent"] = classified_intent

        # Extract referenced_date from prompt (ISO dates, month+year patterns)
        _date_iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", prompt)
        if _date_iso:
            meta["referenced_date"] = _date_iso.group(1)
        else:
            _date_month_year = re.search(
                r"\b((?:January|February|March|April|May|June|July|August|"
                r"September|October|November|December|"
                r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                r"\s+\d{4})\b",
                prompt,
                re.IGNORECASE,
            )
            if _date_month_year:
                meta["referenced_date"] = _date_month_year.group(1)

        auto_capture(
            content=f"{content_prefix}: {prompt[:500]}",
            event_type=event_type,
            metadata=meta,
            session_id=session_id,
            project=cwd,
            entity_id=entity_id,
        )
    except Exception as e:
        _log_hook_error("auto_capture", e)
        return {"output": "\n".join(filter(None, [coord_output, router_output])), "error": None}

    # User-visible confirmation of what was captured
    capture_line = f"[CAPTURED] {content_prefix.lower()}: {prompt[:80].replace(chr(10), ' ').strip()}"
    combined = "\n".join(filter(None, [capture_line, coord_output, router_output]))
    return {"output": combined, "error": None}


# ---------------------------------------------------------------------------
# Pre-push guard (git divergence + branch claims)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Pre-push guard (git divergence + branch claims)
# ---------------------------------------------------------------------------






def handle_pre_push_guard(payload: dict) -> dict:
    """Git push divergence guard + branch claim check.

    Enforces:
      1. git push: blocks if origin has advanced (divergence guard)
      2. git checkout/switch: blocks if target branch is claimed by another agent
      3. git commit: blocks if current branch is claimed by another agent

    Returns exit_code=2 when blocking. Fail-open on errors.
    """
    tool_name = payload.get("tool_name", "")
    if tool_name != "Bash":
        return {"output": "", "error": None}

    input_data = _parse_tool_input(payload)
    command = input_data.get("command", "")
    session_id = payload.get("session_id", "")
    project = payload.get("project", "")

    if re.search(r"\bgit\s+push\b", command):
        try:
            from omega.coordination import get_manager

            get_manager().record_metric("gate_check_medium", session_id=session_id, metadata={"action": "push"})
        except Exception as e:
            _log_hook_error("handle_pre_push_guard", e)
        result = _handle_push_divergence(command, project, session_id)
        if result.get("exit_code"):
            return result
        _handle_auto_claim_branch(command, session_id, project)
        return {"output": "", "error": None}

    if re.search(r"\bgit\s+(?:checkout|switch|commit)\b", command):
        return _handle_branch_claims(command, session_id, project)

    return {"output": "", "error": None}




def _handle_push_divergence(command: str, project: str, session_id: str) -> dict:
    """Check for push divergence. Returns exit_code=2 if origin has advanced."""
    if not project:
        return {"output": "", "error": None}

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project,
        )
        if result.returncode != 0:
            return {"output": "", "error": None}

        subprocess.run(
            ["git", "fetch", "origin", "--quiet"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=project,
        )

        branch = _get_current_branch(project) or "main"

        behind_result = subprocess.run(
            ["git", "log", f"HEAD..origin/{branch}", "--oneline", "--no-decorate", "-10"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project,
        )
        if behind_result.returncode != 0 or not behind_result.stdout.strip():
            # Not behind — log push event
            _log_push_event(project, branch, session_id)
            return {"output": "", "error": None}

        behind_lines = behind_result.stdout.strip().split("\n")
        count = len(behind_lines)

        # Log divergence to coordination
        try:
            from omega.coordination import get_manager

            mgr = get_manager()
            mgr.log_git_event(
                project=project,
                event_type="push_divergence_warning",
                branch=branch,
                message=f"{count} upstream commit(s) detected before push",
                session_id=session_id,
            )
        except Exception as e:
            _log_hook_error("_handle_push_divergence", e)

        lines = [f"\n[GIT-GUARD] BLOCKED: origin/{branch} has {count} commit(s) not in HEAD:"]
        for line in behind_lines[:5]:
            lines.append(f"  {line}")
        if count > 5:
            lines.append(f"  ... and {count - 5} more")
        lines.append("  Run 'git pull --rebase' before pushing to avoid conflicts.")

        return {"output": "\n".join(lines), "error": None, "exit_code": 2}

    except Exception as e:
        _log_hook_error("pre_push_guard", e)
        return {"output": "", "error": None}




def _handle_branch_claims(command: str, session_id: str, project: str) -> dict:
    """Check branch claims for checkout/switch/commit commands."""
    if not session_id:
        return {"output": "", "error": None}

    try:
        if "git commit" in command:
            branch = _get_current_branch(project)
            if branch:
                return _block_if_branch_claimed(session_id, project, branch)
            return {"output": "", "error": None}

        target = _parse_checkout_target(command)
        if target:
            return _block_if_branch_claimed(session_id, project, target)

    except Exception as e:
        _log_hook_error("branch_guard", e)

    return {"output": "", "error": None}




def _block_if_branch_claimed(session_id: str, project: str, branch: str) -> dict:
    """Block if the branch is claimed by another agent."""
    try:
        from omega.coordination import get_manager

        mgr = get_manager()
        info = mgr.check_branch(project, branch)

        if not info.get("claimed"):
            return {"output": "", "error": None}

        if info.get("session_id") == session_id:
            return {"output": "", "error": None}  # Self-claim

        owner_name = _agent_nickname(info.get("session_id", "unknown"))
        owner_task = info.get("task") or "unknown task"
        msg = (
            f"\n[BRANCH-GUARD] BLOCKED: branch '{branch}' is claimed by {owner_name} ({owner_task}).\n"
            f"  Options:\n"
            f"    1. Wait for the other agent to finish\n"
            f"    2. Ask other agent to call omega_branch_release\n"
            f"    3. Use a different feature branch"
        )
        return {"output": msg, "error": None, "exit_code": 2}

    except Exception as e:
        _log_hook_error("branch_guard", e)
        return {"output": "", "error": None}




def _handle_auto_claim_branch(command: str, session_id: str, project: str):
    """Auto-claim the current branch before a push succeeds."""
    if not session_id or not project or not os.path.isdir(project):
        return
    try:
        branch = _get_current_branch(project)
        if not branch or branch == "HEAD":
            return
        from omega.coordination import get_manager

        mgr = get_manager()
        mgr.claim_branch(project=project, branch=branch, session_id=session_id, task="pushing to remote")
    except Exception as e:
        _log_hook_error("_handle_auto_claim_branch", e)




def _log_push_event(project: str, branch: str, session_id: str):
    """Log a push event to coordination."""
    try:
        from omega.coordination import get_manager

        mgr = get_manager()

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project,
        )
        commit_hash = result.stdout.strip() if result.returncode == 0 else None

        mgr.log_git_event(
            project=project,
            event_type="push",
            commit_hash=commit_hash,
            branch=branch,
            session_id=session_id,
        )
    except Exception as e:
        logger.debug("push event logging failed: %s", e)




def _is_broad_add(command: str) -> bool:
    """Detect git add . / git add -A / git add --all / git commit -a patterns."""
    if re.search(r"\bgit\s+add\s+(\.\s*($|&&|\|)|--all\b|-A\b)", command):
        return True
    if re.search(r"\bgit\s+commit\s+.*-[a-zA-Z]*a", command):
        return True
    return False


def _extract_add_paths(command: str) -> list[str] | None:
    """Extract file paths from a git add command. Returns list or None."""
    m = re.search(r"\bgit\s+add\s+(.*?)(\s*&&|\s*\||\s*;|$)", command)
    if not m:
        return None
    args_str = m.group(1).strip()
    paths = []
    for token in args_str.split():
        if token.startswith("-"):
            continue
        paths.append(token)
    return paths if paths else None


def handle_pre_add_guard(payload: dict) -> dict:
    """Git add guard — BLOCKS broad staging and unclaimed file staging.

    Prevents the root cause of mixed commits: agents staging pre-existing
    dirty worktree changes from other agents or prior sessions.

    BLOCKS: git add . / git add -A / git commit -a (always)
    BLOCKS: staging unclaimed files when peers are active
    WARNS: staging unclaimed files in solo mode
    """
    tool_name = payload.get("tool_name", "")
    if tool_name != "Bash":
        return {"output": "", "error": None}

    input_data = _parse_tool_input(payload)
    command = input_data.get("command", "")

    is_git_add = re.search(r"\bgit\s+add\b", command)
    is_commit_a = re.search(r"\bgit\s+commit\s+.*-[a-zA-Z]*a", command)
    if not is_git_add and not is_commit_a:
        return {"output": "", "error": None}

    session_id = payload.get("session_id", "")
    project = payload.get("project", "")

    # --- BLOCK broad staging unconditionally ---
    if _is_broad_add(command):
        lines = [
            "[ADD-GUARD] BLOCKED: broad staging command detected.",
            f"  Command: {command[:120]}",
            "",
            "Never use `git add .`, `git add -A`, or `git commit -a`.",
            "Stage specific files by name: git add <file1> <file2> ...",
        ]

        if session_id:
            try:
                from omega.coordination import get_manager as _gm_add
                mgr = _gm_add()
                own_claims = mgr.get_session_claims(session_id)
                own_files = own_claims.get("file_claims", [])
                if own_files:
                    rel_files = []
                    for f in own_files:
                        if project and f.startswith(project):
                            rel_files.append(os.path.relpath(f, project))
                        else:
                            rel_files.append(f)
                    lines.append("")
                    lines.append(f"Your claimed files ({len(rel_files)}):")
                    for rf in sorted(rel_files)[:20]:
                        lines.append(f"  {rf}")
                    if len(rel_files) > 20:
                        lines.append(f"  +{len(rel_files) - 20} more")

                mgr.record_metric(
                    "gate_blocked",
                    session_id=session_id,
                    metadata={"action": "add_broad", "command": command[:100]},
                )
            except Exception:
                pass

        return {"output": "\n".join(lines), "exit_code": 2, "error": None}

    # --- For specific git add <files>, check against claims ---
    add_paths = _extract_add_paths(command)
    if not add_paths or not session_id:
        return {"output": "", "error": None}

    try:
        from omega.coordination import get_manager as _gm_add2
        mgr = _gm_add2()

        own_claims = mgr.get_session_claims(session_id)
        own_files = own_claims.get("file_claims", [])
        if not own_files:
            return {"output": "", "error": None}

        sessions = mgr.list_sessions(auto_clean=True)
        peers = [s for s in sessions if s.get("session_id") != session_id]
        has_peers = len(peers) > 0

        # Resolve add paths and check against claims
        unclaimed = []
        for path in add_paths:
            try:
                result = subprocess.run(
                    ["git", "diff", "--name-only", "--", path],
                    capture_output=True, text=True, timeout=5, cwd=project or None,
                )
                resolved_files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
            except Exception:
                resolved_files = [path]

            for rf in resolved_files:
                abs_path = os.path.join(project, rf) if project and not os.path.isabs(rf) else rf
                if abs_path not in own_files and rf not in own_files:
                    unclaimed.append(rf)

        if unclaimed:
            if has_peers:
                lines = [
                    f"[ADD-GUARD] BLOCKED: staging {len(unclaimed)} file(s) you didn't edit:",
                ]
                for f in unclaimed[:10]:
                    lines.append(f"  {f}")
                if len(unclaimed) > 10:
                    lines.append(f"  +{len(unclaimed) - 10} more")
                lines.append("")
                lines.append("These files have pre-existing changes from another agent or prior session.")
                lines.append("Stage only files you modified. Edit/Write auto-claims files for you.")
                mgr.record_metric(
                    "gate_blocked",
                    session_id=session_id,
                    metadata={"action": "add_unclaimed", "unclaimed_count": len(unclaimed)},
                )
                return {"output": "\n".join(lines), "exit_code": 2, "error": None}
            else:
                lines = [
                    f"[ADD-GUARD] WARNING: staging {len(unclaimed)} file(s) not in your claim list:",
                ]
                for f in unclaimed[:10]:
                    lines.append(f"  {f}")
                if len(unclaimed) > 10:
                    lines.append(f"  +{len(unclaimed) - 10} more")
                lines.append("")
                lines.append("Did you author these changes? If not, unstage with: git reset HEAD <file>")
                return {"output": "\n".join(lines), "error": None}

    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("pre_add_guard", e)

    return {"output": "", "error": None}


def handle_pre_commit_guard(payload: dict) -> dict:
    """Git commit coordination guard — BLOCKS commits that stage peer-claimed files.

    Surfaces peer activity and file claim overlaps when committing.
    Blocks (exit_code=2) if staged files are claimed by another session,
    preventing mixed-author commits where one agent captures another's work.
    Warns (but allows) if staging files not in own claim list.
    """
    tool_name = payload.get("tool_name", "")
    if tool_name != "Bash":
        return {"output": "", "error": None}

    input_data = _parse_tool_input(payload)
    command = input_data.get("command", "")

    if not re.search(r"\bgit\s+commit\b", command):
        return {"output": "", "error": None}

    session_id = payload.get("session_id", "")
    project = payload.get("project", "")

    # --- Get staged files (needed by both scope check and peer check) ---
    staged_files = []
    try:
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=project or None,
        )
        staged_files = [f.strip() for f in staged.stdout.strip().split("\n") if f.strip()]
    except Exception as e:
        _log_hook_error("handle_pre_commit_guard", e)

    # --- Commit scope / atomicity check (runs independently of coordination) ---
    if staged_files and not os.environ.get("OMEGA_SKIP_SCOPE_CHECK"):
        is_merge = "--amend" in command or "merge" in command.lower()
        if not is_merge:
            # Cluster files by top-level directory
            dir_groups: dict[str, list[str]] = {}
            for sf in staged_files:
                parts = sf.split("/")
                top_dir = parts[0] if len(parts) > 1 else "(root)"
                dir_groups.setdefault(top_dir, []).append(sf)

            num_dirs = len(dir_groups)
            num_files = len(staged_files)

            # Get total lines changed
            total_lines = 0
            try:
                stat_result = subprocess.run(
                    ["git", "diff", "--cached", "--shortstat"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=project or None,
                )
                if stat_result.returncode == 0:
                    for m in re.finditer(r"(\d+) (?:insertion|deletion)", stat_result.stdout):
                        total_lines += int(m.group(1))
            except Exception:
                pass

            # Block if commit is too broad: many files across many concerns
            scope_blocked = (num_files > 10 and num_dirs >= 3) or (total_lines > 500 and num_dirs >= 3)

            if scope_blocked:
                scope_lines = [
                    f"[COMMIT-SCOPE] BLOCKED: {num_files} files across {num_dirs} directories "
                    f"({total_lines} lines). Try smaller, atomic commits.",
                    "",
                    "Suggested splits:",
                ]
                for dir_name, files in sorted(dir_groups.items(), key=lambda x: -len(x[1])):
                    file_list = ", ".join(os.path.basename(f) for f in files[:4])
                    if len(files) > 4:
                        file_list += f" +{len(files) - 4}"
                    scope_lines.append(f"  {dir_name}/ ({len(files)} files): {file_list}")
                scope_lines.append("")
                scope_lines.append("To commit each group separately:")
                for dir_name in sorted(dir_groups, key=lambda d: -len(dir_groups[d])):
                    scope_lines.append(f'  git add {dir_name}/ && git commit -m "<message>"')
                scope_lines.append("")
                scope_lines.append("To override: set OMEGA_SKIP_SCOPE_CHECK=1")

                try:
                    from omega.coordination import get_manager as _gm_scope
                    _gm_scope().record_metric(
                        "gate_blocked",
                        session_id=session_id,
                        metadata={"action": "commit_scope", "files": num_files, "dirs": num_dirs, "lines": total_lines},
                    )
                except Exception:
                    pass

                return {"output": "\n".join(scope_lines), "exit_code": 2, "error": None}

    # --- Schema/handler files without co-staged tests (non-blocking warning) ---
    schema_warn = ""
    if staged_files:
        server_files = [f for f in staged_files if f.startswith("src/omega/server/")]
        test_files = [f for f in staged_files if f.startswith("tests/")]
        if server_files and not test_files:
            schema_warn = (
                "[COMMIT-SCHEMA] Schema/handler files staged without test updates. "
                "Check hardcoded counts in test_uat_coordination.py, test_tool_schemas.py."
            )

    # --- Peer coordination check (requires coordination module) ---
    try:
        from omega.coordination import get_manager

        mgr = get_manager()
        mgr.record_metric("gate_check_medium", session_id=session_id, metadata={"action": "commit"})
        sessions = mgr.list_sessions(auto_clean=True)
        peers = [s for s in sessions if s.get("session_id") != session_id]

        if not peers:
            mgr.record_metric(
                "gate_skipped", session_id=session_id, metadata={"reason": "no_peers", "action": "commit"}
            )
            # Solo mode: still check for unclaimed files (pre-existing dirty worktree)
            if staged_files and session_id:
                try:
                    own_claims = mgr.get_session_claims(session_id)
                    own_files = own_claims.get("file_claims", [])
                    if own_files:  # Only check if we have claims to compare against
                        unclaimed_solo = []
                        for sf in staged_files:
                            full_path = os.path.join(project, sf) if project else sf
                            if full_path not in own_files and sf not in own_files:
                                unclaimed_solo.append(sf)
                        if unclaimed_solo:
                            warn_lines = [
                                f"[COMMIT-SCOPE] WARNING: {len(unclaimed_solo)} staged file(s) not in your claim list:",
                            ]
                            for fname in unclaimed_solo[:10]:
                                warn_lines.append(f"  {fname}")
                            if len(unclaimed_solo) > 10:
                                warn_lines.append(f"  +{len(unclaimed_solo) - 10} more")
                            warn_lines.append("")
                            warn_lines.append("Did you author these changes? If not, unstage with: git reset HEAD <file>")
                            if schema_warn:
                                warn_lines.append(schema_warn)
                            return {"output": "\n".join(warn_lines), "error": None}
                except Exception as e:
                    _log_hook_error("handle_pre_commit_guard", e)
            return {"output": schema_warn, "error": None}

        # Check for peer-claimed file overlaps
        overlapping = []
        for peer in peers:
            try:
                claims = mgr.get_session_claims(peer["session_id"])
                peer_files = claims.get("file_claims", [])
                for sf in staged_files:
                    full_path = os.path.join(project, sf) if project else sf
                    if full_path in peer_files or sf in peer_files:
                        overlapping.append((sf, _agent_nickname(peer["session_id"])))
            except Exception as e:
                _log_hook_error("handle_pre_commit_guard", e)

        # Check for files not in own claim list (warn only)
        unclaimed_by_self = []
        try:
            own_claims = mgr.get_session_claims(session_id)
            own_files = own_claims.get("file_claims", [])
            for sf in staged_files:
                full_path = os.path.join(project, sf) if project else sf
                if full_path not in own_files and sf not in own_files:
                    unclaimed_by_self.append(sf)
        except Exception as e:
            _log_hook_error("handle_pre_commit_guard", e)

        # BLOCK if staging peer-claimed files
        if overlapping:
            lines = [f"[COMMIT-GUARD] BLOCKED: staging {len(overlapping)} file(s) claimed by other agent(s):"]
            for fname, peer_name in overlapping[:10]:
                lines.append(f"  {fname} (claimed by {peer_name})")
            lines.append("")
            lines.append("Unstage peer files with: git reset HEAD <file>")
            lines.append("Or coordinate via omega_send_message to request file release.")
            mgr.record_metric(
                "gate_blocked",
                session_id=session_id,
                metadata={"action": "commit", "overlap_count": len(overlapping)},
            )
            return {"output": "\n".join(lines), "exit_code": 2, "error": None}

        # Build info message
        lines = [f"[COMMIT-COORD] {len(peers)} peer(s) active:"]
        for p in peers:
            p_name = _agent_nickname(p["session_id"])
            p_task = (p.get("task") or "idle")[:50]
            p_proj = os.path.basename(p.get("project", ""))
            lines.append(f"  - {p_name}: {p_task} [{p_proj}]")

        # BLOCK unclaimed staged files in multi-agent mode
        # In solo mode this is just a warning, but with peers active,
        # unclaimed files likely belong to another agent (Cedar incident, Mar 2026).
        if unclaimed_by_self:
            lines.append(f"  [!] {len(unclaimed_by_self)} staged file(s) not in your claim list:")
            for fname in unclaimed_by_self[:5]:
                lines.append(f"     {os.path.basename(fname)}")
            if len(unclaimed_by_self) > 5:
                lines.append(f"     +{len(unclaimed_by_self) - 5} more")
            lines.append("")
            lines.append("  These may belong to another agent. Unstage with: git reset HEAD <file>")
            lines.append("  Or claim them first: omega_file_check + Edit/Write to auto-claim.")
            mgr.record_metric(
                "gate_blocked",
                session_id=session_id,
                metadata={"action": "commit_unclaimed", "unclaimed_count": len(unclaimed_by_self)},
            )
            return {"output": "\n".join(lines), "exit_code": 2, "error": None}

        # Check for peer file reads on staged files (warning only, non-blocking)
        peer_read_warnings = []
        try:
            abs_staged = []
            for sf in staged_files:
                full_path = os.path.join(project, sf) if project else sf
                abs_staged.append(full_path)
            peer_reads = mgr.get_peer_file_reads(session_id, abs_staged)
            for fpath, readers in peer_reads.items():
                for r in readers:
                    reader_name = _agent_nickname(r["session_id"])
                    peer_read_warnings.append(
                        f"     {os.path.basename(fpath)} read by {reader_name} ({r['read_count']}x)"
                    )
        except Exception as e:
            _log_hook_error("handle_pre_commit_guard", e)

        if peer_read_warnings:
            lines.append(f"  [!] {len(peer_read_warnings)} staged file(s) recently read by peers:")
            lines.extend(peer_read_warnings[:5])
            if len(peer_read_warnings) > 5:
                lines.append(f"     +{len(peer_read_warnings) - 5} more")
            lines.append("  These peers may be planning edits. Consider coordinating first.")

        if schema_warn:
            lines.append(schema_warn)

        return {"output": "\n".join(lines), "error": None}

    except Exception as e:
        _log_hook_error("pre_commit_guard", e)
        return {"output": "", "error": None}




def handle_pre_deploy_guard(payload: dict) -> dict:
    """Deploy guard — BLOCKS deployment commands unless coordination gate was run.

    Detects vercel deploy/link/project add and checks if omega_query(event_type="decision")
    was called in the current session. If not, blocks with exit_code=2.

    Gate is cleared by calling omega_query(event_type="decision") — tracked via file marker.
    """
    tool_name = payload.get("tool_name", "")
    if tool_name != "Bash":
        return {"output": "", "error": None}

    input_data = _parse_tool_input(payload)
    command = input_data.get("command", "")

    deploy_patterns = [
        r"\bvercel\s+(?:deploy|link|project\s+add|domains?\s+add)",
        r"\bvercel\s+--prod\b",
        r"\bfly\s+deploy\b",
    ]
    if not any(re.search(p, command) for p in deploy_patterns):
        return {"output": "", "error": None}

    # Record HIGH-tier gate check
    session_id = payload.get("session_id", "")
    try:
        from omega.coordination import get_manager

        get_manager().record_metric("gate_check_high", session_id=session_id, metadata={"command": command[:100]})
    except Exception as e:
        _log_hook_error("handle_pre_deploy_guard", e)
    try:
        from omega.server.handlers import is_deploy_gate_cleared

        gate_ok = is_deploy_gate_cleared(session_id)
    except ImportError:
        gate_ok = False

    if gate_ok:
        # Gate cleared — allow with advisory context
        return {"output": "[DEPLOY-GATE] Gate cleared. Proceeding.", "error": None}

    # Gate NOT cleared — identify what's missing
    project_dir = payload.get("project", "")
    missing = []
    try:
        from omega.server.handlers import _is_coord_status_checked

        # Check each marker individually to give specific guidance
        # Decision marker
        _decision_ok = False
        _gate_candidates = []
        if session_id:
            _gate_candidates.append(_omega_dir() / "gates" / f"{session_id}.gate")
        _gate_candidates.append(_omega_dir() / "gates" / "default.gate")
        for _gf in _gate_candidates:
            if _gf.exists():
                _ts = float(_gf.read_text().strip())
                if (time.time() - _ts) < 1800:
                    _decision_ok = True
                    break
        if not _decision_ok:
            missing.append("omega_query(event_type='decision', query='<target area>')")
        if not _is_coord_status_checked(session_id):
            missing.append("omega_coord_status")
    except (OSError, ValueError):
        missing = ["omega_query(event_type='decision')", "omega_coord_status"]

    lines = [
        "\n[DEPLOY-GATE] BLOCKED: Coordination gate not cleared.",
        "  You MUST run BOTH of these before deploying:",
    ]
    for m in missing:
        lines.append(f"    - {m}  (NOT YET RUN)")
    lines.append("  This prevents deploying without checking peer activity (coordination bug fix).")

    # Surface relevant decisions to help
    try:
        from omega.sqlite_store import SQLiteStore

        store = SQLiteStore()
        project_name = os.path.basename(project_dir.rstrip("/")) if project_dir else ""
        query = f"{project_name} deploy vercel website"
        results = store.query(query=query, event_type="decision", limit=3)
        if results:
            lines.append("  Relevant OMEGA decisions (run the query to clear the gate):")
            for r in results:
                content = r.get("content", "")[:200]
                lines.append(f"  - {content}")
    except Exception as e:
        _log_hook_error("handle_pre_deploy_guard", e)

    return {"output": "\n".join(lines), "exit_code": 2, "error": None}


# ---------------------------------------------------------------------------
# Pre-Agent memory injection reminder
# ---------------------------------------------------------------------------

# Debounce: only remind once per session (not per agent spawn)
_agent_memory_reminded: set[str] = set()
_MAX_AGENT_REMINDED = 100


def handle_pre_agent_memory(payload: dict) -> dict:
    """Remind the primary agent to inject OMEGA context before spawning subagents.

    Non-blocking (exit 0). Fires once per session on Agent tool use.
    Subagents cannot call OMEGA MCP tools, so the primary agent must
    query OMEGA and include results in the agent prompt.
    """
    try:
        session_id = payload.get("session_id", "")
        if not session_id:
            return {"output": "", "error": None}

        # Only remind once per session
        if session_id in _agent_memory_reminded:
            return {"output": "", "error": None}

        _agent_memory_reminded.add(session_id)
        if len(_agent_memory_reminded) > _MAX_AGENT_REMINDED:
            oldest = next(iter(_agent_memory_reminded))
            _agent_memory_reminded.discard(oldest)

        return {
            "output": (
                "[AGENT-MEMORY] Subagents cannot call OMEGA tools. Before spawning:\n"
                "1. omega_query() for task-relevant decisions/preferences/constraints\n"
                "2. Include key results in the agent prompt\n"
                "3. Tell the agent: 'Do NOT fabricate URLs or project details not in this prompt'"
            ),
            "error": None,
        }
    except Exception:
        return {"output": "", "error": None}
