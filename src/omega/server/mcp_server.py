"""OMEGA MCP Server -- MCP server for Claude Code (stdio or HTTP daemon).

Supports two transports:
- **stdio** (default): One process per Claude Code session. Zero-config.
- **http** (daemon): One shared process serving all sessions via Streamable HTTP.
  Set OMEGA_TRANSPORT=http or use `omega serve --daemon`.

Requires the 'server' extra: pip install omega-memory[server]
"""

import atexit
import asyncio
import collections
import logging
import os
import random
import socket
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print(
        "Error: MCP server requires the 'mcp' package.\n"
        "Install with: pip install omega-memory[server]\n"
        "Or directly: pip install mcp>=1.0.0",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Transport configuration ---
_TRANSPORT = os.environ.get("OMEGA_TRANSPORT", "stdio").lower()
_HTTP_HOST = os.environ.get("OMEGA_HTTP_HOST", "127.0.0.1")
_HTTP_PORT = int(os.environ.get("OMEGA_HTTP_PORT", "8377"))
_start_time = time.monotonic()

from omega.server.tool_schemas import TOOL_SCHEMAS as _CORE_SCHEMAS, get_condensed_schemas
from omega.server.handlers import HANDLERS as _CORE_HANDLERS
from omega.server import handlers as _handlers_module

# Start with core memory tools
TOOL_SCHEMAS = list(_CORE_SCHEMAS)
HANDLERS = dict(_CORE_HANDLERS)

# Built-in optional modules (coordination, router, profile, knowledge, entity,
# oracle, typed memory, audit, federation, dreaming, stores, ingest).
# These live in the private `omega_platform` namespace distributed via the
# private Pro delivery path. Core only loads them when an installed extension
# advertises the `pro_tools` capability; it does not trust a local license
# function for entitlement.
_BUILTIN_MODULES = [
    ("omega_platform.server.coord_schemas", "COORD_TOOL_SCHEMAS", "omega_platform.server.coord_handlers", "COORD_HANDLERS"),
    ("omega_platform.router.tool_schemas", "ROUTER_TOOL_SCHEMAS", "omega_platform.router.handlers", "ROUTER_HANDLERS"),
    ("omega_platform.profile.tool_schemas", "PROFILE_TOOL_SCHEMAS", "omega_platform.profile.handlers", "PROFILE_HANDLERS"),
    ("omega_platform.knowledge.tool_schemas", "KNOWLEDGE_TOOL_SCHEMAS", "omega_platform.knowledge.handlers", "KNOWLEDGE_HANDLERS"),
    ("omega_platform.entity.tool_schemas", "ENTITY_TOOL_SCHEMAS", "omega_platform.entity.handlers", "ENTITY_HANDLERS"),
    ("omega_platform.oracle.tool_schemas", "ORACLE_TOOL_SCHEMAS", "omega_platform.oracle.handlers", "ORACLE_HANDLERS"),
    ("omega_platform.ingest.tool_schemas", "INGEST_TOOL_SCHEMAS", "omega_platform.ingest.handlers", "INGEST_HANDLERS"),
    ("omega_platform.stores.tool_schemas", "STORES_TOOL_SCHEMAS", "omega_platform.stores.handlers", "STORES_HANDLERS"),
    ("omega_platform.dreaming.tool_schemas", "DREAMING_TOOL_SCHEMAS", "omega_platform.dreaming.handlers", "DREAMING_HANDLERS"),
    ("omega_platform.audit.tool_schemas", "AUDIT_TOOL_SCHEMAS", "omega_platform.audit.handlers", "AUDIT_HANDLERS"),
    ("omega_platform.federation.tool_schemas", "FEDERATION_TOOL_SCHEMAS", "omega_platform.federation.handlers", "FEDERATION_HANDLERS"),
    ("omega_platform.typed.tool_schemas", "TYPED_TOOL_SCHEMAS", "omega_platform.typed.handlers", "TYPED_HANDLERS"),
]

import importlib

# Check extension capabilities before loading commercial modules
_pro_tools_available = False
try:
    from omega.plugins import has_capability
    _pro_tools_available = has_capability("pro_tools")
except Exception as e:
    logging.getLogger("omega.mcp_server").debug("Capability check failed, defaulting to non-pro: %s", e)
    _pro_tools_available = False

if _pro_tools_available:
    for _schema_mod, _schema_attr, _handler_mod, _handler_attr in _BUILTIN_MODULES:
        try:
            _sm = importlib.import_module(_schema_mod)
            _hm = importlib.import_module(_handler_mod)
            TOOL_SCHEMAS = TOOL_SCHEMAS + getattr(_sm, _schema_attr)
            HANDLERS = {**HANDLERS, **getattr(_hm, _handler_attr)}
        except ImportError:
            pass
else:
    logging.getLogger("omega.server").info(
        "Pro modules available — run 'omega activate <key>' to unlock. "
        "Upgrade at https://omegamax.co/pro?ref=mcp-startup"
    )

# Discover external plugins (e.g. omega-pro)
from omega.plugins import discover_plugins

_discovered_plugins = discover_plugins()
for _plugin in _discovered_plugins:
    if _plugin.TOOL_SCHEMAS:
        TOOL_SCHEMAS = TOOL_SCHEMAS + _plugin.TOOL_SCHEMAS
    if _plugin.HANDLERS:
        HANDLERS = {**HANDLERS, **_plugin.HANDLERS}

# ---------------------------------------------------------------------------
# Condensed Mode (CodeMode-inspired) — expose 2 meta-tools + 3 standalone
# instead of 60+ individual tools to save ~80% context tokens.
# On by default. Disable with OMEGA_CONDENSED=0 if needed.
# ---------------------------------------------------------------------------
_CONDENSED_MODE = os.environ.get("OMEGA_CONDENSED", "1") != "0"

# Give handlers access to the full merged schema list and handler registry
# so omega_tools and omega_call can discover and dispatch all tools.
_handlers_module._ALL_SCHEMAS = TOOL_SCHEMAS
_handlers_module._ALL_HANDLERS.update(HANDLERS)

# Wire plugin retrieval profiles and score modifiers to SQLiteStore (lazy)
def _wire_plugin_retrieval():
    """Register plugin retrieval profiles and score modifiers on the store."""
    try:
        from omega.bridge import _get_store
        store = _get_store()
        for plugin in _discovered_plugins:
            if getattr(plugin, "RETRIEVAL_PROFILES", None):
                store.register_plugin_profiles(plugin.RETRIEVAL_PROFILES)
            for modifier in getattr(plugin, "SCORE_MODIFIERS", []):
                store.register_score_modifier(modifier)
    except Exception as e:
        logger.debug("Plugin profile registration failed: %s", e)

# ---------------------------------------------------------------------------
# Dedicated SQLite executor — serializes all blocking DB/hook work onto a
# single thread, preventing the GIL+GC race in _pysqlite_query_execute that
# caused SIGSEGV crashes under concurrent multi-thread SQLite access.
# Also used as the default asyncio executor to cap thread growth.
# ---------------------------------------------------------------------------
_SQLITE_EXECUTOR = ThreadPoolExecutor(
    max_workers=4 if os.environ.get("OMEGA_TRANSPORT", "").lower() == "http" else 2,
    thread_name_prefix="omega-db",
)

# Dedicated executor for hook handlers — prevents hooks from starving behind
# MCP tool calls that saturate _SQLITE_EXECUTOR. SQLite WAL mode handles
# concurrent reader access safely across both executors.
_HOOK_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="omega-hook")

# RSS memory watchdog threshold (bytes). Default 1 GB for stdio, 4 GB for HTTP daemon.
# Override with OMEGA_RSS_LIMIT_MB env var. HTTP daemon serves 8-10 concurrent Claude
# Code sessions; 2 GB is normal operating load. Self-killing at 2 GB is worse than
# running hot — it breaks ALL connected sessions simultaneously.
_RSS_LIMIT_DEFAULT = "8192" if _TRANSPORT == "http" else "1024"
_RSS_LIMIT_BYTES = int(os.environ.get("OMEGA_RSS_LIMIT_MB", _RSS_LIMIT_DEFAULT)) * 1024 * 1024

# Idle watchdog: exit after this many seconds without a tool call.
# Override with OMEGA_IDLE_TIMEOUT env var. 0 = disabled.
_IDLE_TIMEOUT = int(os.environ.get("OMEGA_IDLE_TIMEOUT", "3600"))
_last_activity: float = time.monotonic()

# Shutdown flag — set True during graceful shutdown to reject new tool calls
_shutting_down: bool = False


def _close_on_exit():
    """Close SQLite store and unregister PID when the MCP server process exits."""
    global _shutting_down
    _shutting_down = True
    # Close UsageTracker (llm_usage.db connection)
    try:
        if _usage_tracker_instance is not None:
            _usage_tracker_instance.close()
    except Exception:
        pass
    # Close CoordinationManager (omega.db connection)
    try:
        from omega_platform.orchestrator.coordination import close_manager
        close_manager()
    except Exception:
        pass
    try:
        from omega.server.pid_registry import unregister_pid
        unregister_pid()
    except Exception:
        pass
    try:
        from omega.bridge import _close_store

        _close_store()
    except Exception as e:
        logger.debug("Store close at exit failed: %s", e)
    # Shutdown background executors to release threads
    for executor in (_SQLITE_EXECUTOR, _HOOK_EXECUTOR):
        try:
            executor.shutdown(wait=False)
        except Exception:
            pass


atexit.register(_close_on_exit)

logger = logging.getLogger("omega.server")


_MCP_INSTRUCTIONS = """\
OMEGA gives you persistent memory across conversations. \
You MUST use it automatically in every conversation.

REQUIRED at the start of EVERY conversation:
1. Call omega_welcome() immediately for context briefing
2. Call omega_protocol() for your operating instructions
3. Follow the protocol it returns

During conversations:
- Before non-trivial tasks: call omega_query() to check for prior context and decisions
- After completing tasks: call omega_store(content, "decision") to save key outcomes
- When the user says "remember": call omega_store(text, "user_preference")
- When context is getting full: call omega_checkpoint() to save state

These tools are your memory. Use them proactively without being asked.\
"""

_MCP_INSTRUCTIONS_CONDENSED = """\
OMEGA gives you persistent memory across conversations. \
You MUST use it automatically in every conversation.

REQUIRED at the start of EVERY conversation:
1. Call omega_welcome() immediately for context briefing
2. Call omega_protocol() for your operating instructions
3. Follow the protocol it returns

During conversations:
- Use omega_store() directly to save memories (decisions, lessons, preferences)
- Use omega_tools() to discover available tools and their parameters
- Use omega_call(tool='name', args={...}) to execute any other OMEGA tool
- When context is getting full: omega_call(tool='omega_checkpoint', args={...})

These tools are your memory. Use them proactively without being asked.\
"""


server = Server(
    "omega-memory",
    instructions=_MCP_INSTRUCTIONS_CONDENSED if _CONDENSED_MODE else _MCP_INSTRUCTIONS,
)

# ---------------------------------------------------------------------------
# Rate limiting — sliding-window counters (no new deps)
# ---------------------------------------------------------------------------
_GLOBAL_RATE_LIMIT = int(os.environ.get("OMEGA_RATE_LIMIT_GLOBAL", "300"))  # per minute
_WRITE_RATE_LIMIT = int(os.environ.get("OMEGA_RATE_LIMIT_WRITE", "60"))  # per minute
_RATE_WINDOW_S = 60.0

_global_timestamps: collections.deque = collections.deque()
_write_timestamps: collections.deque = collections.deque()

_WRITE_TOOLS = frozenset({
    # Core write tools (handlers.py)
    "omega_store", "omega_checkpoint", "omega_remind",
    "omega_memory", "omega_maintain", "omega_reflect",
    # Coord session lifecycle
    "omega_session_register", "omega_session_heartbeat",
    "omega_session_deregister", "omega_session_snapshot",
    # Coord file/branch claims
    "omega_file_claim", "omega_file_release",
    "omega_branch_claim", "omega_branch_release",
    # Coord intents
    "omega_intent_announce",
    # Coord tasks
    "omega_task_create", "omega_task_claim", "omega_task_release", "omega_task_complete",
    "omega_task_fail", "omega_task_cancel", "omega_task_progress",
    "omega_task_deps", "omega_update_task",
    # Coord messaging
    "omega_send_message", "omega_handoff",
    # Coord actions
    "omega_action_claim", "omega_action_complete",
    # Coord goals & decisions
    "omega_goal", "omega_goal_link",
    "omega_decision_register", "omega_decision_revoke",
    # Coord council
    "omega_council",
    # Pro features (not in schemas but used by extended handlers)
    "omega_profile_set", "omega_entity_create", "omega_entity_update",
    "omega_ingest_document",
    "omega_oracle_record", "omega_oracle_resolve",
    "omega_oracle_analyze", "omega_oracle_status",
    "omega_track_statement", "omega_resolve_outcome",
    # Condensed mode meta-tool (rate-limited by inner tool name in call_tool)
    "omega_call",
    # Pro-only mutating tools. Each verdict was taken from the real Pro
    # handler's primary operation; incidental audit/telemetry writes do
    # NOT qualify a tool as a write.
    # coordination
    "omega_dispatch",
    "omega_dispatch_template",
    "omega_global_pause",
    "omega_global_resume",
    "omega_objective_approve_gate",
    "omega_objective_cleanup",
    "omega_objective_complete",
    "omega_objective_create",
    "omega_objective_decompose",
    "omega_objective_dispatch_next",
    "omega_objective_replan_subtask",
    "omega_objective_tick",
    "omega_task_decompose",
    "omega_task_release",
    "omega_task_resolve",
    "omega_trace_deposit",
    "omega_worker_spawns",
    # dreaming
    "omega_dream",
    "omega_dream_apply",
    "omega_dream_discard",
    "omega_dream_revert",
    # entity
    "omega_project_create",
    "omega_project_delete",
    "omega_project_update",
    # federation
    "omega_federation_import",
    "omega_federation_trust_add",
    "omega_federation_trust_remove",
    # ingest
    "omega_ingest_file",
    # profile
    "omega_profile_delete",
    # stores
    "omega_store_archive",
    "omega_store_attach",
    "omega_store_create",
    "omega_store_sync",
    # typed
    "omega_typed_store",
})


def _check_rate_limit(tool_name: str) -> str | None:
    """Return an error message if rate limit exceeded, else None."""
    now = time.monotonic()
    cutoff = now - _RATE_WINDOW_S

    # Prune expired entries
    while _global_timestamps and _global_timestamps[0] < cutoff:
        _global_timestamps.popleft()

    if len(_global_timestamps) >= _GLOBAL_RATE_LIMIT:
        logger.warning("Global rate limit exceeded (%d calls/min)", _GLOBAL_RATE_LIMIT)
        return f"Rate limit exceeded: {_GLOBAL_RATE_LIMIT} calls/min globally. Try again shortly."

    _global_timestamps.append(now)

    # Write-tool tier
    if tool_name in _WRITE_TOOLS:
        while _write_timestamps and _write_timestamps[0] < cutoff:
            _write_timestamps.popleft()
        if len(_write_timestamps) >= _WRITE_RATE_LIMIT:
            logger.warning("Write rate limit exceeded (%d calls/min) for tool=%s", _WRITE_RATE_LIMIT, tool_name)
            return f"Rate limit exceeded: {_WRITE_RATE_LIMIT} write calls/min. Try again shortly."
        _write_timestamps.append(now)

    return None


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return all OMEGA tools (or condensed set if OMEGA_CONDENSED=1)."""
    schemas = get_condensed_schemas(TOOL_SCHEMAS) if _CONDENSED_MODE else TOOL_SCHEMAS
    return [
        Tool(
            name=schema["name"],
            description=schema["description"],
            inputSchema=schema["inputSchema"],
        )
        for schema in schemas
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch tool call to the appropriate handler."""
    global _last_activity
    _last_activity = time.monotonic()
    t0 = time.monotonic()

    # Reject new tool calls during shutdown to prevent docling/thread pool errors
    if _shutting_down:
        return [TextContent(type="text", text="Server is shutting down, please retry.")]

    # Rate limiting — for omega_call, rate-limit against the inner tool name
    rate_name = name
    if name == "omega_call" and arguments.get("tool"):
        rate_name = arguments["tool"]
    rate_err = _check_rate_limit(rate_name)
    if rate_err:
        return [TextContent(type="text", text=rate_err)]

    handler = HANDLERS.get(name)
    if not handler:
        # Check if this is a known Pro tool the user hasn't unlocked
        try:
            from omega.server.tool_schemas import TOOL_CATEGORIES
            if name in TOOL_CATEGORIES:
                try:
                    from omega.telemetry import track_nag
                    track_nag("tool_gate")
                except Exception:
                    pass
                return [TextContent(type="text", text=(
                    f"Tool '{name}' requires OMEGA Pro. "
                    f"**Tell the user:** \"The tool I just tried to use ({name}) requires OMEGA Pro. "
                    "Run `omega upgrade` in terminal to learn more ($19/mo).\""
                ))]
        except Exception:
            pass
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    try:
        # Handlers are async def wrapping synchronous DB/embedding work.
        # Call them directly — the previous run_in_executor + asyncio.run()
        # pattern created a new event loop per call, causing ~1.8 GB/hr memory
        # growth from fragmentation and retained references.
        result = await handler(arguments)
        # Extract text from MCP response format
        content_list = result.get("content", [{}])
        text = content_list[0].get("text", str(result)) if content_list else str(result)

        # Log tool call to usage tracker (fire-and-forget)
        _log_tool_usage(name, arguments, time.monotonic() - t0)

        # Release freed native memory after each tool call to prevent
        # MALLOC_LARGE_REUSABLE accumulation (macOS holds freed pages)
        _force_malloc_release()

        return [TextContent(type="text", text=text)]
    except Exception as e:
        logger.error("Tool %s failed: %s", name, e, exc_info=True)
        error_msg = str(e)
        if "database is locked" in error_msg:
            try:
                from omega.server.pid_registry import format_lock_diagnostic
                diag = format_lock_diagnostic()
                error_msg = (
                    f"Database locked. {diag}. "
                    f"Check `ps aux | grep omega` for stale processes and kill them."
                )
            except Exception:
                error_msg = (
                    "Database locked. Another OMEGA process may be holding the WAL lock. "
                    "Check `ps aux | grep omega` for stale processes."
                )
        return [TextContent(type="text", text=f"Error in {name}: {error_msg}")]


def _log_tool_usage(name: str, arguments: dict, elapsed_s: float) -> None:
    """Fire-and-forget logging of tool calls to UsageTracker."""
    try:
        tracker = _get_usage_tracker()
        if tracker:
            tracker.log_call(
                session_id=arguments.get("session_id"),
                tool_name=name,
                model="mcp-tool-call",  # actual LLM model not available via MCP
                input_tokens=0,
                output_tokens=0,
                duration_ms=int(elapsed_s * 1000),
                project=arguments.get("project"),
            )
    except Exception:
        pass  # Never let usage tracking break tool calls


_usage_tracker_instance = None


def _get_usage_tracker():
    """Lazy singleton for UsageTracker."""
    global _usage_tracker_instance
    if _usage_tracker_instance is None:
        try:
            from omega.usage_tracker import UsageTracker
            _usage_tracker_instance = UsageTracker()
        except Exception:
            return None
    return _usage_tracker_instance


async def _idle_watchdog():
    """Exit the process if no tool call has been received within the timeout."""
    while True:
        await asyncio.sleep(30)
        idle = time.monotonic() - _last_activity
        if idle >= _IDLE_TIMEOUT:
            logger.warning("Idle for %.0fs (limit %ds), shutting down.", idle, _IDLE_TIMEOUT)
            _close_on_exit()
            os._exit(0)


async def _socket_watchdog():
    """Re-create the hook socket if deleted or stale (unresponsive)."""
    if sys.platform == "win32":
        return  # TCP server doesn't need file watchdog

    try:
        from omega.server.hook_server import SOCK_PATH, start_hook_server
    except ImportError:
        logger.debug("hook_server not available, socket watchdog disabled")
        return

    while True:
        await asyncio.sleep(15)
        if not SOCK_PATH:
            continue
        if not SOCK_PATH.exists():
            logger.warning("Hook socket deleted, re-creating...")
            await start_hook_server()
        else:
            # Validate socket is actually ours and responsive
            try:
                r, w = await asyncio.wait_for(
                    asyncio.open_unix_connection(path=str(SOCK_PATH)), timeout=2.0
                )
                w.close()
                await w.wait_closed()
            except (OSError, asyncio.TimeoutError):
                logger.warning("Hook socket unresponsive, re-creating...")
                try:
                    SOCK_PATH.unlink()
                except OSError:
                    pass
                await start_hook_server()


_coord_tick_count = 0


def _run_coordination_tick():
    """Sync helper for periodic coordination maintenance."""
    global _coord_tick_count
    _coord_tick_count += 1
    try:
        from omega_platform.orchestrator.coordination import get_manager
        from omega.server.hook_server import (
            _last_deadlock_push,
            DEADLOCK_PUSH_DEBOUNCE_S,
            _agent_nickname,
        )

        mgr = get_manager()

        # Stale cleanup — internally debounced to 5 min
        try:
            mgr._maybe_clean_stale()
        except Exception as e:
            logger.debug("Stale session cleanup failed: %s", e)

        # Flush audit buffer on every tick (time-based fallback)
        try:
            mgr.flush_audit_buffer()
        except Exception as e:
            logger.debug("Audit buffer flush failed: %s", e)

        # Every 5th tick (~5 min): deadlock detection + notification flush + stale pruning
        if _coord_tick_count % 5 == 0:
            # Prune stale debounce entries (>1h old) to prevent unbounded growth
            try:
                from omega.server.hook_server import _debounce_state
                evicted = _debounce_state.prune_stale(3600)
                if evicted:
                    logger.debug("Pruned %d stale debounce entries", evicted)
            except Exception:
                pass
            # Flush batched notifications (high: 1h, medium: 3h cutoffs)
            try:
                flushed = mgr.flush_notification_batch()
                if flushed:
                    logger.debug("Flushed %d batched notifications", flushed)
            except Exception as e:
                logger.debug("Notification flush failed: %s", e)

            try:
                cycles = mgr.detect_deadlocks()
                if cycles:
                    now_dl = time.monotonic()
                    for cycle in cycles[:2]:
                        cycle_key = str(hash(tuple(sorted(cycle[:-1]))))
                        if cycle_key not in _last_deadlock_push or now_dl - _last_deadlock_push[cycle_key] >= DEADLOCK_PUSH_DEBOUNCE_S:
                            _last_deadlock_push[cycle_key] = now_dl
                            cycle_str = " -> ".join(_agent_nickname(s) for s in cycle)
                            for peer in set(cycle[:-1]):
                                try:
                                    mgr.send_message(
                                        from_session=peer,
                                        subject=f"[DEADLOCK] Circular wait: {cycle_str}",
                                        to_session=peer,
                                        msg_type="inform",
                                        ttl_minutes=30,
                                    )
                                except Exception as e:
                                    logger.debug("Deadlock broadcast failed: %s", e)
            except Exception as e:
                logger.debug("Deadlock detection failed: %s", e)
    except Exception as e:
        logger.debug("Coordination tick failed: %s", e)


async def _coordination_loop():
    """Periodic coordination maintenance — runs even during idle."""
    loop = asyncio.get_running_loop()
    while True:
        # Jitter: 60-90s to desynchronize across processes
        await asyncio.sleep(60 + random.uniform(0, 30))
        try:
            await loop.run_in_executor(_SQLITE_EXECUTOR, _run_coordination_tick)
        except Exception as e:
            logger.debug("Coordination loop tick failed: %s", e)


def _configure_logging():
    """Set up logging with both stderr and rotating file handler."""
    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    log_dir = Path.home() / ".omega" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_file = log_dir / "omega.log"

    # Root logger: WARNING to stderr (default for MCP)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    # File handler: captures WARNING+ with rotation (5MB, 3 backups)
    file_handler = RotatingFileHandler(
        str(log_file), maxBytes=5 * 1024 * 1024, backupCount=3,
    )
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)


def _cleanup_dead_sessions():
    """Crash-recovery: clean sessions whose PIDs no longer exist.

    Runs at startup BEFORE registering the new session, so stale sessions
    from prior crashes are cleaned even when the DB is under contention.
    Uses a short busy_timeout and skips gracefully if the DB is locked.
    """
    try:
        from omega_platform.orchestrator.coordination import get_manager
        mgr = get_manager()
        conn = mgr.get_read_connection()

        rows = conn.execute(
            "SELECT session_id, pid FROM coord_sessions WHERE status = 'active'"
        ).fetchall()

        dead_ids = []
        for sid, pid in rows:
            if pid is None:
                dead_ids.append(sid)
            elif pid > 0:
                try:
                    os.kill(pid, 0)  # Signal 0 = check if alive, no signal sent
                except ProcessLookupError:
                    dead_ids.append(sid)
                except PermissionError:
                    pass  # Process exists but owned by another user — leave it

        if dead_ids:
            now = datetime.now(timezone.utc).isoformat()
            with mgr._lock:
                placeholders = ",".join("?" * len(dead_ids))
                # Release claims and mark stopped
                conn.execute(
                    f"DELETE FROM coord_file_claims WHERE session_id IN ({placeholders})",
                    dead_ids,
                )
                conn.execute(
                    f"DELETE FROM coord_branch_claims WHERE session_id IN ({placeholders})",
                    dead_ids,
                )
                conn.execute(
                    f"UPDATE coord_tasks SET status = 'pending', session_id = NULL, claimed_at = NULL "
                    f"WHERE session_id IN ({placeholders}) AND status = 'in_progress'",
                    dead_ids,
                )
                conn.execute(
                    f"UPDATE coord_sessions SET status = 'stopped', last_heartbeat = ? "
                    f"WHERE session_id IN ({placeholders})",
                    [now] + dead_ids,
                )
                conn.commit()
            # Sync status to cloud so the admin dashboard sees them as stopped
            for sid in dead_ids:
                mgr._cloud_fire("update_session_status", sid, "stopped", now)
                mgr._cloud_fire("delete_session_claims", sid)
                mgr._cloud_fire("delete_session_file_reads", sid)
            logger.warning("Crash recovery: cleaned %d dead sessions: %s",
                          len(dead_ids), [s[:8] for s in dead_ids])
    except Exception as e:
        logger.debug("Crash-recovery cleanup failed (non-fatal): %s", e)


# --- Mach task_info objects (hoisted to module level to avoid per-call leaks) ---
_mach_libc = None
_mach_task_port = None
_MACH_TASK_BASIC_INFO = 20

if sys.platform == "darwin":
    try:
        import ctypes
        import ctypes.util

        _mach_libc = ctypes.CDLL(ctypes.util.find_library("c"))
        _mach_task_port = _mach_libc.mach_task_self()

        class _TaskBasicInfo(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64),
                ("user_time_seconds", ctypes.c_int32),
                ("user_time_microseconds", ctypes.c_int32),
                ("system_time_seconds", ctypes.c_int32),
                ("system_time_microseconds", ctypes.c_int32),
                ("policy", ctypes.c_int32),
                ("suspend_count", ctypes.c_int32),
            ]

        _mach_info_size_words = ctypes.sizeof(_TaskBasicInfo()) // 4

        # malloc_zone_pressure_relief — forces macOS malloc to return
        # MALLOC_LARGE_REUSABLE pages to the kernel. Without this, freed
        # large allocations (httpx responses, ONNX tensors, JSON parsing)
        # stay resident as "reusable" pages, inflating RSS by 2-4 GB.
        _malloc_lib = ctypes.CDLL(ctypes.util.find_library("System"))
        _malloc_lib.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _malloc_lib.malloc_zone_pressure_relief.restype = ctypes.c_size_t

        def _force_malloc_release() -> int:
            """Release freed malloc pages back to OS. Returns bytes released."""
            try:
                return _malloc_lib.malloc_zone_pressure_relief(None, 0)
            except Exception:
                return 0
    except Exception:
        _mach_libc = None

        def _force_malloc_release() -> int:
            return 0
else:
    def _force_malloc_release() -> int:
        return 0


def _get_current_rss_bytes() -> int:
    """Get current RSS in bytes using the most accurate OS-specific method.

    On macOS, resource.getrusage ru_maxrss returns *peak* RSS, not current.
    This function uses Mach task_info for accurate current RSS on macOS,
    falling back to ru_maxrss if the Mach call fails.
    """
    if sys.platform == "darwin" and _mach_libc is not None:
        try:
            info = _TaskBasicInfo()
            count = ctypes.c_uint32(_mach_info_size_words)
            ret = _mach_libc.task_info(
                _mach_task_port, _MACH_TASK_BASIC_INFO,
                ctypes.byref(info), ctypes.byref(count),
            )
            if ret == 0:  # KERN_SUCCESS
                return info.resident_size
        except Exception:
            pass

    if sys.platform == "win32":
        # Windows: GetProcessMemoryInfo via psapi. The `resource` stdlib module
        # is Unix-only, so the getrusage fallback below would crash at import.
        try:
            from ctypes import wintypes

            class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            # GetCurrentProcess() returns a pseudo-handle that has been
            # observed to fail in some configs; OpenProcess on the current
            # PID is more reliable per the Windows bug report.
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False,
                kernel32.GetCurrentProcessId(),
            )
            if handle:
                try:
                    if psapi.GetProcessMemoryInfo(
                        handle, ctypes.byref(counters), counters.cb,
                    ):
                        return counters.WorkingSetSize
                finally:
                    kernel32.CloseHandle(handle)
        except Exception:
            pass
        return 0

    # Fallback: getrusage (peak RSS, not current — but better than nothing)
    try:
        import resource
    except ModuleNotFoundError:
        return 0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024  # KB to bytes on Linux
    return rss


async def _rss_watchdog():
    """Periodically check RSS and gracefully exit if it exceeds the limit.

    Prevents runaway memory growth from crashing the terminal. The MCP
    client (Claude Code) will automatically restart the server on next
    tool call.
    """
    import gc

    while True:
        await asyncio.sleep(15)
        try:
            rss = _get_current_rss_bytes()

            # Proactive memory release when RSS exceeds 50% of limit.
            # macOS malloc holds freed large allocations as MALLOC_LARGE_REUSABLE
            # pages (2-4 GB observed). malloc_zone_pressure_relief forces release.
            if rss > _RSS_LIMIT_BYTES * 0.5:
                gc.collect()
                released = _force_malloc_release()
                rss_after = _get_current_rss_bytes()
                logger.warning(
                    "Memory pressure relief: RSS %.1f MB -> %.1f MB "
                    "(malloc released %d bytes, limit %.0f MB)",
                    rss / 1024**2, rss_after / 1024**2,
                    released, _RSS_LIMIT_BYTES / 1024**2,
                )
                rss = rss_after

            # Tracemalloc snapshots: log top allocations periodically when RSS is high
            if os.environ.get("OMEGA_TRACEMALLOC") and rss > _RSS_LIMIT_BYTES * 0.3:
                import tracemalloc as _tm
                if _tm.is_tracing():
                    # Throttle snapshots to once per 60s
                    _now_snap = time.monotonic()
                    if not hasattr(_rss_watchdog, '_last_snap') or (_now_snap - _rss_watchdog._last_snap) > 60:
                        _rss_watchdog._last_snap = _now_snap
                        snapshot = _tm.take_snapshot()
                        top_stats = snapshot.statistics("lineno")
                        logger.warning("tracemalloc top 15 allocations (RSS %.0f MB):", rss / 1024**2)
                        for stat in top_stats[:15]:
                            logger.warning("  %s", stat)

            if rss > _RSS_LIMIT_BYTES:
                global _shutting_down
                _shutting_down = True
                msg = (
                    "RSS %.0f MB exceeds limit %.0f MB, shutting down to prevent crash."
                    % (rss / 1024**2, _RSS_LIMIT_BYTES / 1024**2)
                )
                # Final tracemalloc snapshot before exit (top 25)
                if os.environ.get("OMEGA_TRACEMALLOC"):
                    import tracemalloc as _tm
                    if _tm.is_tracing():
                        snapshot = _tm.take_snapshot()
                        top_stats = snapshot.statistics("lineno")
                        logger.warning("FINAL tracemalloc top 25 (before kill at RSS %.0f MB):", rss / 1024**2)
                        for stat in top_stats[:25]:
                            logger.warning("  %s", stat)
                # Force flush to stderr (unbuffered) before os._exit
                sys.stderr.write(f"WARNING omega.server: {msg}\n")
                sys.stderr.flush()
                logger.warning(msg)
                # Flush all log handlers before hard exit
                for handler in logging.getLogger().handlers:
                    try:
                        handler.flush()
                    except Exception:
                        pass
                _close_on_exit()
                os._exit(0)
        except Exception as e:
            # Log watchdog errors instead of silently swallowing them
            logger.debug("RSS watchdog check failed: %s", e)


def _check_port_available(host: str, port: int) -> bool:
    """Check if a TCP port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


async def _run_http_transport(hook_srv) -> None:
    """Run the MCP server as a Streamable HTTP daemon via uvicorn.

    Uses StreamableHTTPSessionManager to handle multiple concurrent
    Claude Code sessions over a single shared process.
    """
    try:
        import uvicorn
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Mount, Route
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    except ImportError as e:
        print(
            f"Error: HTTP transport requires additional packages: {e}\n"
            "Install with: pip install omega-memory[server] starlette uvicorn",
            file=sys.stderr,
        )
        sys.exit(1)

    # Fail fast if port is already bound (another daemon running)
    if not _check_port_available(_HTTP_HOST, _HTTP_PORT):
        print(
            f"Error: Port {_HTTP_PORT} already in use on {_HTTP_HOST}.\n"
            f"Another OMEGA daemon may be running. Check: curl http://{_HTTP_HOST}:{_HTTP_PORT}/health",
            file=sys.stderr,
        )
        sys.exit(1)

    session_manager = StreamableHTTPSessionManager(
        app=server,
        json_response=False,
        # Stateless mode: each request is self-contained. OMEGA tool calls are
        # already stateless (state lives in SQLite). This makes daemon restarts
        # invisible to clients — no more "Session not found" errors after crash.
        stateless=True,
    )

    async def health(request):
        """Health check endpoint with process diagnostics."""
        rss = _get_current_rss_bytes()
        return JSONResponse({
            "status": "ok",
            "pid": os.getpid(),
            "rss_mb": round(rss / 1024**2, 1),
            "uptime_s": round(time.monotonic() - _start_time, 1),
            "tool_count": len(TOOL_SCHEMAS),
            "transport": "http",
        })

    import contextlib

    try:
        from omega.server.hook_server import stop_hook_server as _stop_hook_srv
    except ImportError:
        async def _stop_hook_srv(*args, **kwargs):
            pass

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with session_manager.run():
            logger.info(
                "OMEGA MCP daemon listening on http://%s:%d/mcp",
                _HTTP_HOST, _HTTP_PORT,
            )
            yield
        await _stop_hook_srv(hook_srv)

    app = Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Mount("/mcp", app=session_manager.handle_request),
        ],
        lifespan=lifespan,
    )

    config = uvicorn.Config(
        app,
        host=_HTTP_HOST,
        port=_HTTP_PORT,
        log_level="warning",
        timeout_graceful_shutdown=5,
    )
    uv_server = uvicorn.Server(config)

    # Handle SIGTERM gracefully (launchd sends this on unload)
    # Track shutdown task to prevent duplicate asyncio tasks on repeated signals
    _shutdown_task = None

    def _handle_shutdown_signal():
        nonlocal _shutdown_task
        global _shutting_down
        _shutting_down = True
        if _shutdown_task is None or _shutdown_task.done():
            _shutdown_task = asyncio.ensure_future(uv_server.shutdown())

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_shutdown_signal)

    await uv_server.serve()


async def main():
    """Entry point for the OMEGA MCP server."""
    _configure_logging()
    logger.info("Starting OMEGA MCP server...")

    # --- Startup memory baseline ---
    _startup_rss = _get_current_rss_bytes()
    logger.warning(
        "Startup RSS: %.1f MB, PID: %d, transport: %s",
        _startup_rss / 1024**2, os.getpid(), _TRANSPORT,
    )

    # Optional tracemalloc for memory leak diagnosis (enable via env var)
    if os.environ.get("OMEGA_TRACEMALLOC"):
        import tracemalloc
        tracemalloc.start(10)
        logger.warning("tracemalloc enabled for memory leak diagnosis")

    # --- P1 fixes: prevent Metal shader cache growth and cap thread pool ---
    # PyTorch MPS (Metal) allocates an unbounded shader compilation cache that
    # grows ~200 MB/min. Disable it since we only use ONNX for embeddings.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    # Disable in-process cross-encoder reranker in HTTP daemon mode.
    # The bge-reranker-v2-m3 ONNX model costs ~2 GB RSS. With multiple
    # concurrent sessions the daemon easily hits the RSS limit and enters
    # a crash loop. Embedding similarity alone is sufficient for query quality.
    if _TRANSPORT == "http":
        os.environ.setdefault("OMEGA_CROSS_ENCODER", "0")
    # Cap asyncio's default executor — prevents unbounded thread growth from
    # run_in_executor(None, ...) calls (was reaching 49-64 threads at crash).
    # Note: hooks use _HOOK_EXECUTOR (see hook_server/core.py) to prevent
    # starvation behind MCP tool calls. Only MCP tools use _SQLITE_EXECUTOR.
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=2, thread_name_prefix="omega-default"))

    # --- Kill orphaned MCP servers before touching the DB ---
    # Orphans from dead Claude sessions hold DB locks; must be killed first.
    # Skip in HTTP daemon mode — daemon runs under launchd with ppid=1 by design.
    if _TRANSPORT != "http":
        try:
            from omega.server.pid_registry import kill_orphaned_servers
            killed = kill_orphaned_servers()
            if killed:
                logger.warning("Killed %d orphaned MCP server(s) at startup", killed)
        except Exception as e:
            logger.debug("Orphan cleanup at startup failed: %s", e)

    # --- P0/P3: Crash-recovery — clean dead sessions from prior crashes ---
    try:
        await loop.run_in_executor(_SQLITE_EXECUTOR, _cleanup_dead_sessions)
    except Exception as e:
        logger.debug("Crash recovery at startup failed: %s", e)

    # Register this process for lock diagnostics
    try:
        from omega.server.pid_registry import register_pid
        register_pid(
            transport=_TRANSPORT,
            port=_HTTP_PORT if _TRANSPORT == "http" else None,
        )
    except Exception:
        pass

    # Start UDS hook server for fast hook dispatch
    try:
        from omega.server.hook_server import start_hook_server, stop_hook_server
    except ImportError:
        async def start_hook_server(*args, **kwargs):
            return None
        async def stop_hook_server(*args, **kwargs):
            pass

    hook_srv = await start_hook_server()

    # Prewarm embedding: try shared daemon first, fall back to in-process ONNX.
    # Daemon eliminates per-process model duplication (~170MB each).
    async def _prewarm():
        # Try daemon with retries (daemon may be busy handling another MCP server)
        for _attempt in range(3):
            try:
                from omega.embedding_client import get_client

                client = get_client()
                if client is not None:
                    health = client.health()
                    if health and health.get("status") == "ok":
                        logger.info("Embedding daemon available, skipping local model load")
                        return
            except Exception:
                pass
            await asyncio.sleep(1)
        # Fall back to in-process preload
        try:
            from omega.embedding import preload_embedding_model_async
            await preload_embedding_model_async()
        except Exception as e:
            logger.debug("Embedding model prewarm failed: %s", e)

    _prewarm_task = asyncio.create_task(_prewarm())

    # Wire plugin retrieval profiles/modifiers to SQLiteStore.
    # Run in executor to avoid blocking the event loop during store init
    # (store init triggers PRAGMA integrity_check which can take 30+ seconds).
    async def _wire_plugins_async():
        await loop.run_in_executor(_SQLITE_EXECUTOR, _wire_plugin_retrieval)

    _wire_plugins_task = asyncio.create_task(_wire_plugins_async())

    # Enrich MCP instructions with memory stats (runs once at startup)
    def _enrich_instructions():
        try:
            from omega.bridge import _get_store
            store = _get_store()
            count = store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            type_count = store._conn.execute(
                "SELECT COUNT(DISTINCT json_extract(metadata, '$.event_type')) FROM memories"
            ).fetchone()[0]
            stats_line = f"\n\nCurrent state: {count} memories across {type_count} types."
            base = _MCP_INSTRUCTIONS_CONDENSED if _CONDENSED_MODE else _MCP_INSTRUCTIONS
            server.instructions = base + stats_line
        except Exception as e:
            logger.debug("MCP instructions enrichment failed (non-fatal): %s", e)

    async def _enrich_async():
        await _wire_plugins_task  # Wait for store to be initialized
        await loop.run_in_executor(_SQLITE_EXECUTOR, _enrich_instructions)

    _enrich_task = asyncio.create_task(_enrich_async())

    # Start idle watchdog (unless disabled).
    # IMPORTANT: Save reference — unref'd tasks get silently GC'd by asyncio.
    # In HTTP daemon mode, idle timeout should be 0 (managed by launchd).
    if _IDLE_TIMEOUT > 0 and _TRANSPORT != "http":
        _watchdog_task = asyncio.create_task(_idle_watchdog())

    # Socket watchdog — re-creates hook.sock if deleted by another session's stop
    _sock_watchdog_task = asyncio.create_task(_socket_watchdog())

    # Background coordination loop — stale cleanup + deadlock detection even during idle
    _coord_loop_task = asyncio.create_task(_coordination_loop())

    # RSS memory watchdog — graceful exit before memory pressure causes SIGSEGV
    _rss_watchdog_task = asyncio.create_task(_rss_watchdog())

    if _TRANSPORT == "http":
        await _run_http_transport(hook_srv)
    else:
        try:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                )
        finally:
            await stop_hook_server(hook_srv)


if __name__ == "__main__":
    asyncio.run(main())
