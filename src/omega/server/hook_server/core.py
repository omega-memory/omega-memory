"""Hook server infrastructure — dispatch table, UDS/TCP server, start/stop."""

import asyncio
import json
import logging
import sys
import time

logger = logging.getLogger("omega.hook_server")

import omega.server.hook_server as _pkg  # package-level access for SOCK_PATH, HOOK_HOST, HOOK_PORT (tests override)
from .utils import _log_hook_error, _log_timing

# Import all handlers for the dispatch table
from .session import handle_session_start, handle_session_stop
from .memory import handle_surface_memories
from .guards import (
    handle_auto_capture,
    handle_auto_claim_file,
    handle_pre_add_guard,
    handle_pre_file_guard,
    handle_pre_task_guard,
    handle_pre_push_guard,
    handle_pre_commit_guard,
    handle_pre_deploy_guard,
    handle_pre_alignment_gate,
    handle_pre_protocol_gate,
    handle_pre_irreversible_advisor,
    handle_pre_agent_memory,
)
from .insights import handle_pre_insight_surface
from .coordination import handle_coord_session_start, handle_coord_session_stop
from .heartbeat import handle_coord_heartbeat
from .trace import handle_trace_capture
from .assistant import handle_assistant_capture



# ---------------------------------------------------------------------------
# Handler dispatch table — core handlers are always available; commercial
# handlers (coordination, etc.) are loaded when present in the monorepo
# or provided by plugins (e.g. omega-pro).
# ---------------------------------------------------------------------------

# Core memory handlers — always shipped with omega-memory
_CORE_HOOK_HANDLERS = {
    "session_start": handle_session_start,
    "session_stop": handle_session_stop,
    "surface_memories": handle_surface_memories,
    "auto_capture": handle_auto_capture,
    "assistant_capture": handle_assistant_capture,
}

# Commercial handlers — present in monorepo, becomes plugin-provided in open-core
_COMMERCIAL_HOOK_HANDLERS = {
    "coord_session_start": handle_coord_session_start,
    "coord_session_stop": handle_coord_session_stop,
    "coord_heartbeat": handle_coord_heartbeat,
    "auto_claim_file": handle_auto_claim_file,
    "pre_add_guard": handle_pre_add_guard,
    "pre_file_guard": handle_pre_file_guard,
    "pre_task_guard": handle_pre_task_guard,
    "pre_push_guard": handle_pre_push_guard,
    "pre_commit_guard": handle_pre_commit_guard,
    "pre_deploy_guard": handle_pre_deploy_guard,
    "pre_alignment_gate": handle_pre_alignment_gate,
    "pre_protocol_gate": handle_pre_protocol_gate,
    "pre_insight_surface": handle_pre_insight_surface,
    "pre_irreversible_advisor": handle_pre_irreversible_advisor,
    "pre_agent_memory": handle_pre_agent_memory,
    "trace_capture": handle_trace_capture,
}

# Build the dispatch table: core + commercial (if available) + plugins
HOOK_HANDLERS = dict(_CORE_HOOK_HANDLERS)

# Coordination handlers: only register if the coordination module is available.
# In core-only installs (PyPI omega), these remain defined but unreachable —
# hooks-core.json won't wire them, and this guard prevents accidental dispatch.
try:
    import omega.coordination  # noqa: F401

    HOOK_HANDLERS.update(_COMMERCIAL_HOOK_HANDLERS)
except ImportError:
    pass




def register_hook_handler(name: str, handler):
    """Register a hook handler at runtime (for plugins)."""
    HOOK_HANDLERS[name] = handler


# ---------------------------------------------------------------------------
# UDS Server
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# UDS Server
# ---------------------------------------------------------------------------


async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Handle a single hook client connection."""
    t0 = time.monotonic()
    hook_name = "unknown"
    try:
        # Read until EOF — client calls shutdown(SHUT_WR) after sendall()
        chunks = []
        while True:
            chunk = await asyncio.wait_for(reader.read(65536), timeout=10.0)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        if not data:
            writer.close()
            return

        request = json.loads(data.decode("utf-8").strip())

        # Batch mode: {"hooks": ["a", "b", ...], ...}
        # Single mode: {"hook": "a", ...}
        hook_names = request.pop("hooks", None)
        if hook_names:
            hook_name = "+".join(hook_names)
            from omega.server.mcp_server import _HOOK_EXECUTOR
            loop = asyncio.get_running_loop()
            results = []
            for name in hook_names:
                handler = HOOK_HANDLERS.get(name)
                if not handler:
                    results.append({"output": "", "error": f"Unknown hook: {name}"})
                else:
                    r = await loop.run_in_executor(_HOOK_EXECUTOR, handler, request)
                    results.append(r)
                    # Short-circuit on block — skip remaining hooks
                    if r.get("exit_code"):
                        break
            response = {"results": results}
        else:
            hook_name = request.pop("hook", "unknown")
            handler = HOOK_HANDLERS.get(hook_name)
            if not handler:
                response = {"output": "", "error": f"Unknown hook: {hook_name}"}
            else:
                # Run handler in dedicated hook executor to prevent starvation
                from omega.server.mcp_server import _HOOK_EXECUTOR
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(_HOOK_EXECUTOR, handler, request)

        writer.write(json.dumps(response).encode("utf-8"))
        await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        # Hook client disconnected before we could send the response.
        # This is normal when hooks time out (e.g., 3-5s timeout in settings.json).
        logger.debug("Hook client disconnected before response: %s", hook_name)
    except asyncio.TimeoutError:
        try:
            writer.write(json.dumps({"output": "", "error": "timeout"}).encode("utf-8"))
            await writer.drain()
        except Exception:
            pass
    except (json.JSONDecodeError) as e:
        _log_hook_error(f"connection/{hook_name}", e)
        try:
            writer.write(json.dumps({"output": "", "error": str(e)}).encode("utf-8"))
            await writer.drain()
        except Exception:
            pass
    finally:
        elapsed_ms = (time.monotonic() - t0) * 1000
        _log_timing(hook_name, elapsed_ms)
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


_hook_server: asyncio.Server | None = None



async def start_hook_server() -> asyncio.Server | None:
    """Start the hook server. Uses TCP on Windows, Unix domain socket elsewhere."""
    global _hook_server

    try:
        if sys.platform == "win32":
            # Windows: TCP loopback
            _hook_server = await asyncio.start_server(
                handle_connection, host=_pkg.HOOK_HOST, port=_pkg.HOOK_PORT
            )
            logger.info("Hook server listening on %s:%s", _pkg.HOOK_HOST, _pkg.HOOK_PORT)
        else:
            # Unix: domain socket
            _pkg.SOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
            if _pkg.SOCK_PATH.exists():
                _pkg.SOCK_PATH.unlink()
            _hook_server = await asyncio.start_unix_server(handle_connection, path=str(_pkg.SOCK_PATH))
            _pkg.SOCK_PATH.chmod(0o600)
            logger.info("Hook server listening on %s", _pkg.SOCK_PATH)
        return _hook_server
    except Exception as e:
        logger.error("Failed to start hook server: %s", e, exc_info=True)
        return None



async def stop_hook_server(srv: asyncio.Server | None = None):
    """Stop the hook server and clean up socket.

    Only deletes the socket file if this process owns the server,
    to avoid breaking another MCP server's active socket.
    """
    global _hook_server
    server = srv or _hook_server
    if server:
        server.close()
        await server.wait_closed()
        _hook_server = None

        # Only unlink on Unix (TCP doesn't leave socket files)
        if sys.platform != "win32" and _pkg.SOCK_PATH and _pkg.SOCK_PATH.exists():
            try:
                _pkg.SOCK_PATH.unlink()
            except Exception as e:
                logger.debug("Socket unlink failed: %s", e)

