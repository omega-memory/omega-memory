"""Hook daemon core: dispatch table, socket server, start/stop.

Protocol (one request per connection, client half-closes after sending):

    {"hook": "<name>", ...payload}            -> {"output": str, "error": str | None}
    {"hooks": ["<a>", "<b>"], ...payload}     -> {"results": [<response>, ...]}

A response may carry ``exit_code``; a non-zero value tells ``fast_hook.py``
to exit with it, which is how blocking guards veto a tool call. In a batch,
the first non-zero ``exit_code`` short-circuits the remaining hooks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import omega.server.hook_server as _pkg  # SOCK_PATH / HOOK_HOST / HOOK_PORT are read at call time so tests can override them
from .handlers import (
    handle_assistant_capture,
    handle_auto_capture,
    handle_session_start,
    handle_session_stop,
    handle_surface_memories,
)
from .owner_state import clear_owner_state, write_owner_state
from .utils import _log_hook_error, _log_timing

logger = logging.getLogger("omega.hook_server")

HookHandler = Callable[[dict], dict]

# Core handlers, always shipped with omega-memory. Plugins (omega-pro) add
# their own through register_hook_handler().
_CORE_HOOK_HANDLERS: dict[str, HookHandler] = {
    "session_start": handle_session_start,
    "session_stop": handle_session_stop,
    "surface_memories": handle_surface_memories,
    "auto_capture": handle_auto_capture,
    "assistant_capture": handle_assistant_capture,
}

HOOK_HANDLERS: dict[str, HookHandler] = dict(_CORE_HOOK_HANDLERS)

# Handlers touch the SQLite store through the bridge; two workers keep hook
# traffic from starving the MCP tool handlers while still overlapping I/O.
_HOOK_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="omega-hook")

_READ_TIMEOUT_S = 10.0


def register_hook_handler(name: str, handler: HookHandler) -> None:
    """Register or replace a hook handler at runtime (used by plugins)."""
    HOOK_HANDLERS[name] = handler


async def _dispatch(name: str, request: dict) -> dict:
    handler = HOOK_HANDLERS.get(name)
    if handler is None:
        return {"output": "", "error": f"Unknown hook: {name}"}
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_HOOK_EXECUTOR, handler, request)


async def _respond(request: dict) -> dict:
    hook_names = request.pop("hooks", None)
    if hook_names:
        results = []
        for name in hook_names:
            result = await _dispatch(name, request)
            results.append(result)
            if result.get("exit_code"):
                break
        return {"results": results}
    return await _dispatch(request.pop("hook", "unknown"), request)


async def _read_request(reader: asyncio.StreamReader) -> bytes:
    """Read until the client half-closes; an empty body is a liveness probe."""
    chunks = []
    while True:
        chunk = await asyncio.wait_for(reader.read(65536), timeout=_READ_TIMEOUT_S)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Serve one hook client: read the request to EOF, dispatch, write the response."""
    started = time.monotonic()
    hook_name = "unknown"
    data = b""
    try:
        data = await _read_request(reader)
        if not data:
            # The MCP server's socket watchdog connects and hangs up to check
            # we are alive. Nothing to dispatch, nothing worth logging.
            return

        request = json.loads(data.decode("utf-8").strip())
        hook_name = "+".join(request["hooks"]) if request.get("hooks") else request.get("hook", "unknown")
        response = await _respond(request)
        writer.write(json.dumps(response).encode("utf-8"))
        await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        # The client gave up (Claude Code's hook timeout) before we answered.
        logger.debug("hook client disconnected before response: %s", hook_name)
    except asyncio.TimeoutError:
        await _write_error(writer, "timeout")
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        _log_hook_error(f"connection/{hook_name}", error)
        await _write_error(writer, str(error))
    finally:
        if data:
            _log_timing(hook_name, (time.monotonic() - started) * 1000)
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("hook connection close raised", exc_info=True)


async def _write_error(writer: asyncio.StreamWriter, message: str) -> None:
    try:
        writer.write(json.dumps({"output": "", "error": message}).encode("utf-8"))
        await writer.drain()
    except (ConnectionResetError, BrokenPipeError, OSError):
        logger.debug("could not deliver hook error response", exc_info=True)


_hook_server: asyncio.Server | None = None


async def start_hook_server() -> asyncio.Server | None:
    """Start listening: a Unix domain socket, or TCP loopback on Windows.

    Returns None when the socket cannot be bound. The MCP server keeps running
    without the daemon; ``fast_hook.py`` then falls back to its cold path.
    """
    global _hook_server
    try:
        if sys.platform == "win32":
            _hook_server = await asyncio.start_server(handle_connection, host=_pkg.HOOK_HOST, port=_pkg.HOOK_PORT)
            write_owner_state(os.getpid(), "tcp", "ready")
            logger.info("hook server listening on %s:%s", _pkg.HOOK_HOST, _pkg.HOOK_PORT)
        else:
            sock_path = _pkg.SOCK_PATH
            sock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if sock_path.exists():
                sock_path.unlink()
            _hook_server = await asyncio.start_unix_server(handle_connection, path=str(sock_path))
            sock_path.chmod(0o600)
            write_owner_state(os.getpid(), "unix", "ready")
            logger.info("hook server listening on %s", sock_path)
        return _hook_server
    except OSError as error:
        logger.error("failed to start hook server: %s", error, exc_info=True)
        return None


async def stop_hook_server(srv: asyncio.Server | None = None) -> None:
    """Stop the server and remove the socket file this process created."""
    global _hook_server
    server = srv or _hook_server
    if server is None:
        return
    server.close()
    await server.wait_closed()
    _hook_server = None
    if sys.platform != "win32" and _pkg.SOCK_PATH and _pkg.SOCK_PATH.exists():
        try:
            _pkg.SOCK_PATH.unlink()
        except OSError as error:
            logger.debug("socket unlink failed: %s", error)
    clear_owner_state(os.getpid())
