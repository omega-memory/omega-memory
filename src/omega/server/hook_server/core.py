"""Hook daemon core: dispatch table, socket server, start/stop.

Protocol (one request per connection, client half-closes after sending):

    {"hook": "<name>", ...payload}            -> {"output": str, "error": str | None}
    {"hooks": ["<a>", "<b>"], ...payload}     -> {"results": [<response>, ...]}

A response may carry ``exit_code``; a non-zero value tells ``fast_hook.py``
to exit with it, which is how blocking guards veto a tool call. In a batch,
the first non-zero ``exit_code`` short-circuits the remaining hooks.

A response may also carry ``context``: text for the model. On PreToolUse and
PostToolUse ``fast_hook.py`` prints it as Claude Code's additionalContext
JSON, the only way text reaches the model around a tool call.
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
from dataclasses import dataclass

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
# fast_hook.py waits at most 20 s for an answer (pre_push_guard; every other
# hook 5 s). Past this deadline no client is listening, so the connection is
# closed even when its handler is stuck. The handler's thread cannot be
# cancelled: it runs to the end and only its answer is dropped.
_CONNECTION_DEADLINE_S = 30.0
# Closing flushes pending output first; a client that never reads could stall it.
_CLOSE_TIMEOUT_S = 1.0

# Monotonic time of the last real hook request (liveness probes excluded).
# The MCP server's idle watchdog reads it: a session that uses hooks but no
# OMEGA tool is active, and exiting under it leaves every hook dead.
_last_request_at = 0.0


def last_request_at() -> float:
    """Monotonic time of the most recent hook request this process served (0.0 if none)."""
    return _last_request_at


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


@dataclass
class _HookExchange:
    """What one connection asked for, for the timing log."""

    hook_name: str = "unknown"
    # False for the socket watchdog's empty liveness probe.
    requested: bool = False


async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Serve one hook client, then release its socket, even when a handler is stuck.

    The socket used to be closed only once the handler returned. A handler
    blocked on the store held its connection open, and so did every
    connection queued behind it on the two hook workers, long after their
    clients had given up. Each kept a file descriptor until the handler
    came free.
    """
    started = time.monotonic()
    exchange = _HookExchange()
    try:
        await asyncio.wait_for(_serve_request(reader, writer, exchange), timeout=_CONNECTION_DEADLINE_S)
    except asyncio.TimeoutError:
        logger.warning(
            "hook connection passed its %g s deadline, closing: %s", _CONNECTION_DEADLINE_S, exchange.hook_name
        )
    finally:
        await _close_connection(writer)
        if exchange.requested:
            _log_timing(exchange.hook_name, (time.monotonic() - started) * 1000)


async def _serve_request(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, exchange: _HookExchange) -> None:
    """Read one request to EOF, dispatch it, and write the response."""
    global _last_request_at
    try:
        data = await _read_request(reader)
        if not data:
            # The MCP server's socket watchdog connects and hangs up to check
            # we are alive. Nothing to dispatch, nothing worth logging.
            return

        exchange.requested = True
        _last_request_at = time.monotonic()
        request = json.loads(data.decode("utf-8").strip())
        exchange.hook_name = "+".join(request["hooks"]) if request.get("hooks") else request.get("hook", "unknown")
        response = await _respond(request)
        writer.write(json.dumps(response).encode("utf-8"))
        await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        # The client gave up (Claude Code's hook timeout) before we answered.
        logger.debug("hook client disconnected before response: %s", exchange.hook_name)
    except asyncio.TimeoutError:
        # Only the read times out here: the connection deadline cancels this
        # coroutine instead.
        await _write_error(writer, "timeout")
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        _log_hook_error(f"connection/{exchange.hook_name}", error)
        await _write_error(writer, str(error))


async def _close_connection(writer: asyncio.StreamWriter) -> None:
    """Close the client socket, dropping it if the final flush stalls."""
    writer.close()
    try:
        await asyncio.wait_for(writer.wait_closed(), timeout=_CLOSE_TIMEOUT_S)
    except asyncio.TimeoutError:
        writer.transport.abort()
    except OSError:
        # ConnectionResetError / BrokenPipeError: the client already left.
        logger.debug("hook connection close raised", exc_info=True)


async def _write_error(writer: asyncio.StreamWriter, message: str) -> None:
    try:
        writer.write(json.dumps({"output": "", "error": message}).encode("utf-8"))
        await writer.drain()
    except (ConnectionResetError, BrokenPipeError, OSError):
        logger.debug("could not deliver hook error response", exc_info=True)


_hook_server: asyncio.Server | None = None
# (st_dev, st_ino) of the socket file this process bound. The path is shared
# by every session's server and the newest one takes it over, so a server may
# only remove the file while it is still the one it created.
_bound_socket: tuple[int, int] | None = None


def _file_identity(path) -> tuple[int, int] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_dev, stat.st_ino


def release_socket_file() -> None:
    """Remove the socket file and owner record if this process created them.

    Safe to call from ``atexit`` or just before ``os._exit``: it does not
    touch the event loop. A file another session's server has since bound at
    the same path is left alone.
    """
    global _bound_socket
    if sys.platform != "win32" and _bound_socket is not None:
        sock_path = _pkg.SOCK_PATH
        if sock_path and _file_identity(sock_path) == _bound_socket:
            try:
                sock_path.unlink()
            except OSError as error:
                logger.debug("socket unlink failed: %s", error)
        _bound_socket = None
    clear_owner_state(os.getpid())


async def start_hook_server() -> asyncio.Server | None:
    """Start listening: a Unix domain socket, or TCP loopback on Windows.

    Returns None when the socket cannot be bound. The MCP server keeps running
    without the daemon; ``fast_hook.py`` then falls back to its cold path.
    """
    global _hook_server, _bound_socket
    try:
        if sys.platform == "win32":
            _hook_server = await asyncio.start_server(handle_connection, host=_pkg.HOOK_HOST, port=_pkg.HOOK_PORT)
            write_owner_state(os.getpid(), "tcp", "ready")
            logger.info("hook server listening on %s:%s", _pkg.HOOK_HOST, _pkg.HOOK_PORT)
        else:
            sock_path = _pkg.SOCK_PATH
            sock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            # Take the path over from whichever server holds it: that server
            # keeps serving its own clients and, finding the file no longer
            # its own, will not remove ours when it exits.
            if sock_path.exists():
                sock_path.unlink()
            _hook_server = await asyncio.start_unix_server(handle_connection, path=str(sock_path))
            _bound_socket = _file_identity(sock_path)
            sock_path.chmod(0o600)
            write_owner_state(os.getpid(), "unix", "ready")
            logger.info("hook server listening on %s", sock_path)
        return _hook_server
    except OSError as error:
        logger.error("failed to start hook server: %s", error, exc_info=True)
        return None


async def stop_hook_server(srv: asyncio.Server | None = None) -> None:
    """Stop the server and remove the socket file, if this process still owns it."""
    global _hook_server
    server = srv or _hook_server
    if server is None:
        return
    server.close()
    await server.wait_closed()
    _hook_server = None
    release_socket_file()
