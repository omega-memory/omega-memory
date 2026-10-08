"""The hook server must release every client socket, whatever its handlers do.

``handle_connection`` used to close the client socket only after the hook
handler returned, with no deadline. Handlers run on two worker threads; a
handler blocked on the store held its connection open, and so did every
connection queued behind it, long after their clients had given up. Each
kept a file descriptor, and under heavy hook traffic a server can run out
of them and stop answering hooks.

These tests hold a handler the way a stuck worker does, drive many
short-lived clients through a real Unix socket, and count the process's
open descriptors once every client has gone.
"""
import asyncio
import json
import os
import shutil
import socket
import stat
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from omega.server import hook_server
from omega.server.hook_server import core, owner_state

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="counts Unix socket descriptors")

CLIENTS = 40
PROBES = 10
# Sockets the event loop may legitimately open meanwhile.
SLACK = 4


@pytest.fixture
async def running_server(monkeypatch):
    """A hook server on a short socket path (macOS caps AF_UNIX paths at 104 bytes).

    OMEGA_HOME points into the same directory, so the server's timing log
    stays out of the real home directory.
    """
    directory = Path(tempfile.mkdtemp(prefix="omg", dir="/tmp"))
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.bind(str(directory / "probe.sock"))
    except OSError as error:
        pytest.skip(f"AF_UNIX sockets unavailable here: {error}")
    finally:
        probe.close()
    sock_path = directory / "hook.sock"
    monkeypatch.setenv("OMEGA_HOME", str(directory))
    monkeypatch.setattr(hook_server, "SOCK_PATH", sock_path)
    monkeypatch.setattr(owner_state, "OWNER_STATE_PATH", directory / "owner.json")
    server = await hook_server.start_hook_server()
    assert server is not None
    yield sock_path
    await hook_server.stop_hook_server(server)
    shutil.rmtree(directory, ignore_errors=True)


def _open_sockets() -> int:
    """Sockets open in this process. Files other threads open meanwhile (a database, a log) don't count."""
    count = 0
    for name in os.listdir("/dev/fd"):
        try:
            if stat.S_ISSOCK(os.fstat(int(name)).st_mode):
                count += 1
        except OSError:
            continue  # closed between the listing and the stat
    return count


async def _impatient_client(sock_path: Path, hook: str) -> None:
    """A hook client that gives up after 0.2 s, like a fast_hook.py past its timeout."""
    reader, writer = await asyncio.open_unix_connection(str(sock_path))
    try:
        writer.write(json.dumps({"hook": hook, "session_id": "release-test"}).encode())
        writer.write_eof()
        await asyncio.wait_for(reader.read(), timeout=0.2)
    except asyncio.TimeoutError:
        pass
    finally:
        writer.close()
        await writer.wait_closed()


async def _liveness_probe(sock_path: Path) -> None:
    """The MCP server's socket watchdog connects and hangs up without sending anything."""
    _reader, writer = await asyncio.open_unix_connection(str(sock_path))
    writer.close()
    await writer.wait_closed()


async def _drive_traffic(sock_path: Path, hook: str) -> None:
    await asyncio.gather(
        *(_impatient_client(sock_path, hook) for _ in range(CLIENTS)),
        *(_liveness_probe(sock_path) for _ in range(PROBES)),
    )


async def test_a_stuck_handler_releases_sockets_at_the_connection_deadline(running_server, monkeypatch):
    handler_released = threading.Event()
    monkeypatch.setattr(core, "_CONNECTION_DEADLINE_S", 0.5)
    monkeypatch.setitem(
        core.HOOK_HANDLERS,
        "stuck_probe",
        lambda request: handler_released.wait(10) and {"output": "late", "error": None},
    )
    try:
        before = _open_sockets()
        await _drive_traffic(running_server, "stuck_probe")
        await asyncio.sleep(1.0)
        leaked = _open_sockets() - before
    finally:
        # Free both hook workers, or every later test that dispatches a hook waits on them.
        handler_released.set()

    assert leaked <= SLACK, f"{leaked} sockets still open past the connection deadline"


async def test_a_hook_that_answers_in_time_still_gets_its_reply(running_server, monkeypatch):
    monkeypatch.setattr(core, "_CONNECTION_DEADLINE_S", 0.5)
    monkeypatch.setitem(core.HOOK_HANDLERS, "quick_probe", lambda request: {"output": "ok", "error": None})

    reader, writer = await asyncio.open_unix_connection(str(running_server))
    writer.write(json.dumps({"hook": "quick_probe"}).encode())
    writer.write_eof()
    reply = json.loads(await asyncio.wait_for(reader.read(), timeout=5))
    writer.close()
    await writer.wait_closed()

    assert reply == {"output": "ok", "error": None}


async def test_clients_that_stop_reading_cannot_hold_their_sockets(running_server, monkeypatch):
    """Closing waits for the answer to flush; a client that never reads must not stall it for good."""
    monkeypatch.setattr(core, "_CONNECTION_DEADLINE_S", 0.5)
    monkeypatch.setattr(core, "_CLOSE_TIMEOUT_S", 0.2)
    # Far more than the socket and transport buffers hold, so drain() blocks.
    big_answer = {"output": "x" * 1_000_000, "error": None}
    monkeypatch.setitem(core.HOOK_HANDLERS, "big_probe", lambda request: big_answer)
    silent_clients = 10

    before = _open_sockets()
    writers = []
    try:
        for _ in range(silent_clients):
            _reader, writer = await asyncio.open_unix_connection(str(running_server))
            writer.write(json.dumps({"hook": "big_probe"}).encode())
            writer.write_eof()
            writers.append(writer)
        # Never read: each drain() blocks until the connection deadline, and
        # the close after it must not wait for the answer to flush.
        await asyncio.sleep(1.5)
        server_side_open = _open_sockets() - before - silent_clients  # minus the clients' own sockets
    finally:
        for writer in writers:
            writer.transport.abort()

    assert server_side_open <= SLACK, f"the server kept {server_side_open} sockets whose clients never read"
