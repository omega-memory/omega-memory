"""Hook socket lifecycle: who may remove it, what keeps the server alive, how clients fail.

Pinned by the 2026-09-29 audit:

- D1: after the MCP server idle-exited (60 minutes without an OMEGA tool
  call; hook traffic did not count) the socket file stayed behind, and every
  hook then spent 2 s retrying a dead socket before doing nothing.
- D2: one session exiting unlinked the hook socket even when a later session
  had replaced it, cutting every other session off from capture and surfacing.
"""
import asyncio
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from omega.server import hook_server
from omega.server.hook_server import core, owner_state

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


@pytest.fixture
def short_dir():
    """AF_UNIX paths are capped at 104 bytes on macOS; pytest's tmp_path is longer."""
    directory = Path(tempfile.mkdtemp(prefix="omg", dir="/tmp"))
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def daemon_paths(short_dir, monkeypatch):
    sock_path = short_dir / "hook.sock"
    monkeypatch.setattr(hook_server, "SOCK_PATH", sock_path)
    monkeypatch.setattr(owner_state, "OWNER_STATE_PATH", short_dir / "owner.json")
    return sock_path


def _listen_at(path: Path) -> socket.socket:
    """Bind a listening socket at ``path``: another session's server, as far as ours can tell."""
    other = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    other.bind(str(path))
    other.listen(1)
    return other


def _leave_stale_socket(path: Path) -> None:
    """A socket file with nobody listening, as a killed server leaves it."""
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(path))
    dead.close()


# ---------------------------------------------------------------------------
# D2: only the server that created the socket file removes it
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket lifecycle")
async def test_stop_leaves_a_socket_that_another_server_has_since_replaced(daemon_paths):
    ours = await hook_server.start_hook_server()
    daemon_paths.unlink()  # a newer session takes the path over, as start_hook_server does
    other = _listen_at(daemon_paths)
    try:
        await hook_server.stop_hook_server(ours)

        assert daemon_paths.exists(), "stopping must not remove another live server's socket"
    finally:
        other.close()


@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket lifecycle")
async def test_stop_removes_the_socket_it_created(daemon_paths):
    ours = await hook_server.start_hook_server()

    await hook_server.stop_hook_server(ours)

    assert not daemon_paths.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket lifecycle")
async def test_release_on_hard_exit_removes_only_our_own_socket(daemon_paths):
    ours = await hook_server.start_hook_server()
    try:
        core.release_socket_file()
        assert not daemon_paths.exists()

        other = _listen_at(daemon_paths)
        try:
            core.release_socket_file()
            assert daemon_paths.exists(), "a socket we did not create is left alone"
        finally:
            other.close()
    finally:
        ours.close()


# ---------------------------------------------------------------------------
# D1: hook traffic keeps the server alive; hard exits do not leave a stale socket
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket lifecycle")
async def test_a_hook_request_counts_as_activity_but_a_liveness_probe_does_not(daemon_paths):
    server = await hook_server.start_hook_server()
    try:
        before = core.last_request_at()

        reader, writer = await asyncio.open_unix_connection(str(daemon_paths))
        writer.close()  # the socket watchdog's empty probe
        await writer.wait_closed()
        await asyncio.sleep(0.05)
        assert core.last_request_at() == before

        reader, writer = await asyncio.open_unix_connection(str(daemon_paths))
        writer.write(json.dumps({"hook": "no_such_hook"}).encode())
        writer.write_eof()
        await reader.read()
        writer.close()
        assert core.last_request_at() > before
    finally:
        await hook_server.stop_hook_server(server)


def test_idle_time_counts_from_the_latest_tool_call_or_hook_request(monkeypatch):
    from omega.server import mcp_server

    monkeypatch.setattr(mcp_server, "_last_activity", 100.0)
    monkeypatch.setattr(core, "last_request_at", lambda: 900.0)
    assert mcp_server._idle_seconds(1000.0) == 100.0

    monkeypatch.setattr(core, "last_request_at", lambda: 0.0)
    assert mcp_server._idle_seconds(1000.0) == 900.0


@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket lifecycle")
def test_idle_exit_removes_the_hook_socket(short_dir):
    """End to end: a real stdio server with a 2 s idle limit exits and leaves no socket behind."""
    home = short_dir / "h"
    home.mkdir()
    env = {
        **os.environ,
        "HOME": str(home),
        "OMEGA_HOME": str(home / ".omega"),
        "PYTHONPATH": str(SRC_DIR),
        "OMEGA_IDLE_TIMEOUT": "2",
        "OMEGA_SKIP_EMBEDDINGS": "1",
    }
    sock_path = home / ".omega" / "hook.sock"
    server = subprocess.Popen(
        [sys.executable, "-m", "omega.server.mcp_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
    )
    try:
        deadline = time.monotonic() + 30
        while not sock_path.exists() and server.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert sock_path.exists(), server.stderr.read().decode() if server.poll() is not None else "no socket"

        server.wait(timeout=30)

        assert not sock_path.exists(), "idle exit left a stale hook socket"
    finally:
        if server.poll() is None:
            server.kill()
        server.wait()


# ---------------------------------------------------------------------------
# D1: fast_hook gives up at once on a socket whose owner is gone
# ---------------------------------------------------------------------------


def _load_fast_hook(sock_path: Path, owner_path: Path, monkeypatch):
    script = SRC_DIR / "omega" / "hooks" / "fast_hook.py"
    spec = importlib.util.spec_from_file_location("fast_hook_lifecycle", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOCK_PATH = str(sock_path)
    module.OWNER_STATE_PATH = str(owner_path)
    sleeps: list[float] = []
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    return module, sleeps


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket lifecycle")
def test_fast_hook_does_not_retry_a_socket_whose_owner_has_exited(short_dir, monkeypatch):
    sock_path, owner_path = short_dir / "hook.sock", short_dir / "owner.json"
    _leave_stale_socket(sock_path)
    owner_path.write_text(json.dumps({"pid": _dead_pid(), "transport": "unix", "status": "ready"}))
    fast_hook, sleeps = _load_fast_hook(sock_path, owner_path, monkeypatch)

    assert fast_hook._delegate_with_retries("session_start", {}, timeout=1.0) is None
    assert sleeps == []


@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket lifecycle")
def test_fast_hook_does_not_retry_a_socket_nobody_claims(short_dir, monkeypatch):
    sock_path = short_dir / "hook.sock"
    _leave_stale_socket(sock_path)
    fast_hook, sleeps = _load_fast_hook(sock_path, short_dir / "missing-owner.json", monkeypatch)

    assert fast_hook._delegate_with_retries("session_start", {}, timeout=1.0) is None
    assert sleeps == []


@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket lifecycle")
def test_fast_hook_still_retries_while_the_owner_is_alive(short_dir, monkeypatch):
    """A live owner that refuses may be starting up or saturated: retrying is worth it there."""
    sock_path, owner_path = short_dir / "hook.sock", short_dir / "owner.json"
    _leave_stale_socket(sock_path)
    owner_path.write_text(json.dumps({"pid": os.getpid(), "transport": "unix", "status": "ready"}))
    fast_hook, sleeps = _load_fast_hook(sock_path, owner_path, monkeypatch)

    assert fast_hook._delegate_with_retries("session_start", {}, timeout=1.0) is None
    assert len(sleeps) == fast_hook._CONNECT_RETRIES


# ---------------------------------------------------------------------------
# SessionStart fires as the MCP server starts: wait for its socket to appear
# ---------------------------------------------------------------------------


def _answer_once(sock_path: Path, reply: dict) -> threading.Thread:
    """Listen at ``sock_path`` and answer one hook request with ``reply``, as the hook server would."""
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(sock_path))
    listener.listen(1)

    def serve() -> None:
        with listener:
            connection, _ = listener.accept()
            with connection:
                while connection.recv(65536):
                    pass
                connection.sendall(json.dumps(reply).encode())

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return thread


@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket lifecycle")
def test_session_start_waits_for_a_socket_that_appears_after_it_fires(short_dir, monkeypatch):
    sock_path = short_dir / "hook.sock"
    fast_hook, _ = _load_fast_hook(sock_path, short_dir / "owner.json", monkeypatch)
    sleeps: list[float] = []

    def server_starts_meanwhile(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 2:
            _answer_once(sock_path, {"output": "briefing", "error": None})

    monkeypatch.setattr(fast_hook.time, "sleep", server_starts_meanwhile)

    assert fast_hook._delegate_with_retries("session_start", {}, timeout=1.0) == {"output": "briefing", "error": None}
    assert sleeps == [fast_hook._CONNECT_RETRY_DELAY] * 2


@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket lifecycle")
@pytest.mark.parametrize("hook_names", ["session_start", ["session_start", "coord_session_start"]])
def test_session_start_stops_waiting_for_a_socket_at_the_end_of_the_window(hook_names, short_dir, monkeypatch):
    fast_hook, sleeps = _load_fast_hook(short_dir / "hook.sock", short_dir / "owner.json", monkeypatch)

    assert fast_hook._delegate_with_retries(hook_names, {}, timeout=1.0) is None
    assert sleeps == [fast_hook._CONNECT_RETRY_DELAY] * fast_hook._CONNECT_RETRIES
    # The whole wait plus the answer fits inside the hook's timeout in hooks-core.json.
    hooks_json = json.loads((SRC_DIR / "omega" / "data" / "hooks-core.json").read_text())
    assert sum(sleeps) + fast_hook._DAEMON_TIMEOUT_S < hooks_json["SessionStart"][0]["timeout"]


@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket lifecycle")
def test_other_hooks_do_not_wait_for_a_missing_socket(short_dir, monkeypatch):
    fast_hook, sleeps = _load_fast_hook(short_dir / "hook.sock", short_dir / "owner.json", monkeypatch)

    assert fast_hook._delegate_with_retries("surface_memories", {}, timeout=1.0) is None
    assert sleeps == []


@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket lifecycle")
async def test_the_briefing_arrives_when_the_hook_server_starts_after_session_start_fired(short_dir, monkeypatch):
    """End to end: the real client process starts first, the real hook server 0.7 s later."""
    omega_dir = short_dir / ".omega"
    omega_dir.mkdir()
    monkeypatch.setenv("OMEGA_HOME", str(omega_dir))
    monkeypatch.setattr(hook_server, "SOCK_PATH", omega_dir / "hook.sock")
    monkeypatch.setattr(owner_state, "OWNER_STATE_PATH", omega_dir / "hook.sock.owner.json")
    monkeypatch.setitem(core.HOOK_HANDLERS, "session_start", lambda payload: {"output": "briefing", "error": None})

    client = await asyncio.create_subprocess_exec(
        sys.executable, str(SRC_DIR / "omega" / "hooks" / "fast_hook.py"), "session_start",
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "HOME": str(short_dir)},
    )
    # Claude Code writes the payload at once; the client then looks for the socket.
    client.stdin.write(json.dumps({"hook_event_name": "SessionStart", "session_id": "s1"}).encode())
    await client.stdin.drain()
    client.stdin.close()
    await asyncio.sleep(0.7)
    assert not (omega_dir / "hook.sock").exists()
    server = await hook_server.start_hook_server()
    try:
        stdout = await asyncio.wait_for(client.stdout.read(), timeout=15)
        stderr = await client.stderr.read()
        await client.wait()
    finally:
        await hook_server.stop_hook_server(server)

    assert client.returncode == 0, stderr.decode()
    assert stdout.decode() == "briefing\n"


@pytest.mark.skipif(sys.platform == "win32", reason="Unix socket lifecycle")
def test_sigterm_removes_the_hook_socket(short_dir):
    """`claude mcp list` starts the server to probe it, then terminates it: no stale socket either."""
    home = short_dir / "t"
    home.mkdir()
    env = {**os.environ, "HOME": str(home), "OMEGA_HOME": str(home / ".omega"),
           "PYTHONPATH": str(SRC_DIR), "OMEGA_SKIP_EMBEDDINGS": "1"}
    sock_path = home / ".omega" / "hook.sock"
    server = subprocess.Popen(
        [sys.executable, "-m", "omega.server.mcp_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
    )
    try:
        deadline = time.monotonic() + 30
        while not sock_path.exists() and server.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert sock_path.exists(), "no socket"

        server.terminate()
        server.wait(timeout=15)

        assert not sock_path.exists(), "SIGTERM left a stale hook socket"
    finally:
        if server.poll() is None:
            server.kill()
        server.wait()
