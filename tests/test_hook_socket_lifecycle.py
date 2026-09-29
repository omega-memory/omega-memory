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
