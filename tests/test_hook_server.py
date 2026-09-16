"""Hook daemon tests: output capture, core handlers, and the socket protocol.

The daemon is the fast path for every hook ``omega setup`` registers. These
tests pin the contract ``fast_hook.py`` relies on (request/response shapes,
batch short-circuit, socket lifecycle) and check that each core handler runs
the shared hook module and returns its output instead of printing it.
"""
import asyncio
import importlib.util
import json
import shutil
import socket
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from omega.hooks import _output, assistant_capture, auto_capture, surface_memories
from omega.server import hook_server
from omega.server.hook_server import core, owner_state

DECISION_PROMPT = (
    "Let's go with SQLite instead of PostgreSQL for the backend database "
    "since it simplifies local development and testing significantly."
)
FIX_MESSAGE = (
    "I traced the failing upload end to end and compared both sides of the secret. "
    "The fix was to send the cleaned environment value from the frontend instead of the raw one. "
    "After that change every upload succeeded again and the 401 responses stopped completely. "
    "I also added a regression test so the mismatch cannot come back unnoticed in a later release."
)


@pytest.fixture(autouse=True)
def _isolated_daemon(_reset_bridge, tmp_path, monkeypatch):
    """Fresh bridge store, empty daemon state, and no writes under the real home directory."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(surface_memories, "_get_session_tool_names_fast", lambda session_id: [])
    hook_server._debounce_state.reset()
    yield
    hook_server._debounce_state.reset()


@pytest.fixture
def short_socket_dir():
    """AF_UNIX paths are capped at 104 bytes on macOS; pytest's tmp_path is longer."""
    directory = Path(tempfile.mkdtemp(prefix="omg", dir="/tmp"))
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.bind(str(directory / "probe.sock"))
    except OSError as error:
        pytest.skip(f"AF_UNIX sockets unavailable here: {error}")
    finally:
        probe.close()
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
async def running_daemon(short_socket_dir, monkeypatch):
    sock_path = short_socket_dir / "hook.sock"
    monkeypatch.setattr(hook_server, "SOCK_PATH", sock_path)
    monkeypatch.setattr(owner_state, "OWNER_STATE_PATH", short_socket_dir / "owner.json")
    server = await hook_server.start_hook_server()
    assert server is not None
    yield sock_path
    await hook_server.stop_hook_server(server)


def _load_fast_hook(sock_path: Path):
    """Import the real client script with its socket pointed at the test daemon."""
    script = Path(__file__).parent.parent / "src" / "omega" / "hooks" / "fast_hook.py"
    spec = importlib.util.spec_from_file_location("fast_hook_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOCK_PATH = str(sock_path)
    return module


def _stored(event_type: str) -> list:
    from omega.bridge import _get_store

    return _get_store().get_by_type(event_type, limit=20)


# ---------------------------------------------------------------------------
# Output sink
# ---------------------------------------------------------------------------


def test_emit_prints_when_nothing_is_capturing(capsys):
    _output.emit("hello")
    assert capsys.readouterr().out == "hello\n"


def test_capture_collects_lines_and_restores_previous_sink(capsys):
    with _output.capture() as lines:
        _output.emit("inside")
    _output.emit("outside")
    assert lines == ["inside"]
    assert capsys.readouterr().out == "outside\n"


def test_capture_is_isolated_per_thread():
    seen: dict[str, list[str]] = {}
    barrier = threading.Barrier(2)

    def worker(name: str) -> None:
        with _output.capture() as lines:
            barrier.wait()
            _output.emit(f"{name}-1")
            barrier.wait()
            _output.emit(f"{name}-2")
        seen[name] = lines

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == {"a": ["a-1", "a-2"], "b": ["b-1", "b-2"]}


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------


def test_dispatch_table_serves_every_registered_core_hook():
    hooks_json = json.loads((Path(__file__).parent.parent / "src" / "omega" / "data" / "hooks-core.json").read_text())
    registered = {entry["script"].split()[1] for entries in hooks_json.values() for entry in entries}
    assert registered <= set(hook_server.HOOK_HANDLERS)


def test_register_hook_handler_adds_plugin_hook():
    hook_server.register_hook_handler("plugin_probe", lambda payload: {"output": "hi", "error": None})
    try:
        assert hook_server.HOOK_HANDLERS["plugin_probe"]({})["output"] == "hi"
    finally:
        del hook_server.HOOK_HANDLERS["plugin_probe"]


# ---------------------------------------------------------------------------
# Core handlers run the shared hook modules
# ---------------------------------------------------------------------------


def test_auto_capture_stores_decision_and_returns_echo():
    response = hook_server.handle_auto_capture({"prompt": DECISION_PROMPT, "session_id": "s1", "cwd": "/proj"})

    assert response["error"] is None
    assert "[OMEGA] Captured: decision" in response["output"]
    decisions = _stored("decision")
    assert len(decisions) == 1
    assert "SQLite" in decisions[0].content


def test_auto_capture_ignores_prompts_without_signal():
    response = hook_server.handle_auto_capture({"prompt": "please run the tests again", "session_id": "s1"})

    assert response == {"output": "", "error": None}
    assert _stored("decision") == []


def test_auto_capture_cap_is_per_session():
    auto_capture._captures_by_session["full"] = auto_capture.MAX_CAPTURES_PER_SESSION

    hook_server.handle_auto_capture({"prompt": DECISION_PROMPT, "session_id": "full"})
    assert _stored("decision") == []

    hook_server.handle_auto_capture({"prompt": DECISION_PROMPT, "session_id": "fresh"})
    assert len(_stored("decision")) == 1


def test_assistant_capture_stores_fix_sentence():
    response = hook_server.handle_assistant_capture(
        {"last_assistant_message": FIX_MESSAGE, "session_id": "s1", "cwd": "/proj"}
    )

    assert response["error"] is None
    assert response["output"].startswith("[LEARNED] fix:")
    lessons = _stored("lesson_learned")
    assert len(lessons) == 1
    assert lessons[0].content.startswith("Assistant fix: The fix was")


def test_surface_memories_returns_memory_lines_for_edit():
    canned = [{
        "id": "mem-abcdef123456",
        "content": "db.py: wrap every write in a transaction",
        "relevance": 0.91,
        "event_type": "decision",
        "created_at": "2026-09-01T00:00:00+00:00",
        "session_id": "earlier",
    }]
    payload = {"tool_name": "Edit", "tool_input": json.dumps({"file_path": "/proj/db.py"}), "session_id": "s1", "project": "/proj"}

    with patch("omega.bridge.query_structured", return_value=canned):
        response = hook_server.handle_surface_memories(payload)

    assert response["error"] is None
    assert "[MEMORY] Relevant context for db.py:" in response["output"]
    assert "wrap every write in a transaction" in response["output"]


def test_surface_memories_debounces_repeated_touches_of_one_file():
    payload = {"tool_name": "Read", "tool_input": json.dumps({"file_path": "/proj/db.py"}), "session_id": "s1", "project": "/proj"}

    with patch("omega.bridge.query_structured", return_value=[]) as query:
        hook_server.handle_surface_memories(payload)
        hook_server.handle_surface_memories(payload)
        hook_server.handle_surface_memories({**payload, "tool_input": json.dumps({"file_path": "/proj/other.py"})})

    assert query.call_count == 2


def test_surface_memories_captures_bash_error():
    payload = {
        "tool_name": "Bash",
        "tool_input": json.dumps({"command": "pytest"}),
        "tool_output": (
            "Traceback (most recent call last):\n"
            '  File "worker.py", line 42, in run\n'
            "sqlite3.OperationalError: database is locked while the nightly consolidation job "
            "was rewriting the memories table from worker.py"
        ),
        "session_id": "s1",
        "project": "/proj",
    }

    response = hook_server.handle_surface_memories(payload)

    assert response["error"] is None
    assert "[OMEGA] Captured error" in response["output"]
    errors = _stored("error_pattern")
    assert len(errors) == 1
    assert "database is locked" in errors[0].content


def test_standalone_main_still_prints_for_the_fallback_path(capsys):
    auto_capture.main({"prompt": DECISION_PROMPT, "session_id": "s1", "cwd": "/proj"})

    assert capsys.readouterr().out.startswith("[OMEGA] Captured: decision")


def test_session_start_returns_welcome_briefing():
    response = hook_server.handle_session_start({"session_id": "s1", "project": "/proj"})

    assert response["error"] is None
    assert response["output"].startswith("## Welcome back! OMEGA ready")


def test_session_stop_stores_summary_and_releases_session_state():
    hook_server.handle_auto_capture({"prompt": DECISION_PROMPT, "session_id": "s1", "cwd": "/proj"})
    assistant_capture._captures_by_session["s1"] = 3
    hook_server._protocol_calls["s1"] = {"omega_welcome"}

    response = hook_server.handle_session_stop({"session_id": "s1", "project": "/proj"})

    assert response["error"] is None
    assert "## Session complete" in response["output"]
    assert len(_stored("session_summary")) == 1
    assert "s1" not in auto_capture._captures_by_session
    assert "s1" not in assistant_capture._captures_by_session
    assert "s1" not in hook_server._protocol_calls


def test_failing_hook_reports_error_but_keeps_partial_output():
    def broken(payload):
        _output.emit("partial")
        raise RuntimeError("boom")

    with patch.object(hook_server.handlers.auto_capture, "run", broken):
        response = hook_server.handle_auto_capture({"prompt": DECISION_PROMPT, "session_id": "s1"})

    assert response == {"output": "partial", "error": "boom"}


def test_mark_protocol_call_records_session_and_marker(tmp_path):
    hook_server.mark_protocol_call("s1", "omega_welcome")
    hook_server.mark_protocol_call("", "omega_welcome")

    assert hook_server._protocol_calls == {"s1": {"omega_welcome"}}
    assert (tmp_path / ".omega" / "gates" / "s1.omega_welcome").exists()


# ---------------------------------------------------------------------------
# Socket protocol, exercised through the real fast_hook client
# ---------------------------------------------------------------------------


async def test_start_and_stop_manage_socket_and_owner_state(short_socket_dir, monkeypatch):
    sock_path = short_socket_dir / "hook.sock"
    owner_path = short_socket_dir / "owner.json"
    monkeypatch.setattr(hook_server, "SOCK_PATH", sock_path)
    monkeypatch.setattr(owner_state, "OWNER_STATE_PATH", owner_path)

    server = await hook_server.start_hook_server()
    assert server is not None
    assert sock_path.exists()
    assert owner_state.read_owner_state()["status"] == "ready"

    await hook_server.stop_hook_server(server)
    assert not sock_path.exists()
    assert owner_state.read_owner_state() is None


async def test_fast_hook_delegate_round_trip(running_daemon):
    fast_hook = _load_fast_hook(running_daemon)

    response = await asyncio.to_thread(
        fast_hook.delegate, "auto_capture", {"prompt": DECISION_PROMPT, "session_id": "s1", "cwd": "/proj"}
    )

    assert response["error"] is None
    assert "[OMEGA] Captured: decision" in response["output"]
    assert len(_stored("decision")) == 1


async def test_unknown_hook_is_reported_not_raised(running_daemon):
    fast_hook = _load_fast_hook(running_daemon)

    response = await asyncio.to_thread(fast_hook.delegate, "no_such_hook", {})

    assert response == {"output": "", "error": "Unknown hook: no_such_hook"}


async def test_batch_request_returns_one_result_per_hook(running_daemon):
    fast_hook = _load_fast_hook(running_daemon)
    payload = {"prompt": "please run the tests again", "last_assistant_message": "", "session_id": "s1"}

    response = await asyncio.to_thread(fast_hook.delegate, ["assistant_capture", "auto_capture"], payload)

    assert [r["error"] for r in response["results"]] == [None, None]


async def test_batch_short_circuits_on_blocking_exit_code(running_daemon):
    fast_hook = _load_fast_hook(running_daemon)
    hook_server.register_hook_handler("test_block", lambda payload: {"output": "vetoed", "error": None, "exit_code": 2})
    try:
        response = await asyncio.to_thread(
            fast_hook.delegate, ["test_block", "auto_capture"], {"prompt": DECISION_PROMPT, "session_id": "s1"}
        )
    finally:
        del hook_server.HOOK_HANDLERS["test_block"]

    assert len(response["results"]) == 1
    assert response["results"][0]["exit_code"] == 2
    assert _stored("decision") == []


async def test_liveness_probe_is_not_logged_as_a_hook(running_daemon, tmp_path):
    reader, writer = await asyncio.open_unix_connection(str(running_daemon))
    writer.write_eof()
    await reader.read()
    writer.close()
    await writer.wait_closed()

    from omega.server.hook_server.utils import omega_home

    log_path = omega_home() / "hooks.log"
    assert not log_path.exists() or "unknown" not in log_path.read_text()


async def test_malformed_request_gets_error_response(running_daemon):
    reader, writer = await asyncio.open_unix_connection(str(running_daemon))
    writer.write(b"{not json")
    writer.write_eof()
    response = json.loads(await asyncio.wait_for(reader.read(), timeout=5.0))
    writer.close()
    await writer.wait_closed()

    assert response["output"] == ""
    assert response["error"]


# ---------------------------------------------------------------------------
# The MCP server uses this daemon, not the no-op stub
# ---------------------------------------------------------------------------


def test_mcp_server_shares_the_daemon_executor():
    from omega.server import mcp_server

    assert mcp_server._HOOK_EXECUTOR is core._HOOK_EXECUTOR
