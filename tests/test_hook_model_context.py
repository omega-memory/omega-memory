"""Surfaced memories must reach the model, not only Claude Code's debug log.

Claude Code hands a hook's plain stdout to the model only on a few events
(SessionStart and UserPromptSubmit among them). After a tool call it writes
plain stdout to its debug log; text reaches the model only when the whole of
stdout is ``{"hookSpecificOutput": {"hookEventName": ..., "additionalContext":
...}}``. Every release up to 1.5.19 printed the PostToolUse memories as plain
text, so no model ever saw them.

These tests run the real ``fast_hook.py`` as Claude Code does, a separate
process reading the hook payload on stdin, against the real hook server.
"""
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from omega.hooks import _output, surface_memories
from omega.server import hook_server
from omega.server.hook_server import handlers, owner_state

FAST_HOOK = Path(__file__).resolve().parent.parent / "src" / "omega" / "hooks" / "fast_hook.py"

MEMORY = {
    "id": "mem-abcdef123456",
    "content": "db.py: wrap every write in a transaction",
    "relevance": 0.91,
    "event_type": "decision",
    "created_at": "2026-09-01T00:00:00+00:00",
    "session_id": "earlier",
}


@pytest.fixture(autouse=True)
def _isolated_daemon(_reset_bridge, tmp_path, monkeypatch):
    """Fresh store and daemon state; nothing written under the real home directory."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(surface_memories, "_get_session_tool_names_fast", lambda session_id: [])
    hook_server._debounce_state.reset()
    yield
    hook_server._debounce_state.reset()


@pytest.fixture
def client_home():
    """A home for the hook client: its socket path must fit macOS's 104-byte AF_UNIX limit."""
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
async def daemon(client_home, monkeypatch):
    """The hook server, listening where ``fast_hook.py`` looks under ``client_home``."""
    omega_dir = client_home / ".omega"
    omega_dir.mkdir()
    monkeypatch.setattr(hook_server, "SOCK_PATH", omega_dir / "hook.sock")
    monkeypatch.setattr(owner_state, "OWNER_STATE_PATH", omega_dir / "hook.sock.owner.json")
    server = await hook_server.start_hook_server()
    assert server is not None
    yield client_home
    await hook_server.stop_hook_server(server)


async def _run_fast_hook(home: Path, hook: str, payload: dict) -> subprocess.CompletedProcess:
    """Run the client the way Claude Code does: hook name in argv, payload on stdin."""
    return await asyncio.to_thread(
        subprocess.run,
        [sys.executable, str(FAST_HOOK), hook],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "HOME": str(home)},
    )


def _edit_payload(event: str = "PostToolUse") -> dict:
    return {
        "hook_event_name": event,
        "session_id": "s1",
        "cwd": "/proj",
        "tool_name": "Edit",
        "tool_input": {"file_path": "/proj/db.py", "old_string": "a", "new_string": "b"},
        "tool_response": {"filePath": "/proj/db.py"},
    }


# ---------------------------------------------------------------------------
# End to end: real client process, real hook server
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="the test daemon listens on a Unix socket")
async def test_post_tool_use_memories_reach_claude_code_as_additional_context(daemon):
    with patch("omega.bridge.query_structured", return_value=[MEMORY]):
        run = await _run_fast_hook(daemon, "surface_memories", _edit_payload())

    assert run.returncode == 0, run.stderr
    stdout = run.stdout.strip()
    # Claude Code parses stdout as JSON only when it is all JSON.
    assert stdout.startswith("{") and stdout.endswith("}")
    answer = json.loads(stdout)
    assert list(answer) == ["hookSpecificOutput"]
    assert answer["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    context = answer["hookSpecificOutput"]["additionalContext"]
    assert context.startswith(handlers.STORED_DATA_LABEL + "\n")
    assert "[MEMORY] Relevant context for db.py:" in context
    assert "wrap every write in a transaction" in context


@pytest.mark.skipif(sys.platform == "win32", reason="the test daemon listens on a Unix socket")
async def test_nothing_to_surface_prints_nothing(daemon):
    with patch("omega.bridge.query_structured", return_value=[]):
        run = await _run_fast_hook(daemon, "surface_memories", _edit_payload())

    assert run.returncode == 0, run.stderr
    assert run.stdout == ""


@pytest.mark.skipif(sys.platform == "win32", reason="the test daemon listens on a Unix socket")
async def test_pre_tool_use_context_is_answered_under_its_own_event(daemon):
    hook_server.register_hook_handler(
        "probe_context", lambda payload: {"output": "for the transcript", "context": "for the model", "error": None}
    )
    try:
        run = await _run_fast_hook(daemon, "probe_context", _edit_payload("PreToolUse"))
    finally:
        del hook_server.HOOK_HANDLERS["probe_context"]

    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == {
        "hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": "for the model"}
    }


@pytest.mark.skipif(sys.platform == "win32", reason="the test daemon listens on a Unix socket")
async def test_a_block_still_exits_with_its_code_and_gives_its_reason_on_stderr(daemon):
    hook_server.register_hook_handler(
        "probe_block", lambda payload: {"output": "vetoed: secrets in the diff", "error": None, "exit_code": 2}
    )
    try:
        run = await _run_fast_hook(daemon, "probe_block", _edit_payload("PreToolUse"))
    finally:
        del hook_server.HOOK_HANDLERS["probe_block"]

    assert run.returncode == 2
    assert "vetoed: secrets in the diff" in run.stderr


# ---------------------------------------------------------------------------
# Where the client prints what the hooks said
# ---------------------------------------------------------------------------


def _load_fast_hook():
    import importlib.util

    spec = importlib.util.spec_from_file_location("fast_hook_model_context", FAST_HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("event", ["SessionStart", "UserPromptSubmit", "Stop", None])
def test_context_is_plain_text_outside_tool_events(event, capsys):
    """Plain stdout already reaches the model on SessionStart; on Stop additionalContext would restart the turn."""
    payload = {"hook_event_name": event} if event else {}

    _load_fast_hook()._print_answer(payload, [{"output": "report", "context": "memories"}])

    assert capsys.readouterr().out == "report\nmemories\n"


def test_contexts_of_a_batch_share_one_json_answer(capsys):
    results = [{"output": "", "context": "first"}, {"output": "log only"}, {"context": "second"}]

    _load_fast_hook()._print_answer({"hook_event_name": "PostToolUse"}, results)

    answer = json.loads(capsys.readouterr().out)
    assert answer["hookSpecificOutput"]["additionalContext"] == "first\nsecond"


def test_plain_output_on_a_tool_event_prints_as_before(capsys):
    _load_fast_hook()._print_answer({"hook_event_name": "PostToolUse"}, [{"output": "log only"}])

    assert capsys.readouterr().out == "log only\n"


# ---------------------------------------------------------------------------
# The model reads this after every file touch: keep it short
# ---------------------------------------------------------------------------


def _surface(payload: dict) -> str:
    response = hook_server.handle_surface_memories(payload)
    assert response["output"] == ""
    return response["context"]


def test_surfaced_context_is_capped_at_whole_lines():
    lines = [f"line {n:02d} " + "x" * 90 for n in range(60)]

    def chatty(payload):
        for line in lines:
            _output.emit(line)

    with patch.object(handlers.surface_memories, "run", chatty):
        context = _surface(_edit_payload())

    assert len(context) <= handlers.SURFACED_CONTEXT_MAX_CHARS
    label, *body, note = context.split("\n")
    assert label == handlers.STORED_DATA_LABEL
    assert body == lines[: len(body)]
    assert note == f"[OMEGA] {len(lines) - len(body)} more lines not shown."


def test_the_largest_surfacing_keeps_its_top_memories_within_the_cap():
    """Every part at its own cap: three memories, two linked, two exact errors, two lessons."""
    long_text = "Always run the migration inside one transaction and check the row count afterwards " * 4
    memories = [
        {**MEMORY, "id": f"mem-{n:012d}", "content": f"memory {n}: {long_text}", "event_type": "lesson_learned"}
        for n in range(3)
    ]
    store = SimpleNamespace(
        get_related_chain=lambda *args, **kwargs: [
            {"node_id": f"mem-linked-{n}", "content": f"linked {n}: {long_text}", "event_type": "lesson_learned"}
            for n in range(2)
        ],
        phrase_search=lambda *args, **kwargs: [
            {"node_id": f"mem-exact-{n}", "content": f"exact {n}: {long_text}"} for n in range(2)
        ],
    )
    lessons = [{"content": f"lesson {n}: {long_text}", "verified": True} for n in range(2)]

    with (
        patch("omega.bridge.query_structured", return_value=memories),
        patch("omega.bridge._get_store", return_value=store),
        patch("omega.bridge.get_cross_session_lessons", return_value=lessons),
    ):
        context = _surface(_edit_payload())

    assert len(context) <= handlers.SURFACED_CONTEXT_MAX_CHARS
    for n in range(3):
        assert f"memory {n}:" in context
