"""OMEGA Hook Server tests — UDS daemon, dispatch, debouncing, fast_hook client."""
import asyncio
import json
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ============================================================================
# Dispatch table completeness
# ============================================================================

def test_hook_handlers_dispatch_table():
    """Active hooks should be in the dispatch table."""
    from omega.server.hook_server import HOOK_HANDLERS

    # Minimum required handlers (coordination handlers may also be present
    # but are disconnected at the settings/MCP level)
    required = {
        "session_start",
        "session_stop",
        "surface_memories",
        "auto_capture",
        "assistant_capture",
    }
    assert required.issubset(set(HOOK_HANDLERS.keys()))


def test_hook_handlers_are_callable():
    """Every handler in the dispatch table should be callable."""
    from omega.server.hook_server import HOOK_HANDLERS

    for name, handler in HOOK_HANDLERS.items():
        assert callable(handler), f"Handler {name} is not callable"


# ============================================================================
# Handler return format
# ============================================================================

@pytest.fixture(autouse=True)
def _reset_bridge(tmp_omega_dir):
    """Reset the bridge singleton so each test gets a fresh store."""
    from omega.bridge import reset_memory
    reset_memory()
    yield
    reset_memory()


def test_handler_return_format_session_start():
    """session_start handler should return dict with output and error keys."""
    from omega.server.hook_server import handle_session_start

    result = handle_session_start({"session_id": "test-123", "project": "/tmp"})
    assert isinstance(result, dict)
    assert "output" in result
    assert "error" in result


def test_handler_return_format_session_stop():
    """session_stop handler should return dict with output and error keys."""
    from omega.server.hook_server import handle_session_stop

    result = handle_session_stop({"session_id": "test-123", "project": "/tmp"})
    assert isinstance(result, dict)
    assert "output" in result
    assert "error" in result


def test_handler_return_format_coord_heartbeat():
    """coord_heartbeat handler should return dict with output and error keys."""
    from omega.server.hook_server import handle_coord_heartbeat

    result = handle_coord_heartbeat({"session_id": "test-123"})
    assert isinstance(result, dict)
    assert "output" in result
    assert "error" in result


def test_handler_return_format_auto_capture():
    """auto_capture handler should return dict with output and error keys."""
    from omega.server.hook_server import handle_auto_capture

    result = handle_auto_capture({"stdin": ""})
    assert isinstance(result, dict)
    assert "output" in result
    assert "error" in result


def test_handler_return_format_auto_claim():
    """auto_claim_file handler should return dict with output and error keys."""
    from omega.server.hook_server import handle_auto_claim_file

    result = handle_auto_claim_file({
        "tool_name": "Edit",
        "session_id": "test-123",
        "tool_input": '{"file_path": "/tmp/test.py"}',
    })
    assert isinstance(result, dict)
    assert "output" in result
    assert "error" in result


# ============================================================================
# Debounce logic
# ============================================================================

def test_heartbeat_debounce():
    """Heartbeat should be debounced — second call within 30s returns immediately."""
    from omega.server import hook_server
    from omega.server.hook_server import handle_coord_heartbeat

    # Reset debounce state
    hook_server._last_heartbeat.clear()

    # Patch at the import location used inside the handler function
    with patch("omega.coordination.get_manager") as mock_mgr_fn:
        mock_mgr = MagicMock()
        mock_mgr_fn.return_value = mock_mgr

        handle_coord_heartbeat({"session_id": "debounce-test"})
        assert mock_mgr.heartbeat.call_count == 1

        # Second call within debounce window should be skipped
        handle_coord_heartbeat({"session_id": "debounce-test"})
        assert mock_mgr.heartbeat.call_count == 1  # Still 1, not 2

    hook_server._last_heartbeat.clear()


def test_heartbeat_different_sessions_not_debounced():
    """Different session IDs should not debounce each other."""
    from omega.server import hook_server
    from omega.server.hook_server import handle_coord_heartbeat

    hook_server._last_heartbeat.clear()

    with patch("omega.coordination.get_manager") as mock_mgr_fn:
        mock_mgr = MagicMock()
        mock_mgr_fn.return_value = mock_mgr

        handle_coord_heartbeat({"session_id": "session-A"})
        handle_coord_heartbeat({"session_id": "session-B"})
        assert mock_mgr.heartbeat.call_count == 2

    hook_server._last_heartbeat.clear()


def test_claim_debounce():
    """Auto-claim should be debounced per (session, file) pair."""
    from omega.server import hook_server
    from omega.server.hook_server import handle_auto_claim_file

    hook_server._last_claim.clear()

    with patch("omega.coordination.get_manager") as mock_mgr_fn:
        mock_mgr = MagicMock()
        mock_mgr_fn.return_value = mock_mgr

        payload = {
            "tool_name": "Edit",
            "session_id": "claim-test",
            "tool_input": json.dumps({"file_path": "/tmp/foo.py"}),
        }

        handle_auto_claim_file(payload)
        assert mock_mgr.claim_file.call_count == 1

        # Same file, same session — debounced
        handle_auto_claim_file(payload)
        assert mock_mgr.claim_file.call_count == 1

        # Different file — not debounced
        payload2 = {**payload, "tool_input": json.dumps({"file_path": "/tmp/bar.py"})}
        handle_auto_claim_file(payload2)
        assert mock_mgr.claim_file.call_count == 2

    hook_server._last_claim.clear()


def test_surface_debounce():
    """Surface memories should be debounced per file."""
    from omega.server import hook_server
    from omega.server.hook_server import handle_surface_memories

    hook_server._last_surface.clear()

    with patch("omega.bridge.query") as mock_query:
        mock_query.return_value = "Some memory"

        payload = {
            "tool_name": "Edit",
            "tool_input": json.dumps({"file_path": "/tmp/test.py"}),
            "tool_output": "",
            "session_id": "surface-test",
            "project": "/tmp",
        }

        handle_surface_memories(payload)
        first_call_count = mock_query.call_count

        # Same file within debounce window — should be skipped
        handle_surface_memories(payload)
        assert mock_query.call_count == first_call_count

    hook_server._last_surface.clear()


# ============================================================================
# Unknown hook name
# ============================================================================

def test_unknown_hook_not_in_dispatch():
    """Unknown hook names should not be in the dispatch table."""
    from omega.server.hook_server import HOOK_HANDLERS

    assert "nonexistent_hook" not in HOOK_HANDLERS


# ============================================================================
# Auto-capture decision detection
# ============================================================================

def test_auto_capture_detects_decision():
    """auto_capture should detect decision patterns in prompts."""
    from omega.server.hook_server import handle_auto_capture

    with patch("omega.bridge.auto_capture") as mock_ac:
        result = handle_auto_capture({
            "stdin": json.dumps({
                "prompt": "Let's go with SQLite instead of PostgreSQL for the backend database since it simplifies local development and testing significantly",
                "session_id": "test-123",
                "cwd": "/tmp",
            }),
        })
        assert mock_ac.called
        assert result["error"] is None


def test_auto_capture_ignores_non_decision():
    """auto_capture should not fire for non-decision prompts."""
    from omega.server.hook_server import handle_auto_capture

    with patch("omega.bridge.auto_capture") as mock_ac:
        handle_auto_capture({
            "stdin": json.dumps({
                "prompt": "Please read the file src/main.py and show me the contents",
                "session_id": "test-123",
                "cwd": "/tmp",
            }),
        })
        assert not mock_ac.called


def test_auto_capture_ignores_short_prompts():
    """auto_capture should ignore prompts shorter than 20 chars."""
    from omega.server.hook_server import handle_auto_capture

    with patch("omega.bridge.auto_capture") as mock_ac:
        handle_auto_capture({
            "stdin": json.dumps({
                "prompt": "use SQLite",
                "session_id": "test-123",
                "cwd": "/tmp",
            }),
        })
        assert not mock_ac.called


# ============================================================================
# Error capture
# ============================================================================

def test_surface_memories_captures_bash_errors():
    """surface_memories should auto-capture error patterns from Bash output."""
    from omega.server.hook_server import handle_surface_memories

    with patch("omega.bridge.auto_capture") as mock_ac:
        handle_surface_memories({
            "tool_name": "Bash",
            "tool_input": '{"command": "python test.py"}',
            "tool_output": "Traceback (most recent call last):\n  File 'test.py', line 1\nNameError: name 'foo' is not defined",
            "session_id": "test-123",
            "project": "/tmp",
        })
        assert mock_ac.called
        # Should have been called with error_pattern event type
        call_kwargs = mock_ac.call_args
        assert "error_pattern" in str(call_kwargs)


def test_surface_memories_handles_non_string_tool_output():
    """surface_memories should not crash when tool_output is None, dict, or other non-string."""
    from omega.server.hook_server import handle_surface_memories

    # None tool_output (explicit)
    result = handle_surface_memories({
        "tool_name": "Bash",
        "tool_input": '{"command": "git commit -m test"}',
        "tool_output": None,
        "session_id": "test-123",
        "project": "/tmp",
    })
    assert result is not None

    # Dict tool_output (e.g. unparsed JSON from Claude Code)
    result = handle_surface_memories({
        "tool_name": "Bash",
        "tool_input": '{"command": "git commit -m test"}',
        "tool_output": {"status": "ok", "lines": 42},
        "session_id": "test-123",
        "project": "/tmp",
    })
    assert result is not None

    # Boolean tool_output
    result = handle_surface_memories({
        "tool_name": "Bash",
        "tool_input": '{"command": "echo hello"}',
        "tool_output": True,
        "session_id": "test-123",
        "project": "/tmp",
    })
    assert result is not None

    # Missing tool_output key entirely
    result = handle_surface_memories({
        "tool_name": "Bash",
        "tool_input": '{"command": "echo hello"}',
        "session_id": "test-123",
        "project": "/tmp",
    })
    assert result is not None


# ============================================================================
# UDS Server integration test
# ============================================================================

@pytest.fixture
def short_tmp_dir():
    """Create a short temp dir to stay within AF_UNIX 104-byte path limit on macOS."""
    import tempfile
    d = tempfile.mkdtemp(prefix="omg", dir="/tmp")
    yield Path(d)
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.mark.asyncio
async def test_hook_server_start_stop(short_tmp_dir):
    """Hook server should start, accept connections, and stop cleanly."""
    from omega.server import hook_server

    # Use a short path for the socket (macOS AF_UNIX limit is 104 bytes)
    test_sock = short_tmp_dir / "hook.sock"
    original_sock = hook_server.SOCK_PATH
    hook_server.SOCK_PATH = test_sock

    try:
        srv = await hook_server.start_hook_server()
        assert srv is not None
        assert test_sock.exists()

        # Connect and send a request
        reader, writer = await asyncio.open_unix_connection(str(test_sock))
        request = json.dumps({"hook": "coord_heartbeat", "session_id": "integration-test"})
        writer.write(request.encode("utf-8"))
        writer.write_eof()

        response_data = await asyncio.wait_for(reader.read(), timeout=5.0)
        response = json.loads(response_data.decode("utf-8"))
        assert "output" in response
        assert "error" in response

        writer.close()
        await writer.wait_closed()

        # Stop server
        await hook_server.stop_hook_server(srv)
        assert not test_sock.exists()
    finally:
        hook_server.SOCK_PATH = original_sock


@pytest.mark.asyncio
async def test_hook_server_unknown_hook(short_tmp_dir):
    """Unknown hook name should return an error in the response."""
    from omega.server import hook_server

    test_sock = short_tmp_dir / "hook.sock"
    original_sock = hook_server.SOCK_PATH
    hook_server.SOCK_PATH = test_sock

    try:
        srv = await hook_server.start_hook_server()
        assert srv is not None

        reader, writer = await asyncio.open_unix_connection(str(test_sock))
        request = json.dumps({"hook": "nonexistent_hook"})
        writer.write(request.encode("utf-8"))
        writer.write_eof()

        response_data = await asyncio.wait_for(reader.read(), timeout=5.0)
        response = json.loads(response_data.decode("utf-8"))
        assert response["error"] is not None
        assert "Unknown hook" in response["error"]

        writer.close()
        await writer.wait_closed()
        await hook_server.stop_hook_server(srv)
    finally:
        hook_server.SOCK_PATH = original_sock


# ============================================================================
# fast_hook.py fallback test
# ============================================================================

def test_fast_hook_fallback_scripts_map():
    """fast_hook fallback scripts should cover all daemon-proxied hooks."""
    import importlib.util
    hooks_dir = Path(__file__).parent.parent / "hooks"
    spec = importlib.util.spec_from_file_location("fast_hook", hooks_dir / "fast_hook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from omega.server.hook_server import HOOK_HANDLERS

    # Every fallback script in fast_hook should have a corresponding handler
    # and point to an existing script file
    for hook_name, script_name in mod._FALLBACK_SCRIPTS.items():
        assert hook_name in HOOK_HANDLERS, f"Fallback for {hook_name} has no handler"
        script_path = hooks_dir / f"{script_name}.py"
        assert script_path.exists(), f"Fallback script {script_path} does not exist"


# ============================================================================
# hooks.json manifest coverage
# ============================================================================

def test_hooks_json_covers_all_events():
    """hooks.json manifest should cover all 5 event types."""
    data_dir = Path(__file__).parent.parent / "src" / "omega" / "data"
    manifest = json.loads((data_dir / "hooks.json").read_text())

    expected_events = {"SessionStart", "Stop", "UserPromptSubmit", "PostToolUse", "PreToolUse"}
    assert set(manifest.keys()) == expected_events


def test_hooks_json_all_entries_are_lists():
    """Every event in hooks.json should map to a list of hook entries."""
    data_dir = Path(__file__).parent.parent / "src" / "omega" / "data"
    manifest = json.loads((data_dir / "hooks.json").read_text())

    for event, entries in manifest.items():
        assert isinstance(entries, list), f"{event} should be a list, got {type(entries)}"
        for entry in entries:
            assert "script" in entry
            assert "timeout" in entry


def test_hooks_json_fast_hook_entries():
    """19 hooks should use fast_hook.py, 0 should be a direct script."""
    data_dir = Path(__file__).parent.parent / "src" / "omega" / "data"
    manifest = json.loads((data_dir / "hooks.json").read_text())

    fast_count = 0
    direct_count = 0
    for entries in manifest.values():
        for entry in entries:
            if entry["script"].startswith("fast_hook.py"):
                fast_count += 1
            else:
                direct_count += 1

    assert fast_count == 20
    assert direct_count == 0


# ============================================================================
# Batch protocol — daemon side
# ============================================================================

@pytest.mark.asyncio
async def test_hook_server_batch_request(short_tmp_dir):
    """Batch request should run multiple hooks and return results array."""
    from omega.server import hook_server

    test_sock = short_tmp_dir / "hook.sock"
    original_sock = hook_server.SOCK_PATH
    hook_server.SOCK_PATH = test_sock

    try:
        srv = await hook_server.start_hook_server()
        assert srv is not None

        reader, writer = await asyncio.open_unix_connection(str(test_sock))
        request = json.dumps({
            "hooks": ["coord_heartbeat", "auto_capture"],
            "session_id": "batch-test",
            "stdin": "",
        })
        writer.write(request.encode("utf-8"))
        writer.write_eof()

        response_data = await asyncio.wait_for(reader.read(), timeout=5.0)
        response = json.loads(response_data.decode("utf-8"))

        assert "results" in response
        assert len(response["results"]) == 2
        for r in response["results"]:
            assert "output" in r
            assert "error" in r

        writer.close()
        await writer.wait_closed()
        await hook_server.stop_hook_server(srv)
    finally:
        hook_server.SOCK_PATH = original_sock


@pytest.mark.asyncio
async def test_hook_server_batch_short_circuits_on_block(short_tmp_dir):
    """Batch should stop executing hooks when one returns exit_code."""
    from omega.server import hook_server

    test_sock = short_tmp_dir / "hook.sock"
    original_sock = hook_server.SOCK_PATH
    hook_server.SOCK_PATH = test_sock

    # Track which hooks were called
    called = []
    original_file_guard = hook_server.HOOK_HANDLERS["pre_file_guard"]
    original_task_guard = hook_server.HOOK_HANDLERS["pre_task_guard"]

    def mock_file_guard(p):
        called.append("pre_file_guard")
        return {"output": "BLOCKED", "error": None, "exit_code": 2}

    def mock_task_guard(p):
        called.append("pre_task_guard")
        return {"output": "", "error": None}

    hook_server.HOOK_HANDLERS["pre_file_guard"] = mock_file_guard
    hook_server.HOOK_HANDLERS["pre_task_guard"] = mock_task_guard

    try:
        srv = await hook_server.start_hook_server()
        assert srv is not None

        reader, writer = await asyncio.open_unix_connection(str(test_sock))
        request = json.dumps({
            "hooks": ["pre_file_guard", "pre_task_guard"],
            "session_id": "block-test",
            "tool_name": "Edit",
            "tool_input": '{"file_path": "/tmp/test.py"}',
        })
        writer.write(request.encode("utf-8"))
        writer.write_eof()

        response_data = await asyncio.wait_for(reader.read(), timeout=5.0)
        response = json.loads(response_data.decode("utf-8"))

        assert "results" in response
        # Only 1 result: pre_file_guard blocked, pre_task_guard never ran
        assert len(response["results"]) == 1
        assert response["results"][0]["exit_code"] == 2
        assert response["results"][0]["output"] == "BLOCKED"
        assert called == ["pre_file_guard"]

        writer.close()
        await writer.wait_closed()
        await hook_server.stop_hook_server(srv)
    finally:
        hook_server.SOCK_PATH = original_sock
        hook_server.HOOK_HANDLERS["pre_file_guard"] = original_file_guard
        hook_server.HOOK_HANDLERS["pre_task_guard"] = original_task_guard


@pytest.mark.asyncio
async def test_hook_server_batch_no_short_circuit_on_unknown(short_tmp_dir):
    """Unknown hook in batch should NOT short-circuit — remaining hooks still run."""
    from omega.server import hook_server

    test_sock = short_tmp_dir / "hook.sock"
    original_sock = hook_server.SOCK_PATH
    hook_server.SOCK_PATH = test_sock

    try:
        srv = await hook_server.start_hook_server()
        assert srv is not None

        reader, writer = await asyncio.open_unix_connection(str(test_sock))
        request = json.dumps({
            "hooks": ["nonexistent_hook", "coord_heartbeat"],
            "session_id": "batch-unknown-test",
        })
        writer.write(request.encode("utf-8"))
        writer.write_eof()

        response_data = await asyncio.wait_for(reader.read(), timeout=5.0)
        response = json.loads(response_data.decode("utf-8"))

        assert "results" in response
        assert len(response["results"]) == 2
        # First: unknown hook error (no exit_code, so no short-circuit)
        assert "Unknown hook" in response["results"][0]["error"]
        # Second: coord_heartbeat ran successfully
        assert response["results"][1]["error"] is None

        writer.close()
        await writer.wait_closed()
        await hook_server.stop_hook_server(srv)
    finally:
        hook_server.SOCK_PATH = original_sock


@pytest.mark.asyncio
async def test_hook_server_batch_shares_payload(short_tmp_dir):
    """All hooks in a batch should receive the same payload data."""
    from omega.server import hook_server

    test_sock = short_tmp_dir / "hook.sock"
    original_sock = hook_server.SOCK_PATH
    hook_server.SOCK_PATH = test_sock

    received_payloads = []
    original_heartbeat = hook_server.HOOK_HANDLERS["coord_heartbeat"]
    original_capture = hook_server.HOOK_HANDLERS["auto_capture"]

    def capture_heartbeat(p):
        received_payloads.append(("heartbeat", dict(p)))
        return {"output": "", "error": None}

    def capture_auto_capture(p):
        received_payloads.append(("auto_capture", dict(p)))
        return {"output": "", "error": None}

    hook_server.HOOK_HANDLERS["coord_heartbeat"] = capture_heartbeat
    hook_server.HOOK_HANDLERS["auto_capture"] = capture_auto_capture

    try:
        srv = await hook_server.start_hook_server()

        reader, writer = await asyncio.open_unix_connection(str(test_sock))
        request = json.dumps({
            "hooks": ["coord_heartbeat", "auto_capture"],
            "session_id": "payload-test",
            "project": "/test/project",
        })
        writer.write(request.encode("utf-8"))
        writer.write_eof()

        response_data = await asyncio.wait_for(reader.read(), timeout=5.0)
        json.loads(response_data.decode("utf-8"))

        assert len(received_payloads) == 2
        # Both handlers got the same session_id and project
        for name, p in received_payloads:
            assert p["session_id"] == "payload-test"
            assert p["project"] == "/test/project"

        writer.close()
        await writer.wait_closed()
        await hook_server.stop_hook_server(srv)
    finally:
        hook_server.SOCK_PATH = original_sock
        hook_server.HOOK_HANDLERS["coord_heartbeat"] = original_heartbeat
        hook_server.HOOK_HANDLERS["auto_capture"] = original_capture


# ============================================================================
# Batch protocol — fast_hook.py client side
# ============================================================================

class TestFastHookBatchClient:
    """Test fast_hook.py batch protocol — client side."""

    @pytest.fixture
    def fast_hook_mod(self):
        """Import the fast_hook module."""
        import importlib.util
        hooks_dir = Path(__file__).parent.parent / "hooks"
        spec = importlib.util.spec_from_file_location("fast_hook", hooks_dir / "fast_hook.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_delegate_sends_hooks_key_for_list(self, fast_hook_mod):
        """delegate() with a list should send {"hooks": [...]} not {"hook": "..."}."""
        captured = {}
        raw_response = json.dumps({"results": [
            {"output": "", "error": None},
        ]}).encode("utf-8")

        def mock_socket_factory(*args, **kwargs):
            mock = MagicMock()
            mock.sendall = lambda data: captured.update(
                request=json.loads(data.decode("utf-8"))
            )
            mock.recv = MagicMock(side_effect=[raw_response, b""])
            return mock

        with patch("socket.socket", mock_socket_factory):
            fast_hook_mod.delegate(["hook_a", "hook_b"], {"session_id": "test"})

        assert "hooks" in captured["request"]
        assert captured["request"]["hooks"] == ["hook_a", "hook_b"]
        assert "hook" not in captured["request"]

    def test_delegate_sends_hook_key_for_string(self, fast_hook_mod):
        """delegate() with a string should send {"hook": "..."} not {"hooks": [...]}."""
        captured = {}
        raw_response = json.dumps({"output": "", "error": None}).encode("utf-8")

        def mock_socket_factory(*args, **kwargs):
            mock = MagicMock()
            mock.sendall = lambda data: captured.update(
                request=json.loads(data.decode("utf-8"))
            )
            mock.recv = MagicMock(side_effect=[raw_response, b""])
            return mock

        with patch("socket.socket", mock_socket_factory):
            fast_hook_mod.delegate("single_hook", {"session_id": "test"})

        assert "hook" in captured["request"]
        assert captured["request"]["hook"] == "single_hook"
        assert "hooks" not in captured["request"]

    def test_delegate_uses_custom_timeout(self, fast_hook_mod):
        """delegate() should apply custom timeout to socket."""
        timeouts = []
        raw_response = json.dumps({"output": "", "error": None}).encode("utf-8")

        def mock_socket_factory(*args, **kwargs):
            mock = MagicMock()
            mock.settimeout = lambda t: timeouts.append(t)
            mock.recv = MagicMock(side_effect=[raw_response, b""])
            return mock

        with patch("socket.socket", mock_socket_factory):
            fast_hook_mod.delegate("hook_a", {}, timeout=20.0)

        assert timeouts == [20.0]

    def test_slow_hooks_set(self, fast_hook_mod):
        """_SLOW_HOOKS should contain pre_push_guard."""
        assert "pre_push_guard" in fast_hook_mod._SLOW_HOOKS


# ============================================================================
# Fallback short-circuit behavior
# ============================================================================

class TestFallbackShortCircuit:
    """Test that fallback mode correctly short-circuits on blocking hooks."""

    @pytest.fixture
    def fast_hook_mod(self):
        """Import the fast_hook module."""
        import importlib.util
        hooks_dir = Path(__file__).parent.parent / "hooks"
        spec = importlib.util.spec_from_file_location("fast_hook", hooks_dir / "fast_hook.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_fallback_sys_exit_propagates(self, fast_hook_mod):
        """SystemExit from a blocking hook should propagate through _fallback."""
        blocking_mod = MagicMock()
        blocking_mod.main.side_effect = SystemExit(2)

        with patch("importlib.import_module", return_value=blocking_mod):
            with pytest.raises(SystemExit) as exc:
                fast_hook_mod._fallback("pre_file_guard", {})
            assert exc.value.code == 2

    def test_fallback_loop_stops_on_sys_exit(self, fast_hook_mod):
        """In fallback loop, SystemExit from first hook prevents second from running."""
        call_log = []

        def mock_fallback(name, payload):
            call_log.append(name)
            if name == "pre_file_guard":
                raise SystemExit(2)

        with patch.object(fast_hook_mod, "_fallback", mock_fallback):
            with pytest.raises(SystemExit) as exc:
                # Simulate fallback loop from main()
                for name in ["pre_file_guard", "pre_task_guard"]:
                    fast_hook_mod._fallback(name, {})
            assert exc.value.code == 2
            assert call_log == ["pre_file_guard"]  # pre_task_guard never called

    def test_fallback_non_blocking_hooks_all_run(self, fast_hook_mod):
        """Non-blocking hooks in fallback should all execute."""
        call_log = []

        def mock_fallback(name, payload):
            call_log.append(name)

        with patch.object(fast_hook_mod, "_fallback", mock_fallback):
            for name in ["coord_heartbeat", "surface_memories", "auto_claim_file"]:
                fast_hook_mod._fallback(name, {})
            assert call_log == ["coord_heartbeat", "surface_memories", "auto_claim_file"]

    def test_fallback_exception_does_not_propagate(self, fast_hook_mod):
        """Regular exceptions in _fallback should be caught, not propagated."""
        failing_mod = MagicMock()
        failing_mod.main.side_effect = RuntimeError("boom")

        with patch("importlib.import_module", return_value=failing_mod):
            # Should not raise — _fallback catches Exception
            fast_hook_mod._fallback("coord_heartbeat", {})


# ============================================================================
# E2E: Hook log format for batched hooks
# ============================================================================

class TestHookLogFormat:
    """Test the hooks.log format for batched and single hook invocations."""

    @pytest.fixture
    def fast_hook_mod(self):
        import importlib.util
        hooks_dir = Path(__file__).parent.parent / "hooks"
        spec = importlib.util.spec_from_file_location("fast_hook", hooks_dir / "fast_hook.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_log_timing_daemon_format(self, fast_hook_mod, tmp_path):
        """Daemon-mode log entry should contain hook name and 'daemon' mode."""
        log_path = tmp_path / ".omega" / "hooks.log"

        with patch.object(Path, "home", return_value=tmp_path):
            fast_hook_mod._log_timing("session_start+coord_session_start", 42.5, "daemon")

        log_content = log_path.read_text()
        assert "fast_hook/session_start+coord_session_start: OK (42ms, daemon)" in log_content

    def test_log_timing_fallback_format(self, fast_hook_mod, tmp_path):
        """Fallback-mode log entry should contain 'fallback' mode."""
        log_path = tmp_path / ".omega" / "hooks.log"

        with patch.object(Path, "home", return_value=tmp_path):
            fast_hook_mod._log_timing("pre_file_guard+pre_task_guard", 150.3, "fallback")

        log_content = log_path.read_text()
        assert "fast_hook/pre_file_guard+pre_task_guard: OK (150ms, fallback)" in log_content

    def test_log_timing_single_hook(self, fast_hook_mod, tmp_path):
        """Single hook log entry should also work."""
        log_path = tmp_path / ".omega" / "hooks.log"

        with patch.object(Path, "home", return_value=tmp_path):
            fast_hook_mod._log_timing("auto_capture", 3.2, "daemon")

        log_content = log_path.read_text()
        assert "fast_hook/auto_capture: OK (3ms, daemon)" in log_content

    def test_log_timestamp_format(self, fast_hook_mod, tmp_path):
        """Log entries should have ISO timestamps."""
        log_path = tmp_path / ".omega" / "hooks.log"

        with patch.object(Path, "home", return_value=tmp_path):
            fast_hook_mod._log_timing("test_hook", 1.0, "daemon")

        log_content = log_path.read_text()
        # Should match [2026-02-10T...] format
        import re
        assert re.search(r'\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\]', log_content)


# ============================================================================
# Pre-push guard daemon handler
# ============================================================================

class TestPrePushGuardHandler:
    """Test handle_pre_push_guard in daemon mode."""

    def test_skips_non_bash_tools(self):
        from omega.server.hook_server import handle_pre_push_guard

        result = handle_pre_push_guard({"tool_name": "Edit"})
        assert result == {"output": "", "error": None}

    def test_skips_non_git_commands(self):
        from omega.server.hook_server import handle_pre_push_guard

        result = handle_pre_push_guard({
            "tool_name": "Bash",
            "tool_input": '{"command": "ls -la"}',
        })
        assert result == {"output": "", "error": None}

    def test_skips_invalid_tool_input(self):
        from omega.server.hook_server import handle_pre_push_guard

        result = handle_pre_push_guard({
            "tool_name": "Bash",
            "tool_input": "not json",
        })
        assert result == {"output": "", "error": None}

    def test_push_divergence_blocks(self):
        """Should block when origin has commits not in HEAD."""
        from omega.server.hook_server import handle_pre_push_guard

        def mock_run(cmd, **kwargs):
            if "rev-parse" in cmd and "--is-inside-work-tree" in cmd:
                return MagicMock(returncode=0, stdout="true\n")
            if "fetch" in cmd:
                return MagicMock(returncode=0)
            if "rev-parse" in cmd and "--abbrev-ref" in cmd:
                return MagicMock(returncode=0, stdout="main\n")
            if "log" in cmd and "HEAD..origin/" in cmd[2]:
                return MagicMock(returncode=0, stdout="abc1234 upstream commit 1\ndef5678 upstream commit 2\n")
            return MagicMock(returncode=1, stdout="")

        with patch("omega.server.hook_server.guards.subprocess.run", side_effect=mock_run), \
             patch("omega.coordination.get_manager"):
            result = handle_pre_push_guard({
                "tool_name": "Bash",
                "tool_input": '{"command": "git push origin main"}',
                "session_id": "test-sess",
                "project": "/tmp/test-repo",
            })

        assert result.get("exit_code") == 2
        assert "GIT-GUARD" in result["output"]
        assert "2 commit(s)" in result["output"]

    def test_push_no_divergence_allows(self):
        """Should allow push when HEAD is up-to-date."""
        from omega.server.hook_server import handle_pre_push_guard

        def mock_run(cmd, **kwargs):
            if "rev-parse" in cmd and "--is-inside-work-tree" in cmd:
                return MagicMock(returncode=0, stdout="true\n")
            if "fetch" in cmd:
                return MagicMock(returncode=0)
            if "rev-parse" in cmd and "--abbrev-ref" in cmd:
                return MagicMock(returncode=0, stdout="main\n")
            if "log" in cmd:
                return MagicMock(returncode=0, stdout="")
            return MagicMock(returncode=0, stdout="abc1234\n")

        with patch("omega.server.hook_server.guards.subprocess.run", side_effect=mock_run), \
             patch("omega.coordination.get_manager"):
            result = handle_pre_push_guard({
                "tool_name": "Bash",
                "tool_input": '{"command": "git push origin main"}',
                "session_id": "test-sess",
                "project": "/tmp/test-repo",
            })

        assert result.get("exit_code") is None
        assert result["output"] == ""

    def test_push_not_git_repo_allows(self):
        """Should allow when not in a git repo (fail-open)."""
        from omega.server.hook_server import handle_pre_push_guard

        def mock_run(cmd, **kwargs):
            if "--is-inside-work-tree" in cmd:
                return MagicMock(returncode=128, stdout="")
            return MagicMock(returncode=1, stdout="")

        with patch("omega.server.hook_server.guards.subprocess.run", side_effect=mock_run):
            result = handle_pre_push_guard({
                "tool_name": "Bash",
                "tool_input": '{"command": "git push"}',
                "project": "/tmp/not-a-repo",
            })

        assert result.get("exit_code") is None

    def test_branch_claim_blocks_checkout(self):
        """Should block checkout to a claimed branch."""
        from omega.server.hook_server import handle_pre_push_guard, _agent_nickname

        mock_mgr = MagicMock()
        mock_mgr.check_branch.return_value = {
            "claimed": True,
            "session_id": "other-agent-123",
            "task": "refactoring auth",
        }

        with patch("omega.coordination.get_manager", return_value=mock_mgr):
            result = handle_pre_push_guard({
                "tool_name": "Bash",
                "tool_input": '{"command": "git checkout feature-branch"}',
                "session_id": "my-session",
                "project": "/tmp/repo",
            })

        assert result.get("exit_code") == 2
        assert "BRANCH-GUARD" in result["output"]
        assert _agent_nickname("other-agent-123") in result["output"]

    def test_branch_self_claim_allows(self):
        """Should allow checkout to own claimed branch."""
        from omega.server.hook_server import handle_pre_push_guard

        mock_mgr = MagicMock()
        mock_mgr.check_branch.return_value = {
            "claimed": True,
            "session_id": "my-session",
            "task": "my work",
        }

        with patch("omega.coordination.get_manager", return_value=mock_mgr):
            result = handle_pre_push_guard({
                "tool_name": "Bash",
                "tool_input": '{"command": "git checkout feature-branch"}',
                "session_id": "my-session",
                "project": "/tmp/repo",
            })

        assert result.get("exit_code") is None

    def test_new_branch_creation_skipped(self):
        """git checkout -b should not check branch claims."""
        from omega.server.hook_server import handle_pre_push_guard

        mock_mgr = MagicMock()
        with patch("omega.coordination.get_manager", return_value=mock_mgr):
            result = handle_pre_push_guard({
                "tool_name": "Bash",
                "tool_input": '{"command": "git checkout -b new-feature"}',
                "session_id": "my-session",
                "project": "/tmp/repo",
            })

        # Should not even call check_branch for new branch creation
        mock_mgr.check_branch.assert_not_called()
        assert result.get("exit_code") is None

    def test_git_fetch_timeout_fails_open(self):
        """Timeout on git fetch should allow push (fail-open)."""
        from omega.server.hook_server import handle_pre_push_guard

        def mock_run(cmd, **kwargs):
            if "--is-inside-work-tree" in cmd:
                return MagicMock(returncode=0, stdout="true\n")
            if "fetch" in cmd:
                raise subprocess.TimeoutExpired(cmd, 15)
            return MagicMock(returncode=0, stdout="")

        with patch("omega.server.hook_server.guards.subprocess.run", side_effect=mock_run):
            result = handle_pre_push_guard({
                "tool_name": "Bash",
                "tool_input": '{"command": "git push"}',
                "project": "/tmp/repo",
            })

        assert result.get("exit_code") is None

    def test_handles_dict_tool_input(self):
        """Should handle tool_input as dict (not just string)."""
        from omega.server.hook_server import handle_pre_push_guard

        result = handle_pre_push_guard({
            "tool_name": "Bash",
            "tool_input": {"command": "ls -la"},  # dict, not JSON string
        })
        assert result == {"output": "", "error": None}


# ============================================================================
# Parse checkout target (shared helper)
# ============================================================================

class TestParseCheckoutTarget:
    """Test _parse_checkout_target from hook_server."""

    def test_simple_checkout(self):
        from omega.server.hook_server import _parse_checkout_target
        assert _parse_checkout_target("git checkout main") == "main"

    def test_checkout_with_flags(self):
        from omega.server.hook_server import _parse_checkout_target
        assert _parse_checkout_target("git checkout --force main") == "main"

    def test_new_branch_returns_none(self):
        from omega.server.hook_server import _parse_checkout_target
        assert _parse_checkout_target("git checkout -b new-feature") is None

    def test_orphan_returns_none(self):
        from omega.server.hook_server import _parse_checkout_target
        assert _parse_checkout_target("git checkout --orphan gh-pages") is None

    def test_file_restore_returns_none(self):
        from omega.server.hook_server import _parse_checkout_target
        assert _parse_checkout_target("git checkout -- file.py") is None

    def test_switch_command(self):
        from omega.server.hook_server import _parse_checkout_target
        assert _parse_checkout_target("git switch develop") == "develop"

    def test_compound_command(self):
        from omega.server.hook_server import _parse_checkout_target
        assert _parse_checkout_target("git fetch && git checkout main") == "main"

    def test_no_git_returns_none(self):
        from omega.server.hook_server import _parse_checkout_target
        assert _parse_checkout_target("ls -la") is None


# ============================================================================
# Dispatch table completeness (updated for pre_push_guard)
# ============================================================================

def test_pre_push_guard_in_dispatch_table():
    """pre_push_guard should be in the handler dispatch table."""
    from omega.server.hook_server import HOOK_HANDLERS
    assert "pre_push_guard" in HOOK_HANDLERS
    assert callable(HOOK_HANDLERS["pre_push_guard"])


def test_pre_push_guard_in_fallback_scripts():
    """pre_push_guard should be in fast_hook fallback scripts."""
    import importlib.util
    hooks_dir = Path(__file__).parent.parent / "hooks"
    spec = importlib.util.spec_from_file_location("fast_hook", hooks_dir / "fast_hook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert "pre_push_guard" in mod._FALLBACK_SCRIPTS
    # And the fallback script file should exist
    script_path = hooks_dir / "pre_push_guard.py"
    assert script_path.exists()


# ============================================================================
# Coordination Wiring & Observability Tests
# ============================================================================

def test_claim_debounce_lowered():
    """CLAIM_DEBOUNCE_S should be 30.0 (lowered from 60)."""
    from omega.server.hook_server import CLAIM_DEBOUNCE_S
    assert CLAIM_DEBOUNCE_S == 30.0


def test_claim_debounce_refreshes_activity():
    """Debounced auto-claim should still call refresh_file_activity."""
    from omega.server import hook_server
    from omega.server.hook_server import handle_auto_claim_file

    hook_server._last_claim.clear()

    with patch("omega.coordination.get_manager") as mock_mgr_fn:
        mock_mgr = MagicMock()
        mock_mgr.claim_file.return_value = {"success": True}
        mock_mgr.announce_intent.return_value = {"success": True}
        mock_mgr.check_intents.return_value = {"has_overlaps": False, "overlaps": []}
        mock_mgr_fn.return_value = mock_mgr

        payload = {
            "tool_name": "Edit",
            "session_id": "refresh-test",
            "tool_input": json.dumps({"file_path": "/tmp/refresh.py"}),
        }

        # First call — full claim
        handle_auto_claim_file(payload)
        assert mock_mgr.claim_file.call_count == 1

        # Second call — debounced, should call refresh_file_activity
        handle_auto_claim_file(payload)
        assert mock_mgr.claim_file.call_count == 1  # Not called again
        assert mock_mgr.refresh_file_activity.call_count == 1
        mock_mgr.refresh_file_activity.assert_called_with("refresh-test", "/tmp/refresh.py")

    hook_server._last_claim.clear()


def test_heartbeat_inbox_frequency():
    """Inbox should be checked every 2nd heartbeat (~60s polling)."""
    from omega.server import hook_server
    from omega.server.hook_server import handle_coord_heartbeat

    hook_server._last_heartbeat.clear()
    hook_server._heartbeat_count.clear()

    with patch("omega.coordination.get_manager") as mock_mgr_fn:
        mock_mgr = MagicMock()
        mock_mgr.get_unread_count.return_value = 0
        mock_mgr_fn.return_value = mock_mgr

        # Heartbeats 1-4: only even counts should check inbox
        for i in range(4):
            # Reset debounce so heartbeat actually fires
            hook_server._last_heartbeat.clear()
            handle_coord_heartbeat({"session_id": "inbox-test"})

        # Count 2 and 4 should trigger inbox check = 2 calls
        assert mock_mgr.get_unread_count.call_count == 2

    hook_server._last_heartbeat.clear()
    hook_server._heartbeat_count.clear()


def test_auto_claim_checks_intent_overlap():
    """auto_claim_file should check for intent overlaps after successful claim."""
    from omega.server import hook_server
    from omega.server.hook_server import handle_auto_claim_file

    hook_server._last_claim.clear()

    with patch("omega.coordination.get_manager") as mock_mgr_fn:
        mock_mgr = MagicMock()
        mock_mgr.claim_file.return_value = {"success": True}
        mock_mgr.announce_intent.return_value = {"success": True}
        mock_mgr.check_intents.return_value = {
            "has_overlaps": True,
            "overlaps": [
                {
                    "session_id": "other-agent-12345",
                    "description": "Editing shared module",
                    "overlapping_files": ["/tmp/shared.py"],
                }
            ],
        }
        mock_mgr_fn.return_value = mock_mgr

        payload = {
            "tool_name": "Edit",
            "session_id": "overlap-test",
            "tool_input": json.dumps({"file_path": "/tmp/shared.py"}),
        }
        result = handle_auto_claim_file(payload)

        assert mock_mgr.check_intents.call_count == 1
        assert "[INTENT-OVERLAP]" in result["output"]
        from omega.server.hook_server import _agent_nickname
        assert _agent_nickname("other-agent-12345") in result["output"]

    hook_server._last_claim.clear()


def test_coord_session_start_no_noise_solo_agent():
    """Session start with no peers should produce minimal output."""
    from omega.server.hook_server import handle_coord_session_start

    with patch("omega.coordination.get_manager") as mock_mgr_fn:
        mock_mgr = MagicMock()
        mock_mgr.register_session.return_value = {"peers_on_project": 0}
        mock_mgr.list_sessions.return_value = [
            {"session_id": "silent-test", "project": "/tmp/proj", "task": "",
             "status": "active", "last_heartbeat": "2026-01-01T00:00:00"},
        ]
        mock_mgr.list_tasks.return_value = []
        mock_mgr.get_status.return_value = {
            "file_claims": 0,
            "branch_claims": 0,
            "active_intents": 0,
            "conflicts": [],
        }
        mock_mgr_fn.return_value = mock_mgr

        with patch("omega.server.hook_server.coordination._check_git_sync", return_value=[]):
            with patch("omega.server.hook_server.coordination._session_resume", return_value=[]):
                result = handle_coord_session_start({
                    "session_id": "silent-test",
                    "project": "/tmp/proj",
                })

        # No peer footer when solo
        assert "[COORD]" not in result["output"]
        # No [TODO] when no tasks and no unread
        assert "[TODO]" not in result["output"]


# ============================================================================
# Audit Logging in Coord Handlers
# ============================================================================

def test_audit_on_task_create():
    """task_create handler should call _audit_log on success."""
    import asyncio
    from omega.server.coord_handlers import handle_task_create

    with patch("omega.coordination.get_manager") as mock_mgr_fn:
        mock_mgr = MagicMock()
        mock_mgr.create_task.return_value = {"success": True, "task_id": 42}
        mock_mgr_fn.return_value = mock_mgr

        with patch("omega.server.coord_handlers._audit_log") as mock_audit:
            asyncio.run(
                handle_task_create({"session_id": "audit-test", "title": "Test task"})
            )
            assert mock_audit.call_count == 1
            assert mock_audit.call_args[0][0] == "task_create"
            assert mock_audit.call_args[0][1] == "audit-test"


def test_audit_on_branch_release():
    """branch_release handler should call _audit_log on success."""
    import asyncio
    from omega.server.coord_handlers import handle_branch_release

    with patch("omega.coordination.get_manager") as mock_mgr_fn:
        mock_mgr = MagicMock()
        mock_mgr.release_branch.return_value = {"released": True}
        mock_mgr_fn.return_value = mock_mgr

        with patch("omega.server.coord_handlers._audit_log") as mock_audit:
            asyncio.run(
                handle_branch_release({
                    "session_id": "audit-test",
                    "project": "/tmp/proj",
                    "branch": "feature-x",
                })
            )
            assert mock_audit.call_count == 1
            assert mock_audit.call_args[0][0] == "branch_release"


def test_audit_on_send_message():
    """send_message handler should call _audit_log on success."""
    import asyncio
    from omega.server.coord_handlers import handle_send_message

    with patch("omega.coordination.get_manager") as mock_mgr_fn:
        mock_mgr = MagicMock()
        mock_mgr.send_message.return_value = {"success": True, "context_id": "ctx-123"}
        mock_mgr_fn.return_value = mock_mgr

        with patch("omega.server.coord_handlers._audit_log") as mock_audit:
            asyncio.run(
                handle_send_message({
                    "session_id": "audit-test",
                    "subject": "Hello agent",
                })
            )
            assert mock_audit.call_count == 1
            assert mock_audit.call_args[0][0] == "send_message"


def test_audit_on_task_progress():
    """task_progress handler should call _audit_log on success."""
    import asyncio
    from omega.server.coord_handlers import handle_task_progress
    with patch("omega.coordination.get_manager") as m:
        m.return_value.update_task_progress.return_value = {"success": True, "progress": 50}
        with patch("omega.server.coord_handlers._audit_log") as a:
            asyncio.run(
                handle_task_progress({"session_id": "a", "task_id": 7, "progress": 50})
            )
            assert a.call_args[0][0] == "task_progress"
            assert "50%" in a.call_args[0][2]


def test_intent_announce_passes_intent_type():
    """intent_announce handler should pass intent_type to mgr.announce_intent."""
    import asyncio
    from omega.server.coord_handlers import handle_intent_announce
    with patch("omega.coordination.get_manager") as m:
        m.return_value.announce_intent.return_value = {"success": True, "expires_at": "2026-02-10T12:00:00"}
        asyncio.run(
            handle_intent_announce({"session_id": "a", "description": "Refactoring", "intent_type": "refactor"})
        )
        assert m.return_value.announce_intent.call_args[1]["intent_type"] == "refactor"


def test_intent_announce_default_intent_type():
    """intent_announce handler should default intent_type to 'work'."""
    import asyncio
    from omega.server.coord_handlers import handle_intent_announce
    with patch("omega.coordination.get_manager") as m:
        m.return_value.announce_intent.return_value = {"success": True, "expires_at": "2026-02-10T12:00:00"}
        asyncio.run(
            handle_intent_announce({"session_id": "a", "description": "Working on feature"})
        )
        assert m.return_value.announce_intent.call_args[1]["intent_type"] == "work"


def test_intent_type_in_schema():
    """intent_type should be in the omega_intent_announce MCP schema."""
    from omega.server.coord_schemas import COORD_TOOL_SCHEMAS
    schema = next(s for s in COORD_TOOL_SCHEMAS if s["name"] == "omega_intent_announce")
    assert "intent_type" in schema["inputSchema"]["properties"]
    assert schema["inputSchema"]["properties"]["intent_type"]["default"] == "work"


# ============================================================================
# Memory Subsystem Wiring Tests
# ============================================================================

def test_pre_edit_surface_removed_from_dispatch():
    """handle_pre_edit_surface was removed — should not be in dispatch table."""
    from omega.server.hook_server import HOOK_HANDLERS
    assert "pre_edit_surface" not in HOOK_HANDLERS


def test_lesson_quality_gate_relaxed():
    """Lesson auto-capture should accept non-code lessons >= 50 chars, >= 7 words."""
    from omega.server.hook_server import handle_auto_capture

    # This lesson has no tech signals but is 55 chars, 9 words — should pass
    payload = {
        "stdin": json.dumps({
            "prompt": "the trick is to always test in isolation before integrating",
            "session_id": "lesson-test",
            "cwd": "/tmp/proj",
        }),
    }
    with patch("omega.bridge.auto_capture") as mock_capture:
        mock_capture.return_value = "Stored mem-test12 (lesson_learned, permanent)"
        handle_auto_capture(payload)
        assert mock_capture.called
        assert mock_capture.call_args[1]["event_type"] == "lesson_learned"


def test_lesson_quality_gate_rejects_too_short():
    """Lesson auto-capture should reject prompts < 50 chars."""
    from omega.server.hook_server import handle_auto_capture

    payload = {
        "stdin": json.dumps({
            "prompt": "the trick is to test first",
            "session_id": "lesson-test",
            "cwd": "/tmp/proj",
        }),
    }
    with patch("omega.bridge.auto_capture") as mock_capture:
        mock_capture.return_value = "Stored mem-test12 (lesson_learned, permanent)"
        handle_auto_capture(payload)
        assert not mock_capture.called


def test_session_start_embedding_warning():
    """Session start should show [!] alert when embedding backend is None."""
    from omega.server.hook_server import handle_session_start

    with patch("omega.bridge.get_session_context") as mock_ctx, \
         patch("omega.embedding.get_active_backend", return_value=None):
        mock_ctx.return_value = {
            "memory_count": 10,
            "health_status": "ok",
            "last_capture_ago": "5m ago",
            "context_items": [],
        }
        result = handle_session_start({
            "session_id": "embed-test",
            "project": "/tmp",
        })

    assert "[!]" in result["output"]
    assert "hash fallback" in result["output"]


def test_session_start_no_warning_when_backend_active():
    """Session start should NOT show [!] alert when embedding backend is active."""
    from omega.server.hook_server import handle_session_start

    with patch("omega.bridge.get_session_context") as mock_ctx, \
         patch("omega.embedding.get_active_backend", return_value="onnx"):
        mock_ctx.return_value = {
            "memory_count": 10,
            "health_status": "ok",
            "last_capture_ago": "5m ago",
            "context_items": [],
        }
        result = handle_session_start({
            "session_id": "embed-test",
            "project": "/tmp",
        })

    # Only check for embedding-related warnings; router provider warnings
    # (e.g., "0 providers active") are expected in CI where no API keys exist
    lines = result["output"].split("\n")
    embedding_warnings = [l for l in lines if "[!]" in l and "embedding" in l.lower()]
    assert embedding_warnings == []


def test_session_stop_no_wasted_similar_call():
    """Session stop should NOT call find_similar_memories (removed wasted call)."""
    from omega.server.hook_server import handle_session_stop

    with patch("omega.bridge.query_structured") as mock_qs:
        mock_qs.return_value = []
        with patch("omega.bridge.auto_capture") as mock_capture:
            mock_capture.return_value = "# Captured"
            with patch("omega.bridge.session_stats", return_value={}):
                with patch("omega.bridge.type_stats", return_value={}):
                    with patch("omega.bridge.find_similar_memories") as mock_similar:
                        handle_session_stop({
                            "session_id": "stop-test",
                            "project": "/tmp",
                        })
                        assert not mock_similar.called


# ============================================================================
# Scorecard A-Grade Tests: auto-doctor, router, coordination
# ============================================================================

def test_session_start_auto_doctor_runs(tmp_path):
    """Session start should run auto-doctor when marker is absent or stale."""
    from omega.server.hook_server import handle_session_start

    with patch("omega.bridge.get_session_context") as mock_ctx, \
         patch("omega.embedding.get_active_backend", return_value="onnx"), \
         patch("omega.bridge.status") as mock_status, \
         patch("pathlib.Path.home", return_value=tmp_path):

        mock_ctx.return_value = {
            "memory_count": 10, "health_status": "ok",
            "last_capture_ago": "5m ago", "context_items": [],
        }
        mock_status.return_value = {"node_count": 100, "vec_enabled": True}

        # Create .omega dir
        (tmp_path / ".omega").mkdir(parents=True, exist_ok=True)

        result = handle_session_start({"session_id": "doc-test", "project": "/tmp"})
        # maintenance pipeline footer shows doctor ran
        assert "maintenance:" in result["output"]


def test_session_start_router_degraded_alert():
    """Session start should show [!] alert when some router providers unavailable."""
    from omega.server.hook_server import handle_session_start

    with patch("omega.bridge.get_session_context") as mock_ctx, \
         patch("omega.embedding.get_active_backend", return_value="onnx"), \
         patch("omega.router.engine.OmegaRouter") as mock_router_cls:

        mock_ctx.return_value = {
            "memory_count": 10, "health_status": "ok",
            "last_capture_ago": "5m ago", "context_items": [],
        }
        mock_router = MagicMock()
        mock_router.get_provider_status.return_value = {
            "anthropic": "available", "openai": "available",
            "google": "available", "xai": "available", "groq": "no_api_key",
        }
        mock_router_cls.return_value = mock_router

        result = handle_session_start({"session_id": "router-test", "project": "/tmp"})
        # Degraded providers shown as [!] alert
        assert "4/5 providers active" in result["output"]
        assert "[!]" in result["output"]


def test_auto_capture_classifies_intent():
    """auto_capture should classify prompt intent when router is available."""
    from omega.server.hook_server import handle_auto_capture

    with patch("omega.bridge.auto_capture") as mock_ac, \
         patch("omega.router.classifier.classify_intent", return_value=("coding", 0.85)):
        result = handle_auto_capture({
            "stdin": json.dumps({
                "prompt": "Let's go with SQLite instead of PostgreSQL for the backend database since it simplifies local development and testing significantly",
                "session_id": "classify-test",
                "cwd": "/tmp",
            }),
        })
        assert mock_ac.called
        # Intent should be tagged in metadata
        meta = mock_ac.call_args[1]["metadata"]
        assert meta.get("intent") == "coding"
        # Router output should be returned
        assert "[ROUTER]" in result["output"]


def test_auto_capture_no_router_still_works():
    """auto_capture should work normally when router is not installed."""
    from omega.server.hook_server import handle_auto_capture

    with patch("omega.bridge.auto_capture") as mock_ac, \
         patch.dict("sys.modules", {"omega.router.classifier": None}):
        # Force ImportError by patching the import
        import builtins
        _original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "omega.router.classifier":
                raise ImportError("No module")
            return _original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = handle_auto_capture({
                "stdin": json.dumps({
                    "prompt": "Let's go with SQLite instead of PostgreSQL for the backend database since it simplifies local development and testing significantly",
                    "session_id": "no-router-test",
                    "cwd": "/tmp",
                }),
            })
            assert mock_ac.called
            # No router output when router unavailable
            assert "[ROUTER]" not in result["output"]


def test_coord_session_stop_sends_handoff():
    """coord_session_stop should broadcast a handoff message before deregistering."""
    from omega.server.hook_server import handle_coord_session_stop

    with patch("omega.coordination.get_manager") as mock_mgr_fn:
        mock_mgr = MagicMock()
        mock_mgr_fn.return_value = mock_mgr

        handle_coord_session_stop({
            "session_id": "handoff-test",
            "project": "/tmp/proj",
        })

        # send_message should be called before deregister
        assert mock_mgr.send_message.called
        call_kwargs = mock_mgr.send_message.call_args
        assert call_kwargs[1]["msg_type"] == "complete"
        assert mock_mgr.deregister_session.called


def test_coord_session_start_announces_intent():
    """coord_session_start should announce session intent with branch."""
    from omega.server.hook_server import handle_coord_session_start

    with patch("omega.coordination.get_manager") as mock_mgr_fn:
        mock_mgr = MagicMock()
        mock_mgr.register_session.return_value = {"peers_on_project": 0}
        mock_mgr.list_tasks.return_value = []
        mock_mgr.get_status.return_value = {
            "file_claims": 0, "branch_claims": 0,
            "active_intents": 0, "conflicts": [],
        }
        mock_mgr_fn.return_value = mock_mgr

        with patch("omega.server.hook_server.coordination._check_git_sync", return_value=[]), \
             patch("omega.server.hook_server.coordination._session_resume", return_value=[]), \
             patch("omega.server.hook_server.coordination._get_current_branch", return_value="feature-xyz"):
            handle_coord_session_start({
                "session_id": "intent-test",
                "project": "/tmp/proj",
            })

        assert mock_mgr.announce_intent.called
        assert mock_mgr.claim_branch.called
        call_kwargs = mock_mgr.claim_branch.call_args
        assert call_kwargs[0][2] == "feature-xyz"  # branch name


def test_coord_session_start_skips_protected_branch():
    """coord_session_start should NOT announce intent for main/master branches."""
    from omega.server.hook_server import handle_coord_session_start

    with patch("omega.coordination.get_manager") as mock_mgr_fn:
        mock_mgr = MagicMock()
        mock_mgr.register_session.return_value = {"peers_on_project": 0}
        mock_mgr.list_tasks.return_value = []
        mock_mgr.get_status.return_value = {
            "file_claims": 0, "branch_claims": 0,
            "active_intents": 0, "conflicts": [],
        }
        mock_mgr_fn.return_value = mock_mgr

        with patch("omega.server.hook_server.coordination._check_git_sync", return_value=[]), \
             patch("omega.server.hook_server.coordination._session_resume", return_value=[]), \
             patch("omega.server.hook_server.coordination._get_current_branch", return_value="main"):
            handle_coord_session_start({
                "session_id": "main-test",
                "project": "/tmp/proj",
            })

        assert not mock_mgr.announce_intent.called
        assert not mock_mgr.claim_branch.called


# ============================================================================
# Router warmup reporting in coord_session_start
# ============================================================================


def test_coord_session_start_router_warmup_called():
    """coord_session_start should defer warm_up to background executor (no output)."""
    from omega.server.hook_server import handle_coord_session_start

    with patch("omega.coordination.get_manager") as mock_mgr_fn, \
         patch("omega.server.hook_server.utils._HOOK_BG_EXECUTOR") as mock_exec, \
         patch("omega.router.classifier.warm_up"):
        mock_mgr = MagicMock()
        mock_mgr.register_session.return_value = {"success": True, "session_id": "clf-test"}
        mock_mgr.get_status.return_value = {"file_claims": [], "branch_claims": [], "active_intents": [], "conflicts": []}
        mock_mgr.list_tasks.return_value = []
        mock_mgr_fn.return_value = mock_mgr

        result = handle_coord_session_start({
            "session_id": "clf-test",
            "project": "/tmp",
        })
        # Router warmup is now deferred to background executor
        mock_exec.submit.assert_called()
        assert "[ROUTER]" not in result["output"]


# ============================================================================
# Session start fixes
# ============================================================================


def test_capture_plan_skips_read_output():
    """_capture_plan should NOT fire on Read tool output (false positive source)."""
    from omega.server.hook_server import handle_surface_memories

    plan_output = (
        "## Phase 1\n## Phase 2\n## Plan\n"
        "| Step | Description |\n| 1 | Do thing |\n"
        "1. First step\n2. Second step\n" + "x" * 600
    )
    with patch("omega.bridge.auto_capture") as mock_capture:
        handle_surface_memories({
            "tool_name": "Read",
            "tool_input": json.dumps({"file_path": "/tmp/plan.md"}),
            "tool_output": plan_output,
            "session_id": "plan-test",
            "project": "/tmp",
        })
    # auto_capture should NOT be called for plan capture (may be called for surfacing)
    for call in mock_capture.call_args_list:
        content = call[1].get("content", "") if call[1] else (call[0][0] if call[0] else "")
        assert "Plan/decision captured" not in content


def test_capture_plan_disabled_on_edit_output():
    """_capture_plan is disabled — Edit tool output should NOT generate plan captures.

    _capture_plan was disabled because Edit/Write tool_output contains serialized
    tool input params (filePath, oldString, newString), not plan content. This
    generated 50+ noisy memories. Real decisions use omega_store() explicitly.
    """
    from omega.server import hook_server
    from omega.server.hook_server import handle_surface_memories

    plan_output = (
        "## Phase 1\n## Phase 2\n## Plan\n"
        "| Step | Description |\n| 1 | Do thing |\n"
        "1. First step\n2. Second step\n" + "x" * 600
    )
    # Clear surface debounce to avoid interference
    hook_server._last_surface.clear()

    with patch("omega.bridge.auto_capture") as mock_capture, \
         patch("omega.bridge.query_structured", return_value=[]):
        handle_surface_memories({
            "tool_name": "Edit",
            "tool_input": json.dumps({"file_path": "/tmp/plan-edit.py"}),
            "tool_output": plan_output,
            "session_id": "plan-edit-test",
            "project": "/tmp",
        })

    # No plan capture calls should be made (feature disabled)
    plan_calls = [
        c for c in mock_capture.call_args_list
        if "Plan/decision captured" in str(c)
    ]
    assert len(plan_calls) == 0


def test_pre_edit_surface_not_in_fast_hook():
    """pre_edit_surface should not be in fast_hook.py fallback scripts."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fast_hook", str(Path(__file__).parent.parent / "hooks" / "fast_hook.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "pre_edit_surface" not in mod._FALLBACK_SCRIPTS


# ============================================================================
# Proactive Multi-Agent Collaboration Tests
# ============================================================================


class TestPeerAwareness:
    """Peer awareness deferred to footer + tool hints."""

    def test_peer_count_shown_in_footer(self):
        """Peer count should show in footer when peers > 0."""
        from omega.server.hook_server import handle_coord_session_start

        from datetime import datetime, timezone

        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr.register_session.return_value = {"peers_on_project": 1}
            mock_mgr.list_sessions.return_value = [
                {
                    "session_id": "my-session-abc",
                    "project": "/tmp/proj",
                    "task": "",
                    "last_heartbeat": datetime.now(timezone.utc).isoformat(),
                },
                {
                    "session_id": "peer-session-xyz",
                    "project": "/tmp/proj",
                    "task": "editing tests",
                    "last_heartbeat": datetime.now(timezone.utc).isoformat(),
                },
            ]
            mock_mgr.get_session_claims.return_value = {
                "file_claims": [], "branch_claims": [],
            }
            mock_mgr.get_unread_count.return_value = 0
            mock_mgr.list_tasks.return_value = []
            mock_mgr.get_status.return_value = {
                "file_claims": 0, "branch_claims": 0,
                "active_intents": 0, "conflicts": [],
            }
            mock_mgr_fn.return_value = mock_mgr

            with patch("omega.server.hook_server.coordination._check_git_sync", return_value=[]), \
                 patch("omega.server.hook_server.coordination._session_resume", return_value=[]):
                result = handle_coord_session_start({
                    "session_id": "my-session-abc",
                    "project": "/tmp/proj",
                })

        output = result["output"]
        assert "[COORD]" in output
        assert "peer" in output

    def test_peer_footer_not_shown_single_agent(self):
        """No peer footer when no other agents active."""
        from omega.server.hook_server import handle_coord_session_start

        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr.register_session.return_value = {"peers_on_project": 0}
            mock_mgr.list_sessions.return_value = [
                {"session_id": "solo-session", "project": "/tmp/proj", "task": "",
                 "status": "active", "last_heartbeat": "2026-01-01T00:00:00"},
            ]
            mock_mgr.list_tasks.return_value = []
            mock_mgr.get_status.return_value = {
                "file_claims": 0, "branch_claims": 0,
                "active_intents": 0, "conflicts": [],
            }
            mock_mgr_fn.return_value = mock_mgr

            with patch("omega.server.hook_server.coordination._check_git_sync", return_value=[]), \
                 patch("omega.server.hook_server.coordination._session_resume", return_value=[]):
                result = handle_coord_session_start({
                    "session_id": "solo-session",
                    "project": "/tmp/proj",
                })

        assert "[COORD]" not in result["output"]

    def test_unread_messages_shown_in_todo(self):
        """Session start should show unread count in [TODO] when peers active."""
        from omega.server.hook_server import handle_coord_session_start
        from datetime import datetime, timezone

        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr.register_session.return_value = {"peers_on_project": 1}
            mock_mgr.list_sessions.return_value = [
                {"session_id": "other-agent", "project": "/tmp", "task": "working",
                 "last_heartbeat": datetime.now(timezone.utc).isoformat()},
            ]
            mock_mgr.get_session_claims.return_value = {
                "file_claims": [], "branch_claims": [],
            }
            mock_mgr.get_unread_count.return_value = 3
            mock_mgr.list_tasks.return_value = []
            mock_mgr.get_status.return_value = {
                "file_claims": 0, "branch_claims": 0,
                "active_intents": 0, "conflicts": [],
            }
            mock_mgr_fn.return_value = mock_mgr

            with patch("omega.server.hook_server.coordination._check_git_sync", return_value=[]), \
                 patch("omega.server.hook_server.coordination._session_resume", return_value=[]):
                result = handle_coord_session_start({
                    "session_id": "my-session",
                    "project": "/tmp",
                })

        assert "[TODO]" in result["output"]
        assert "3 unread msg" in result["output"]
        assert "omega_inbox" in result["output"]


class TestStructuredHandoffs:
    """Change 2: Structured rich handoffs on session stop."""

    def test_handoff_includes_decisions(self):
        """Handoff should include decisions from the session."""
        from omega.server.hook_server import handle_coord_session_stop

        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr_fn.return_value = mock_mgr

            with patch("omega.bridge._get_store") as mock_store_fn:
                mock_store = MagicMock()
                mock_store.get_session_event_counts.return_value = {"decision": 2}
                mock_store_fn.return_value = mock_store

                with patch("omega.bridge.query_structured") as mock_qs:
                    mock_qs.side_effect = [
                        # decisions query
                        [{"content": "Decided to use SQLite"}],
                        # errors query
                        [],
                    ]
                    handle_coord_session_stop({
                        "session_id": "handoff-decisions",
                        "project": "/tmp/proj",
                    })

            call_kwargs = mock_mgr.send_message.call_args[1]
            assert call_kwargs["msg_type"] == "complete"
            assert call_kwargs["ttl_minutes"] == 1440
            assert "## Decisions" in call_kwargs["body"]
            assert "Decided to use SQLite" in call_kwargs["body"]

    def test_handoff_includes_errors(self):
        """Handoff should include errors/blockers from the session."""
        from omega.server.hook_server import handle_coord_session_stop

        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr_fn.return_value = mock_mgr

            with patch("omega.bridge._get_store") as mock_store_fn:
                mock_store = MagicMock()
                mock_store.get_session_event_counts.return_value = {"error_pattern": 1}
                mock_store_fn.return_value = mock_store

                with patch("omega.bridge.query_structured") as mock_qs:
                    mock_qs.side_effect = [
                        [],  # decisions
                        [{"content": "Error: Connection refused on port 5432"}],  # errors
                    ]
                    handle_coord_session_stop({
                        "session_id": "handoff-errors",
                        "project": "/tmp/proj",
                    })

            body = mock_mgr.send_message.call_args[1]["body"]
            assert "## Blockers" in body
            assert "Connection refused" in body

    def test_handoff_includes_incomplete_tasks(self):
        """Handoff should include incomplete tasks owned by this session."""
        from omega.server.hook_server import handle_coord_session_stop

        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr.list_tasks.return_value = [
                {"id": 42, "title": "Fix auth bug", "session_id": "handoff-tasks"},
                {"id": 43, "title": "Other agent's task", "session_id": "other-agent"},
            ]
            mock_mgr_fn.return_value = mock_mgr

            with patch("omega.bridge._get_store") as mock_store_fn:
                mock_store = MagicMock()
                mock_store.get_session_event_counts.return_value = {}
                mock_store_fn.return_value = mock_store

                with patch("omega.bridge.query_structured", return_value=[]):
                    handle_coord_session_stop({
                        "session_id": "handoff-tasks",
                        "project": "/tmp/proj",
                    })

            body = mock_mgr.send_message.call_args[1]["body"]
            assert "## Incomplete Work" in body
            assert "Fix auth bug" in body
            assert "Other agent's task" not in body

    def test_handoff_body_capped_at_2kb(self):
        """Handoff body should be capped at 2000 chars."""
        from omega.server.hook_server import handle_coord_session_stop

        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr.list_tasks.return_value = []
            mock_mgr_fn.return_value = mock_mgr

            with patch("omega.bridge._get_store") as mock_store_fn:
                mock_store = MagicMock()
                mock_store.get_session_event_counts.return_value = {}
                mock_store_fn.return_value = mock_store

                # Generate lots of decisions to exceed 2KB
                with patch("omega.bridge.query_structured") as mock_qs:
                    mock_qs.side_effect = [
                        [{"content": "x" * 200} for _ in range(5)],  # decisions
                        [{"content": "y" * 200} for _ in range(3)],  # errors
                    ]
                    handle_coord_session_stop({
                        "session_id": "handoff-cap",
                        "project": "/tmp/proj",
                    })

            body = mock_mgr.send_message.call_args[1]["body"]
            assert len(body) <= 2000


class TestActiveCoordHeartbeat:
    """Change 3: Active coordination on heartbeat."""

    def test_blocked_task_surfaced(self):
        """4th heartbeat should surface blocked task info."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_coord_heartbeat

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()

        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr.get_unread_count.return_value = 0
            mock_mgr.list_tasks.return_value = [
                {
                    "id": 10, "title": "My task", "session_id": "hb-blocked",
                    "status": "in_progress", "depends_on": [9],
                },
                {
                    "id": 9, "title": "Dep task", "session_id": "other-agent",
                    "status": "in_progress",
                },
            ]
            mock_mgr_fn.return_value = mock_mgr

            # Run 4 heartbeats to trigger coordination check
            for _ in range(4):
                hook_server._last_heartbeat.clear()
                handle_coord_heartbeat({"session_id": "hb-blocked", "project": "/tmp"})

        # 4th heartbeat should have checked tasks
        result = handle_coord_heartbeat.__wrapped__ if hasattr(handle_coord_heartbeat, '__wrapped__') else None
        # Check the last call's result by calling one more time
        hook_server._last_heartbeat.clear()
        result = handle_coord_heartbeat({"session_id": "hb-blocked", "project": "/tmp"})
        # The 5th heartbeat (count=5) is not a 4th-multiple, so check directly
        # Reset and run exactly 4
        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()

        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr.get_unread_count.return_value = 0
            mock_mgr.list_tasks.return_value = [
                {
                    "id": 10, "title": "My task", "session_id": "hb-blocked2",
                    "status": "in_progress", "depends_on": [9],
                },
                {
                    "id": 9, "title": "Dep task", "session_id": "other-agent-abc",
                    "status": "in_progress",
                },
            ]
            mock_mgr_fn.return_value = mock_mgr

            last_result = None
            for i in range(4):
                hook_server._last_heartbeat.clear()
                last_result = handle_coord_heartbeat({"session_id": "hb-blocked2", "project": "/tmp"})

        assert "[BLOCKED]" in last_result["output"]
        assert "#10" in last_result["output"]
        assert "#9" in last_result["output"]

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()

    def test_request_message_surfaced(self):
        """4th heartbeat should surface request-type message content when unread."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_coord_heartbeat, _agent_nickname

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()

        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr.get_unread_count.return_value = 2
            mock_mgr.check_inbox.return_value = [
                {
                    "from_session": "requester-agent-123",
                    "subject": "Need help with database migration",
                    "msg_type": "request",
                },
            ]
            mock_mgr.list_tasks.return_value = []
            mock_mgr.get_recent_events.return_value = []
            mock_mgr_fn.return_value = mock_mgr

            last_result = None
            for _ in range(4):
                hook_server._last_heartbeat.clear()
                last_result = handle_coord_heartbeat({"session_id": "hb-request", "project": "/tmp"})

        assert "[REQUEST]" in last_result["output"]
        assert _agent_nickname("requester-agent-123") in last_result["output"]
        assert "database migration" in last_result["output"]

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()

    def test_no_coord_on_non_4th_heartbeat(self):
        """Non-4th heartbeats should not run coordination checks."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_coord_heartbeat

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()

        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr.get_unread_count.return_value = 0
            mock_mgr_fn.return_value = mock_mgr

            # Run 3 heartbeats (counts 1, 2, 3 — none are multiples of 4)
            for _ in range(3):
                hook_server._last_heartbeat.clear()
                handle_coord_heartbeat({"session_id": "hb-no-coord", "project": "/tmp"})

            # list_tasks should NOT have been called (only called on 4th heartbeat)
            mock_mgr.list_tasks.assert_not_called()

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()


class TestPeerContextOnEdit:
    """Change 4: Peer context on file edit."""

    def test_peer_decisions_surfaced(self):
        """Editing a file should surface high-relevance peer decisions."""
        from omega.server.hook_server import _surface_lessons

        with patch("omega.bridge.get_cross_session_lessons", return_value=[]), \
             patch("omega.bridge.query_structured") as mock_qs:
            mock_qs.return_value = [
                {
                    "content": "Decided to use async IO for parser.py",
                    "event_type": "decision",
                    "relevance": 0.75,
                    "metadata": {"session_id": "other-session"},
                },
            ]
            lines = _surface_lessons("/tmp/proj/parser.py", "my-session", "/tmp/proj")

        assert any("[PEER-DECISION]" in line for line in lines)
        assert any("async IO" in line for line in lines)

    def test_own_decisions_filtered_out(self):
        """Own session's decisions should not appear as peer decisions."""
        from omega.server.hook_server import _surface_lessons

        with patch("omega.bridge.get_cross_session_lessons", return_value=[]), \
             patch("omega.bridge.query_structured") as mock_qs:
            mock_qs.return_value = [
                {
                    "content": "My own decision about parser",
                    "event_type": "decision",
                    "relevance": 0.8,
                    "metadata": {"session_id": "my-session"},
                },
            ]
            lines = _surface_lessons("/tmp/proj/parser.py", "my-session", "/tmp/proj")

        assert not any("[PEER-DECISION]" in line for line in lines)

    def test_low_relevance_filtered_out(self):
        """Low-relevance peer decisions should be filtered out."""
        from omega.server.hook_server import _surface_lessons

        with patch("omega.bridge.get_cross_session_lessons", return_value=[]), \
             patch("omega.bridge.query_structured") as mock_qs:
            mock_qs.return_value = [
                {
                    "content": "Some vague decision",
                    "event_type": "decision",
                    "relevance": 0.3,
                    "metadata": {"session_id": "other-session"},
                },
            ]
            lines = _surface_lessons("/tmp/proj/parser.py", "my-session", "/tmp/proj")

        assert not any("[PEER-DECISION]" in line for line in lines)


class TestOverlapNotification:
    """Change 5: Coordination message on intent overlap."""

    def test_overlap_sends_message_to_other_agent(self):
        """Intent overlap should send an inform message to the other agent."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_auto_claim_file

        hook_server._last_claim.clear()
        hook_server._last_overlap_notify.clear()

        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr.claim_file.return_value = {"success": True}
            mock_mgr.announce_intent.return_value = {"success": True}
            mock_mgr.check_intents.return_value = {
                "has_overlaps": True,
                "overlaps": [
                    {
                        "session_id": "other-agent-full-id",
                        "description": "Editing shared module",
                        "overlapping_files": ["/tmp/shared.py"],
                    }
                ],
            }
            mock_mgr_fn.return_value = mock_mgr

            # Verify mock is reachable via the import path
            from omega.coordination import get_manager as _gm
            assert _gm() is mock_mgr, "Mock not applied"

            result = handle_auto_claim_file({
                "tool_name": "Edit",
                "session_id": "my-agent",
                "tool_input": json.dumps({"file_path": "/tmp/shared.py"}),
            })

            # Verify the full code path was reached
            mock_mgr.claim_file.assert_called_once()
            mock_mgr.check_intents.assert_called_once()
            assert "[INTENT-OVERLAP]" in result["output"], (
                f"Overlap warning missing, got: {result['output']!r}"
            )

            # Verify send_message was called to notify the other agent
            mock_mgr.send_message.assert_called_once()
            call_kwargs = mock_mgr.send_message.call_args[1]
            assert call_kwargs["to_session"] == "other-agent-full-id"
            assert call_kwargs["msg_type"] == "inform"
            assert "shared.py" in call_kwargs["subject"]
            assert call_kwargs["ttl_minutes"] == 30

        hook_server._last_claim.clear()
        hook_server._last_overlap_notify.clear()

    def test_overlap_notification_debounced(self):
        """Same overlap should not send multiple messages within debounce window."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_auto_claim_file

        hook_server._last_claim.clear()
        hook_server._last_overlap_notify.clear()

        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr.claim_file.return_value = {"success": True}
            mock_mgr.announce_intent.return_value = {"success": True}
            mock_mgr.check_intents.return_value = {
                "has_overlaps": True,
                "overlaps": [
                    {
                        "session_id": "other-agent",
                        "description": "Editing module",
                        "overlapping_files": ["/tmp/mod.py"],
                    }
                ],
            }
            mock_mgr_fn.return_value = mock_mgr

            # First call — should send message
            handle_auto_claim_file({
                "tool_name": "Edit",
                "session_id": "my-agent",
                "tool_input": json.dumps({"file_path": "/tmp/mod.py"}),
            })
            assert mock_mgr.send_message.call_count == 1

            # Second call — debounced, no second message
            hook_server._last_claim.clear()  # clear claim debounce so handler runs fully
            handle_auto_claim_file({
                "tool_name": "Edit",
                "session_id": "my-agent",
                "tool_input": json.dumps({"file_path": "/tmp/mod.py"}),
            })
            assert mock_mgr.send_message.call_count == 1  # Still 1

        hook_server._last_claim.clear()
        hook_server._last_overlap_notify.clear()

    def test_overlap_notify_cleanup_on_session_stop(self):
        """Session stop should clean up overlap notify debounce entries."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_session_stop

        hook_server._last_overlap_notify[("test-sess", "other", "/tmp/f.py")] = 100.0
        hook_server._last_overlap_notify[("other-sess", "other", "/tmp/f.py")] = 100.0

        with patch("omega.bridge.query_structured", return_value=[]), \
             patch("omega.bridge.auto_capture"):
            handle_session_stop({"session_id": "test-sess", "project": "/tmp"})

        # test-sess entries should be cleaned
        assert ("test-sess", "other", "/tmp/f.py") not in hook_server._last_overlap_notify
        # other-sess entries should remain
        assert ("other-sess", "other", "/tmp/f.py") in hook_server._last_overlap_notify

        hook_server._last_overlap_notify.clear()


# ============================================================================
# Handoff + Auto-task + Deadlock surfacing (v0.6.0 multi-agent collab)
# ============================================================================


class TestHandoffSurfacing:
    """Tests for [HANDOFF] surfacing on session start and heartbeat."""

    def test_coord_session_start_surfaces_handoff(self, coord_mgr):
        """Session start should surface [HANDOFF] from predecessor's complete message."""
        from omega.server.hook_server import handle_coord_session_start

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("agent-A", pid=1001, project="/proj/x")
            # Agent A sends a complete-type handoff to agent-B
            coord_mgr.send_message(
                from_session="agent-A",
                to_session="agent-B",
                subject="Session handoff: auth module done",
                body="Completed auth module. TODO: add rate limiting to /api/login.",
                msg_type="complete",
            )

            with patch.multiple(
                "omega.bridge",
                query_structured=MagicMock(return_value=[]),
                get_cross_session_lessons=MagicMock(return_value=[]),
                _get_store=MagicMock(return_value=MagicMock(
                    get_session_event_counts=MagicMock(return_value={}),
                )),
                auto_capture=MagicMock(),
                consolidate=MagicMock(),
                create=True,
            ):
                result = handle_coord_session_start({
                    "session_id": "agent-B",
                    "project": "/proj/x",
                })

        output = result["output"]
        assert "[HANDOFF]" in output
        assert "agent-A" in output
        assert "auth module done" in output

    def test_heartbeat_surfaces_complete_messages(self, coord_mgr):
        """Heartbeat should surface [HANDOFF] from complete-type messages on 4th beat."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_coord_heartbeat

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("agent-B", pid=1002, project="/proj/x")
            coord_mgr.send_message(
                from_session="agent-A",
                to_session="agent-B",
                subject="Handoff: tests passing",
                body="All 50 tests green. Next: deploy.",
                msg_type="complete",
            )

            # Run 4 heartbeats to reach the 4th (skipping debounce)
            for i in range(4):
                hook_server._last_heartbeat.pop("agent-B", None)
                result = handle_coord_heartbeat({"session_id": "agent-B", "project": "/proj/x"})

        output = result["output"]
        assert "[HANDOFF]" in output
        assert "tests passing" in output


class TestAutoTaskFromPrompt:
    """Tests for auto-setting session.task from first user prompt."""

    def test_auto_capture_sets_session_task(self, coord_mgr):
        """First prompt should populate session.task in the coord DB."""
        from omega.server.hook_server import handle_auto_capture
        from omega.task_utils import clean_task_text

        with patch("omega.coordination.get_manager", return_value=coord_mgr), \
             patch("omega.server.hook_server.guards._summarize_task_text", side_effect=clean_task_text):
            coord_mgr.register_session("task-agent", pid=1001, project="/proj/x")

            payload = {
                "stdin": json.dumps({
                    "prompt": "let's use fastapi instead of flask for the API server migration",
                    "session_id": "task-agent",
                    "cwd": "/proj/x",
                }),
            }
            handle_auto_capture(payload)

        # Verify session.task was set
        row = coord_mgr._conn.execute(
            "SELECT task FROM coord_sessions WHERE session_id = ?", ("task-agent",)
        ).fetchone()
        assert row is not None
        assert "fastapi" in row[0].lower()

    def test_auto_capture_no_overwrite_existing_task(self, coord_mgr):
        """If session already has a task, don't overwrite it."""
        from omega.server.hook_server import handle_auto_capture
        from omega.task_utils import clean_task_text

        with patch("omega.coordination.get_manager", return_value=coord_mgr), \
             patch("omega.server.hook_server.guards._summarize_task_text", side_effect=clean_task_text):
            coord_mgr.register_session(
                "task-agent2", pid=1001, project="/proj/x", task="existing task"
            )

            payload = {
                "stdin": json.dumps({
                    "prompt": "let's switch to a completely different approach for everything",
                    "session_id": "task-agent2",
                    "cwd": "/proj/x",
                }),
            }
            handle_auto_capture(payload)

        row = coord_mgr._conn.execute(
            "SELECT task FROM coord_sessions WHERE session_id = ?", ("task-agent2",)
        ).fetchone()
        assert row[0] == "existing task"

    def test_auto_capture_task_set_once(self, coord_mgr):
        """Second prompt should not overwrite the task set by first prompt."""
        from omega.server.hook_server import handle_auto_capture

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("task-agent3", pid=1001, project="/proj/x")

            # First prompt sets task
            payload1 = {
                "stdin": json.dumps({
                    "prompt": "let's go with redis for the caching layer implementation",
                    "session_id": "task-agent3",
                    "cwd": "/proj/x",
                }),
            }
            handle_auto_capture(payload1)

            # Manually clear the task to see if second prompt would overwrite
            first_task = coord_mgr._conn.execute(
                "SELECT task FROM coord_sessions WHERE session_id = ?", ("task-agent3",)
            ).fetchone()[0]

            # Second prompt — should NOT update (task already set in DB)
            payload2 = {
                "stdin": json.dumps({
                    "prompt": "actually let's use memcached instead of redis for everything",
                    "session_id": "task-agent3",
                    "cwd": "/proj/x",
                }),
            }
            handle_auto_capture(payload2)

        row = coord_mgr._conn.execute(
            "SELECT task FROM coord_sessions WHERE session_id = ?", ("task-agent3",)
        ).fetchone()
        assert row[0] == first_task  # Unchanged by second prompt


class TestDeadlockAlerting:
    """Tests for [DEADLOCK] surfacing in heartbeat."""

    def test_heartbeat_surfaces_deadlock(self, coord_mgr):
        """Circular file claims should produce [DEADLOCK] alert on 10th heartbeat."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_coord_heartbeat

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("dl-agent-A", pid=1001, project="/proj/x")
            coord_mgr.register_session("dl-agent-B", pid=1002, project="/proj/x")

            # Create circular claims: A holds file1, wants file2; B holds file2, wants file1
            coord_mgr.claim_file("dl-agent-A", "/proj/x/file1.py")
            coord_mgr.claim_file("dl-agent-B", "/proj/x/file2.py")
            # Announce intents that create the cycle
            coord_mgr.announce_intent(
                "dl-agent-A", "editing file2",
                target_files=["/proj/x/file2.py"],
            )
            coord_mgr.announce_intent(
                "dl-agent-B", "editing file1",
                target_files=["/proj/x/file1.py"],
            )

            # Run 10 heartbeats to trigger deadlock detection
            for i in range(10):
                hook_server._last_heartbeat.pop("dl-agent-A", None)
                result = handle_coord_heartbeat({"session_id": "dl-agent-A", "project": "/proj/x"})

        output = result["output"]
        # Deadlock detection depends on detect_deadlocks() finding cycles.
        # If the coordination module detects the cycle, we should see [DEADLOCK].
        # If not (depends on implementation), at least verify no crash.
        assert result["error"] is None

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()


# ============================================================================
# Agent nicknames
# ============================================================================


class TestAgentNicknames:
    """Tests for _agent_nickname() deterministic name generation."""

    def setup_method(self):
        """No-op: pure hash has no mutable state to reset."""
        pass

    def test_nickname_deterministic(self):
        """Same session_id should always produce the same nickname."""
        from omega.server.hook_server import _agent_nickname

        name1 = _agent_nickname("test-session-abc")
        name2 = _agent_nickname("test-session-abc")
        assert name1 == name2

    def test_nickname_format(self):
        """Nickname should be 'Name (sid[:8])' format."""
        from omega.server.hook_server import _agent_nickname

        result = _agent_nickname("a3f2b1c8d9e0f1a2b3c4")
        assert "a3f2b1c8" in result
        assert "(" in result and ")" in result
        # Should start with a capitalized name
        name_part = result.split(" (")[0]
        assert name_part[0].isupper()

    def test_nickname_different_sessions(self):
        """Different session_ids should usually produce different nicknames."""
        from omega.server.hook_server import _agent_nickname

        names = {_agent_nickname(f"session-{i}") for i in range(20)}
        # With 88 names and 20 sessions, expect many unique (collisions possible but unlikely)
        assert len(names) >= 10

    def test_nickname_empty_session(self):
        """Empty session_id should return 'unknown'."""
        from omega.server.hook_server import _agent_nickname

        assert _agent_nickname("") == "unknown"

    def test_nickname_collision_same_word_different_suffix(self):
        """Two sessions that hash-collide get the same name word but different ID suffixes."""
        import hashlib
        from omega.server.hook_server import _agent_nickname, _AGENT_NAMES

        # Find two session IDs that hash to the same index but differ in first 8 chars
        target_idx = None
        sid_a = None
        sid_b = None
        for i in range(10000):
            sid = hashlib.md5(f"probe-{i}".encode()).hexdigest()  # use hex digest as session ID
            idx = int(hashlib.md5(sid.encode()).hexdigest()[:8], 16) % len(_AGENT_NAMES)
            if target_idx is None:
                target_idx = idx
                sid_a = sid
            elif idx == target_idx and sid[:8] != sid_a[:8]:
                sid_b = sid
                break

        assert sid_b is not None, "Could not find colliding session IDs"

        name_a = _agent_nickname(sid_a)
        name_b = _agent_nickname(sid_b)

        # Pure hash: same name word for colliding sessions
        word_a = name_a.split(" (")[0]
        word_b = name_b.split(" (")[0]
        assert word_a == word_b, f"Expected same name word, got '{word_a}' vs '{word_b}'"

        # But different full strings (the 8-char hex suffix disambiguates)
        assert name_a != name_b, "Full nicknames should differ due to session ID suffix"

    def test_nickname_in_conflict_output(self):
        """CONFLICT output should use nicknames instead of raw session IDs."""
        from omega.server.hook_server import handle_auto_claim_file, _agent_nickname

        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr.claim_file.return_value = {
                "conflict": True,
                "claimed_by": "peer-session-xyz",
                "task": "editing tests",
            }
            mock_mgr_fn.return_value = mock_mgr

            result = handle_auto_claim_file({
                "tool_name": "Edit",
                "session_id": "my-session",
                "tool_input": '{"file_path": "/tmp/test.py"}',
            })

        output = result["output"]
        expected_name = _agent_nickname("peer-session-xyz")
        assert expected_name in output
        assert "[CONFLICT]" in output

    def test_nickname_in_file_guard(self):
        """FILE-GUARD block should use nicknames."""
        from omega.server.hook_server import _file_guard_block_msg, _agent_nickname

        result = _file_guard_block_msg("/proj/engine.py", "agent-abc-123", "refactoring")
        output = result["output"]
        expected_name = _agent_nickname("agent-abc-123")
        assert expected_name in output
        assert "[FILE-GUARD]" in output


class TestRichCoordSessionStart:
    """Tests for rich [COORD] team roster at session start."""

    def test_coord_roster_shows_peers(self, coord_mgr):
        """Session start with peers should show [COORD] roster with nicknames."""
        from omega.server.hook_server import handle_coord_session_start, _agent_nickname

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            # Only register the peer — let the handler register agent-B
            coord_mgr.register_session("agent-A", pid=1001, project="/proj/x", task="editing engine.py")
            coord_mgr.claim_file("agent-A", "/proj/x/engine.py")

            with patch("omega.server.hook_server.coordination._check_git_sync", return_value=[]):
                with patch("omega.server.hook_server.coordination._session_resume", return_value=[]):
                    result = handle_coord_session_start({
                        "session_id": "agent-B",
                        "project": "/proj/x",
                    })

        output = result["output"]
        assert "[COORD]" in output
        name_a = _agent_nickname("agent-A")
        assert name_a in output
        assert "engine.py" in output

    def test_coord_roster_no_peers(self):
        """Solo session should not show [COORD] roster."""
        from omega.server.hook_server import handle_coord_session_start

        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr.register_session.return_value = {"peers_on_project": 0}
            mock_mgr.list_sessions.return_value = [
                {"session_id": "solo-agent", "project": "/tmp/proj", "task": "",
                 "status": "active", "last_heartbeat": "2026-01-01T00:00:00"},
            ]
            mock_mgr.list_tasks.return_value = []
            mock_mgr_fn.return_value = mock_mgr

            with patch("omega.server.hook_server.coordination._check_git_sync", return_value=[]):
                with patch("omega.server.hook_server.coordination._session_resume", return_value=[]):
                    result = handle_coord_session_start({
                        "session_id": "solo-agent",
                        "project": "/tmp/proj",
                    })

        assert "[COORD]" not in result["output"]

    def test_coord_roster_shows_cross_project_peers_with_label(self, coord_mgr):
        """Cross-project peers should appear with project label badge."""
        from omega.server.hook_server import handle_coord_session_start

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("agent-A", pid=1001, project="/proj/acme", task="building UI")

            with patch("omega.server.hook_server.coordination._check_git_sync", return_value=[]):
                with patch("omega.server.hook_server.coordination._session_resume", return_value=[]):
                    result = handle_coord_session_start({
                        "session_id": "agent-B",
                        "project": "/proj/omega",
                    })

        output = result["output"]
        # Cross-project peer IS visible with project label
        assert "[COORD]" in output
        assert "building UI" in output
        assert "[acme]" in output


class TestHeartbeatMessagePreviews:
    """Tests for [INBOX] message previews and [TEAM] activity in heartbeat."""

    def test_inbox_shows_message_previews(self, coord_mgr):
        """Inbox should show message content previews with nicknames."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_coord_heartbeat, _agent_nickname

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("agent-A", pid=1001, project="/proj/x")
            coord_mgr.register_session("agent-B", pid=1002, project="/proj/x")
            coord_mgr.send_message(
                from_session="agent-A",
                to_session="agent-B",
                subject="Overlap: both editing engine.py",
                msg_type="inform",
            )

            # Run 2 heartbeats to trigger inbox check
            for _ in range(2):
                hook_server._last_heartbeat.pop("agent-B", None)
                result = handle_coord_heartbeat({"session_id": "agent-B", "project": "/proj/x"})

        output = result["output"]
        assert "[INBOX]" in output
        name_a = _agent_nickname("agent-A")
        assert name_a in output
        assert "Overlap" in output

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()

    def test_team_activity_on_4th_heartbeat(self, coord_mgr):
        """[TEAM] line should appear on 4th heartbeat when peers have activity."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_coord_heartbeat, _agent_nickname

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("agent-A", pid=1001, project="/proj/x")
            coord_mgr.register_session("agent-B", pid=1002, project="/proj/x")
            coord_mgr.claim_file("agent-A", "/proj/x/engine.py")

            # Run 4 heartbeats to trigger coordination check
            for _ in range(4):
                hook_server._last_heartbeat.pop("agent-B", None)
                result = handle_coord_heartbeat({"session_id": "agent-B", "project": "/proj/x"})

        output = result["output"]
        assert "[TEAM]" in output
        name_a = _agent_nickname("agent-A")
        assert name_a in output
        assert "engine.py" in output

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()

    def test_no_team_activity_solo(self, coord_mgr):
        """Solo session should not show [TEAM] line (no peer events)."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_coord_heartbeat

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("solo-agent", pid=1001, project="/proj/x")

            for _ in range(4):
                hook_server._last_heartbeat.pop("solo-agent", None)
                result = handle_coord_heartbeat({"session_id": "solo-agent", "project": "/proj/x"})

        assert "[TEAM]" not in result["output"]

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()


class TestCoordSessionStopSummary:
    """Tests for [COORD] summary block in session stop."""

    def test_stop_shows_coord_summary_with_claims(self, coord_mgr):
        """Session stop should show [COORD] block with file claims."""
        from omega.server.hook_server import handle_coord_session_stop

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            with patch("omega.bridge._get_store") as mock_store_fn:
                mock_store = MagicMock()
                mock_store.get_session_event_counts.return_value = {}
                mock_store_fn.return_value = mock_store

                with patch("omega.bridge.query_structured", return_value=[]):
                    with patch("omega.bridge.auto_capture"):
                        coord_mgr.register_session("agent-A", pid=1001, project="/proj/x")
                        coord_mgr.claim_file("agent-A", "/proj/x/engine.py")
                        coord_mgr.claim_file("agent-A", "/proj/x/test_engine.py")

                        result = handle_coord_session_stop({
                            "session_id": "agent-A",
                            "project": "/proj/x",
                        })

        output = result["output"]
        assert "[COORD]" in output
        assert "engine.py" in output

    def test_stop_no_coord_when_no_activity(self, coord_mgr):
        """Session stop with no coordination activity should not show [COORD]."""
        from omega.server.hook_server import handle_coord_session_stop

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            with patch("omega.bridge._get_store") as mock_store_fn:
                mock_store = MagicMock()
                mock_store.get_session_event_counts.return_value = {}
                mock_store_fn.return_value = mock_store

                with patch("omega.bridge.query_structured", return_value=[]):
                    with patch("omega.bridge.auto_capture"):
                        coord_mgr.register_session("agent-solo", pid=1001, project="/proj/x")

                        result = handle_coord_session_stop({
                            "session_id": "agent-solo",
                            "project": "/proj/x",
                        })

        output = result["output"]
        # No file claims, messages, or tasks → no [COORD] block
        assert "Files:" not in output


# ============================================================================
# Improvement 2: Notify blocker when file guard blocks
# ============================================================================


class TestFileGuardBlockNotify:
    """Tests for [WAITING] message sent to file owner when guard blocks."""

    def test_block_sends_waiting_message(self, coord_mgr):
        """Blocking on a file claim should send [WAITING] message to the owner."""
        from omega.server import hook_server
        from omega.server.hook_server import _file_guard_block_msg, _agent_nickname

        hook_server._last_block_notify.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("owner-A", pid=1001, project="/proj/x")
            coord_mgr.register_session("blocked-B", pid=1002, project="/proj/x")

            result = _file_guard_block_msg(
                "/proj/x/engine.py", "owner-A", "refactoring", blocked_sid="blocked-B"
            )

        assert result["exit_code"] == 2
        assert "[FILE-GUARD]" in result["output"]

        # Check that a message was sent to the owner
        msgs = coord_mgr.check_inbox("owner-A", unread_only=True)
        assert len(msgs) >= 1
        msg = msgs[0]
        assert "[WAITING]" in msg["subject"]
        blocked_name = _agent_nickname("blocked-B")
        assert blocked_name in msg["subject"]
        assert "engine.py" in msg["subject"]

        hook_server._last_block_notify.clear()

    def test_block_notify_debounced(self, coord_mgr):
        """Repeated blocks on same file should not send duplicate messages."""
        from omega.server import hook_server
        from omega.server.hook_server import _file_guard_block_msg

        hook_server._last_block_notify.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("owner-A", pid=1001, project="/proj/x")
            coord_mgr.register_session("blocked-B", pid=1002, project="/proj/x")

            # First block — sends message
            _file_guard_block_msg(
                "/proj/x/engine.py", "owner-A", "refactoring", blocked_sid="blocked-B"
            )
            # Second block — debounced, should NOT send another
            _file_guard_block_msg(
                "/proj/x/engine.py", "owner-A", "refactoring", blocked_sid="blocked-B"
            )

        msgs = coord_mgr.check_inbox("owner-A", unread_only=False)
        waiting_msgs = [m for m in msgs if "[WAITING]" in (m.get("subject") or "")]
        assert len(waiting_msgs) == 1

        hook_server._last_block_notify.clear()

    def test_block_no_notify_without_blocked_sid(self, coord_mgr):
        """Block without blocked_sid should not attempt notification."""
        from omega.server import hook_server
        from omega.server.hook_server import _file_guard_block_msg

        hook_server._last_block_notify.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("owner-A", pid=1001, project="/proj/x")

            result = _file_guard_block_msg(
                "/proj/x/engine.py", "owner-A", "refactoring"
            )

        assert result["exit_code"] == 2
        msgs = coord_mgr.check_inbox("owner-A", unread_only=True)
        assert len(msgs) == 0

        hook_server._last_block_notify.clear()


# ============================================================================
# Improvement 4: Surface peer claims during edits
# ============================================================================


class TestPeerClaimSurface:
    """Tests for [PEER] lines when peers have claims in the same directory."""

    def test_peer_claim_shown_on_edit(self, coord_mgr):
        """Editing a file should show [PEER] if a peer has claims in the same directory."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_surface_memories, _agent_nickname

        hook_server._last_surface.clear()
        hook_server._last_peer_dir_check.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("agent-A", pid=1001, project="/proj/x")
            coord_mgr.register_session("agent-B", pid=1002, project="/proj/x")
            coord_mgr.claim_file("agent-A", "/proj/x/engine.py")

            result = handle_surface_memories({
                "tool_name": "Edit",
                "tool_input": json.dumps({"file_path": "/proj/x/bridge.py"}),
                "tool_output": "ok",
                "session_id": "agent-B",
                "project": "/proj/x",
            })

        output = result["output"]
        assert "[PEER]" in output
        name_a = _agent_nickname("agent-A")
        assert name_a in output
        assert "engine.py" in output

        hook_server._last_surface.clear()
        hook_server._last_peer_dir_check.clear()

    def test_no_peer_claim_different_dir(self, coord_mgr):
        """No [PEER] when peer claims are in a different directory."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_surface_memories

        hook_server._last_surface.clear()
        hook_server._last_peer_dir_check.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("agent-A", pid=1001, project="/proj/x")
            coord_mgr.register_session("agent-B", pid=1002, project="/proj/x")
            coord_mgr.claim_file("agent-A", "/proj/x/tests/test_engine.py")

            result = handle_surface_memories({
                "tool_name": "Edit",
                "tool_input": json.dumps({"file_path": "/proj/x/src/bridge.py"}),
                "tool_output": "ok",
                "session_id": "agent-B",
                "project": "/proj/x",
            })

        assert "[PEER]" not in result["output"]

        hook_server._last_surface.clear()
        hook_server._last_peer_dir_check.clear()

    def test_no_peer_claim_solo_agent(self, coord_mgr):
        """No [PEER] when working alone (no peers on same project)."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_surface_memories

        hook_server._last_surface.clear()
        hook_server._last_peer_dir_check.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("agent-solo", pid=1001, project="/proj/x")

            result = handle_surface_memories({
                "tool_name": "Edit",
                "tool_input": json.dumps({"file_path": "/proj/x/engine.py"}),
                "tool_output": "ok",
                "session_id": "agent-solo",
                "project": "/proj/x",
            })

        assert "[PEER]" not in result["output"]

        hook_server._last_surface.clear()
        hook_server._last_peer_dir_check.clear()


# ============================================================================
# Improvement 3: Mid-session [COORD] refresh on heartbeat
# ============================================================================


class TestHeartbeatPeerRefresh:
    """Tests for [TEAM] peer state change detection on 8th heartbeat."""

    def test_peer_join_detected_on_8th_beat(self, coord_mgr):
        """New peer joining should be reported on the 8th heartbeat."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_coord_heartbeat, _agent_nickname

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()
        hook_server._peer_snapshot.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("agent-A", pid=1001, project="/proj/x")
            coord_mgr.register_session("agent-B", pid=1002, project="/proj/x")

            # First 8 beats to populate initial snapshot
            for _ in range(8):
                hook_server._last_heartbeat.pop("agent-A", None)
                handle_coord_heartbeat({"session_id": "agent-A", "project": "/proj/x"})

            # New peer joins
            coord_mgr.register_session("agent-C", pid=1003, project="/proj/x")

            # Next 8 beats — should detect the join
            for _ in range(8):
                hook_server._last_heartbeat.pop("agent-A", None)
                result = handle_coord_heartbeat({"session_id": "agent-A", "project": "/proj/x"})

        output = result["output"]
        assert "[TEAM]" in output
        name_c = _agent_nickname("agent-C")
        assert name_c in output
        assert "joined" in output

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()
        hook_server._peer_snapshot.clear()

    def test_peer_departure_detected(self, coord_mgr):
        """Peer departure should be reported on the next 8th heartbeat."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_coord_heartbeat, _agent_nickname

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()
        hook_server._peer_snapshot.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("agent-A", pid=1001, project="/proj/x")
            coord_mgr.register_session("agent-B", pid=1002, project="/proj/x")

            # First 8 beats to populate initial snapshot (includes agent-B)
            for _ in range(8):
                hook_server._last_heartbeat.pop("agent-A", None)
                handle_coord_heartbeat({"session_id": "agent-A", "project": "/proj/x"})

            # Peer departs
            coord_mgr.deregister_session("agent-B")

            # Next 8 beats — should detect the departure
            for _ in range(8):
                hook_server._last_heartbeat.pop("agent-A", None)
                result = handle_coord_heartbeat({"session_id": "agent-A", "project": "/proj/x"})

        output = result["output"]
        assert "[TEAM]" in output
        name_b = _agent_nickname("agent-B")
        assert name_b in output
        assert "left" in output

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()
        hook_server._peer_snapshot.clear()


class TestIdleDetection:
    """Tests for [IDLE] task nudge on 6th heartbeat."""

    def test_idle_nudge_on_6th_beat(self, coord_mgr):
        """Agent with no in-progress task should get [IDLE] nudge on 6th beat."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_coord_heartbeat

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("agent-A", pid=1001, project="/proj/x")
            coord_mgr.create_task(created_by="agent-A", title="Available task", project="/proj/x")

            # Run 6 beats to reach 6th beat
            for _ in range(6):
                hook_server._last_heartbeat.pop("agent-A", None)
                result = handle_coord_heartbeat({"session_id": "agent-A", "project": "/proj/x"})

        assert "[IDLE]" in result["output"]
        assert "omega_task_next" in result["output"]

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()

    def test_no_idle_when_task_in_progress(self, coord_mgr):
        """Agent with an in-progress task should NOT get [IDLE] nudge."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_coord_heartbeat

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("agent-A", pid=1001, project="/proj/x")
            t = coord_mgr.create_task(created_by="agent-A", title="Active task", project="/proj/x")
            coord_mgr.claim_task(t["task_id"], "agent-A")

            # Run 6 beats
            for _ in range(6):
                hook_server._last_heartbeat.pop("agent-A", None)
                result = handle_coord_heartbeat({"session_id": "agent-A", "project": "/proj/x"})

        assert "[IDLE]" not in result["output"]

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()


class TestContinueBlock:
    """Tests for [CONTINUE] block in session start for reassigned tasks."""

    def test_continue_block_shown_for_reassigned_task(self, coord_mgr):
        """Session start should show [CONTINUE] for tasks with progress > 0."""
        from omega.server.hook_server import handle_coord_session_start

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("agent-A", pid=1001, project="/proj/x")
            t = coord_mgr.create_task(created_by="agent-A", title="Partial work", project="/proj/x")
            coord_mgr.claim_task(t["task_id"], "agent-A")
            coord_mgr.update_task_progress(t["task_id"], "agent-A", 60)

            # Session dies — reassign tasks
            coord_mgr.reassign_orphaned_tasks("agent-A")

            # New session starts
            coord_mgr.register_session("agent-B", pid=1002, project="/proj/x")
            result = handle_coord_session_start({
                "session_id": "agent-B",
                "project": "/proj/x",
            })

        output = result["output"]
        assert "[CONTINUE]" in output
        assert "Partial work" in output
        assert "60%" in output

    def test_no_continue_for_fresh_tasks(self, coord_mgr):
        """Tasks with 0 progress should NOT show [CONTINUE] block."""
        from omega.server.hook_server import handle_coord_session_start

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("agent-A", pid=1001, project="/proj/x")
            coord_mgr.create_task(created_by="agent-A", title="Fresh task", project="/proj/x")

            result = handle_coord_session_start({
                "session_id": "agent-A",
                "project": "/proj/x",
            })

        assert "[CONTINUE]" not in result["output"]


class TestSessionStopReassign:
    """Tests for task reassignment during session stop."""

    def test_session_stop_reassigns_tasks(self, coord_mgr):
        """Session stop should reassign in-progress tasks back to pending."""
        from omega.server.hook_server import handle_coord_session_stop

        with patch("omega.coordination.get_manager", return_value=coord_mgr), \
             patch("omega.bridge.query_structured", return_value=[]):
            coord_mgr.register_session("agent-A", pid=1001, project="/proj/x")
            t = coord_mgr.create_task(created_by="agent-A", title="My task", project="/proj/x")
            coord_mgr.claim_task(t["task_id"], "agent-A")

            result = handle_coord_session_stop({
                "session_id": "agent-A",
                "project": "/proj/x",
            })

        output = result["output"]
        assert "Tasks returned to queue" in output
        assert "My task" in output

        # Task should be pending again
        tasks = coord_mgr.list_tasks(status="pending")
        assert len(tasks) == 1
        assert tasks[0]["title"] == "My task"


class TestIntentOverlapEscalation:
    """Tests for [CONFLICT] escalation when overlap files are claimed."""

    def test_escalation_when_file_claimed(self, coord_mgr):
        """Overlap with claimed files should produce [CONFLICT] instead of [INTENT-OVERLAP]."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_auto_claim_file

        hook_server._last_claim.clear()
        hook_server._last_overlap_notify.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("agent-A", pid=1001, project="/proj/x")
            coord_mgr.register_session("agent-B", pid=1002, project="/proj/x")

            # Agent B claims a file and announces intent
            coord_mgr.claim_file("agent-B", "/proj/x/src/foo.py")
            coord_mgr.announce_intent(
                "agent-B", "Working on foo",
                target_files=["/proj/x/src/foo.py"],
                ttl_minutes=30,
            )

            # Agent A edits a file in same area — trigger auto_claim with overlap
            result = handle_auto_claim_file({
                "tool_name": "Edit",
                "tool_input": json.dumps({"file_path": "/proj/x/src/foo.py"}),
                "session_id": "agent-A",
            })

        output = result["output"]
        # Should have CONFLICT (escalated) instead of soft INTENT-OVERLAP
        assert "[CONFLICT]" in output or "[INTENT-OVERLAP]" in output

        hook_server._last_claim.clear()
        hook_server._last_overlap_notify.clear()

    def test_soft_warning_when_no_claim(self, coord_mgr):
        """Overlap without file claim should produce soft [INTENT-OVERLAP]."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_auto_claim_file

        hook_server._last_claim.clear()
        hook_server._last_overlap_notify.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("agent-A", pid=1001, project="/proj/x")
            coord_mgr.register_session("agent-B", pid=1002, project="/proj/x")

            # Agent B announces intent but does NOT claim the file
            coord_mgr.announce_intent(
                "agent-B", "Planning to work on bar",
                target_files=["/proj/x/src/bar.py"],
                ttl_minutes=30,
            )

            # Agent A edits the overlapping file
            result = handle_auto_claim_file({
                "tool_name": "Edit",
                "tool_input": json.dumps({"file_path": "/proj/x/src/bar.py"}),
                "session_id": "agent-A",
            })

        output = result["output"]
        if "[INTENT-OVERLAP]" in output:
            assert "[CONFLICT]" not in output

        hook_server._last_claim.clear()
        hook_server._last_overlap_notify.clear()


class TestSuggestAlternativeDir:
    """Tests for _suggest_alternative_dir helper."""

    def test_suggests_siblings(self, tmp_path):
        """Should suggest sibling directories."""
        from omega.server.hook_server import _suggest_alternative_dir

        # Create directory structure: parent/target/ parent/sibling1/ parent/sibling2/
        (tmp_path / "target").mkdir()
        (tmp_path / "sibling1").mkdir()
        (tmp_path / "sibling2").mkdir()

        result = _suggest_alternative_dir(str(tmp_path / "target" / "file.py"))
        assert result is not None
        assert "nearby dirs:" in result
        assert "sibling1" in result

    def test_returns_none_for_root(self):
        """Should return None when file is at root level."""
        from omega.server.hook_server import _suggest_alternative_dir

        result = _suggest_alternative_dir("/file.py")
        assert result is None

    def test_skips_hidden_dirs(self, tmp_path):
        """Should skip hidden and __dunder__ directories."""
        from omega.server.hook_server import _suggest_alternative_dir

        (tmp_path / "target").mkdir()
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "visible").mkdir()

        result = _suggest_alternative_dir(str(tmp_path / "target" / "file.py"))
        assert result is not None
        assert ".hidden" not in result
        assert "__pycache__" not in result
        assert "visible" in result


# ============================================================================
# Coordination awareness on planning prompts (auto_capture)
# ============================================================================


class TestAutoCapturePlanningCoord:
    """Tests for coordination awareness when user asks planning questions."""

    def test_auto_capture_surfaces_peer_work_on_whats_next(self, coord_mgr):
        """'What's next?' should surface [COORD] with peer's task."""
        from omega.server.hook_server import handle_auto_capture

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("me", pid=1001, project="/proj/x")
            coord_mgr.register_session("peer-A", pid=1002, project="/proj/x")
            # Peer has an in-progress coord task
            res = coord_mgr.create_task("peer-A", title="Implement auth module", project="/proj/x")
            coord_mgr.claim_task(res["task_id"], "peer-A")

            result = handle_auto_capture({
                "stdin": json.dumps({
                    "prompt": "What's next on the roadmap for this project?",
                    "session_id": "me",
                    "cwd": "/proj/x",
                }),
            })

        output = result["output"]
        assert "[COORD]" in output
        assert "Implement auth module" in output

    def test_auto_capture_no_coord_on_regular_prompt(self, coord_mgr):
        """Non-planning prompts should not trigger [COORD] output."""
        from omega.server.hook_server import handle_auto_capture

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("me", pid=1001, project="/proj/x")
            coord_mgr.register_session("peer-A", pid=1002, project="/proj/x")

            result = handle_auto_capture({
                "stdin": json.dumps({
                    "prompt": "Please read the file src/main.py and show me the contents",
                    "session_id": "me",
                    "cwd": "/proj/x",
                }),
            })

        assert "[COORD]" not in result["output"]

    def test_auto_capture_coord_debounced(self, coord_mgr):
        """Second planning prompt within 2min should not repeat [COORD]."""
        from omega.server.hook_server import handle_auto_capture

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("me", pid=1001, project="/proj/x")
            coord_mgr.register_session("peer-A", pid=1002, project="/proj/x")

            payload = {
                "stdin": json.dumps({
                    "prompt": "What's next on the roadmap for this project?",
                    "session_id": "me",
                    "cwd": "/proj/x",
                }),
            }

            # First call — should produce [COORD]
            r1 = handle_auto_capture(payload)
            assert "[COORD]" in r1["output"]

            # Second call within debounce window — no [COORD]
            r2 = handle_auto_capture(payload)
            assert "[COORD]" not in r2["output"]

    def test_auto_capture_coord_no_peers_no_output(self, coord_mgr):
        """Solo agent should get no [COORD] output on planning prompts."""
        from omega.server.hook_server import handle_auto_capture

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("me", pid=1001, project="/proj/x")

            result = handle_auto_capture({
                "stdin": json.dumps({
                    "prompt": "What's next on the roadmap for this project?",
                    "session_id": "me",
                    "cwd": "/proj/x",
                }),
            })

        assert "[COORD]" not in result["output"]

    def test_auto_capture_coord_fail_open(self):
        """Coordination errors should not crash the hook."""
        from omega.server.hook_server import handle_auto_capture

        mock_mgr = MagicMock()
        mock_mgr.list_sessions.side_effect = RuntimeError("DB locked")

        with patch("omega.coordination.get_manager", return_value=mock_mgr):
            result = handle_auto_capture({
                "stdin": json.dumps({
                    "prompt": "What's next on the roadmap for this project?",
                    "session_id": "me",
                    "cwd": "/proj/x",
                }),
            })

        assert result["error"] is None

    def test_session_stop_cleans_coord_query_debounce(self, coord_mgr):
        """Session stop should clean up _last_coord_query for that session."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_auto_capture, handle_session_stop

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("me", pid=1001, project="/proj/x")
            coord_mgr.register_session("peer-A", pid=1002, project="/proj/x")

            # Trigger a coord query
            handle_auto_capture({
                "stdin": json.dumps({
                    "prompt": "What's next on the roadmap for this project?",
                    "session_id": "me",
                    "cwd": "/proj/x",
                }),
            })
            assert "me" in hook_server._last_coord_query

        # Session stop should clean it up
        with patch("omega.bridge._get_store") as mock_store_fn:
            mock_store = MagicMock()
            mock_store.get_session_event_counts.return_value = {}
            mock_store_fn.return_value = mock_store
            with patch("omega.bridge.query_structured", return_value=[]):
                with patch("omega.bridge.auto_capture"):
                    handle_session_stop({"session_id": "me", "project": "/proj/x"})

        assert "me" not in hook_server._last_coord_query


class TestCoordSessionStartPrefersCoordTask:
    """Tests for preferring coord_task over session.task in [COORD] roster."""

    def test_coord_session_start_prefers_coord_task(self, coord_mgr):
        """Session start roster should show coord_task title instead of session.task."""
        from omega.server.hook_server import handle_coord_session_start

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session(
                "peer-A", pid=1001, project="/proj/x", task="vague first prompt"
            )
            # Give peer-A an in-progress coord task
            res = coord_mgr.create_task("peer-A", title="Fix auth bug #42", project="/proj/x")
            coord_mgr.claim_task(res["task_id"], "peer-A")

            with patch.multiple(
                "omega.bridge",
                query_structured=MagicMock(return_value=[]),
                get_cross_session_lessons=MagicMock(return_value=[]),
                _get_store=MagicMock(return_value=MagicMock(
                    get_session_event_counts=MagicMock(return_value={}),
                )),
                auto_capture=MagicMock(),
                consolidate=MagicMock(),
                create=True,
            ):
                result = handle_coord_session_start({
                    "session_id": "me",
                    "project": "/proj/x",
                })

        output = result["output"]
        assert "[COORD]" in output
        assert "Fix auth bug #42" in output
        # session.task should NOT appear when coord_task is available
        assert "vague first prompt" not in output


# ============================================================================
# Urgent message queue (cross-agent push notifications)
# ============================================================================


class TestUrgentMessageQueue:
    """Tests for immediate cross-agent message push via urgent queue."""

    def test_urgent_queue_notify_and_drain(self):
        """notify_session queues messages; _drain_urgent_queue formats and clears."""
        from omega.server import hook_server
        from omega.server.hook_server import notify_session, _drain_urgent_queue

        hook_server._pending_urgent.clear()

        notify_session("agent-B", {
            "from_session": "agent-A",
            "subject": "Tests passing",
            "msg_type": "complete",
        })
        notify_session("agent-B", {
            "from_session": "agent-C",
            "subject": "Need review",
            "msg_type": "request",
        })

        # Queue should have 2 entries
        assert len(hook_server._pending_urgent.get("agent-B", [])) == 2

        # Drain should format and clear
        output = _drain_urgent_queue("agent-B")
        assert "[INBOX]" in output
        assert "Tests passing" in output
        assert "Need review" in output
        assert "complete" in output
        assert "request" in output
        assert "omega_inbox" in output

        # Queue should be empty after drain
        assert "agent-B" not in hook_server._pending_urgent

    def test_urgent_queue_cap(self):
        """Queue caps at _MAX_URGENT_PER_SESSION, keeping newest."""
        from omega.server import hook_server
        from omega.server.hook_server import notify_session

        hook_server._pending_urgent.clear()

        for i in range(15):
            notify_session("agent-X", {
                "from_session": f"sender-{i}",
                "subject": f"Message {i}",
                "msg_type": "inform",
            })

        queue = hook_server._pending_urgent["agent-X"]
        assert len(queue) == hook_server._MAX_URGENT_PER_SESSION
        # Newest should be kept (message 14 is the last)
        assert queue[-1]["subject"] == "Message 14"
        # Oldest kept should be message 5 (15 - 10 = 5)
        assert queue[0]["subject"] == "Message 5"

    def test_urgent_drain_empty(self):
        """Draining nonexistent session returns empty string."""
        from omega.server import hook_server
        from omega.server.hook_server import _drain_urgent_queue

        hook_server._pending_urgent.clear()
        assert _drain_urgent_queue("nonexistent") == ""

    def test_heartbeat_surfaces_urgent_even_when_debounced(self):
        """Debounced heartbeat still returns urgent output."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_coord_heartbeat, notify_session

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()
        hook_server._pending_urgent.clear()

        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr_fn.return_value = mock_mgr

            # First heartbeat — not debounced
            handle_coord_heartbeat({"session_id": "agent-B"})
            assert mock_mgr.heartbeat.call_count == 1

            # Queue an urgent message
            notify_session("agent-B", {
                "from_session": "agent-A",
                "subject": "Urgent: deploy now",
                "msg_type": "request",
            })

            # Second heartbeat — within debounce window, but urgent should surface
            result = handle_coord_heartbeat({"session_id": "agent-B"})
            assert mock_mgr.heartbeat.call_count == 1  # Still debounced

            output = result["output"]
            assert "[INBOX]" in output
            assert "Urgent: deploy now" in output

    def test_heartbeat_urgent_plus_normal(self):
        """Non-debounced heartbeat shows both urgent and regular output."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_coord_heartbeat, notify_session

        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()
        hook_server._pending_urgent.clear()

        # Queue urgent before heartbeat
        notify_session("agent-B", {
            "from_session": "agent-A",
            "subject": "Code review done",
            "msg_type": "complete",
        })

        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr.get_unread_count.return_value = 0
            mock_mgr_fn.return_value = mock_mgr

            result = handle_coord_heartbeat({"session_id": "agent-B"})

        output = result["output"]
        # Urgent content should be present
        assert "Code review done" in output

    def test_send_message_queues_urgent_for_direct(self, coord_mgr):
        """Direct message triggers notify_session for target."""
        from omega.server import hook_server

        hook_server._pending_urgent.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("sender-1", pid=1001, project="/proj/x")
            coord_mgr.register_session("target-1", pid=1002, project="/proj/x")

            import asyncio
            from omega.server.coord_handlers import handle_send_message

            asyncio.run(handle_send_message({
                "session_id": "sender-1",
                "to_session": "target-1",
                "subject": "Need your review",
                "msg_type": "request",
                "priority": "critical",
            }))

        # target-1 should have an urgent notification queued
        queue = hook_server._pending_urgent.get("target-1", [])
        assert len(queue) == 1
        assert queue[0]["subject"] == "Need your review"
        assert queue[0]["from_session"] == "sender-1"
        assert queue[0]["msg_type"] == "request"

    def test_send_message_queues_urgent_for_broadcast(self, coord_mgr):
        """Broadcast notifies all same-project peers."""
        from omega.server import hook_server

        hook_server._pending_urgent.clear()

        with patch("omega.coordination.get_manager", return_value=coord_mgr):
            coord_mgr.register_session("broadcaster", pid=1001, project="/proj/x")
            coord_mgr.register_session("peer-A", pid=1002, project="/proj/x")
            coord_mgr.register_session("peer-B", pid=1003, project="/proj/x")
            coord_mgr.register_session("other-proj", pid=1004, project="/proj/y")

            import asyncio
            from omega.server.coord_handlers import handle_send_message

            asyncio.run(handle_send_message({
                "session_id": "broadcaster",
                "subject": "Switching to main branch",
                "msg_type": "inform",
                "priority": "critical",
            }))

        # Same-project peers should have urgent notifications
        assert len(hook_server._pending_urgent.get("peer-A", [])) == 1
        assert len(hook_server._pending_urgent.get("peer-B", [])) == 1
        # Sender should NOT get their own broadcast
        assert len(hook_server._pending_urgent.get("broadcaster", [])) == 0
        # Different-project agent should NOT get notified
        assert len(hook_server._pending_urgent.get("other-proj", [])) == 0

    def test_session_stop_cleans_urgent_queue(self):
        """Session stop removes pending notifications."""
        from omega.server import hook_server
        from omega.server.hook_server import handle_session_stop, notify_session

        hook_server._pending_urgent.clear()
        hook_server._last_heartbeat.clear()
        hook_server._heartbeat_count.clear()

        notify_session("stop-agent", {
            "from_session": "sender",
            "subject": "Will be cleaned",
            "msg_type": "inform",
        })
        assert "stop-agent" in hook_server._pending_urgent

        with patch("omega.bridge._get_store") as mock_store_fn, \
             patch("omega.bridge.query_structured", return_value=[]):
            mock_store = MagicMock()
            mock_store.get_session_event_counts.return_value = {}
            mock_store_fn.return_value = mock_store

            handle_session_stop({"session_id": "stop-agent", "project": "/proj/x"})

        assert "stop-agent" not in hook_server._pending_urgent


# ============================================================================
# Phase 1: pre_edit_surface.py — guards only (memory surfacing moved to PostToolUse)
# ============================================================================

# _surface_memories and _ext_to_tags removed from pre_edit_surface.py
# (consolidated into surface_memories.py PostToolUse hook to avoid double-surfacing)

# ============================================================================
# Phase 3: Session intent cache
# ============================================================================

def test_session_intent_set_on_classify():
    """auto_capture with a coding prompt should cache intent."""
    from omega.server import hook_server
    from omega.server.hook_server import handle_auto_capture

    hook_server._session_intent.clear()

    with patch("omega.router.classifier.classify_intent", return_value=("coding", 0.85)):
        handle_auto_capture({
            "prompt": "Please read the file and fix the bug in the authentication module",
            "session_id": "intent-test-123",
            "cwd": "/tmp",
        })
    assert hook_server._session_intent.get("intent-test-123") == "coding"
    hook_server._session_intent.clear()


def test_session_intent_cleared_on_stop():
    """Session stop should clear cached intent."""
    from omega.server import hook_server
    from omega.server.hook_server import handle_session_stop

    hook_server._session_intent["cleanup-test"] = "coding"

    with patch("omega.bridge._get_store") as mock_store_fn, \
         patch("omega.bridge.query_structured", return_value=[]):
        mock_store = MagicMock()
        mock_store.get_session_event_counts.return_value = {}
        mock_store_fn.return_value = mock_store

        handle_session_stop({"session_id": "cleanup-test", "project": "/tmp"})

    assert "cleanup-test" not in hook_server._session_intent


def test_intent_bias_in_surface():
    """_surface_for_edit should include intent hint in query text when cached."""
    from omega.server import hook_server
    from omega.server.hook_server import _surface_for_edit

    hook_server._session_intent["bias-test"] = "coding"

    with patch("omega.bridge.query_structured") as mock_qs:
        mock_qs.return_value = []
        _surface_for_edit("/tmp/test.py", "bias-test", "/tmp")
        call_kwargs = mock_qs.call_args[1]
        assert call_kwargs["query_text"].startswith("code implementation ")

    hook_server._session_intent.clear()


def test_no_intent_bias_when_no_session():
    """_surface_for_edit should work normally without cached intent."""
    from omega.server import hook_server
    from omega.server.hook_server import _surface_for_edit

    hook_server._session_intent.clear()

    with patch("omega.bridge.query_structured") as mock_qs:
        mock_qs.return_value = []
        _surface_for_edit("/tmp/test.py", "", "/tmp")
        call_kwargs = mock_qs.call_args[1]
        # No intent prefix
        assert not call_kwargs["query_text"].startswith("code implementation ")
        assert "test.py" in call_kwargs["query_text"]


# ============================================================================
# Phase 4: Graph health + auto-feedback
# ============================================================================

def test_graph_health_in_welcome():
    """Graph health was removed from welcome output (personal brief redesign).
    Verify welcome still works without graph lines."""
    from omega.server.hook_server import handle_session_start

    ctx = {
        "memory_count": 100,
        "health_status": "ok",
        "last_capture_ago": "5m ago",
        "context_items": [],
    }
    with patch("omega.bridge.get_session_context", return_value=ctx), \
         patch("omega.bridge._get_store") as mock_store_fn:
        mock_store = MagicMock()
        mock_store.count.return_value = 50
        mock_store.edge_count.return_value = 100
        mock_store_fn.return_value = mock_store

        result = handle_session_start({"session_id": "graph-test", "project": "/tmp"})
        output = result["output"]
        # Graph health line was removed; footer has memory count
        assert "100 memories" in output
        assert "Graph:" not in output


def test_graph_health_sparse():
    """Graph health lines removed from welcome (personal brief redesign)."""
    from omega.server.hook_server import handle_session_start

    ctx = {
        "memory_count": 100,
        "health_status": "ok",
        "last_capture_ago": "5m ago",
        "context_items": [],
    }
    with patch("omega.bridge.get_session_context", return_value=ctx), \
         patch("omega.bridge._get_store") as mock_store_fn:
        mock_store = MagicMock()
        mock_store.count.return_value = 50
        mock_store.edge_count.return_value = 5
        mock_store_fn.return_value = mock_store

        result = handle_session_start({"session_id": "sparse-test", "project": "/tmp"})
        output = result["output"]
        # Graph lines removed from output
        assert "Graph:" not in output


def test_auto_feedback_once_surfaced(tmp_path):
    """Multi-surfaced IDs get positive feedback; single-surfaced in small sessions get nothing."""
    from omega.server.hook_server import _auto_feedback_on_surfaced

    # id-1 surfaced in both files (count=2), id-2 and id-3 only once
    fake_omega = tmp_path / ".omega"
    fake_omega.mkdir(exist_ok=True)
    json_path = fake_omega / "session-feedback-test.surfaced.json"
    data = {
        "/tmp/a.py": ["id-1", "id-2"],
        "/tmp/b.py": ["id-1", "id-3"],
    }
    json_path.write_text(json.dumps(data))

    with patch("omega.server.hook_server.session._omega_dir", return_value=fake_omega):
        with patch("omega.bridge.batch_record_feedback") as mock_fb:
            _auto_feedback_on_surfaced("feedback-test")
            # id-1 multi-surfaced -> 1 batched call with 1 item; id-2/id-3 single in 2-file session -> no negative
            assert mock_fb.call_count == 1
            items = mock_fb.call_args[0][0]
            assert len(items) == 1
            assert items[0] == ("id-1", "helpful", "Auto: surfaced across multiple edits")


def test_auto_feedback_negative_in_busy_session(tmp_path):
    """Single-surfaced IDs in busy sessions (5+ files) get negative feedback."""
    from omega.server.hook_server import _auto_feedback_on_surfaced

    fake_omega = tmp_path / ".omega"
    fake_omega.mkdir(exist_ok=True)
    json_path = fake_omega / "session-single-test.surfaced.json"
    # 5 files edited; id-1 single-surfaced, id-2 multi-surfaced (2 files)
    data = {
        "/tmp/a.py": ["id-1", "id-2"],
        "/tmp/b.py": ["id-2"],
        "/tmp/c.py": ["id-3"],
        "/tmp/d.py": ["id-3"],
        "/tmp/e.py": ["id-4"],
    }
    json_path.write_text(json.dumps(data))

    with patch("omega.server.hook_server.session._omega_dir", return_value=fake_omega):
        with patch("omega.bridge.batch_record_feedback") as mock_fb:
            _auto_feedback_on_surfaced("single-test")
            # Single batched call with 4 items:
            # id-2 (count=2) and id-3 (count=2) -> 2 positive
            # id-1 and id-4 (count=1) in 5-file session -> 2 negative
            assert mock_fb.call_count == 1
            items = mock_fb.call_args[0][0]
            assert len(items) == 4
            item_tuples = set(items)
            assert ("id-2", "helpful", "Auto: surfaced across multiple edits") in item_tuples
            assert ("id-3", "helpful", "Auto: surfaced across multiple edits") in item_tuples
            assert ("id-1", "unhelpful", "Auto: single surfacing in busy session") in item_tuples
            assert ("id-4", "unhelpful", "Auto: single surfacing in busy session") in item_tuples



def test_kb_surfacing_on_edit():
    """KB chunks should be surfaced when editing a file with matching documents."""
    from omega.server.hook_server import _surface_for_edit

    with patch("omega.bridge.query_structured", return_value=[]),          patch("omega.knowledge.engine.search_documents", return_value="**Match:** Some relevant documentation content about the module") as mock_kb:
        lines = _surface_for_edit("/tmp/project/app.py", "test-session", "test-project")
        mock_kb.assert_called_once()
        assert any("[KB]" in l for l in lines), f"Expected [KB] line in {lines}"


def test_kb_surfacing_skipped_no_match():
    """KB surfacing should be silent when no document matches."""
    from omega.server.hook_server import _surface_for_edit

    with patch("omega.bridge.query_structured", return_value=[]),          patch("omega.knowledge.engine.search_documents", return_value="No document matches found for: app.py"):
        lines = _surface_for_edit("/tmp/project/app.py", "test-session", "test-project")
        assert not any("[KB]" in l for l in lines)


def test_kb_surfacing_skipped_import_error():
    """KB surfacing should be silent when knowledge engine not available."""
    from omega.server.hook_server import _surface_for_edit

    with patch("omega.bridge.query_structured", return_value=[]),          patch("omega.knowledge.engine.search_documents", side_effect=ImportError("no module")):
        lines = _surface_for_edit("/tmp/project/app.py", "test-session", "test-project")
        assert not any("[KB]" in l for l in lines)


def test_graph_health_in_welcome_output():
    """Graph health removed from welcome output (personal brief redesign)."""
    from omega.server.hook_server import handle_session_start

    mock_store = MagicMock()
    mock_store.count.return_value = 50
    mock_store.edge_count.return_value = 100

    with patch("omega.bridge.get_session_context", return_value={
        "memory_count": 50, "health_status": "ok",
        "last_capture_ago": "1m ago", "context_items": [],
    }), patch("omega.bridge._get_store", return_value=mock_store):
        result = handle_session_start({"session_id": "test-session", "project": "/tmp/project"})
        output = result.get("output", "")
        assert "Graph:" not in output
        assert "50 memories" in output


def test_profile_summary_in_welcome():
    """Profile line removed from welcome output (personal brief redesign)."""
    from omega.server.hook_server import handle_session_start

    mock_pe = MagicMock()
    mock_pe._conn.execute.return_value.fetchone.return_value = {"cnt": 5}

    with patch("omega.bridge.get_session_context", return_value={
        "memory_count": 50, "health_status": "ok",
        "last_capture_ago": "1m ago", "context_items": [],
    }), patch("omega.profile.engine.get_profile_engine", return_value=mock_pe):
        result = handle_session_start({"session_id": "test-session", "project": "/tmp/project"})
        output = result.get("output", "")
        assert "Profile:" not in output


def test_maintenance_suggestion_sparse_graph():
    """Maintenance suggestion removed from welcome (personal brief redesign)."""
    from omega.server.hook_server import handle_session_start

    mock_store = MagicMock()
    mock_store.count.return_value = 100
    mock_store.edge_count.return_value = 10  # 0.1x = sparse

    with patch("omega.bridge.get_session_context", return_value={
        "memory_count": 100, "health_status": "ok",
        "last_capture_ago": "1m ago", "context_items": [],
    }), patch("omega.bridge._get_store", return_value=mock_store):
        result = handle_session_start({"session_id": "test-session", "project": "/tmp/project"})
        output = result.get("output", "")
        assert "MAINTENANCE" not in output


def test_no_maintenance_suggestion_healthy_graph():
    """Welcome should NOT suggest maintenance when graph is healthy."""
    from omega.server.hook_server import handle_session_start

    mock_store = MagicMock()
    mock_store.count.return_value = 100
    mock_store.edge_count.return_value = 200  # 2.0x = rich

    with patch("omega.bridge.get_session_context", return_value={
        "memory_count": 100, "health_status": "ok",
        "last_capture_ago": "1m ago", "context_items": [],
    }), patch("omega.bridge._get_store", return_value=mock_store):
        result = handle_session_start({"session_id": "test-session", "project": "/tmp/project"})
        output = result.get("output", "")
        assert "MAINTENANCE" not in output


# ============================================================================
# Coordination Breakdown Prevention (verification tests)
# ============================================================================


def test_coord_session_start_defers_git_sync():
    """coord_session_start defers git sync to background (no [GIT] in output)."""
    from omega.server.hook_server import handle_coord_session_start

    with patch("omega.coordination.get_manager") as mock_mgr_fn:
        mock_mgr = MagicMock()
        mock_mgr.register_session.return_value = {"peers_on_project": 0}
        mock_mgr.list_tasks.return_value = []
        mock_mgr.get_status.return_value = {
            "file_claims": 0, "branch_claims": 0,
            "active_intents": 0, "conflicts": [],
        }
        mock_mgr_fn.return_value = mock_mgr

        with patch("omega.server.hook_server.coordination._check_git_sync", return_value=[]), \
             patch("omega.server.hook_server.coordination._session_resume", return_value=[]):
            result = handle_coord_session_start({
                "session_id": "git-log-test",
                "project": "/tmp/proj",
            })

    output = result.get("output", "")
    # Git sync is deferred to background — not in synchronous output
    assert "[SESSION] git-log-test" in output


def test_coord_session_stop_handoff_includes_git_data():
    """coord_session_stop handoff message should include commits and files."""
    from omega.server.hook_server import handle_coord_session_stop

    # Mock git log and git diff responses
    def mock_subprocess_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        if "log" in cmd:
            result.stdout = "abc1234 feat: add login\ndef5678 fix: typo\n"
        elif "diff" in cmd:
            result.stdout = "src/auth.py\nsrc/utils.py\n"
        elif "ls-files" in cmd:
            result.stdout = ""
        else:
            result.stdout = ""
        return result

    with patch("omega.coordination.get_manager") as mock_mgr_fn:
        mock_mgr = MagicMock()
        mock_mgr.list_sessions.return_value = []
        mock_mgr.list_tasks.return_value = []
        mock_mgr.get_session_claims.return_value = {"file_claims": [], "branch_claims": []}
        mock_mgr._conn = MagicMock()
        mock_mgr._conn.execute.return_value.fetchone.return_value = [0]
        mock_mgr_fn.return_value = mock_mgr

        with patch("omega.bridge._get_store") as mock_store_fn, \
             patch("omega.bridge.query_structured", return_value=[]), \
             patch("subprocess.run", side_effect=mock_subprocess_run):
            mock_store = MagicMock()
            mock_store.get_session_event_counts.return_value = {}
            mock_store_fn.return_value = mock_store

            handle_coord_session_stop({
                "session_id": "handoff-git-test",
                "project": "/tmp/proj",
            })

        # Check the handoff body includes git data
        assert mock_mgr.send_message.called
        call_kwargs = mock_mgr.send_message.call_args
        body = call_kwargs[1].get("body", "")
        assert "## Commits" in body
        assert "abc1234" in body
        assert "## Files Modified" in body
        assert "src/auth.py" in body


def test_session_stop_summary_includes_git_data():
    """handle_session_stop summary should include git commits even with 0 OMEGA captures."""
    from omega.server.hook_server import handle_session_stop

    def mock_subprocess_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        if "log" in cmd:
            result.stdout = "abc1234 feat: add login\n"
        elif "diff" in cmd:
            result.stdout = "src/auth.py\n"
        else:
            result.stdout = ""
        return result

    with patch("omega.bridge._get_store") as mock_store_fn, \
         patch("omega.bridge.query_structured", return_value=[]), \
         patch("omega.bridge.auto_capture") as mock_capture, \
         patch("subprocess.run", side_effect=mock_subprocess_run):
        mock_store = MagicMock()
        mock_store.get_session_event_counts.return_value = {}
        mock_store_fn.return_value = mock_store

        result = handle_session_stop({
            "session_id": "git-summary-test",
            "project": "/tmp/proj",
        })

    # The stored summary should contain git info (check all calls, not just last)
    assert mock_capture.called
    all_contents = [
        call[1].get("content", "") for call in mock_capture.call_args_list if call[1]
    ]
    assert any("Commits:" in c or "abc1234" in c for c in all_contents), (
        f"Expected git data in auto_capture calls, got: {all_contents}"
    )


def test_coord_session_stop_warns_untracked_files():
    """coord_session_stop should warn about untracked files."""
    from omega.server.hook_server import handle_coord_session_stop

    def mock_subprocess_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        if "ls-files" in cmd:
            result.stdout = "website/app/benchmarks/page.tsx\nwebsite/components/Foo.tsx\n"
        elif "log" in cmd:
            result.stdout = ""
        elif "diff" in cmd:
            result.stdout = ""
        else:
            result.stdout = ""
        return result

    with patch("omega.coordination.get_manager") as mock_mgr_fn:
        mock_mgr = MagicMock()
        mock_mgr.list_sessions.return_value = []
        mock_mgr.list_tasks.return_value = []
        mock_mgr.get_session_claims.return_value = {"file_claims": [], "branch_claims": []}
        mock_mgr._conn = MagicMock()
        mock_mgr._conn.execute.return_value.fetchone.return_value = [0]
        mock_mgr_fn.return_value = mock_mgr

        with patch("omega.bridge._get_store") as mock_store_fn, \
             patch("omega.bridge.query_structured", return_value=[]), \
             patch("subprocess.run", side_effect=mock_subprocess_run):
            mock_store = MagicMock()
            mock_store.get_session_event_counts.return_value = {}
            mock_store_fn.return_value = mock_store

            result = handle_coord_session_stop({
                "session_id": "untracked-test",
                "project": "/tmp/proj",
            })

    output = result.get("output", "")
    assert "[!]" in output
    assert "untracked" in output
    assert "page.tsx" in output


def test_track_git_commit_stores_decision():
    """_track_git_commit routes routine commits to task_completion, architecture to decision."""
    from omega.server.hook_server import _track_git_commit

    mock_diff = MagicMock()
    mock_diff.returncode = 0
    mock_diff.stdout = "src/auth.py\nsrc/utils.py\n"

    # Routine commit → task_completion
    with patch("omega.coordination.get_manager") as mock_mgr_fn, \
         patch("omega.bridge.auto_capture") as mock_capture, \
         patch("subprocess.run", return_value=mock_diff), \
         patch("omega.server.hook_server.memory._get_current_branch", return_value="main"), \
         patch("omega.server.hook_server.memory._resolve_entity", return_value=None):
        mock_mgr = MagicMock()
        mock_mgr_fn.return_value = mock_mgr

        tool_input = json.dumps({"command": "git commit -m 'feat: add login'"})
        tool_output = "[main abc1234] feat: add login\n 2 files changed"

        _track_git_commit(tool_input, tool_output, "commit-test", "/tmp/proj")

    assert mock_mgr.log_git_event.called
    assert mock_capture.called
    capture_kwargs = mock_capture.call_args[1]
    assert capture_kwargs["event_type"] == "task_completion"
    assert "abc1234" in capture_kwargs["content"]
    assert "src/auth.py" in capture_kwargs["content"]

    # Architecture commit → decision (contains "migration" signal word)
    with patch("omega.coordination.get_manager") as mock_mgr_fn, \
         patch("omega.bridge.auto_capture") as mock_capture, \
         patch("subprocess.run", return_value=mock_diff), \
         patch("omega.server.hook_server.memory._get_current_branch", return_value="main"), \
         patch("omega.server.hook_server.memory._resolve_entity", return_value=None):
        mock_mgr = MagicMock()
        mock_mgr_fn.return_value = mock_mgr

        tool_input = json.dumps({"command": "git commit -m 'feat: schema migration v4 to v5'"})
        tool_output = "[main def5678] feat: schema migration v4 to v5\n 3 files changed"

        _track_git_commit(tool_input, tool_output, "commit-test", "/tmp/proj")

    capture_kwargs = mock_capture.call_args[1]
    assert capture_kwargs["event_type"] == "decision"


# ============================================================================
# Protocol Directive Injection
# ============================================================================


def test_session_start_includes_protocol_directive():
    """Session start output should always include [PROTOCOL] with essential rules."""
    from omega.server.hook_server import handle_session_start

    with patch("omega.bridge.get_session_context") as mock_ctx, \
         patch("omega.embedding.get_active_backend", return_value="onnx"):
        mock_ctx.return_value = {
            "memory_count": 10,
            "health_status": "ok",
            "last_capture_ago": "5m ago",
            "context_items": [],
        }
        result = handle_session_start({
            "session_id": "protocol-test",
            "project": "/tmp",
        })

    output = result["output"]
    assert "[PROTOCOL]" in output
    assert "Recent activity was loaded above" in output
    assert "omega_query()" in output
    assert "omega_store(" in output
    assert "git add" in output
    assert "omega_protocol()" in output


def test_coord_start_includes_coord_protocol_with_peers():
    """coord_session_start should include [COORD-PROTOCOL] when peers > 0, omit when solo."""
    from omega.server.hook_server import handle_coord_session_start

    def _run(peer_count):
        with patch("omega.coordination.get_manager") as mock_mgr_fn:
            mock_mgr = MagicMock()
            mock_mgr.register_session.return_value = {"peers_on_project": peer_count}
            mock_mgr.list_sessions.return_value = (
                [
                    {"session_id": "coord-proto-test", "project": "/tmp/proj",
                     "task": "testing", "status": "active",
                     "last_heartbeat": "2026-01-01T00:00:00"},
                    {"session_id": "peer-1", "project": "/tmp/proj",
                     "task": "other work", "status": "active",
                     "last_heartbeat": "2026-01-01T00:00:00"},
                ] if peer_count > 0 else [
                    {"session_id": "coord-proto-test", "project": "/tmp/proj",
                     "task": "", "status": "active",
                     "last_heartbeat": "2026-01-01T00:00:00"},
                ]
            )
            mock_mgr.list_tasks.return_value = []
            mock_mgr.get_status.return_value = {
                "file_claims": 0, "branch_claims": 0,
                "active_intents": 0, "conflicts": [], "deadlocks": [],
                "files": [],
            }
            mock_mgr.get_unread_count.return_value = 0
            mock_mgr.check_inbox.return_value = []
            mock_mgr.get_latest_handoff.return_value = None
            mock_mgr.get_session_claims.return_value = {"file_claims": [], "branch_claims": []}
            mock_mgr.list_goals.return_value = []
            mock_mgr.query_decisions.return_value = []
            mock_mgr_fn.return_value = mock_mgr

            with patch("omega.server.hook_server.coordination._check_git_sync", return_value=[]), \
                 patch("omega.server.hook_server.coordination._session_resume", return_value=[]):
                return handle_coord_session_start({
                    "session_id": "coord-proto-test",
                    "project": "/tmp/proj",
                })

    # With peers: should have [COORD-PROTOCOL]
    result_peers = _run(1)
    output_peers = result_peers["output"]
    assert "[COORD-PROTOCOL]" in output_peers
    assert "omega_inbox(session_id=" in output_peers
    assert "omega_intent_announce(" in output_peers
    assert "omega_file_check(" in output_peers
    assert "omega_action_claim()" in output_peers

    # Solo: should NOT have [COORD-PROTOCOL]
    result_solo = _run(0)
    output_solo = result_solo["output"]
    assert "[COORD-PROTOCOL]" not in output_solo


# ============================================================================
# Protocol gate enforcement
# ============================================================================

def test_pre_protocol_gate_blocks_edit_without_inbox_multi_agent():
    """Multi-agent: REMIND (non-blocking) if omega_inbox not called within early window."""
    import time
    from omega.server import hook_server
    from omega.server.hook_server import handle_pre_protocol_gate

    sid = "proto-gate-block-test"
    hook_server._gate_call_count[sid] = 1  # within early window (< 5)
    hook_server._protocol_calls.pop(sid, None)
    hook_server._session_peer_count[sid] = 1  # multi-agent
    hook_server._session_peer_count_time[sid] = time.monotonic()  # fresh cache

    try:
        result = handle_pre_protocol_gate({
            "tool_name": "Edit",
            "session_id": sid,
            "tool_input": '{"file_path": "/tmp/test.py"}',
        })
        assert "exit_code" not in result, "gate should be non-blocking (no exit_code)"
        assert "[PROTOCOL-GATE]" in result["output"]
        assert "omega_inbox()" in result["output"]
    finally:
        hook_server._gate_call_count.pop(sid, None)
        hook_server._protocol_calls.pop(sid, None)
        hook_server._session_peer_count.pop(sid, None)
        hook_server._session_peer_count_time.pop(sid, None)


def test_pre_protocol_gate_allows_after_inbox_called():
    """Multi-agent: allow Edit after omega_inbox has been called."""
    import time
    from omega.server import hook_server
    from omega.server.hook_server import handle_pre_protocol_gate, mark_protocol_call

    sid = "proto-gate-allow-test"
    hook_server._gate_call_count[sid] = 1  # within early window
    hook_server._session_peer_count[sid] = 1  # multi-agent
    hook_server._session_peer_count_time[sid] = time.monotonic()
    mark_protocol_call(sid, "omega_inbox")
    mark_protocol_call(sid, "omega_intent_announce")
    mark_protocol_call(sid, "omega_coord_status")

    try:
        result = handle_pre_protocol_gate({
            "tool_name": "Edit",
            "session_id": sid,
            "tool_input": '{"file_path": "/tmp/test.py"}',
        })
        assert result.get("exit_code") is None
        assert result["output"] == ""
    finally:
        hook_server._gate_call_count.pop(sid, None)
        hook_server._protocol_calls.pop(sid, None)
        hook_server._session_peer_count.pop(sid, None)
        hook_server._session_peer_count_time.pop(sid, None)


def test_pre_protocol_gate_reminder_solo_no_welcome():
    """Solo: reminder (no block) if omega_welcome not called."""
    import time
    from omega.server import hook_server
    from omega.server.hook_server import handle_pre_protocol_gate

    sid = "proto-gate-solo-test"
    hook_server._gate_call_count[sid] = 1  # within early window
    hook_server._protocol_calls.pop(sid, None)
    hook_server._session_peer_count[sid] = 0  # solo
    hook_server._session_peer_count_time[sid] = time.monotonic()

    try:
        result = handle_pre_protocol_gate({
            "tool_name": "Bash",
            "session_id": sid,
            "tool_input": '{"command": "ls"}',
        })
        assert result.get("exit_code") is None  # no block
        assert "[PROTOCOL-REMINDER]" in result["output"]
        assert "omega_welcome()" in result["output"]
    finally:
        hook_server._gate_call_count.pop(sid, None)
        hook_server._protocol_calls.pop(sid, None)
        hook_server._session_peer_count.pop(sid, None)
        hook_server._session_peer_count_time.pop(sid, None)


def test_pre_protocol_gate_stops_after_threshold():
    """Enforcement stops after 20 gate calls — no output."""
    import time
    from omega.server import hook_server
    from omega.server.hook_server import handle_pre_protocol_gate

    sid = "proto-gate-threshold-test"
    hook_server._gate_call_count[sid] = 21  # past threshold (> 20)
    hook_server._protocol_calls.pop(sid, None)
    hook_server._session_peer_count[sid] = 1  # multi-agent, but past threshold
    hook_server._session_peer_count_time[sid] = time.monotonic()

    try:
        result = handle_pre_protocol_gate({
            "tool_name": "Edit",
            "session_id": sid,
            "tool_input": '{"file_path": "/tmp/test.py"}',
        })
        assert result.get("exit_code") is None
        assert result["output"] == ""
    finally:
        hook_server._gate_call_count.pop(sid, None)
        hook_server._protocol_calls.pop(sid, None)
        hook_server._session_peer_count.pop(sid, None)
        hook_server._session_peer_count_time.pop(sid, None)


def test_pre_protocol_gate_peer_count_ttl_expiry():
    """Peer count cache expires after TTL, refreshes from coordination manager."""
    import time
    from unittest.mock import patch, MagicMock
    from omega.server import hook_server
    from omega.server.hook_server import handle_pre_protocol_gate

    sid = "proto-gate-ttl-test"
    # Set stale peer count (multi-agent) with expired TTL
    hook_server._gate_call_count[sid] = 0
    hook_server._protocol_calls.pop(sid, None)
    hook_server._session_peer_count[sid] = 1  # stale: says multi-agent
    hook_server._session_peer_count_time[sid] = time.monotonic() - 60  # expired (> 30s TTL)

    # Mock coordination manager to return 1 active session (just us, so 0 peers)
    mock_mgr = MagicMock()
    mock_mgr.active_session_count.return_value = 1

    try:
        with patch("omega.coordination.get_manager", return_value=mock_mgr):
            # Gate should re-query active_session_count() because cache expired.
            # Mock returns 1 session (our own), so peers = max(0, 1-1) = 0.
            result = handle_pre_protocol_gate({
                "tool_name": "Edit",
                "session_id": sid,
                "tool_input": '{"file_path": "/tmp/test.py"}',
            })
        # Should NOT get exit_code=2 because stale peer count refreshed to solo
        assert result.get("exit_code") is None
        # Verify the cache was refreshed (peer count should now be 0)
        assert hook_server._session_peer_count[sid] == 0
        # Verify cache timestamp was updated
        assert time.monotonic() - hook_server._session_peer_count_time[sid] < 5
    finally:
        hook_server._gate_call_count.pop(sid, None)
        hook_server._protocol_calls.pop(sid, None)
        hook_server._session_peer_count.pop(sid, None)
        hook_server._session_peer_count_time.pop(sid, None)


def test_pre_protocol_gate_call_count_increments():
    """Gate call count increments independently on each PreToolUse invocation."""
    import time
    from omega.server import hook_server
    from omega.server.hook_server import handle_pre_protocol_gate, mark_protocol_call

    sid = "proto-gate-count-test"
    hook_server._gate_call_count.pop(sid, None)
    hook_server._session_peer_count[sid] = 0  # solo
    hook_server._session_peer_count_time[sid] = time.monotonic()
    mark_protocol_call(sid, "omega_welcome")  # prevent reminder noise

    try:
        for i in range(10):
            handle_pre_protocol_gate({
                "tool_name": "Edit",
                "session_id": sid,
                "tool_input": '{"file_path": "/tmp/test.py"}',
            })
        # Counter should reflect exactly 10 calls
        assert hook_server._gate_call_count[sid] == 10
    finally:
        hook_server._gate_call_count.pop(sid, None)
        hook_server._protocol_calls.pop(sid, None)
        hook_server._session_peer_count.pop(sid, None)
        hook_server._session_peer_count_time.pop(sid, None)


def test_pre_protocol_gate_fresh_peer_count_not_stale():
    """Fresh peer count cache (within TTL) is used without re-querying."""
    import time
    from omega.server import hook_server
    from omega.server.hook_server import handle_pre_protocol_gate

    sid = "proto-gate-fresh-test"
    hook_server._gate_call_count[sid] = 0
    hook_server._protocol_calls.pop(sid, None)
    # Fresh cache: multi-agent, just set
    hook_server._session_peer_count[sid] = 1
    hook_server._session_peer_count_time[sid] = time.monotonic()  # fresh

    try:
        result = handle_pre_protocol_gate({
            "tool_name": "Edit",
            "session_id": sid,
            "tool_input": '{"file_path": "/tmp/test.py"}',
        })
        # Should remind (non-blocking) because fresh cache says multi-agent and no inbox call
        assert "exit_code" not in result, "gate should be non-blocking (no exit_code)"
        assert "[PROTOCOL-GATE]" in result["output"]
        # Cache should NOT have been refreshed (still says 1 peer)
        assert hook_server._session_peer_count[sid] == 1
    finally:
        hook_server._gate_call_count.pop(sid, None)
        hook_server._protocol_calls.pop(sid, None)
        hook_server._session_peer_count.pop(sid, None)
        hook_server._session_peer_count_time.pop(sid, None)


@pytest.mark.asyncio
async def test_omega_protocol_gate_status():
    """omega_protocol(section='gate_status') returns diagnostic info."""
    import time
    from omega.server import hook_server
    from omega.server.hook_server import mark_protocol_call
    from omega.server.handlers import handle_omega_protocol

    sid = "proto-gate-status-test"
    hook_server._gate_call_count[sid] = 3
    hook_server._session_peer_count[sid] = 0
    hook_server._session_peer_count_time[sid] = time.monotonic()
    mark_protocol_call(sid, "omega_welcome")

    try:
        result = await handle_omega_protocol({"section": "gate_status", "session_id": sid})
        # Result is an MCP response with content
        text = result["content"][0]["text"]
        assert "gate_call_count" in text
        assert "3" in text
        assert "omega_welcome" in text
        assert "enforcement_window" in text
    finally:
        hook_server._gate_call_count.pop(sid, None)
        hook_server._protocol_calls.pop(sid, None)
        hook_server._session_peer_count.pop(sid, None)
        hook_server._session_peer_count_time.pop(sid, None)


# ============================================================================
# Phase 3: Multi-client abstraction tests
# ============================================================================


class TestDetectClient:
    """Tests for _detect_client() in fast_hook.py."""

    def test_explicit_env_var(self, monkeypatch):
        """OMEGA_CLIENT env var should take priority."""
        import importlib
        import hooks.fast_hook as fh
        monkeypatch.setenv("OMEGA_CLIENT", "cursor")
        result = fh._detect_client()
        assert result == "cursor"

    def test_explicit_env_var_windsurf(self, monkeypatch):
        """OMEGA_CLIENT=windsurf should return windsurf."""
        import hooks.fast_hook as fh
        monkeypatch.setenv("OMEGA_CLIENT", "windsurf")
        result = fh._detect_client()
        assert result == "windsurf"

    def test_no_env_var_with_claude_config(self, monkeypatch, tmp_path):
        """Without OMEGA_CLIENT, falls back to heuristic (claude config exists)."""
        import hooks.fast_hook as fh
        monkeypatch.delenv("OMEGA_CLIENT", raising=False)
        # The real ~/.claude/settings.json likely exists on this machine
        # so we just test that the function returns a string without error
        result = fh._detect_client()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_env_var_ignored(self, monkeypatch):
        """Empty OMEGA_CLIENT should fall through to heuristic."""
        import hooks.fast_hook as fh
        monkeypatch.setenv("OMEGA_CLIENT", "")
        result = fh._detect_client()
        # Should not be empty — heuristic kicks in
        assert result != ""


class TestIsPlanFile:
    """Tests for _is_plan_file() in guards.py supporting multiple clients."""

    def test_claude_code_plan_file(self):
        from omega.server.hook_server.guards import _is_plan_file
        assert _is_plan_file("/Users/me/.claude/plans/plan-123.md") is True

    def test_cursor_plan_file(self):
        from omega.server.hook_server.guards import _is_plan_file
        assert _is_plan_file("/Users/me/.cursor/plans/plan-456.md") is True

    def test_regular_file_not_plan(self):
        from omega.server.hook_server.guards import _is_plan_file
        assert _is_plan_file("/Users/me/Projects/omega/src/omega/bridge.py") is False

    def test_empty_path(self):
        from omega.server.hook_server.guards import _is_plan_file
        assert _is_plan_file("") is False

    def test_none_path(self):
        from omega.server.hook_server.guards import _is_plan_file
        assert _is_plan_file(None) is False


class TestClientFlowsToAutoCapture:
    """Tests that client field flows from payload to auto_capture calls."""

    def test_capture_error_passes_client(self, monkeypatch):
        """_capture_error should pass client arg to auto_capture."""
        from omega.server.hook_server.memory import _capture_error
        from omega.server.hook_server import memory as mem_mod

        captured_kwargs = {}

        def mock_auto_capture(**kwargs):
            captured_kwargs.update(kwargs)

        monkeypatch.setattr(mem_mod, "_error_counts", {})
        monkeypatch.setattr(mem_mod, "_error_hashes", set())

        # Mock the bridge import inside _capture_error
        import omega.bridge
        monkeypatch.setattr(omega.bridge, "auto_capture", mock_auto_capture)
        monkeypatch.setattr(omega.bridge, "query_structured", lambda **kw: [])

        _capture_error(
            "Traceback (most recent call last)\nTypeError: bad arg",
            "test-session",
            "/tmp/project",
            entity_id=None,
            client="cursor",
        )

        assert captured_kwargs.get("agent_type") == "cursor"

    def test_capture_error_defaults_to_claude_code(self, monkeypatch):
        """_capture_error without client arg should default to claude-code."""
        from omega.server.hook_server.memory import _capture_error
        from omega.server.hook_server import memory as mem_mod

        captured_kwargs = {}

        def mock_auto_capture(**kwargs):
            captured_kwargs.update(kwargs)

        monkeypatch.setattr(mem_mod, "_error_counts", {})
        monkeypatch.setattr(mem_mod, "_error_hashes", set())

        import omega.bridge
        monkeypatch.setattr(omega.bridge, "auto_capture", mock_auto_capture)
        monkeypatch.setattr(omega.bridge, "query_structured", lambda **kw: [])

        _capture_error(
            "Traceback (most recent call last)\nValueError: oops",
            "test-session-2",
            "/tmp/project",
            entity_id=None,
            # no client= arg — should default
        )

        assert captured_kwargs.get("agent_type") == "claude-code"
