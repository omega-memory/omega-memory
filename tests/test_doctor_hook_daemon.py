"""`omega doctor` must notice when the hook daemon is missing or unreachable.

Issue #76 went unnoticed for months because doctor only checked that the
hook entries and script paths existed. These tests pin the three signals it
now reports: the daemon's socket state, whether an MCP server is running
without one, and how many core hook runs hooks.log shows as skipped.
"""
import argparse
import json
import shutil
import socket
import sqlite3
import tempfile
from pathlib import Path

import pytest

from omega import cli
from omega.server import hook_server

SKIPPED = "[2026-09-15T02:40:01] fast_hook/{}: OK (0ms, skipped)"
DAEMON = "[2026-09-16T13:04:26] fast_hook/{}: OK (148ms, daemon)"


def test_count_skipped_core_hooks_counts_only_core_hooks_skipped_for_lack_of_daemon():
    lines = [
        SKIPPED.format("auto_capture"),
        SKIPPED.format("surface_memories"),
        SKIPPED.format("session_start+coord_session_start"),
        SKIPPED.format("coord_heartbeat"),  # Pro hook: not a core hook
        DAEMON.format("auto_capture"),  # served by the daemon: not skipped
        "[2026-09-16T13:04:26] hook_server/auto_capture: OK (148ms)",
    ]
    assert cli._count_skipped_core_hooks(lines) == 3


@pytest.fixture
def short_socket_dir():
    directory = Path(tempfile.mkdtemp(prefix="omg", dir="/tmp"))
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


def test_probe_reports_absent_when_no_socket_file(short_socket_dir, monkeypatch):
    monkeypatch.setattr(hook_server, "SOCK_PATH", short_socket_dir / "hook.sock")

    state, detail = cli._probe_hook_daemon()

    assert state == "absent"
    assert detail.endswith("hook.sock")


def test_probe_reports_listening_when_a_server_accepts(short_socket_dir, monkeypatch):
    sock_path = short_socket_dir / "hook.sock"
    monkeypatch.setattr(hook_server, "SOCK_PATH", sock_path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(1)
    try:
        state, _ = cli._probe_hook_daemon()
    finally:
        server.close()

    assert state == "listening"


def test_probe_reports_stale_when_nothing_answers_on_the_file(short_socket_dir, monkeypatch):
    sock_path = short_socket_dir / "hook.sock"
    monkeypatch.setattr(hook_server, "SOCK_PATH", sock_path)
    sock_path.write_text("")  # a plain file where a socket used to be

    state, detail = cli._probe_hook_daemon()

    assert state == "stale"
    assert "hook.sock" in detail


def test_probe_reports_unavailable_when_daemon_module_is_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def missing_hook_server(name, *args, **kwargs):
        if name == "omega.server.hook_server":
            raise ImportError("No module named 'omega.server.hook_server'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_hook_server)

    state, detail = cli._probe_hook_daemon()

    assert state == "unavailable"
    assert "hook_server" in detail


def _run_doctor_json(tmp_path, monkeypatch, capsys) -> dict:
    db_path = tmp_path / "omega.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE memories (id TEXT, content TEXT, metadata TEXT)")
    conn.execute("CREATE VIRTUAL TABLE memories_fts USING fts5(content)")
    conn.execute("CREATE TABLE memories_vec (rowid INTEGER PRIMARY KEY, embedding BLOB)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(cli, "OMEGA_DIR", tmp_path)
    monkeypatch.setattr(cli, "BGE_MODEL_DIR", tmp_path / "no-model")
    monkeypatch.setattr(cli, "MINILM_MODEL_DIR", tmp_path / "no-model")
    monkeypatch.setattr(cli, "SETTINGS_JSON_PATH", tmp_path / "no-settings.json")
    with pytest.raises(SystemExit):
        cli.cmd_doctor(argparse.Namespace(json=True, client=None))
    return json.loads(capsys.readouterr().out)


def _messages(report: dict, status: str) -> list[str]:
    return [c["message"] for c in report["checks"] if c["status"] == status]


def test_doctor_warns_when_server_runs_without_a_socket(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_probe_hook_daemon", lambda timeout=1.0: ("absent", "/x/hook.sock"))
    monkeypatch.setattr(cli, "_mcp_servers_running", lambda: True)

    report = _run_doctor_json(tmp_path, monkeypatch, capsys)

    assert any("no hook socket" in m and "core hooks are being skipped" in m for m in _messages(report, "warn"))


def test_doctor_is_calm_when_nothing_is_running(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_probe_hook_daemon", lambda timeout=1.0: ("absent", "/x/hook.sock"))
    monkeypatch.setattr(cli, "_mcp_servers_running", lambda: False)

    report = _run_doctor_json(tmp_path, monkeypatch, capsys)

    assert any(m.startswith("Hook daemon not running") for m in _messages(report, "ok"))
    assert not any("hook socket" in m for m in _messages(report, "warn"))


def test_doctor_treats_a_stale_socket_as_benign_when_no_server_runs(tmp_path, monkeypatch, capsys):
    """`claude mcp list` (run by doctor itself) starts and kills a server, leaving a socket behind."""
    monkeypatch.setattr(cli, "_probe_hook_daemon", lambda timeout=1.0: ("stale", "/x/hook.sock"))
    monkeypatch.setattr(cli, "_mcp_servers_running", lambda: False)

    report = _run_doctor_json(tmp_path, monkeypatch, capsys)

    assert any(m.startswith("Hook socket is stale") for m in _messages(report, "ok"))
    assert not any("hook socket" in m.lower() for m in _messages(report, "warn"))


def test_doctor_warns_on_a_stale_socket_when_a_server_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_probe_hook_daemon", lambda timeout=1.0: ("stale", "/x/hook.sock"))
    monkeypatch.setattr(cli, "_mcp_servers_running", lambda: True)

    report = _run_doctor_json(tmp_path, monkeypatch, capsys)

    assert any("nothing answers on the hook socket" in m for m in _messages(report, "warn"))


def test_doctor_fails_when_daemon_module_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_probe_hook_daemon", lambda timeout=1.0: ("unavailable", "No module named x"))

    report = _run_doctor_json(tmp_path, monkeypatch, capsys)

    assert any("Hook daemon module not importable" in m for m in _messages(report, "fail"))


def test_doctor_reports_skipped_core_hooks_from_the_log(tmp_path, monkeypatch, capsys):
    (tmp_path / "hooks.log").write_text("\n".join([SKIPPED.format("auto_capture")] * 4 + [DAEMON.format("session_stop")]) + "\n")
    monkeypatch.setattr(cli, "_probe_hook_daemon", lambda timeout=1.0: ("absent", "/x/hook.sock"))
    monkeypatch.setattr(cli, "_mcp_servers_running", lambda: False)

    report = _run_doctor_json(tmp_path, monkeypatch, capsys)

    assert any(m.startswith("4 core hook run(s)") for m in _messages(report, "warn"))
