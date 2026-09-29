"""The opt-in MCP HTTP daemon must not be usable by a web page (audit finding E1).

A page can POST to loopback directly or reach it through DNS rebinding. The
daemon now validates Host and Origin and requires a bearer key by default;
`omega serve migrate-config` writes that key into the Claude Code entries it
configures, so the clients Core sets up keep working.
"""
import argparse
import asyncio
import json
import os
import shutil
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("starlette", reason="requires the 'server' extra")

from omega import cli  # noqa: E402
from omega.server import mcp_auth  # noqa: E402

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
KEY = "test-key-0123456789"
PORT = 18555
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
}
MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


@pytest.fixture
def omega_home(tmp_path, monkeypatch):
    home = tmp_path / ".omega"
    monkeypatch.setenv("OMEGA_HOME", str(home))
    monkeypatch.delenv(mcp_auth.API_KEY_ENV, raising=False)
    return home


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def test_key_is_generated_once_private_and_reused(omega_home):
    first = mcp_auth.resolve_api_key()

    assert len(first) >= 32
    assert mcp_auth.api_key_path() == omega_home / "mcp_api_key"
    assert stat.S_IMODE(mcp_auth.api_key_path().stat().st_mode) == 0o600
    assert mcp_auth.resolve_api_key() == first


def test_env_key_overrides_the_file(omega_home, monkeypatch):
    mcp_auth.resolve_api_key()
    monkeypatch.setenv(mcp_auth.API_KEY_ENV, "from-env")

    assert mcp_auth.resolve_api_key() == "from-env"


def test_read_only_callers_never_create_a_key(omega_home):
    assert mcp_auth.resolve_api_key(create=False) is None
    assert not mcp_auth.api_key_path().exists()


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from starlette.testclient import TestClient

    from omega.server import mcp_server

    app = mcp_server._build_http_app("127.0.0.1", PORT, KEY)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as test_client:
        yield test_client


def test_mcp_request_without_the_key_is_refused(client):
    response = client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)

    assert response.status_code == 401


def test_mcp_request_with_a_wrong_key_is_refused(client):
    response = client.post("/mcp", json=INITIALIZE, headers={**MCP_HEADERS, "Authorization": "Bearer nope"})

    assert response.status_code == 401


def test_mcp_request_with_the_key_is_served_at_the_url_migrate_config_writes(client):
    response = client.post("/mcp", json=INITIALIZE, headers={**MCP_HEADERS, "Authorization": f"Bearer {KEY}"})

    assert response.status_code == 200
    assert "serverInfo" in response.text


def test_dns_rebinding_host_is_refused_even_with_the_key(client):
    headers = {**MCP_HEADERS, "Authorization": f"Bearer {KEY}", "Host": f"attacker.example:{PORT}"}

    assert client.post("/mcp", json=INITIALIZE, headers=headers).status_code == 421


def test_foreign_origin_is_refused_even_with_the_key(client):
    headers = {**MCP_HEADERS, "Authorization": f"Bearer {KEY}", "Origin": "http://attacker.example"}

    assert client.post("/mcp", json=INITIALIZE, headers=headers).status_code == 403


def test_health_needs_no_key_but_still_checks_host(client):
    assert client.get("/health").status_code == 200
    assert client.get("/health", headers={"Host": f"attacker.example:{PORT}"}).status_code == 421


# ---------------------------------------------------------------------------
# migrate-config: the Claude Code entries Core writes carry the key
# ---------------------------------------------------------------------------

DAEMON_URL = f"http://{cli._DEFAULT_HTTP_HOST}:{cli._DEFAULT_HTTP_PORT}/mcp"
STDIO = {"type": "stdio", "command": "/venv/bin/python3", "args": ["-m", "omega.server.mcp_server"], "env": {}}


def _migrate(tmp_path, monkeypatch, config: dict) -> dict:
    claude_json = tmp_path / ".claude.json"
    claude_json.write_text(json.dumps(config))
    monkeypatch.setattr(cli, "CLAUDE_JSON_PATH", claude_json)
    cli._serve_migrate_config(argparse.Namespace())
    return json.loads(claude_json.read_text())


def test_migrate_config_moves_user_and_project_entries_to_the_daemon_with_the_key(tmp_path, omega_home, monkeypatch):
    migrated = _migrate(tmp_path, monkeypatch, {
        "mcpServers": {"omega-memory": STDIO, "other": {"type": "stdio", "command": "x"}},
        "projects": {"/p": {"mcpServers": {"omega-memory": STDIO}}},
    })

    expected = {"type": "http", "url": DAEMON_URL, "headers": {"Authorization": f"Bearer {mcp_auth.resolve_api_key()}"}}
    assert migrated["mcpServers"]["omega-memory"] == expected
    assert migrated["projects"]["/p"]["mcpServers"]["omega-memory"] == expected
    assert migrated["mcpServers"]["other"] == {"type": "stdio", "command": "x"}


def test_migrate_config_adds_the_key_to_entries_migrated_before_auth_existed(tmp_path, omega_home, monkeypatch):
    migrated = _migrate(tmp_path, monkeypatch, {"mcpServers": {"omega-memory": {"type": "http", "url": DAEMON_URL}}})

    headers = migrated["mcpServers"]["omega-memory"]["headers"]
    assert headers == {"Authorization": f"Bearer {mcp_auth.resolve_api_key()}"}


# ---------------------------------------------------------------------------
# doctor: an http entry the daemon will refuse
# ---------------------------------------------------------------------------


def _doctor_fails(tmp_path, monkeypatch, capsys, entry: dict) -> list[str]:
    (tmp_path / ".claude.json").write_text(json.dumps({"mcpServers": {"omega-memory": entry}}))
    conn = sqlite3.connect(str(tmp_path / "omega.db"))
    conn.execute("CREATE TABLE memories (id TEXT, content TEXT, metadata TEXT)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(cli, "CLAUDE_JSON_PATH", tmp_path / ".claude.json")
    monkeypatch.setattr(cli, "OMEGA_DIR", tmp_path)
    monkeypatch.setattr(cli, "SETTINGS_JSON_PATH", tmp_path / "no-settings.json")
    monkeypatch.setattr(cli, "CLAUDE_MD_PATH", tmp_path / "no-claude.md")
    monkeypatch.setattr(cli, "_probe_hook_daemon", lambda timeout=1.0: ("absent", "/x/hook.sock"))
    monkeypatch.setattr(cli, "_mcp_servers_running", lambda: False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit):
        cli.cmd_doctor(argparse.Namespace(json=True, client=None))
    report = json.loads(capsys.readouterr().out)
    return [c["message"] for c in report["checks"] if c["status"] == "fail"]


def test_doctor_fails_an_http_entry_without_the_key(tmp_path, omega_home, monkeypatch, capsys):
    failures = _doctor_fails(tmp_path, monkeypatch, capsys, {"type": "http", "url": DAEMON_URL})

    assert any("omega serve migrate-config" in m for m in failures)


def test_doctor_fails_an_http_entry_with_a_stale_key(tmp_path, omega_home, monkeypatch, capsys):
    mcp_auth.resolve_api_key()
    entry = {"type": "http", "url": DAEMON_URL, "headers": {"Authorization": "Bearer old"}}

    failures = _doctor_fails(tmp_path, monkeypatch, capsys, entry)

    assert any("omega serve migrate-config" in m for m in failures)


# ---------------------------------------------------------------------------
# End to end: a real daemon, and a real MCP client configured like Claude Code
# ---------------------------------------------------------------------------


def _free_port() -> int:
    while True:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        if port not in (8377, 8378):  # the owner's live daemons
            return port


@pytest.fixture
def running_daemon():
    home = Path(tempfile.mkdtemp(prefix="omg", dir="/tmp"))  # short: AF_UNIX path limit
    port = _free_port()
    env = {
        **os.environ,
        "HOME": str(home),
        "OMEGA_HOME": str(home / ".omega"),
        "PYTHONPATH": str(SRC_DIR),
        "OMEGA_TRANSPORT": "http",
        "OMEGA_HTTP_PORT": str(port),
        "OMEGA_SKIP_EMBEDDINGS": "1",
    }
    env.pop(mcp_auth.API_KEY_ENV, None)
    daemon = subprocess.Popen(
        [sys.executable, "-m", "omega.server.mcp_server"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env,
    )
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline and daemon.poll() is None:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail(f"daemon did not come up: {daemon.stderr.read().decode() if daemon.poll() is not None else ''}")
        key = (home / ".omega" / "mcp_api_key").read_text().strip()
        yield port, key
    finally:
        daemon.terminate()
        try:
            daemon.wait(timeout=15)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait()
        shutil.rmtree(home, ignore_errors=True)


def test_real_daemon_refuses_a_page_and_serves_a_configured_client(running_daemon):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    port, key = running_daemon
    url = f"http://127.0.0.1:{port}/mcp"

    anonymous = urllib.request.Request(url, data=json.dumps(INITIALIZE).encode(), headers=MCP_HEADERS, method="POST")
    with pytest.raises(urllib.error.HTTPError) as refused:
        urllib.request.urlopen(anonymous, timeout=5)
    assert refused.value.code == 401

    async def use_like_claude_code() -> str:
        async with streamablehttp_client(url, headers=mcp_auth.authorization_header(key)) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = {tool.name for tool in (await session.list_tools()).tools}
                assert "omega_store" in names
                await session.call_tool("omega_store", {"content": "The billing service retries webhooks 5 times", "event_type": "decision"})
                result = await session.call_tool("omega_call", {"tool": "omega_query", "args": {"query": "billing webhooks retries"}})
                return result.content[0].text

    assert "billing service retries webhooks" in asyncio.run(use_like_claude_code())
