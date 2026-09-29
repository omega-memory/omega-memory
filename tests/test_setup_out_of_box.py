"""What a new user gets from `omega setup`, `omega hooks setup` and `omega doctor`.

Pinned by the 2026-09-29 bug-team audit of a clean install:

- C1: with Pro installed, setup read `omega/data/hooks.json`, which Core does
  not ship, so no hooks were installed and the summary still said [OK].
  Doctor counted any hook command containing "omega" as an OMEGA hook.
- C3: a fresh install got all-MiniLM, whose similarities fall under the 0.60
  vector floor, so the documented first query found nothing.
- C4: without the `[server]` extra, setup and doctor said OK although the MCP
  server could not start.
- C5: hook commands were not quoted, so an install path with a space failed.
- C6: `--dry-run` wrote files and downloaded models; the CLI ignored
  OMEGA_HOME; `omega status` looked up the wrong Pro package name.
"""
import argparse
import inspect
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from omega import cli

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
CORE_MANIFEST = json.loads((SRC_DIR / "omega" / "data" / "hooks-core.json").read_text())


@pytest.fixture
def claude_home(tmp_path, monkeypatch):
    """Point every path setup and doctor touch at a temp home."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setattr(cli, "SETTINGS_JSON_PATH", home / ".claude" / "settings.json")
    monkeypatch.setattr(cli, "CLAUDE_MD_PATH", home / ".claude" / "CLAUDE.md")
    monkeypatch.setattr(cli, "CLAUDE_JSON_PATH", home / ".claude.json")
    monkeypatch.setattr(cli, "OMEGA_DIR", home / ".omega")
    monkeypatch.setattr(cli, "BGE_MODEL_DIR", home / "models" / "bge")
    monkeypatch.setattr(cli, "MINILM_MODEL_DIR", home / "models" / "minilm")
    monkeypatch.setattr(cli, "CLAUDE_SCRIPTS_DIR", home / ".claude" / "scripts")
    monkeypatch.setattr(cli, "_resolve_python_path", lambda: "/usr/bin/python3")
    return home


@pytest.fixture
def core_only_data_dir(tmp_path, monkeypatch):
    """A package data dir that ships only the core manifest, like every Core wheel."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "hooks-core.json").write_text(json.dumps(CORE_MANIFEST))
    for name in ("claude-md-fragment.md", "claude-md-fragment-pro.md"):
        (data_dir / name).write_text((SRC_DIR / "omega" / "data" / name).read_text())
    monkeypatch.setattr(cli, "DATA_DIR", data_dir)
    return data_dir


def _settings(home: Path) -> dict:
    return json.loads((home / ".claude" / "settings.json").read_text())


def _commands(settings: dict) -> list[str]:
    return [h["command"] for entries in settings.get("hooks", {}).values() for e in entries for h in e["hooks"]]


# ---------------------------------------------------------------------------
# C1: the manifest setup installs, and what the summary says about it
# ---------------------------------------------------------------------------


def test_hooks_install_from_core_manifest_when_pro_is_present_but_ships_no_manifest(
    claude_home, core_only_data_dir, monkeypatch
):
    monkeypatch.setattr(cli, "_has_commercial_modules", lambda: True)

    cli._inject_settings_hooks(claude_home / "hooks")

    commands = _commands(_settings(claude_home))
    expected = sorted(entry["script"].split()[1] for entries in CORE_MANIFEST.values() for entry in entries)
    assert sorted(shlex.split(c)[-1] for c in commands) == expected


def test_hooks_install_from_full_manifest_when_one_is_shipped(claude_home, core_only_data_dir):
    full = {"SessionStart": [{"script": "fast_hook.py session_start+coord_session_start", "timeout": 5, "matcher": ""}]}
    (core_only_data_dir / "hooks.json").write_text(json.dumps(full))

    cli._inject_settings_hooks(claude_home / "hooks")

    assert [shlex.split(c)[-1] for c in _commands(_settings(claude_home))] == ["session_start+coord_session_start"]


def test_malformed_settings_json_is_a_setup_failure_and_is_left_untouched(claude_home, core_only_data_dir):
    settings_path = claude_home / ".claude" / "settings.json"
    settings_path.write_text("{ not json")

    with pytest.raises(cli.HookSetupError, match="malformed"):
        cli._inject_settings_hooks(claude_home / "hooks")

    assert settings_path.read_text() == "{ not json"


def test_setup_summary_reports_a_failed_hooks_step_as_fail_not_ok(claude_home, core_only_data_dir, monkeypatch, capsys):
    def no_manifest(hooks_src):
        raise FileNotFoundError("omega/data/hooks.json")

    monkeypatch.setattr(cli, "_inject_settings_hooks", no_manifest)
    monkeypatch.setattr(cli, "_install_embedding_model", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_download_reranker_model", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_register_claude_code_mcp", lambda python_path, errors: True)

    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_setup(argparse.Namespace(client="claude-code", hooks_only=False, dry_run=False, download_model=False))

    out = capsys.readouterr().out
    assert exit_info.value.code == 1
    assert "[FAIL] Hooks (settings.json): omega/data/hooks.json" in out
    assert "[OK] Hooks (settings.json)" not in out
    assert "[FAIL] 1" not in out


def test_hooks_setup_command_installs_core_hooks_with_pro_present(claude_home, core_only_data_dir, monkeypatch, capsys):
    hooks_src = claude_home / "hooks"
    hooks_src.mkdir()
    (hooks_src / "fast_hook.py").write_text("")
    monkeypatch.setattr(cli, "_resolve_hooks_src", lambda: hooks_src)
    monkeypatch.setattr(cli, "_has_commercial_modules", lambda: True)

    cli.cmd_hooks(argparse.Namespace(hooks_command="setup"))

    assert len(_commands(_settings(claude_home))) == 5
    assert "ERROR" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# C5: quoting, and repairing entries written before quoting
# ---------------------------------------------------------------------------


def test_hook_commands_survive_paths_with_spaces(claude_home, core_only_data_dir, monkeypatch):
    python = "/Users/Jo Doe/venv/bin/python3"
    hooks_src = claude_home / "Application Support" / "omega" / "hooks"
    monkeypatch.setattr(cli, "_resolve_python_path", lambda: python)

    cli._inject_settings_hooks(hooks_src)

    start = _settings(claude_home)["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    assert shlex.split(start) == [python, str(hooks_src / "fast_hook.py"), "session_start"]


def test_plain_paths_are_written_without_quotes_so_existing_installs_do_not_churn(claude_home, core_only_data_dir):
    cli._inject_settings_hooks(claude_home / "hooks")

    start = _settings(claude_home)["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    assert start == f"/usr/bin/python3 {claude_home / 'hooks' / 'fast_hook.py'} session_start"


def test_unquoted_entry_from_an_older_setup_is_repaired_in_place_not_duplicated(
    claude_home, core_only_data_dir, monkeypatch
):
    python = "/Users/Jo Doe/venv/bin/python3"
    hooks_src = claude_home / "My Hooks"
    monkeypatch.setattr(cli, "_resolve_python_path", lambda: python)
    legacy = f"{python} {hooks_src / 'fast_hook.py'} session_start"
    (claude_home / ".claude" / "settings.json").write_text(json.dumps({
        "hooks": {"SessionStart": [{"hooks": [{"command": legacy, "timeout": 5000, "type": "command"}], "matcher": ""}]}
    }))

    cli._inject_settings_hooks(hooks_src)

    entries = _settings(claude_home)["hooks"]["SessionStart"]
    assert len(entries) == 1
    assert shlex.split(entries[0]["hooks"][0]["command"]) == [python, str(hooks_src / "fast_hook.py"), "session_start"]


def test_hooks_doctor_parses_quoted_commands(claude_home, core_only_data_dir, monkeypatch, capsys):
    python = claude_home / "py bin" / "python3"
    python.parent.mkdir()
    python.write_text("")
    hooks_src = claude_home / "hook dir"
    hooks_src.mkdir()
    (hooks_src / "fast_hook.py").write_text("")
    monkeypatch.setattr(cli, "_resolve_python_path", lambda: str(python))
    monkeypatch.setattr(cli, "_resolve_hooks_src", lambda: hooks_src)
    monkeypatch.setattr(cli, "_probe_hook_daemon", lambda timeout=1.0: ("absent", "/x/hook.sock"))
    cli._inject_settings_hooks(hooks_src)

    cli.cmd_hooks(argparse.Namespace(hooks_command="doctor"))

    out = capsys.readouterr().out
    assert "5/5 OMEGA hooks configured" in out
    assert "BROKEN" not in out


# ---------------------------------------------------------------------------
# C1: doctor checks the entries setup would write, not any command with "omega"
# ---------------------------------------------------------------------------


def _doctor_report(home: Path, monkeypatch, capsys, client: str | None = "claude-code", mcp_ok: bool = True) -> dict:
    capsys.readouterr()  # drop output from arranging the test (hook injection)
    conn = sqlite3.connect(str(home / "omega.db"))
    conn.execute("CREATE TABLE memories (id TEXT, content TEXT, metadata TEXT)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(cli, "OMEGA_DIR", home)
    monkeypatch.setattr(cli, "_probe_hook_daemon", lambda timeout=1.0: ("absent", "/x/hook.sock"))
    monkeypatch.setattr(cli, "_mcp_servers_running", lambda: False)
    monkeypatch.setattr(cli, "_mcp_importable", lambda python_path: mcp_ok)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        cli.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="omega-memory: registered", stderr=""),
    )
    with pytest.raises(SystemExit):
        cli.cmd_doctor(argparse.Namespace(json=True, client=client))
    return json.loads(capsys.readouterr().out)


def _by_status(report: dict, status: str) -> list[str]:
    return [c["message"] for c in report["checks"] if c["status"] == status]


def test_doctor_fails_when_settings_has_no_omega_hooks_even_if_commands_mention_omega(
    claude_home, core_only_data_dir, monkeypatch, capsys
):
    unrelated = "/Users/omega-fan/bin/python3 /Users/omega-fan/lint.py"
    (claude_home / ".claude" / "settings.json").write_text(json.dumps({
        "hooks": {event: [{"hooks": [{"command": unrelated, "type": "command"}], "matcher": ""}]
                  for event in ("SessionStart", "Stop", "PostToolUse", "UserPromptSubmit")}
    }))

    report = _doctor_report(claude_home, monkeypatch, capsys)

    failures = " ".join(_by_status(report, "fail"))
    assert "0/5 OMEGA hooks configured" in failures
    assert "omega hooks setup" in failures
    assert not any("hook configured" in m for m in _by_status(report, "ok"))


def test_doctor_passes_when_every_manifest_hook_is_wired_to_real_paths(
    claude_home, core_only_data_dir, monkeypatch, capsys
):
    python = claude_home / "python3"
    python.write_text("")
    hooks_src = claude_home / "hooks"
    hooks_src.mkdir()
    (hooks_src / "fast_hook.py").write_text("")
    monkeypatch.setattr(cli, "_resolve_python_path", lambda: str(python))
    monkeypatch.setattr(cli, "_resolve_hooks_src", lambda: hooks_src)
    cli._inject_settings_hooks(hooks_src)

    report = _doctor_report(claude_home, monkeypatch, capsys)

    assert "5/5 OMEGA hooks configured" in _by_status(report, "ok")
    assert not [m for m in _by_status(report, "fail") if "hook" in m.lower()]


def test_doctor_fails_a_hook_whose_script_no_longer_exists(claude_home, core_only_data_dir, monkeypatch, capsys):
    python = claude_home / "python3"
    python.write_text("")
    monkeypatch.setattr(cli, "_resolve_python_path", lambda: str(python))
    monkeypatch.setattr(cli, "_resolve_hooks_src", lambda: claude_home / "gone")
    cli._inject_settings_hooks(claude_home / "gone")

    report = _doctor_report(claude_home, monkeypatch, capsys)

    assert any("script not found" in m and "fast_hook.py" in m for m in _by_status(report, "fail"))


# ---------------------------------------------------------------------------
# C4: the MCP server needs the `mcp` package
# ---------------------------------------------------------------------------


def test_setup_does_not_register_an_mcp_server_that_cannot_start(claude_home, core_only_data_dir, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_mcp_importable", lambda python_path: False)
    ran = []
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: ran.append(a) or pytest.fail("must not run claude"))
    errors: list = []

    done = cli._setup_claude_code(errors, claude_home / "hooks")

    assert ran == []
    assert "MCP server registration" not in done
    assert any('pip install "omega-memory[server]"' in str(e) for e in errors)


def test_doctor_fails_when_the_mcp_package_is_missing(claude_home, core_only_data_dir, monkeypatch, capsys):
    report = _doctor_report(claude_home, monkeypatch, capsys, client=None, mcp_ok=False)

    assert any('pip install "omega-memory[server]"' in m for m in _by_status(report, "fail"))


# ---------------------------------------------------------------------------
# C3: a fresh install gets the model the retrieval thresholds are tuned for
# ---------------------------------------------------------------------------


def test_fresh_install_downloads_bge_small_not_minilm(claude_home, monkeypatch):
    fetched = []

    def fake_bge(target_dir, errors_ref):
        fetched.append(target_dir)
        return True

    monkeypatch.setattr(cli, "_download_bge_model", fake_bge)
    monkeypatch.setattr(cli, "_download_minilm_model", lambda *a: pytest.fail("fresh installs must not get MiniLM"))
    done: list = []

    cli._install_embedding_model(False, [], done)

    assert fetched == [cli.BGE_MODEL_DIR]
    assert done == ["Embedding model (bge-small-en-v1.5)"]


# ---------------------------------------------------------------------------
# C6: --dry-run, OMEGA_HOME, status, tool counts
# ---------------------------------------------------------------------------


def test_dry_run_changes_nothing_and_downloads_nothing(claude_home, core_only_data_dir, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_download_file", lambda *a: pytest.fail("dry run must not download"))
    monkeypatch.setattr(cli, "_download_reranker_model", lambda *a: pytest.fail("dry run must not download"))
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: pytest.fail("dry run must not run commands"))
    monkeypatch.setattr(cli, "_mcp_importable", lambda python_path: True)
    before = sorted(p.relative_to(claude_home) for p in claude_home.rglob("*"))

    cli.cmd_setup(argparse.Namespace(client="claude-code", hooks_only=False, dry_run=True, download_model=False))

    assert sorted(p.relative_to(claude_home) for p in claude_home.rglob("*")) == before
    out = capsys.readouterr().out
    assert "dry run" in out.lower()
    assert "would" in out


def test_cli_honours_omega_home(tmp_path):
    env = {**os.environ, "OMEGA_HOME": str(tmp_path / "custom-home"), "PYTHONPATH": str(SRC_DIR)}
    result = subprocess.run(
        [sys.executable, "-c", "import omega.cli as c; print(c.OMEGA_DIR)"],
        capture_output=True, text=True, env=env, check=True,
    )
    assert result.stdout.strip() == str(tmp_path / "custom-home")


def test_status_reports_the_pro_package_by_its_published_name(tmp_path, monkeypatch, capsys):
    from importlib.metadata import PackageNotFoundError

    def installed(name):
        if name == "omega-memory-pro":
            return "1.5.12"
        raise PackageNotFoundError(name)

    monkeypatch.setattr(cli, "OMEGA_DIR", tmp_path)
    monkeypatch.setattr(cli, "BGE_MODEL_DIR", tmp_path / "no-model")
    monkeypatch.setattr(cli, "MINILM_MODEL_DIR", tmp_path / "no-model")
    monkeypatch.setattr(cli, "_pkg_version", installed)

    cli.cmd_status(argparse.Namespace(json=True))

    assert json.loads(capsys.readouterr().out)["platform_version"] == "1.5.12"


def test_cli_prints_no_hard_coded_tool_counts():
    """The counts drifted (98, 95, 53, 25 ...) against a measured 16 + 120; say what, not how many."""
    source = inspect.getsource(cli)
    assert not re.findall(r"[\"'][^\"'\n]*\b\d+ (?:more |Pro |coordination )?tools\b", source)


def test_claude_md_dry_run_writes_nothing_even_when_a_backup_already_exists(claude_home, core_only_data_dir):
    claude_md = claude_home / ".claude" / "CLAUDE.md"
    claude_md.write_text("# mine\n")
    claude_md.with_suffix(".md.pre-omega").write_text("# older backup\n")

    cli._inject_claude_md(dry_run=True)

    assert claude_md.read_text() == "# mine\n"
