"""Tests for the self-audit council's signal gatherers."""
import sqlite3
import sys
import threading
import types
from datetime import datetime, timedelta, timezone

import pytest

from omega.council import Council

NOW = datetime.now(timezone.utc)


def _just_outside(days):
    return (NOW - timedelta(days=days, seconds=1)).isoformat()


def _just_inside(days):
    return (NOW - timedelta(days=days) + timedelta(hours=1)).isoformat()


@pytest.fixture
def coord(monkeypatch):
    """Pro's coordination tables, which Pro writes with isoformat() timestamps."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE coord_audit (tool_name TEXT, result_summary TEXT, created_at TEXT)")
    conn.execute(
        "CREATE TABLE coord_external_actions "
        "(action_type TEXT, action_target TEXT, status TEXT, created_at TEXT)"
    )

    class FakeManager:
        _lock = threading.Lock()

        def get_read_connection(self):
            return conn

    coordination = types.ModuleType("omega_platform.orchestrator.coordination")
    coordination.get_manager = lambda: FakeManager()
    monkeypatch.setitem(sys.modules, "omega_platform", types.ModuleType("omega_platform"))
    monkeypatch.setitem(sys.modules, "omega_platform.orchestrator", types.ModuleType("omega_platform.orchestrator"))
    monkeypatch.setitem(sys.modules, "omega_platform.orchestrator.coordination", coordination)
    return conn


@pytest.fixture
def council(tmp_path):
    (tmp_path / "platform_health.md").write_text("# Platform health")
    return Council("platform_health", config_dir=str(tmp_path))


def test_signal_windows_count_only_rows_inside_them(coord, council):
    """Windows compare stored isoformat text with a bound in the same format.

    Comparing it with SQLite datetime() text sorted 'T' after ' ', so rows from
    the cutoff's calendar day counted as inside every window.
    """
    coord.executemany(
        "INSERT INTO coord_audit VALUES ('mcp__omega-memory__omega_store', 'error: disk full', ?)",
        [(_just_outside(1),), (_just_inside(1),)],
    )
    coord.executemany(
        "INSERT INTO coord_audit VALUES ('mcp__omega-memory__omega_query', 'ok', ?)",
        [(_just_outside(7),), (_just_inside(7),)],
    )
    coord.executemany(
        "INSERT INTO coord_external_actions VALUES ('publish', 'pypi', 'completed', ?)",
        [(_just_outside(1),), (_just_inside(1),)],
    )

    assert council._get_tool_failures(project=None) == [
        {"tool": "mcp__omega-memory__omega_store", "failures": 1},
    ]
    assert len(council._get_external_actions(project=None)) == 1
    # Both error rows are inside the 7-day window; only one query row is.
    assert {r["tool"]: r["calls_7d"] for r in council._get_tool_usage()} == {
        "mcp__omega-memory__omega_store": 2,
        "mcp__omega-memory__omega_query": 1,
    }
