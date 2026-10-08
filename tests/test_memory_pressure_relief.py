"""The memory watchdog relieves pressure when memory has grown, not on every check.

With both models loaded a server stays above half its memory limit. The
watchdog used to force a garbage collection and log a WARNING on every 15 s
check from then on.
"""

import logging

import pytest

mcp_server = pytest.importorskip("omega.server.mcp_server")

MB = 1024 * 1024


@pytest.fixture
def watchdog(monkeypatch):
    """A 1,000 MB limit, a settable RSS, and a count of the reliefs performed."""
    state = {"rss": 0, "reliefs": 0, "frees": 0}

    def fake_release():
        state["reliefs"] += 1
        state["rss"] -= state["frees"]
        return state["frees"]

    monkeypatch.setattr(mcp_server, "_RSS_LIMIT_BYTES", 1000 * MB)
    monkeypatch.setattr(mcp_server, "_relief_floor_bytes", 0)
    monkeypatch.setattr(mcp_server, "_force_malloc_release", fake_release)
    monkeypatch.setattr(mcp_server, "_get_current_rss_bytes", lambda: state["rss"])
    return state


def _check(state):
    return mcp_server._relieve_memory_pressure(state["rss"])


def test_nothing_happens_below_half_the_limit(watchdog, caplog):
    watchdog["rss"] = 400 * MB

    with caplog.at_level(logging.DEBUG, logger="omega.server"):
        assert _check(watchdog) == 400 * MB

    assert watchdog["reliefs"] == 0
    assert "Memory pressure relief" not in caplog.text


def test_steady_memory_above_half_is_relieved_once(watchdog, caplog):
    watchdog["rss"] = 600 * MB

    with caplog.at_level(logging.DEBUG, logger="omega.server"):
        for _ in range(20):  # five minutes of checks
            _check(watchdog)

    relief_logs = [r for r in caplog.records if "Memory pressure relief" in r.getMessage()]
    assert watchdog["reliefs"] == 1
    assert [r.levelno for r in relief_logs] == [logging.INFO]


def test_relief_runs_again_after_memory_grows(watchdog):
    watchdog["rss"] = 600 * MB
    _check(watchdog)

    watchdog["rss"] = 640 * MB
    _check(watchdog)
    assert watchdog["reliefs"] == 1

    watchdog["rss"] = 670 * MB
    _check(watchdog)
    assert watchdog["reliefs"] == 2


def test_relief_returns_the_memory_left_afterwards(watchdog):
    watchdog["rss"] = 700 * MB
    watchdog["frees"] = 250 * MB

    assert _check(watchdog) == 450 * MB


def test_still_near_the_limit_after_relief_is_a_warning(watchdog, caplog):
    watchdog["rss"] = 900 * MB

    with caplog.at_level(logging.DEBUG, logger="omega.server"):
        _check(watchdog)

    relief_logs = [r for r in caplog.records if "Memory pressure relief" in r.getMessage()]
    assert [r.levelno for r in relief_logs] == [logging.WARNING]
