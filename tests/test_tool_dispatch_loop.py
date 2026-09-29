"""Tool calls and SessionStart maintenance must not stall the loop that serves hooks (audit finding D4).

Tool handlers are ``async def`` but do synchronous DB and embedding work.
Awaited on the server's event loop, which also accepts hook connections,
they held every other session's hooks: a hook went from 33 ms to ~400 ms
behind one query. The first SessionStart of a day also ran consolidation,
compaction and backup before answering (2.5 s at 10k memories, against a
5 s hook budget).
"""
import asyncio
import threading
import time

import pytest

from omega.server import mcp_server
from omega.server.hook_server import handlers


async def test_tool_handlers_run_off_the_server_loop(monkeypatch):
    seen: list[str] = []

    async def record_thread(arguments):
        seen.append(threading.current_thread().name)
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setitem(mcp_server.HANDLERS, "omega_test_thread", record_thread)

    result = await mcp_server.call_tool("omega_test_thread", {})

    assert result[0].text == "ok"
    assert seen and seen[0] != threading.current_thread().name


async def test_server_loop_keeps_serving_while_a_handler_blocks(monkeypatch):
    async def slow_sync_work(arguments):
        time.sleep(0.5)  # what a real handler's SQLite or ONNX call does to its loop
        return {"content": [{"type": "text", "text": "done"}]}

    monkeypatch.setitem(mcp_server.HANDLERS, "omega_test_slow", slow_sync_work)
    gaps: list[float] = []

    async def ticker():
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.01)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    tick = asyncio.create_task(ticker())
    try:
        result = await mcp_server.call_tool("omega_test_slow", {})
    finally:
        tick.cancel()

    assert result[0].text == "done"
    assert max(gaps) < 0.2, f"server loop stalled for {max(gaps):.2f}s"


async def test_handler_errors_still_come_back_as_tool_errors(monkeypatch):
    async def fails(arguments):
        raise ValueError("boom")

    monkeypatch.setitem(mcp_server.HANDLERS, "omega_test_fails", fails)

    result = await mcp_server.call_tool("omega_test_fails", {})

    assert result[0].text == "Error in omega_test_fails: boom"


def test_session_start_answers_before_running_periodic_maintenance(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    briefings: list[bool] = []

    def slow_maintenance():
        started.set()
        release.wait(5)

    monkeypatch.setattr(handlers.session_start, "run_periodic_maintenance", slow_maintenance)
    monkeypatch.setattr(handlers.session_start, "run", lambda payload, maintenance=True: briefings.append(maintenance))

    t0 = time.monotonic()
    response = handlers.handle_session_start({"session_id": "s1", "project": "/proj"})
    elapsed = time.monotonic() - t0

    try:
        assert response["error"] is None
        assert briefings == [False], "the briefing must not run maintenance inline"
        assert elapsed < 1.0
        assert started.wait(2), "maintenance was never started"
    finally:
        release.set()
