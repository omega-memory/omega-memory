"""Memories from earlier sessions must be reachable from the current one.

A store search given a ``session_id`` returns only that session's memories;
the session-stop summaries rely on that. Two callers passed the current
session only to say who was asking, and so never saw anything an earlier
session had stored: the file hook that surfaces memories after Read and Edit,
and ``omega_query`` whenever the agent sent its session id.
"""

import asyncio

import pytest


def run_async(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _fresh_state(tmp_omega_dir):
    from omega.bridge import reset_memory

    reset_memory()
    yield
    reset_memory()


def _recording(seen: dict, result):
    def fake(**kwargs):
        seen.update(kwargs)
        return result

    return fake


class TestStoreContract:
    """What the callers below are working around, pinned so it stays deliberate."""

    def test_a_search_given_a_session_returns_only_that_session(self, store):
        store.store(content="alpha runbook: rotate the deploy key monthly", session_id="earlier", skip_inference=True)
        store.store(content="alpha runbook: rotate the signing key yearly", session_id="current", skip_inference=True)

        sessions = {r.metadata.get("session_id") for r in store.query("alpha runbook rotate key", session_id="current")}

        assert "earlier" not in sessions

    def test_a_search_given_no_session_reaches_every_session(self, store):
        store.store(content="alpha runbook: rotate the deploy key monthly", session_id="earlier", skip_inference=True)
        store.store(content="alpha runbook: rotate the signing key yearly", session_id="current", skip_inference=True)

        sessions = {r.metadata.get("session_id") for r in store.query("alpha runbook rotate key")}

        assert sessions == {"earlier", "current"}


class TestFileSurfacing:
    def test_the_file_hook_does_not_narrow_to_the_current_session(self, monkeypatch):
        import omega.bridge as bridge
        from omega.hooks import surface_memories

        seen: dict = {}
        monkeypatch.setattr(bridge, "query_structured", _recording(seen, []))

        surface_memories._surface_for_edit("/proj/app/config.py", "session-now", "/proj")

        assert seen["project"] == "/proj"
        assert seen["context_file"] == "/proj/app/config.py"
        assert seen.get("session_id") is None

    def test_a_memory_from_an_earlier_session_is_surfaced(self, monkeypatch):
        import omega.bridge as bridge
        from omega.hooks import surface_memories
        from omega.hooks._output import capture, captured_text

        earlier = {
            "id": "mem-0123456789ab",
            "content": "Decision: keep TIMEOUT_SECONDS at 30 in config.py",
            "event_type": "decision",
            "relevance": 0.9,
            "session_id": "session-earlier",
            "created_at": "",
        }
        monkeypatch.setattr(bridge, "query_structured", _recording({}, [earlier]))

        with capture() as lines:
            surface_memories._surface_for_edit("/proj/app/config.py", "session-now", "/proj")

        assert "keep TIMEOUT_SECONDS at 30" in captured_text(lines)


class TestOmegaQuery:
    def test_session_id_alone_searches_every_session(self, monkeypatch):
        import omega.bridge as bridge
        from omega.server.handlers import handle_omega_query

        seen: dict = {}
        monkeypatch.setattr(bridge, "query", _recording(seen, "Results: 0\n"))

        result = run_async(handle_omega_query({"query": "deploy key", "session_id": "abc-123"}))

        assert not result.get("isError", False)
        assert seen["session_id"] is None

    def test_scope_session_narrows_to_the_callers_session(self, monkeypatch):
        import omega.bridge as bridge
        from omega.server.handlers import handle_omega_query

        seen: dict = {}
        monkeypatch.setattr(bridge, "query", _recording(seen, "Results: 0\n"))

        run_async(handle_omega_query({"query": "deploy key", "session_id": "abc-123", "scope": "session"}))

        assert seen["session_id"] == "abc-123"
        assert seen["scope"] == "session"

    def test_schema_documents_scope_and_session_id(self):
        from omega.server.tool_schemas import TOOL_SCHEMAS

        properties = next(t for t in TOOL_SCHEMAS if t["name"] == "omega_query")["inputSchema"]["properties"]

        assert properties["scope"]["enum"] == ["project", "session"]
        assert "scope" in properties["session_id"]["description"]
