"""Tests for the ``max_chars`` parameter of omega_query.

Results used to render every memory as a fixed 200-character preview, so an MCP
client could never read a long memory. ``max_chars`` keeps that default and lets
the caller widen it or, with 0, render the full content.
"""

import asyncio

import pytest


def run_async(coro):
    return asyncio.run(coro)


def _text(result: dict) -> str:
    return result["content"][0]["text"]


def _is_error(result: dict) -> bool:
    return result.get("isError", False)


LONG_MARKER = "zebra-quartz-lantern"
LONG_CONTENT = (
    f"{LONG_MARKER} decision record: " + "the migration runbook has many steps. " * 12 + "END-OF-MEMORY"
)


@pytest.fixture(autouse=True)
def _fresh_state(tmp_omega_dir):
    from omega.bridge import reset_memory

    reset_memory()
    yield
    reset_memory()


def _store_long_memory() -> None:
    from omega.server.handlers import handle_omega_store

    result = run_async(handle_omega_store({"content": LONG_CONTENT, "event_type": "decision"}))
    assert not _is_error(result)


class TestContentPreview:
    def test_default_cuts_at_200_with_marker(self):
        from omega.bridge import _content_preview

        preview = _content_preview(LONG_CONTENT)
        assert preview == LONG_CONTENT[:200] + "..."

    def test_short_content_is_untouched(self):
        from omega.bridge import _content_preview

        assert _content_preview("short", 200) == "short"

    def test_exact_length_is_untouched(self):
        from omega.bridge import _content_preview

        content = "x" * 200
        assert _content_preview(content, 200) == content

    def test_zero_returns_full_content(self):
        from omega.bridge import _content_preview

        assert _content_preview(LONG_CONTENT, 0) == LONG_CONTENT

    def test_custom_width(self):
        from omega.bridge import _content_preview

        assert _content_preview(LONG_CONTENT, 50) == LONG_CONTENT[:50] + "..."


class TestPhraseModeMaxChars:
    def test_default_preview_is_unchanged(self):
        from omega.server.handlers import handle_omega_query

        _store_long_memory()
        result = run_async(handle_omega_query({"query": LONG_MARKER, "mode": "phrase"}))
        assert not _is_error(result)
        text = _text(result)
        assert LONG_CONTENT[:200] + "..." in text
        assert "END-OF-MEMORY" not in text

    def test_zero_returns_full_content(self):
        from omega.server.handlers import handle_omega_query

        _store_long_memory()
        result = run_async(handle_omega_query({"query": LONG_MARKER, "mode": "phrase", "max_chars": 0}))
        assert not _is_error(result)
        text = _text(result)
        assert LONG_CONTENT in text
        assert LONG_CONTENT[:200] + "..." not in text

    def test_custom_width(self):
        from omega.server.handlers import handle_omega_query

        _store_long_memory()
        result = run_async(handle_omega_query({"query": LONG_MARKER, "mode": "phrase", "max_chars": 60}))
        assert not _is_error(result)
        text = _text(result)
        assert LONG_CONTENT[:60] + "..." in text
        assert "END-OF-MEMORY" not in text

    def test_negative_is_clamped_to_full_content(self):
        from omega.server.handlers import handle_omega_query

        _store_long_memory()
        result = run_async(handle_omega_query({"query": LONG_MARKER, "mode": "phrase", "max_chars": -5}))
        assert not _is_error(result)
        assert LONG_CONTENT in _text(result)

    def test_invalid_value_falls_back_to_default(self):
        from omega.server.handlers import handle_omega_query

        _store_long_memory()
        result = run_async(handle_omega_query({"query": LONG_MARKER, "mode": "phrase", "max_chars": "wide"}))
        assert not _is_error(result)
        assert LONG_CONTENT[:200] + "..." in _text(result)


class TestSemanticModeMaxChars:
    def test_semantic_passes_max_chars_to_bridge(self, monkeypatch):
        import omega.bridge as bridge
        from omega.server.handlers import handle_omega_query

        seen = {}

        def fake_query(**kwargs):
            seen.update(kwargs)
            return "Results: 0\n"

        monkeypatch.setattr(bridge, "query", fake_query)
        result = run_async(handle_omega_query({"query": "anything", "max_chars": 0}))
        assert not _is_error(result)
        assert seen["max_chars"] == 0

    def test_semantic_default_is_200(self, monkeypatch):
        import omega.bridge as bridge
        from omega.server.handlers import handle_omega_query

        seen = {}

        def fake_query(**kwargs):
            seen.update(kwargs)
            return "Results: 0\n"

        monkeypatch.setattr(bridge, "query", fake_query)
        run_async(handle_omega_query({"query": "anything"}))
        assert seen["max_chars"] == 200

    def test_bridge_query_renders_full_content_when_zero(self):
        from omega.bridge import query

        _store_long_memory()
        text = query(query_text=LONG_MARKER, limit=5, max_chars=0)
        if LONG_MARKER in text:
            assert LONG_CONTENT in text
            assert LONG_CONTENT[:200] + "..." not in text

    def test_unified_passes_max_chars_to_memory_search(self, monkeypatch):
        import omega.bridge as bridge
        from omega.server.handlers import handle_omega_query

        seen = {}

        def fake_query(**kwargs):
            seen.update(kwargs)
            return "Results: 0\n"

        monkeypatch.setattr(bridge, "query", fake_query)
        result = run_async(handle_omega_query({"query": "anything", "mode": "unified", "max_chars": 0}))
        assert not _is_error(result)
        assert seen["max_chars"] == 0


class TestSchema:
    def test_omega_query_schema_declares_max_chars(self):
        from omega.server.tool_schemas import TOOL_SCHEMAS

        schema = next(t for t in TOOL_SCHEMAS if t["name"] == "omega_query")
        prop = schema["inputSchema"]["properties"]["max_chars"]
        assert prop["type"] == "integer"
        assert prop["default"] == 200
        assert prop["minimum"] == 0
