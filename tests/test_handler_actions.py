"""Tests for handler actions added/updated in v0.11.0.

Covers:
  - handle_omega_reflect: pro-only module graceful fallback
  - omega_memory action=link: manual edge creation
  - omega_memory action=flagged: flagged memory listing
  - omega_memory action=supersede: manual supersession
  - omega_stats action=forgetting_log: pro-only graceful fallback
  - omega_stats action=dedup: dedup stats
  - omega_stats action=milestones: milestone progress
  - handle_omega_browse: browse by type/session/recent
"""
from unittest.mock import patch

import pytest

from omega.server.handlers import (
    HANDLERS,
    handle_omega_reflect,
    handle_omega_memory,
    handle_omega_stats,
    handle_omega_browse,
)
from omega.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    """Real SQLiteStore in a temp directory."""
    db_path = str(tmp_path / "test.db")
    s = SQLiteStore(db_path)
    return s


@pytest.fixture
def mock_get_store(store):
    """Patch _get_store to return our real SQLiteStore."""
    with patch("omega.bridge._get_store", return_value=store):
        yield store


# ---------------------------------------------------------------------------
# omega_reflect
# ---------------------------------------------------------------------------


class TestOmegaReflect:
    """Tests for handle_omega_reflect — core module."""

    @pytest.mark.asyncio
    async def test_handler_in_registry(self):
        assert "omega_reflect" in HANDLERS

    @pytest.mark.asyncio
    async def test_stale_action_succeeds(self):
        """omega.reflect is core — stale action should work."""
        result = await handle_omega_reflect({"action": "stale"})
        assert not result.get("isError")

    @pytest.mark.asyncio
    async def test_unknown_action_returns_error(self):
        """Unknown action should return an error."""
        result = await handle_omega_reflect({"action": "bogus"})
        assert result.get("isError")


# ---------------------------------------------------------------------------
# omega_memory action=link
# ---------------------------------------------------------------------------


class TestOmegaMemoryLink:
    @pytest.mark.asyncio
    async def test_link_success(self, mock_get_store):
        store = mock_get_store
        # Store two memories
        id1 = store.store("Memory A", metadata={"event_type": "decision"})
        id2 = store.store("Memory B", metadata={"event_type": "decision"})

        result = await handle_omega_memory({
            "action": "link", "memory_id": id1, "target_id": id2, "edge_type": "related"
        })
        assert not result.get("isError")
        assert "Linked" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_link_missing_memory_id(self, mock_get_store):
        result = await handle_omega_memory({
            "action": "link", "memory_id": "", "target_id": "some-id"
        })
        assert result.get("isError")
        assert "memory_id" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_link_missing_target_id(self, mock_get_store):
        result = await handle_omega_memory({
            "action": "link", "memory_id": "some-id"
        })
        assert result.get("isError")
        assert "target_id" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# omega_memory action=flagged
# ---------------------------------------------------------------------------


class TestOmegaMemoryFlagged:
    @pytest.mark.asyncio
    async def test_no_flagged(self, mock_get_store):
        result = await handle_omega_memory({"action": "flagged"})
        assert not result.get("isError")
        assert "No memories flagged" in result["content"][0]["text"] or "No flagged" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_with_flagged_memories(self, mock_get_store):
        store = mock_get_store
        node_id = store.store("Bad memory", metadata={"event_type": "decision", "feedback_score": -5})
        result = await handle_omega_memory({"action": "flagged"})
        assert not result.get("isError")
        text = result["content"][0]["text"]
        assert "Flagged" in text or "score=" in text


# ---------------------------------------------------------------------------
# omega_memory action=supersede
# ---------------------------------------------------------------------------


class TestOmegaMemorySupersede:
    @pytest.mark.asyncio
    async def test_supersede_success(self, mock_get_store):
        store = mock_get_store
        id1 = store.store("Old decision", metadata={"event_type": "decision"})
        id2 = store.store("New decision", metadata={"event_type": "decision"})

        result = await handle_omega_memory({
            "action": "supersede", "memory_id": id1, "target_id": id2
        })
        assert not result.get("isError")
        assert "superseded" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_supersede_missing_memory_id(self, mock_get_store):
        result = await handle_omega_memory({
            "action": "supersede", "memory_id": "", "target_id": "some-id"
        })
        assert result.get("isError")

    @pytest.mark.asyncio
    async def test_supersede_without_target_is_simple_retirement(self, mock_get_store):
        result = await handle_omega_memory({
            "action": "supersede", "memory_id": "some-id"
        })
        assert result.get("isError")


# ---------------------------------------------------------------------------
# omega_stats action=forgetting_log
# ---------------------------------------------------------------------------


class TestOmegaStatsForgettingLog:
    @pytest.mark.asyncio
    async def test_forgetting_log_works(self, mock_get_store):
        """forgetting_log action should work in core build."""
        result = await handle_omega_stats({"action": "forgetting_log"})
        assert not result.get("isError")


# ---------------------------------------------------------------------------
# omega_stats action=dedup
# ---------------------------------------------------------------------------


class TestOmegaStatsDedup:
    @pytest.mark.asyncio
    async def test_dedup_stats(self, mock_get_store):
        store = mock_get_store
        store.store("Memory 1", metadata={"event_type": "decision"})
        store.store("Memory 2", metadata={"event_type": "decision"})

        result = await handle_omega_stats({"action": "dedup"})
        assert not result.get("isError")
        text = result["content"][0]["text"]
        assert "dedup" in text.lower()
        assert "0" in text  # dedup counters present


# ---------------------------------------------------------------------------
# omega_stats action=milestones
# ---------------------------------------------------------------------------


class TestOmegaStatsMilestones:
    @pytest.mark.asyncio
    async def test_milestones_empty(self, mock_get_store):
        result = await handle_omega_stats({"action": "milestones"})
        assert not result.get("isError")
        text = result["content"][0]["text"]
        assert "milestone" in text.lower() or "streak" in text.lower()

    @pytest.mark.asyncio
    async def test_milestones_with_data(self, mock_get_store):
        store = mock_get_store
        # Use very distinct content to avoid dedup
        topics = [
            "We chose PostgreSQL for the orders database",
            "Authentication uses JWT tokens not sessions",
            "Frontend built with React and TypeScript strict mode",
            "Deployed on AWS ECS with Fargate launch type",
            "Monitoring uses Datadog with custom dashboards",
        ]
        for topic in topics:
            store.store(topic, metadata={"event_type": "decision"})

        result = await handle_omega_stats({"action": "milestones"})
        text = result["content"][0]["text"]
        assert "milestone" in text.lower() or "streak" in text.lower()


# ---------------------------------------------------------------------------
# omega_browse
# ---------------------------------------------------------------------------


class TestOmegaBrowse:
    @pytest.mark.asyncio
    async def test_browse_recent_empty(self, mock_get_store):
        result = await handle_omega_browse({"browse_by": "recent"})
        assert not result.get("isError")
        assert "memor" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_browse_recent_with_data(self, mock_get_store):
        store = mock_get_store
        store.store("First memory", metadata={"event_type": "decision"})
        store.store("Second memory", metadata={"event_type": "lesson_learned"})

        result = await handle_omega_browse({"browse_by": "recent"})
        text = result["content"][0]["text"]
        assert "memor" in text.lower()

    @pytest.mark.asyncio
    async def test_browse_by_type(self, mock_get_store):
        store = mock_get_store
        store.store("Decision 1", metadata={"event_type": "decision"})
        store.store("Decision 2", metadata={"event_type": "decision"})
        store.store("Lesson 1", metadata={"event_type": "lesson_learned"})

        result = await handle_omega_browse({"browse_by": "type"})
        text = result["content"][0]["text"]
        assert "type" in text.lower() or "decision" in text

    @pytest.mark.asyncio
    async def test_browse_respects_limit(self, mock_get_store):
        store = mock_get_store
        for i in range(10):
            store.store(f"Memory {i}", metadata={"event_type": "decision"})

        result = await handle_omega_browse({"browse_by": "recent", "limit": 3})
        text = result["content"][0]["text"]
        lines = [l for l in text.split("\n") if l.strip().startswith("[")]
        assert len(lines) <= 3

    @pytest.mark.asyncio
    async def test_browse_by_session(self, mock_get_store):
        store = mock_get_store
        store.store("Sess memory", metadata={"event_type": "decision", "session_id": "sess-abc123"})

        result = await handle_omega_browse({"browse_by": "session", "session_id": "sess-abc123"})
        text = result["content"][0]["text"]
        assert "session" in text.lower()


# ---------------------------------------------------------------------------
# omega_stats unknown action
# ---------------------------------------------------------------------------


class TestOmegaStatsUnknown:
    @pytest.mark.asyncio
    async def test_unknown_action(self, mock_get_store):
        result = await handle_omega_stats({"action": "bogus"})
        assert result.get("isError")
        assert "Unknown" in result["content"][0]["text"]


class TestOmegaMemoryUnknown:
    @pytest.mark.asyncio
    async def test_unknown_action(self, mock_get_store):
        result = await handle_omega_memory({"action": "bogus"})
        assert result.get("isError")
        assert "Unknown" in result["content"][0]["text"]
