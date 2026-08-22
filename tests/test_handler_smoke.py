"""Smoke tests for MCP handler layer.

Exercises handlers end-to-end with a real SQLiteStore, verifying the
full path from handler arguments to MCP-compatible response dicts.
"""

import os
import re

import pytest
from unittest.mock import patch

from omega.server.handlers import HANDLERS


# ============================================================================
# Helpers
# ============================================================================


def _text(result: dict) -> str:
    """Extract text from an MCP response."""
    return result["content"][0]["text"]


def _is_error(result: dict) -> bool:
    """Check if an MCP response is an error."""
    return result.get("isError", False)


def _extract_node_id(store_response_text: str) -> str:
    """Extract the node ID from a store response like 'Stored abc123def (memory, permanent)'."""
    match = re.search(r"Stored\s+(\S+)\s+\(", store_response_text)
    assert match, f"Could not extract node ID from: {store_response_text}"
    return match.group(1)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_bridge(tmp_omega_dir):
    """Reset the bridge singleton so each test gets a fresh store."""
    from omega.bridge import reset_memory
    reset_memory()
    yield
    reset_memory()


# ============================================================================
# 1. omega_store
# ============================================================================


class TestOmegaStore:
    @pytest.mark.asyncio
    async def test_store_returns_success(self):
        """Store a memory and verify success response format."""
        result = await HANDLERS["omega_store"]({"content": "Test memory content"})
        assert not _is_error(result)
        text = _text(result)
        assert "Stored" in text
        assert "memory" in text

    @pytest.mark.asyncio
    async def test_store_empty_content_returns_error(self):
        """Storing with empty content should return an error."""
        result = await HANDLERS["omega_store"]({"content": ""})
        assert _is_error(result)
        assert "required" in _text(result).lower()

    @pytest.mark.asyncio
    async def test_store_with_text_alias(self):
        """The 'text' param should work as an alias for 'content'."""
        result = await HANDLERS["omega_store"]({"text": "Alias memory"})
        assert not _is_error(result)
        assert "Stored" in _text(result)

    @pytest.mark.asyncio
    async def test_store_auto_scopes_project_from_cwd(self):
        """When no project is passed, store should auto-scope from os.getcwd()."""
        fake_cwd = "/Users/test/Projects/omega"
        with patch("omega.server.handlers.os.getcwd", return_value=fake_cwd):
            result = await HANDLERS["omega_store"]({"content": "CWD-scoped memory"})
        assert not _is_error(result)
        node_id = _extract_node_id(_text(result))

        # Verify the memory was stored with the CWD as project
        from omega.bridge import _get_store
        stored = _get_store().get_node(node_id)
        assert stored is not None
        assert stored.metadata.get("project") == fake_cwd

    @pytest.mark.asyncio
    async def test_store_explicit_project_overrides_cwd(self):
        """Explicit project param should take precedence over CWD."""
        explicit = "/Users/test/Projects/acme-app"
        with patch("omega.server.handlers.os.getcwd", return_value="/some/other/dir"):
            result = await HANDLERS["omega_store"]({
                "content": "Explicitly scoped",
                "project": explicit,
            })
        assert not _is_error(result)
        node_id = _extract_node_id(_text(result))

        from omega.bridge import _get_store
        stored = _get_store().get_node(node_id)
        assert stored is not None
        assert stored.metadata.get("project") == explicit


# ============================================================================
# 2. omega_query
# ============================================================================


class TestOmegaQuery:
    @pytest.mark.asyncio
    async def test_store_then_query(self):
        """Store a memory and then query for it."""
        await HANDLERS["omega_store"]({"content": "The capital of France is Paris"})
        result = await HANDLERS["omega_query"]({"query": "capital of France"})
        assert not _is_error(result)
        text = _text(result)
        assert "Paris" in text

    @pytest.mark.asyncio
    async def test_query_empty_returns_error(self):
        """Querying with empty string should return an error."""
        result = await HANDLERS["omega_query"]({"query": ""})
        assert _is_error(result)


# ============================================================================
# 3-5. omega_delete_memory
# ============================================================================


class TestOmegaDeleteMemory:
    @pytest.mark.asyncio
    async def test_store_then_delete(self):
        """Store a memory, then delete it successfully."""
        store_result = await HANDLERS["omega_store"]({"content": "Ephemeral note"})
        node_id = _extract_node_id(_text(store_result))

        result = await HANDLERS["omega_delete_memory"]({"memory_id": node_id})
        assert not _is_error(result)
        assert "Deleted" in _text(result)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_error(self):
        """Deleting a non-existent memory ID should return an error."""
        result = await HANDLERS["omega_delete_memory"]({"memory_id": "nonexistent-id-12345"})
        assert _is_error(result)
        assert "not found" in _text(result).lower()

    @pytest.mark.asyncio
    async def test_session_ownership_blocks_foreign_delete(self):
        """Deleting a memory from a different session should be blocked without force."""
        store_result = await HANDLERS["omega_store"]({
            "content": "Owned by session A",
            "session_id": "session-aaa",
        })
        node_id = _extract_node_id(_text(store_result))

        # Attempt delete from a different session (should fail)
        result = await HANDLERS["omega_delete_memory"]({
            "memory_id": node_id,
            "caller_session_id": "session-bbb",
        })
        assert _is_error(result)
        assert "Ownership" in _text(result)

        # Force override should succeed
        result = await HANDLERS["omega_delete_memory"]({
            "memory_id": node_id,
            "caller_session_id": "session-bbb",
            "force": True,
        })
        assert not _is_error(result)
        assert "Deleted" in _text(result)


# ============================================================================
# 6. omega_feedback
# ============================================================================


class TestOmegaFeedback:
    @pytest.mark.asyncio
    async def test_store_then_feedback(self):
        """Store a memory and then record helpful feedback on it."""
        store_result = await HANDLERS["omega_store"]({"content": "Useful lesson about testing"})
        node_id = _extract_node_id(_text(store_result))

        result = await HANDLERS["omega_feedback"]({
            "memory_id": node_id,
            "rating": "helpful",
        })
        assert not _is_error(result)
        text = _text(result)
        assert "helpful" in text.lower()
        assert "Feedback recorded" in text


# ============================================================================
# 7-8. omega_clear_session
# ============================================================================


class TestOmegaClearSession:
    @pytest.mark.asyncio
    async def test_clear_session_returns_count(self):
        """Store memories in a session, clear it, verify count reported."""
        session_id = "test-session-clear"
        await HANDLERS["omega_store"]({"content": "Python decorator patterns for caching optimization", "session_id": session_id})
        await HANDLERS["omega_store"]({"content": "Database migration strategy using Alembic rollback", "session_id": session_id})

        result = await HANDLERS["omega_clear_session"]({"session_id": session_id})
        assert not _is_error(result)
        text = _text(result)
        assert "Cleared" in text
        # Should report at least 2 removed (may be more due to auto-extracted facts)
        match = re.search(r"(\d+) memories removed", text)
        assert match
        assert int(match.group(1)) >= 2

    @pytest.mark.asyncio
    async def test_clear_session_ownership_blocks_foreign(self):
        """Clearing another session should be blocked without force."""
        result = await HANDLERS["omega_clear_session"]({
            "session_id": "target-session",
            "caller_session_id": "other-session",
        })
        assert _is_error(result)
        assert "Ownership" in _text(result)


# ============================================================================
# 9. omega_consolidate
# ============================================================================


class TestOmegaConsolidate:
    @pytest.mark.asyncio
    async def test_consolidate_on_empty_store(self):
        """Consolidation on an empty store should succeed without errors."""
        result = await HANDLERS["omega_consolidate"]({"wait": True})
        assert not _is_error(result)
        text = _text(result)
        assert "Consolidation" in text


# ============================================================================
# 10-12. omega_backup
# ============================================================================


class TestOmegaBackup:
    @pytest.mark.asyncio
    async def test_export_to_valid_path(self, tmp_omega_dir):
        """Export to a valid path under the safe directory."""
        filepath = str(tmp_omega_dir / "export-test.json")
        with patch("omega.server.handlers._SAFE_EXPORT_DIR", tmp_omega_dir):
            result = await HANDLERS["omega_backup"]({
                "mode": "export",
                "filepath": filepath,
                "wait": True,
            })
        assert not _is_error(result)
        text = _text(result)
        assert "Export" in text

    @pytest.mark.asyncio
    async def test_export_then_import(self, tmp_omega_dir):
        """Export memories, then import them back."""
        await HANDLERS["omega_store"]({"content": "Roundtrip memory for backup"})
        filepath = str(tmp_omega_dir / "roundtrip.json")

        with patch("omega.server.handlers._SAFE_EXPORT_DIR", tmp_omega_dir):
            export_result = await HANDLERS["omega_backup"]({
                "mode": "export",
                "filepath": filepath,
                "wait": True,
            })
            assert not _is_error(export_result)

            import_result = await HANDLERS["omega_backup"]({
                "mode": "import",
                "filepath": filepath,
                "wait": True,
            })
            assert not _is_error(import_result)
            assert "Import" in _text(import_result)

    @pytest.mark.asyncio
    async def test_invalid_path_returns_error(self):
        """Exporting to a path outside the safe directory should return an error."""
        result = await HANDLERS["omega_backup"]({
            "mode": "export",
            "filepath": "/etc/passwd",
        })
        assert _is_error(result)
        assert "Path must be under" in _text(result)


# ============================================================================
# 13-14. omega_checkpoint / omega_resume_task
# ============================================================================


class TestOmegaCheckpoint:
    @pytest.mark.asyncio
    async def test_store_checkpoint(self):
        """Store a checkpoint and verify success."""
        result = await HANDLERS["omega_checkpoint"]({
            "task_title": "Smoke test task",
            "progress": "Step 1 of 3 complete",
            "plan": "Complete all three steps",
            "next_steps": "Do step 2",
        })
        assert not _is_error(result)
        text = _text(result)
        assert "Checkpoint" in text
        assert "Smoke test task" in text


class TestOmegaResumeTask:
    @pytest.mark.asyncio
    async def test_checkpoint_then_resume(self):
        """Store a checkpoint and then resume it."""
        await HANDLERS["omega_checkpoint"]({
            "task_title": "Resume smoke test",
            "progress": "Halfway done",
            "next_steps": "Finish the second half",
        })

        result = await HANDLERS["omega_resume_task"]({
            "task_title": "Resume smoke test",
        })
        assert not _is_error(result)
        text = _text(result)
        assert "checkpoint" in text.lower()
        assert "Halfway done" in text


# ============================================================================
# 15. omega_memory action=supersede
# ============================================================================


class TestOmegaMemorySupersede:
    @pytest.mark.asyncio
    async def test_supersede_marks_old_and_links_replacement(self):
        """memory_id is retired and target_id remains the active replacement."""
        old_result = await HANDLERS["omega_store"](
            {"content": "Deploy to staging every Monday morning"}
        )
        replacement_result = await HANDLERS["omega_store"](
            {"content": "Deploy to staging after every approved release"}
        )
        old_id = _extract_node_id(_text(old_result))
        replacement_id = _extract_node_id(_text(replacement_result))

        result = await HANDLERS["omega_memory"]({
            "action": "supersede",
            "memory_id": old_id,
            "target_id": replacement_id,
            "reason": "no longer relevant",
        })
        assert not _is_error(result)
        text = _text(result)
        assert "Superseded" in text
        assert "no longer relevant" in text
        assert old_id in text
        assert replacement_id in text

        from omega.bridge import _get_store

        store = _get_store()
        old = store.get_node(old_id, track_access=False)
        replacement = store.get_node(replacement_id, track_access=False)
        assert old.metadata["superseded"] is True
        assert not replacement.metadata.get("superseded", False)
        assert store._conn.execute(
            """SELECT COUNT(*) FROM edges
               WHERE source_id = ? AND target_id = ? AND edge_type = 'supersedes'""",
            (replacement_id, old_id),
        ).fetchone()[0] == 1

    @pytest.mark.asyncio
    async def test_supersede_already_superseded(self):
        """Superseding an already-superseded memory returns informative message."""
        store_result = await HANDLERS["omega_store"]({"content": "Use Redis for caching in all services"})
        node_id = _extract_node_id(_text(store_result))

        # Supersede once
        await HANDLERS["omega_memory"]({
            "action": "supersede",
            "target_id": node_id,
        })
        # Supersede again
        result = await HANDLERS["omega_memory"]({
            "action": "supersede",
            "target_id": node_id,
        })
        assert not _is_error(result)
        text = _text(result)
        assert "already superseded" in text.lower()

    @pytest.mark.asyncio
    async def test_supersede_nonexistent_returns_error(self):
        """Superseding a non-existent memory ID returns an error."""
        result = await HANDLERS["omega_memory"]({
            "action": "supersede",
            "target_id": "mem-nonexistent999",
        })
        assert _is_error(result)
        assert "not found" in _text(result).lower()

    @pytest.mark.asyncio
    async def test_supersede_missing_target_id_returns_error(self):
        """Supersede without target_id should return an error."""
        result = await HANDLERS["omega_memory"]({
            "action": "supersede",
        })
        assert _is_error(result)
        assert "target_id" in _text(result).lower()


# ============================================================================
# 16. omega_health
# ============================================================================


class TestOmegaHealth:
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Health check should return status information."""
        result = await HANDLERS["omega_health"]({})
        assert not _is_error(result)
        text = _text(result)
        # Health output contains status indicators
        assert any(keyword in text.upper() for keyword in ("HEALTHY", "OK", "WARNING", "CRITICAL", "STATUS"))
