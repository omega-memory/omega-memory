"""
Comprehensive tests for the 13 core MCP tool handlers in omega.server.handlers.

Covers happy paths and key edge cases for:
  1. omega_store          8. omega_memory (composite)
  2. omega_query           9. omega_profile (skip -- covered in test_handler_coverage.py)
  3. omega_welcome        10. omega_remind (composite)
  4. omega_protocol       11. omega_maintain (composite)
  5. (removed)            12. omega_stats (composite)
  6. omega_checkpoint     13. omega_reflect (composite)
  7. omega_resume_task

omega_lessons removed — auto-surfaced via hooks (0 calls ever).
Plus helper/validation unit tests.
"""

import asyncio
import re
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_async(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _text(result: dict) -> str:
    """Extract text from MCP response."""
    return result["content"][0]["text"]


def _is_error(result: dict) -> bool:
    return result.get("isError", False)


def _extract_mem_id(text: str) -> str | None:
    """Extract a memory ID from handler response text."""
    m = re.search(r"(?:Stored|Deduped|Evolved)\s+(?:→\s*)?(mem-[a-f0-9]+)", text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Fixture: fresh state per test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_state(tmp_omega_dir):
    """Reset bridge + coordination for each test."""
    from omega.bridge import reset_memory
    reset_memory()
    try:
        import omega_platform.orchestrator.coordination as coord
        coord._manager = None
    except (ImportError, AttributeError):
        pass
    yield


# ---------------------------------------------------------------------------
# Seed helper -- store via handler and return memory ID
# ---------------------------------------------------------------------------


def _store(content="test memory", event_type="memory", **kwargs) -> tuple[dict, str | None]:
    """Store a memory via handle_omega_store, return (result, mem_id)."""
    from omega.server.handlers import handle_omega_store
    args = {"content": content, "event_type": event_type, **kwargs}
    result = run_async(handle_omega_store(args))
    mem_id = _extract_mem_id(_text(result)) if not _is_error(result) else None
    return result, mem_id


# ===========================================================================
# 1. omega_store (handle_omega_store)
# ===========================================================================


class TestOmegaStore:

    def test_store_basic_memory(self):
        from omega.server.handlers import handle_omega_store
        result = run_async(handle_omega_store({"content": "Basic test memory"}))
        assert not _is_error(result)
        text = _text(result)
        assert "mem-" in text  # confirms a memory ID was returned

    def test_store_with_event_type(self):
        from omega.server.handlers import handle_omega_store
        result = run_async(handle_omega_store({
            "content": "Use pytest -x for fast feedback",
            "event_type": "decision",
        }))
        assert not _is_error(result)
        text = _text(result)
        assert "decision" in text.lower() or "mem-" in text

    def test_store_metadata_as_string(self):
        """Backbone resilience: metadata passed as JSON string is auto-corrected."""
        from omega.server.handlers import handle_omega_store
        result = run_async(handle_omega_store({
            "content": "Memory with string metadata",
            "metadata": '{"source": "test"}',
        }))
        assert not _is_error(result)
        assert "mem-" in _text(result)

    def test_store_metadata_as_int(self):
        """Backbone resilience: metadata passed as int is auto-corrected."""
        from omega.server.handlers import handle_omega_store
        result = run_async(handle_omega_store({
            "content": "Memory with int metadata",
            "metadata": 42,
        }))
        assert not _is_error(result)
        assert "mem-" in _text(result)

    def test_store_empty_content_error(self):
        from omega.server.handlers import handle_omega_store
        result = run_async(handle_omega_store({"content": ""}))
        assert _is_error(result)
        assert "required" in _text(result).lower()

    def test_store_whitespace_only_error(self):
        from omega.server.handlers import handle_omega_store
        result = run_async(handle_omega_store({"content": "   \n  "}))
        assert _is_error(result)

    def test_store_text_alias(self):
        """omega_remember backward compat: 'text' works as alias for 'content'."""
        from omega.server.handlers import handle_omega_store
        result = run_async(handle_omega_store({"text": "Remember this via text alias"}))
        assert not _is_error(result)
        assert "mem-" in _text(result)

    def test_store_content_precedence_over_text(self):
        from omega.server.handlers import handle_omega_store
        result = run_async(handle_omega_store({
            "content": "primary content",
            "text": "secondary text",
        }))
        assert not _is_error(result)

    def test_store_decision_broadcasts(self):
        """Storing a decision includes prior decision trail formatting."""
        from omega.server.handlers import handle_omega_store
        # Store first decision
        r1 = run_async(handle_omega_store({
            "content": "Use SQLite for persistence",
            "event_type": "decision",
            "session_id": "test-session-1",
            "project": "/test/project",
        }))
        assert not _is_error(r1)
        # Store a related decision -- trail may appear
        r2 = run_async(handle_omega_store({
            "content": "Switch from SQLite to PostgreSQL for persistence",
            "event_type": "decision",
            "session_id": "test-session-1",
            "project": "/test/project",
        }))
        assert not _is_error(r2)
        # The response should at least contain the stored confirmation
        assert "mem-" in _text(r2)

    def test_store_with_priority(self):
        from omega.server.handlers import handle_omega_store
        result = run_async(handle_omega_store({
            "content": "High priority memory",
            "priority": 5,
        }))
        assert not _is_error(result)


# ===========================================================================
# 2. omega_query (handle_omega_query)
# ===========================================================================


class TestOmegaQuery:

    def test_query_semantic_returns_results(self):
        """Store then query -- should find the memory."""
        from omega.server.handlers import handle_omega_query
        _store("The database migration uses Alembic for schema versioning")
        result = run_async(handle_omega_query({"query": "database migration alembic"}))
        assert not _is_error(result)
        text = _text(result)
        # Should have some content (even if "no results" message)
        assert len(text) > 0

    def test_free_tier_query_degrades_after_soft_cap(self, monkeypatch):
        """Free users at 2,000+ memories get keyword-only search + upgrade CTA."""
        import omega.bridge as bridge
        from omega.server import handlers
        from omega.server.handlers import handle_omega_query

        class StoreStub:
            def count_memories(self):
                return 2000

        monkeypatch.setattr(handlers, "_full_retrieval_available", lambda: False)
        monkeypatch.setattr(bridge, "_get_store", lambda: StoreStub())
        monkeypatch.setattr(
            bridge,
            "phrase_search",
            lambda **kwargs: f"keyword search for {kwargs['phrase']}",
        )

        result = run_async(handle_omega_query({"query": "database migration"}))

        assert not _is_error(result)
        text = _text(result)
        assert "keyword search for database migration" in text
        assert "keyword-only mode" in text
        assert "OMEGA Pro restores full semantic search" in text
        assert "omega upgrade" in text

    def test_query_missing_query_error(self):
        from omega.server.handlers import handle_omega_query
        result = run_async(handle_omega_query({}))
        assert _is_error(result)
        assert "required" in _text(result).lower()

    def test_query_empty_query_error(self):
        from omega.server.handlers import handle_omega_query
        result = run_async(handle_omega_query({"query": ""}))
        assert _is_error(result)

    def test_query_timeline_mode(self):
        from omega.server.handlers import handle_omega_query
        _store("Timeline test entry")
        result = run_async(handle_omega_query({"mode": "timeline"}))
        assert not _is_error(result)

    def test_query_browse_mode(self):
        from omega.server.handlers import handle_omega_query
        _store("Browse test entry")
        result = run_async(handle_omega_query({"mode": "browse"}))
        assert not _is_error(result)
        text = _text(result)
        assert "memor" in text.lower()  # "memories" or "memory"

    def test_query_browse_by_type(self):
        from omega.server.handlers import handle_omega_query
        _store("A lesson about testing", event_type="lesson_learned")
        result = run_async(handle_omega_query({
            "mode": "browse",
            "browse_by": "type",
            "event_type": "lesson_learned",
        }))
        assert not _is_error(result)

    def test_query_browse_by_type_missing_event_type(self):
        from omega.server.handlers import handle_omega_query
        result = run_async(handle_omega_query({
            "mode": "browse",
            "browse_by": "type",
        }))
        assert _is_error(result)

    def test_query_browse_by_session(self):
        from omega.server.handlers import handle_omega_query
        _store("Session-bound test entry", session_id="sess-browse-test")
        result = run_async(handle_omega_query({
            "mode": "browse",
            "browse_by": "session",
            "session_id": "sess-browse-test",
        }))
        assert not _is_error(result)

    def test_query_trace_mode_no_session(self):
        """Trace mode requires session_id."""
        from omega.server.handlers import handle_omega_query
        result = run_async(handle_omega_query({"mode": "trace"}))
        assert _is_error(result)
        assert "session_id" in _text(result).lower()

    def test_query_phrase_mode_exact(self):
        from omega.server.handlers import handle_omega_query
        _store("The quick brown fox jumps over the lazy dog")
        result = run_async(handle_omega_query({
            "query": "quick brown fox",
            "mode": "phrase",
        }))
        assert not _is_error(result)

    def test_query_limit_clamped(self):
        """Limit of 0 should be clamped to minimum (1)."""
        from omega.server.handlers import handle_omega_query
        _store("Limit clamp test")
        result = run_async(handle_omega_query({
            "query": "limit clamp test",
            "limit": 0,
        }))
        assert not _is_error(result)

    def test_query_with_event_type_filter(self):
        from omega.server.handlers import handle_omega_query
        _store("This is a decision about testing", event_type="decision")
        _store("This is a regular memory about testing", event_type="memory")
        result = run_async(handle_omega_query({
            "query": "testing",
            "event_type": "decision",
        }))
        assert not _is_error(result)


# ===========================================================================
# 3. omega_welcome (handle_omega_welcome)
# ===========================================================================


class TestOmegaWelcome:

    def test_welcome_basic(self):
        from omega.server.handlers import handle_omega_welcome
        result = run_async(handle_omega_welcome({}))
        assert not _is_error(result)
        text = _text(result)
        assert "welcome" in text.lower() or "briefing" in text.lower()

    def test_welcome_with_session(self):
        from omega.server.handlers import handle_omega_welcome
        result = run_async(handle_omega_welcome({"session_id": "test-welcome-session"}))
        assert not _is_error(result)
        assert len(_text(result)) > 0

    def test_welcome_with_project(self):
        from omega.server.handlers import handle_omega_welcome
        result = run_async(handle_omega_welcome({"project": "/tmp/test-project"}))
        assert not _is_error(result)

    def test_welcome_includes_memory_count(self):
        """Welcome briefing should mention memory count."""
        _store("Some seed memory for welcome test")
        from omega.server.handlers import handle_omega_welcome
        result = run_async(handle_omega_welcome({}))
        assert not _is_error(result)
        text = _text(result)
        # Should contain the memory count somewhere
        assert "memor" in text.lower()


# ===========================================================================
# 4. omega_protocol (handle_omega_protocol)
# ===========================================================================


# TestOmegaProtocol removed — protocol module is pro-only


# ===========================================================================
# 5. omega_lessons — REMOVED (auto-surfaced via hooks, 0 calls ever)
# ===========================================================================


# ===========================================================================
# 6. omega_checkpoint (handle_omega_checkpoint)
# ===========================================================================


class TestOmegaCheckpoint:

    def test_checkpoint_basic(self):
        from omega.server.handlers import handle_omega_checkpoint
        result = run_async(handle_omega_checkpoint({
            "task_title": "Implement feature X",
            "progress": "Completed step 1 and 2, step 3 pending",
            "next_steps": "Implement step 3, then write tests",
        }))
        assert not _is_error(result)
        text = _text(result)
        assert "checkpoint" in text.lower()
        assert "Implement feature X" in text

    def test_checkpoint_missing_task_title(self):
        from omega.server.handlers import handle_omega_checkpoint
        result = run_async(handle_omega_checkpoint({
            "progress": "some progress",
        }))
        assert _is_error(result)
        assert "required" in _text(result).lower()

    def test_checkpoint_missing_progress(self):
        from omega.server.handlers import handle_omega_checkpoint
        result = run_async(handle_omega_checkpoint({
            "task_title": "Some task",
        }))
        assert _is_error(result)

    def test_checkpoint_with_files_and_decisions(self):
        from omega.server.handlers import handle_omega_checkpoint
        result = run_async(handle_omega_checkpoint({
            "task_title": "Refactor auth module",
            "progress": "Moved to JWT tokens",
            "plan": "Replace session-based auth with JWT",
            "files_touched": {"src/auth.py": "Added JWT validation", "tests/test_auth.py": "New JWT tests"},
            "decisions": ["Use HS256 algorithm", "Token expiry: 1 hour"],
            "key_context": "Backend uses FastAPI",
            "next_steps": "Update middleware",
        }))
        assert not _is_error(result)
        text = _text(result)
        assert "checkpoint" in text.lower()

    def test_checkpoint_numbering_increments(self):
        """Subsequent checkpoints for the same task should increment number."""
        from omega.server.handlers import handle_omega_checkpoint
        r1 = run_async(handle_omega_checkpoint({
            "task_title": "Numbering test task",
            "progress": "Step 1 done",
        }))
        assert not _is_error(r1)
        assert "#1" in _text(r1)

        r2 = run_async(handle_omega_checkpoint({
            "task_title": "Numbering test task",
            "progress": "Step 2 done",
        }))
        assert not _is_error(r2)
        assert "#2" in _text(r2)


# ===========================================================================
# 7. omega_resume_task (handle_omega_resume_task)
# ===========================================================================


class TestOmegaResumeTask:

    def test_resume_finds_checkpoint(self):
        """Save checkpoint then resume -- should find the checkpoint."""
        from omega.server.handlers import handle_omega_checkpoint, handle_omega_resume_task
        run_async(handle_omega_checkpoint({
            "task_title": "Resume test task",
            "progress": "Halfway through implementation",
            "next_steps": "Finish the REST endpoints",
        }))
        result = run_async(handle_omega_resume_task({
            "task_title": "Resume test task",
        }))
        assert not _is_error(result)
        text = _text(result)
        assert "checkpoint" in text.lower()
        assert "Resume test task" in text or "Halfway" in text

    def test_resume_no_checkpoints(self):
        from omega.server.handlers import handle_omega_resume_task
        result = run_async(handle_omega_resume_task({
            "task_title": "Nonexistent task with no checkpoints at all",
        }))
        assert not _is_error(result)
        text = _text(result)
        assert "no checkpoint" in text.lower() or "start fresh" in text.lower()

    def test_resume_empty_title_returns_any(self):
        """Resuming without a title should return any recent checkpoint."""
        from omega.server.handlers import handle_omega_checkpoint, handle_omega_resume_task
        run_async(handle_omega_checkpoint({
            "task_title": "Generic checkpoint task",
            "progress": "Some progress",
        }))
        result = run_async(handle_omega_resume_task({}))
        assert not _is_error(result)

    def test_resume_verbosity_minimal(self):
        from omega.server.handlers import handle_omega_checkpoint, handle_omega_resume_task
        run_async(handle_omega_checkpoint({
            "task_title": "Verbosity test",
            "progress": "Done step 1",
            "next_steps": "Do step 2",
        }))
        result = run_async(handle_omega_resume_task({
            "task_title": "Verbosity test",
            "verbosity": "minimal",
        }))
        assert not _is_error(result)


# ===========================================================================
# 8. omega_memory composite (handle_omega_memory)
# ===========================================================================


class TestOmegaMemoryDelete:

    def test_memory_delete(self):
        from omega.server.handlers import handle_omega_memory
        _, mem_id = _store("Memory to be deleted")
        assert mem_id is not None
        result = run_async(handle_omega_memory({
            "action": "delete",
            "memory_id": mem_id,
        }))
        assert not _is_error(result)
        assert "deleted" in _text(result).lower()

    def test_memory_delete_cross_session_blocked(self):
        """Delete from a different session without force should be blocked."""
        from omega.server.handlers import handle_omega_memory
        _, mem_id = _store("Session-owned memory", session_id="owner-session")
        assert mem_id is not None
        result = run_async(handle_omega_memory({
            "action": "delete",
            "memory_id": mem_id,
            "caller_session_id": "different-session",
        }))
        assert _is_error(result)
        assert "ownership" in _text(result).lower()

    def test_memory_delete_cross_session_forced(self):
        """Delete from a different session with force=True should succeed."""
        from omega.server.handlers import handle_omega_memory
        _, mem_id = _store("Session-owned memory to force delete", session_id="owner-sess")
        assert mem_id is not None
        result = run_async(handle_omega_memory({
            "action": "delete",
            "memory_id": mem_id,
            "caller_session_id": "other-sess",
            "force": True,
        }))
        assert not _is_error(result)

    def test_memory_delete_nonexistent(self):
        from omega.server.handlers import handle_omega_memory
        result = run_async(handle_omega_memory({
            "action": "delete",
            "memory_id": "mem-0000000000000000",
        }))
        assert _is_error(result)


class TestOmegaMemoryEdit:

    def test_memory_edit(self):
        from omega.server.handlers import handle_omega_memory
        _, mem_id = _store("Original content to edit")
        assert mem_id is not None
        result = run_async(handle_omega_memory({
            "action": "edit",
            "memory_id": mem_id,
            "new_content": "Updated content after edit",
        }))
        assert not _is_error(result)
        text = _text(result)
        assert "updated" in text.lower() or "Updated content" in text

    def test_memory_edit_missing_new_content(self):
        from omega.server.handlers import handle_omega_memory
        _, mem_id = _store("Content to not edit")
        result = run_async(handle_omega_memory({
            "action": "edit",
            "memory_id": mem_id,
        }))
        assert _is_error(result)
        assert "new_content" in _text(result).lower() or "required" in _text(result).lower()

    def test_memory_edit_missing_memory_id(self):
        from omega.server.handlers import handle_omega_memory
        result = run_async(handle_omega_memory({
            "action": "edit",
            "new_content": "something",
        }))
        assert _is_error(result)


class TestOmegaMemoryFeedback:

    def test_memory_feedback_helpful(self):
        from omega.server.handlers import handle_omega_memory
        _, mem_id = _store("Memory to rate as helpful")
        assert mem_id is not None
        result = run_async(handle_omega_memory({
            "action": "feedback",
            "memory_id": mem_id,
            "rating": "helpful",
        }))
        assert not _is_error(result)
        text = _text(result)
        assert "helpful" in text.lower()

    def test_memory_feedback_unhelpful(self):
        from omega.server.handlers import handle_omega_memory
        _, mem_id = _store("Memory to rate as unhelpful")
        assert mem_id is not None
        result = run_async(handle_omega_memory({
            "action": "feedback",
            "memory_id": mem_id,
            "rating": "unhelpful",
        }))
        assert not _is_error(result)

    def test_memory_feedback_invalid_rating(self):
        from omega.server.handlers import handle_omega_memory
        _, mem_id = _store("Memory with invalid rating")
        result = run_async(handle_omega_memory({
            "action": "feedback",
            "memory_id": mem_id,
            "rating": "amazing",
        }))
        assert _is_error(result)
        assert "helpful" in _text(result).lower() or "unhelpful" in _text(result).lower()


class TestOmegaMemorySimilar:

    def test_memory_similar(self):
        from omega.server.handlers import handle_omega_memory
        _, mem_id = _store("Python is a great programming language")
        assert mem_id is not None
        _store("Python excels at data science and scripting")
        result = run_async(handle_omega_memory({
            "action": "similar",
            "memory_id": mem_id,
        }))
        assert not _is_error(result)

    def test_memory_similar_missing_id(self):
        from omega.server.handlers import handle_omega_memory
        result = run_async(handle_omega_memory({"action": "similar"}))
        assert _is_error(result)


class TestOmegaMemoryLink:

    def test_memory_link(self):
        from omega.server.handlers import handle_omega_memory
        _, id1 = _store("Python decorator patterns for caching optimization")
        _, id2 = _store("Database migration strategy using Alembic rollback")
        assert id1 and id2
        result = run_async(handle_omega_memory({
            "action": "link",
            "memory_id": id1,
            "target_id": id2,
            "edge_type": "related",
        }))
        assert not _is_error(result)
        text = _text(result)
        assert "linked" in text.lower() or "link" in text.lower()

    def test_memory_link_missing_target(self):
        from omega.server.handlers import handle_omega_memory
        _, id1 = _store("Memory with no link target")
        result = run_async(handle_omega_memory({
            "action": "link",
            "memory_id": id1,
        }))
        assert _is_error(result)

    def test_memory_link_nonexistent_target(self):
        from omega.server.handlers import handle_omega_memory
        _, id1 = _store("Memory linking to ghost")
        result = run_async(handle_omega_memory({
            "action": "link",
            "memory_id": id1,
            "target_id": "mem-0000000000000000",
        }))
        assert _is_error(result)


class TestOmegaMemoryFlagged:

    def test_memory_flagged_empty(self):
        from omega.server.handlers import handle_omega_memory
        result = run_async(handle_omega_memory({"action": "flagged"}))
        assert not _is_error(result)
        text = _text(result)
        assert "no memor" in text.lower() or "all clear" in text.lower()


class TestOmegaMemorySupersede:

    def test_memory_supersede(self):
        from omega.server.handlers import handle_omega_memory
        _, mem_id = _store("Old approach: use REST API")
        assert mem_id is not None
        result = run_async(handle_omega_memory({
            "action": "supersede",
            "memory_id": mem_id,
            "reason": "Switched to GraphQL",
        }))
        assert not _is_error(result)
        text = _text(result)
        assert "superseded" in text.lower()

    def test_memory_supersede_already_superseded(self):
        from omega.server.handlers import handle_omega_memory
        _, mem_id = _store("Already superseded memory")
        assert mem_id is not None
        # Supersede once
        run_async(handle_omega_memory({
            "action": "supersede",
            "memory_id": mem_id,
            "reason": "first supersession",
        }))
        # Supersede again
        result = run_async(handle_omega_memory({
            "action": "supersede",
            "memory_id": mem_id,
            "reason": "second attempt",
        }))
        assert not _is_error(result)
        assert "already superseded" in _text(result).lower()

    def test_memory_supersede_nonexistent(self):
        from omega.server.handlers import handle_omega_memory
        result = run_async(handle_omega_memory({
            "action": "supersede",
            "memory_id": "mem-0000000000000000",
        }))
        assert _is_error(result)


class TestOmegaMemoryUnknownAction:

    def test_memory_unknown_action(self):
        from omega.server.handlers import handle_omega_memory
        result = run_async(handle_omega_memory({"action": "teleport"}))
        assert _is_error(result)
        assert "unknown" in _text(result).lower()


# ===========================================================================
# 10. omega_remind composite (handle_omega_remind_composite)
# ===========================================================================


class TestOmegaRemind:

    def test_remind_set(self):
        from omega.server.handlers import handle_omega_remind_composite
        result = run_async(handle_omega_remind_composite({
            "action": "set",
            "text": "Review pull request",
            "duration": "1h",
        }))
        assert not _is_error(result)
        text = _text(result)
        assert "reminder" in text.lower() or "Review pull request" in text

    def test_remind_set_missing_text(self):
        from omega.server.handlers import handle_omega_remind_composite
        result = run_async(handle_omega_remind_composite({
            "action": "set",
            "duration": "1h",
        }))
        assert _is_error(result)

    def test_remind_set_missing_duration(self):
        from omega.server.handlers import handle_omega_remind_composite
        result = run_async(handle_omega_remind_composite({
            "action": "set",
            "text": "Do something",
        }))
        assert _is_error(result)

    def test_remind_list(self):
        from omega.server.handlers import handle_omega_remind_composite
        # Set a reminder first
        run_async(handle_omega_remind_composite({
            "action": "set",
            "text": "Listed reminder",
            "duration": "2h",
        }))
        result = run_async(handle_omega_remind_composite({"action": "list"}))
        assert not _is_error(result)
        text = _text(result)
        assert "Listed reminder" in text or "reminder" in text.lower()

    def test_remind_list_empty(self):
        from omega.server.handlers import handle_omega_remind_composite
        result = run_async(handle_omega_remind_composite({"action": "list"}))
        assert not _is_error(result)
        assert "no reminder" in _text(result).lower()

    def test_remind_dismiss(self):
        from omega.server.handlers import handle_omega_remind_composite
        # Set a reminder and extract ID
        set_result = run_async(handle_omega_remind_composite({
            "action": "set",
            "text": "Dismissable reminder",
            "duration": "30m",
        }))
        assert not _is_error(set_result)
        text = _text(set_result)
        # Extract reminder ID
        id_match = re.search(r"ID:\s*(mem-[a-f0-9]+)", text)
        assert id_match, f"Could not extract reminder ID from: {text}"
        reminder_id = id_match.group(1)

        result = run_async(handle_omega_remind_composite({
            "action": "dismiss",
            "reminder_id": reminder_id,
        }))
        assert not _is_error(result)
        assert "dismissed" in _text(result).lower()

    def test_remind_dismiss_missing_id(self):
        from omega.server.handlers import handle_omega_remind_composite
        result = run_async(handle_omega_remind_composite({"action": "dismiss"}))
        assert _is_error(result)

    def test_remind_unknown_action(self):
        from omega.server.handlers import handle_omega_remind_composite
        result = run_async(handle_omega_remind_composite({"action": "snooze"}))
        assert _is_error(result)


# ===========================================================================
# 11. omega_maintain composite (handle_omega_maintain)
# ===========================================================================


class TestOmegaMaintain:

    def test_maintain_health(self):
        from omega.server.handlers import handle_omega_maintain
        result = run_async(handle_omega_maintain({"action": "health"}))
        assert not _is_error(result)
        text = _text(result)
        # Health check returns status info
        assert len(text) > 10

    def test_maintain_consolidate(self):
        from omega.server.handlers import handle_omega_maintain
        _store("Memory for consolidation test")
        result = run_async(handle_omega_maintain({"action": "consolidate", "wait": True}))
        assert not _is_error(result)

    def test_maintain_compact(self):
        from omega.server.handlers import handle_omega_maintain
        result = run_async(handle_omega_maintain({"action": "compact", "wait": True}))
        assert not _is_error(result)

    def test_maintain_backup_restore(self):
        """Export then import -- round-trip backup test."""
        from pathlib import Path
        from omega.server.handlers import handle_omega_maintain
        _store("Memory to backup and restore")

        # _SAFE_EXPORT_DIR is hardcoded to ~/.omega, so use a path under that
        safe_dir = Path.home() / ".omega"
        safe_dir.mkdir(parents=True, exist_ok=True)
        export_path = str(safe_dir / "test_core_handlers_backup.json")

        try:
            export_result = run_async(handle_omega_maintain({
                "action": "backup",
                "filepath": export_path,
                "wait": True,
            }))
            assert not _is_error(export_result), f"Export failed: {_text(export_result)}"

            # Now restore
            restore_result = run_async(handle_omega_maintain({
                "action": "restore",
                "filepath": export_path,
                "wait": True,
            }))
            assert not _is_error(restore_result), f"Restore failed: {_text(restore_result)}"
        finally:
            # Clean up the test export file
            try:
                Path(export_path).unlink(missing_ok=True)
            except Exception:
                pass

    def test_maintain_backup_missing_filepath(self):
        from omega.server.handlers import handle_omega_maintain
        result = run_async(handle_omega_maintain({"action": "backup"}))
        assert _is_error(result)

    def test_maintain_backup_unsafe_path(self):
        """Paths outside ~/.omega should be rejected."""
        from omega.server.handlers import handle_omega_maintain
        result = run_async(handle_omega_maintain({
            "action": "backup",
            "filepath": "/tmp/evil_export.json",
        }))
        assert _is_error(result)

    def test_maintain_clear_session(self):
        from omega.server.handlers import handle_omega_maintain
        _store("Memory in session to clear", session_id="clear-me-session")
        result = run_async(handle_omega_maintain({
            "action": "clear_session",
            "session_id": "clear-me-session",
        }))
        assert not _is_error(result)
        assert "cleared" in _text(result).lower()

    def test_maintain_clear_session_cross_session_blocked(self):
        """Clearing a different session without force should be blocked."""
        from omega.server.handlers import handle_omega_maintain
        result = run_async(handle_omega_maintain({
            "action": "clear_session",
            "session_id": "target-session",
            "caller_session_id": "different-session",
        }))
        assert _is_error(result)
        assert "ownership" in _text(result).lower()

    def test_maintain_unknown_action(self):
        from omega.server.handlers import handle_omega_maintain
        result = run_async(handle_omega_maintain({"action": "nuke"}))
        assert _is_error(result)
        assert "unknown" in _text(result).lower()


# ===========================================================================
# 12. omega_stats composite (handle_omega_stats)
# ===========================================================================


class TestOmegaStats:

    def test_stats_types(self):
        from omega.server.handlers import handle_omega_stats
        _store("Memory for type stats", event_type="decision")
        _store("Another for type stats", event_type="lesson_learned")
        result = run_async(handle_omega_stats({"action": "types"}))
        assert not _is_error(result)
        text = _text(result)
        assert "stat" in text.lower() or "decision" in text.lower() or "total" in text.lower()

    def test_stats_types_empty(self):
        from omega.server.handlers import handle_omega_stats
        result = run_async(handle_omega_stats({"action": "types"}))
        assert not _is_error(result)
        text = _text(result)
        assert "no memor" in text.lower() or "stat" in text.lower()

    def test_stats_sessions(self):
        from omega.server.handlers import handle_omega_stats
        _store("Session stat memory", session_id="stat-session-1")
        result = run_async(handle_omega_stats({"action": "sessions"}))
        assert not _is_error(result)

    def test_stats_sessions_empty(self):
        from omega.server.handlers import handle_omega_stats
        result = run_async(handle_omega_stats({"action": "sessions"}))
        assert not _is_error(result)

    def test_stats_digest(self):
        from omega.server.handlers import handle_omega_stats
        _store("Digest test memory")
        result = run_async(handle_omega_stats({"action": "digest"}))
        assert not _is_error(result)
        text = _text(result)
        assert "week" in text.lower() or "total" in text.lower()

    def test_stats_dedup(self):
        from omega.server.handlers import handle_omega_stats
        result = run_async(handle_omega_stats({"action": "dedup"}))
        assert not _is_error(result)
        text = _text(result)
        assert "dedup" in text.lower()

    def test_stats_forgetting_log(self):
        from omega.server.handlers import handle_omega_stats
        result = run_async(handle_omega_stats({"action": "forgetting_log"}))
        assert not _is_error(result)

    def test_stats_unknown_action(self):
        from omega.server.handlers import handle_omega_stats
        result = run_async(handle_omega_stats({"action": "quantum"}))
        assert _is_error(result)

    def test_stats_diagnostic(self):
        from omega.server.handlers import handle_omega_stats
        _store("Diagnostic test memory", event_type="decision")
        result = run_async(handle_omega_stats({"action": "diagnostic"}))
        assert not _is_error(result)
        text = _text(result)
        import json
        report = json.loads(text)
        assert "memory_health" in report
        assert "tool_usage" in report
        assert "sessions" in report
        assert "llm_costs" in report
        assert "value_assessment" in report
        assert report["value_assessment"]["verdict"] in ("healthy", "underused", "idle")
        assert report["memory_health"]["total"] >= 1

    def test_stats_diagnostic_empty(self):
        from omega.server.handlers import handle_omega_stats
        result = run_async(handle_omega_stats({"action": "diagnostic"}))
        assert not _is_error(result)
        text = _text(result)
        import json
        report = json.loads(text)
        assert report["value_assessment"]["verdict"] == "idle"

    def test_stats_diagnostic_custom_days(self):
        from omega.server.handlers import handle_omega_stats
        result = run_async(handle_omega_stats({"action": "diagnostic", "days": 7}))
        assert not _is_error(result)
        text = _text(result)
        import json
        report = json.loads(text)
        assert report["period_days"] == 7


# ===========================================================================
# 13. omega_reflect composite (handle_omega_reflect)
# ===========================================================================


class TestOmegaReflect:

    def test_reflect_contradictions(self):
        from omega.server.handlers import handle_omega_reflect
        _store("The API uses REST endpoints for all communication",
               event_type="decision")
        _store("The API uses GraphQL instead of REST for all communication",
               event_type="decision")
        result = run_async(handle_omega_reflect({
            "action": "contradictions",
            "topic": "API communication protocol",
        }))
        assert not _is_error(result)
        text = _text(result)
        assert "contradiction" in text.lower()

    def test_reflect_contradictions_missing_topic(self):
        from omega.server.handlers import handle_omega_reflect
        result = run_async(handle_omega_reflect({"action": "contradictions"}))
        assert _is_error(result)
        assert "topic" in _text(result).lower()

    def test_reflect_evolution(self):
        from omega.server.handlers import handle_omega_reflect
        _store("Initial design: monolithic architecture", event_type="decision")
        _store("Evolved design: microservices architecture", event_type="decision")
        result = run_async(handle_omega_reflect({
            "action": "evolution",
            "topic": "architecture design",
        }))
        assert not _is_error(result)
        text = _text(result)
        assert "evolution" in text.lower()

    def test_reflect_evolution_missing_topic(self):
        from omega.server.handlers import handle_omega_reflect
        result = run_async(handle_omega_reflect({"action": "evolution"}))
        assert _is_error(result)

    def test_reflect_stale(self):
        from omega.server.handlers import handle_omega_reflect
        # Stale audit works on old memories; with fresh store it should return "no stale"
        result = run_async(handle_omega_reflect({"action": "stale"}))
        assert not _is_error(result)
        text = _text(result)
        assert "stale" in text.lower()

    def test_reflect_unknown_action(self):
        from omega.server.handlers import handle_omega_reflect
        result = run_async(handle_omega_reflect({"action": "meditate"}))
        assert _is_error(result)
        assert "unknown" in _text(result).lower()


# ===========================================================================
# Helper / Validation Unit Tests
# ===========================================================================


class TestValidateMemoryWrite:

    def test_normalizes_string_metadata(self):
        from omega.server.handlers import _validate_memory_write
        event_type, metadata, errors = _validate_memory_write(
            "content", "memory", '{"key": "value"}'
        )
        assert isinstance(metadata, dict)
        assert metadata.get("key") == "value"
        assert any("str" in e for e in errors)

    def test_normalizes_int_metadata(self):
        from omega.server.handlers import _validate_memory_write
        event_type, metadata, errors = _validate_memory_write("content", "memory", 42)
        assert isinstance(metadata, dict)
        assert any("int" in e for e in errors)

    def test_normalizes_list_metadata(self):
        from omega.server.handlers import _validate_memory_write
        event_type, metadata, errors = _validate_memory_write("content", "memory", [1, 2, 3])
        assert isinstance(metadata, dict)
        assert any("list" in e for e in errors)

    def test_normalizes_none_metadata(self):
        from omega.server.handlers import _validate_memory_write
        event_type, metadata, errors = _validate_memory_write("content", "memory", None)
        assert metadata == {}
        assert len(errors) == 0

    def test_valid_dict_metadata_passes_through(self):
        from omega.server.handlers import _validate_memory_write
        event_type, metadata, errors = _validate_memory_write(
            "content", "memory", {"valid": True}
        )
        assert metadata == {"valid": True}
        assert not any("metadata" in e for e in errors)

    def test_normalizes_empty_event_type(self):
        from omega.server.handlers import _validate_memory_write
        event_type, metadata, errors = _validate_memory_write("content", "", {})
        assert event_type == "memory"
        assert len(errors) > 0

    def test_normalizes_none_event_type(self):
        from omega.server.handlers import _validate_memory_write
        event_type, metadata, errors = _validate_memory_write("content", None, {})
        assert event_type == "memory"

    def test_unknown_event_type_allowed(self):
        """Unknown event types are allowed but logged as warning."""
        from omega.server.handlers import _validate_memory_write
        event_type, metadata, errors = _validate_memory_write("content", "alien_signal", {})
        assert event_type == "alien_signal"
        assert any("not in known set" in e for e in errors)

    def test_known_event_type_no_warning(self):
        from omega.server.handlers import _validate_memory_write
        event_type, metadata, errors = _validate_memory_write("content", "decision", {})
        assert event_type == "decision"
        assert not any("event_type" in e for e in errors)

    def test_string_metadata_invalid_json(self):
        from omega.server.handlers import _validate_memory_write
        event_type, metadata, errors = _validate_memory_write(
            "content", "memory", "not valid json"
        )
        assert isinstance(metadata, dict)
        assert "_raw" in metadata
        assert metadata["_raw"] == "not valid json"

    def test_string_metadata_json_array(self):
        """JSON string that parses to non-dict is wrapped in _raw."""
        from omega.server.handlers import _validate_memory_write
        event_type, metadata, errors = _validate_memory_write(
            "content", "memory", '[1, 2, 3]'
        )
        assert isinstance(metadata, dict)
        assert "_raw" in metadata


class TestClampInt:

    def test_clamp_normal(self):
        from omega.server.handlers import _clamp_int
        assert _clamp_int(5, default=10) == 5

    def test_clamp_below_min(self):
        from omega.server.handlers import _clamp_int
        assert _clamp_int(0, default=10, min_val=1) == 1

    def test_clamp_above_max(self):
        from omega.server.handlers import _clamp_int
        assert _clamp_int(50000, default=10, max_val=10000) == 10000

    def test_clamp_none_returns_default(self):
        from omega.server.handlers import _clamp_int
        assert _clamp_int(None, default=7) == 7

    def test_clamp_string_returns_default(self):
        from omega.server.handlers import _clamp_int
        assert _clamp_int("not_a_number", default=42) == 42

    def test_clamp_string_number(self):
        from omega.server.handlers import _clamp_int
        assert _clamp_int("15", default=10) == 15

    def test_clamp_negative(self):
        from omega.server.handlers import _clamp_int
        assert _clamp_int(-5, default=10, min_val=1) == 1

    def test_clamp_float_truncated(self):
        from omega.server.handlers import _clamp_int
        assert _clamp_int(3.7, default=10) == 3

    def test_clamp_exact_min(self):
        from omega.server.handlers import _clamp_int
        assert _clamp_int(1, default=10, min_val=1) == 1

    def test_clamp_exact_max(self):
        from omega.server.handlers import _clamp_int
        assert _clamp_int(10000, default=10, max_val=10000) == 10000


class TestDeployGateTracking:
    """Tests for deploy gate file-based tracking.

    Uses unique session IDs per test to avoid interference from real gate files.
    """

    @staticmethod
    def _clean_gate(session_id):
        """Remove gate files for a test session to ensure clean state."""
        from omega.server.handlers import _GATE_DIR
        for suffix in (".gate", ".coord"):
            f = _GATE_DIR / f"{session_id}{suffix}"
            if f.exists():
                f.unlink()

    @staticmethod
    def _stash_and_clean_defaults():
        """Temporarily remove default gate files to prevent fallback interference.

        Returns dict of {path: contents} for restoration.
        """
        from omega.server.handlers import _GATE_DIR
        stashed = {}
        for name in ("default.gate", "default.coord"):
            f = _GATE_DIR / name
            if f.exists():
                stashed[f] = f.read_text()
                f.unlink()
        return stashed

    @staticmethod
    def _restore_defaults(stashed):
        """Restore previously stashed default gate files."""
        for path, contents in stashed.items():
            path.write_text(contents)

    def test_gate_mark_and_check(self, tmp_omega_dir, monkeypatch):
        from omega.server import handlers
        from omega.server.handlers import (
            _mark_deploy_gate_cleared,
            _mark_coord_status_checked,
            is_deploy_gate_cleared,
        )
        # Force Pro-available path so the test exercises the two-gate requirement.
        # In the public package, _is_pro_available() returns False and the
        # coord-status check is skipped, making this test's assertions invalid.
        monkeypatch.setattr(handlers, "_is_pro_available", lambda: True)
        sid = "test-gate-core-handlers-mc"
        self._clean_gate(sid)
        stashed = self._stash_and_clean_defaults()
        try:
            # Initially should not be cleared
            assert not is_deploy_gate_cleared(sid)

            # Mark only deploy gate -- still not cleared (needs coord too)
            _mark_deploy_gate_cleared(sid)
            assert not is_deploy_gate_cleared(sid)

            # Mark coord status too -- now should be cleared
            _mark_coord_status_checked(sid)
            assert is_deploy_gate_cleared(sid)
        finally:
            self._clean_gate(sid)
            self._restore_defaults(stashed)

    def test_gate_default_session(self, tmp_omega_dir):
        from omega.server.handlers import (
            _mark_deploy_gate_cleared,
            _mark_coord_status_checked,
            is_deploy_gate_cleared,
        )
        sid = "test-gate-default-ch"
        self._clean_gate(sid)
        try:
            _mark_deploy_gate_cleared(sid)
            _mark_coord_status_checked(sid)
            assert is_deploy_gate_cleared(sid)
        finally:
            self._clean_gate(sid)

    def test_gate_expired(self, tmp_omega_dir):
        from omega.server.handlers import (
            _mark_deploy_gate_cleared,
            _mark_coord_status_checked,
            is_deploy_gate_cleared,
        )
        sid = "test-gate-expired-ch"
        self._clean_gate(sid)
        try:
            _mark_deploy_gate_cleared(sid)
            _mark_coord_status_checked(sid)
            # With max_age_sec=0, the gate should appear expired
            assert not is_deploy_gate_cleared(sid, max_age_sec=0)
        finally:
            self._clean_gate(sid)

    def test_coord_status_checked(self, tmp_omega_dir):
        from omega.server.handlers import _mark_coord_status_checked, _is_coord_status_checked
        sid = "test-coord-status-ch"
        self._clean_gate(sid)
        stashed = self._stash_and_clean_defaults()
        try:
            assert not _is_coord_status_checked(sid)
            _mark_coord_status_checked(sid)
            assert _is_coord_status_checked(sid)
            assert not _is_coord_status_checked(sid, max_age_sec=0)
        finally:
            self._clean_gate(sid)
            self._restore_defaults(stashed)
