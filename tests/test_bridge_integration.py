"""Bridge integration tests -- real SQLiteStore, no mocking.

Tests the public bridge API end-to-end with a temporary OMEGA directory
and a fresh SQLiteStore per test (via the _reset_bridge fixture).
"""

import os
import pytest

from omega.bridge import (
    auto_capture,
    clear_session,
    delete_memory,
    edit_memory,
    export_memories,
    import_memories,
    query,
    reset_memory,
    status,
    store,
    welcome,
)


# ---------------------------------------------------------------------------
# Fixture: reset bridge singleton between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_bridge(tmp_omega_dir):
    """Reset the bridge singleton so each test gets a fresh store."""
    reset_memory()
    yield
    reset_memory()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_node_id(result_str: str) -> str:
    """Extract the node ID from a store() return string like 'Stored mem-abc123 ...'."""
    # Format: "Stored <id> (<event_type>, <ttl>)"
    parts = result_str.split()
    if len(parts) >= 2 and parts[0] == "Stored":
        return parts[1]
    raise ValueError(f"Could not extract node ID from: {result_str!r}")


# ============================================================================
# 1. store -- basic
# ============================================================================


def test_store_basic():
    """Store a memory and verify the returned confirmation string."""
    result = store("The quick brown fox jumped over the lazy dog near the riverbank")
    assert isinstance(result, str)
    assert "Stored" in result or "Deduped" in result or "Evolved" in result
    # Default event_type is "memory"
    if "Stored" in result:
        assert "memory" in result


# ============================================================================
# 2. store -- with metadata and event_type
# ============================================================================


def test_store_with_metadata():
    """Store with explicit event_type and metadata, verify they flow through."""
    result = store(
        "Always run pytest before committing Python changes to the repository",
        event_type="lesson_learned",
        metadata={"source": "test", "tags": ["testing", "ci"]},
    )
    assert isinstance(result, str)
    assert "Stored" in result or "Deduped" in result or "Evolved" in result
    if "Stored" in result:
        assert "lesson_learned" in result


# ============================================================================
# 3. query -- basic
# ============================================================================


def test_query_basic():
    """Store a memory then query for it; results should contain the content."""
    store(
        "Postgres connection pooling reduces latency for high-traffic applications",
        event_type="lesson_learned",
    )
    result = query("postgres connection pooling latency")
    assert isinstance(result, str)
    # The query result should surface the stored content
    assert "Postgres" in result or "postgres" in result or "pooling" in result


# ============================================================================
# 4. query -- event_type filter
# ============================================================================


def test_query_with_event_type_filter():
    """Store different event types, query with filter for a specific one."""
    store(
        "Redis caching dramatically improves response times for read-heavy workloads",
        event_type="lesson_learned",
    )
    store(
        "Decided to use Redis for session storage instead of Memcached for this project",
        event_type="decision",
    )

    # Query with event_type filter -- should only find the decision
    result = query("Redis caching session storage", event_type="decision")
    assert isinstance(result, str)
    # The decision should appear in results
    assert "session storage" in result or "Memcached" in result or "decision" in result.lower()


# ============================================================================
# 5. query -- session scope
# ============================================================================


def test_query_with_session_scope():
    """Store with different session IDs, query scoped to one session."""
    store(
        "Session alpha: configured Nginx reverse proxy for load balancing the cluster",
        event_type="memory",
        session_id="session-alpha-111",
    )
    store(
        "Session beta: set up Cloudflare DNS records for the production domain",
        event_type="memory",
        session_id="session-beta-222",
    )

    result = query(
        "Nginx proxy load balancing",
        session_id="session-alpha-111",
        scope="session",
    )
    assert isinstance(result, str)
    # Session-scoped query should find session-alpha content
    # (may or may not exclude beta depending on implementation, but alpha should appear)


# ============================================================================
# 6. delete_memory -- success
# ============================================================================


def test_delete_memory():
    """Store a memory then delete it; verify success response."""
    result_str = store("Temporary test memory that will be deleted shortly after creation")
    node_id = _extract_node_id(result_str)

    result = delete_memory(node_id)
    assert isinstance(result, dict)
    assert result["success"] is True
    assert result["deleted_id"] == node_id


# ============================================================================
# 7. delete_memory -- non-existent
# ============================================================================


def test_delete_memory_nonexistent():
    """Deleting a non-existent memory should return an error response."""
    result = delete_memory("mem-does-not-exist-at-all-12345")
    assert isinstance(result, dict)
    assert result["success"] is False
    assert "error" in result


# ============================================================================
# 8. edit_memory
# ============================================================================


def test_edit_memory():
    """Store a memory, edit it, and verify old/new content previews."""
    original_text = "Original content for testing the edit memory bridge function"
    result_str = store(original_text)
    node_id = _extract_node_id(result_str)

    new_text = "Updated content after editing the memory through the bridge layer"
    result = edit_memory(node_id, new_text)
    assert isinstance(result, dict)
    assert result["success"] is True
    assert result["id"] == node_id
    assert "Original" in result["old_content_preview"]
    assert "Updated" in result["new_content_preview"]


# ============================================================================
# 9. clear_session
# ============================================================================


def test_clear_session():
    """Store memories in two sessions, clear one, verify count."""
    sid_keep = "session-keep-aaa"
    sid_clear = "session-clear-bbb"

    store(
        "Memory in the session that will be kept after clearing the other session",
        session_id=sid_keep,
    )
    store(
        "The azure butterfly migration pattern occurs between November and March across the Pacific",
        session_id=sid_clear,
    )
    store(
        "Quantum entanglement was experimentally verified by Alain Aspect in 1982 using Bell tests",
        session_id=sid_clear,
    )

    result = clear_session(sid_clear)
    assert isinstance(result, dict)
    assert result["session_id"] == sid_clear
    assert result["removed"] >= 2

    # Verify the kept session's memory is still queryable
    q = query("session that will be kept", session_id=sid_keep)
    assert isinstance(q, str)


# ============================================================================
# 10. export / import round trip
# ============================================================================


def test_export_import_roundtrip(tmp_omega_dir):
    """Store memories, export, reset, import, then query to verify."""
    store(
        "Roundtrip test memory: always validate exports before deploying to production",
        event_type="lesson_learned",
    )
    store(
        "Roundtrip test decision: chose PostgreSQL over MySQL for the new microservice",
        event_type="decision",
    )

    export_path = str(tmp_omega_dir / "export_test.json")

    # Export
    export_result = export_memories(export_path)
    assert isinstance(export_result, str)
    assert "Export" in export_result
    assert os.path.exists(export_path)

    # Reset the store
    reset_memory()

    # Import
    import_result = import_memories(export_path, clear_existing=True)
    assert isinstance(import_result, str)
    assert "Import" in import_result

    # Query to verify data survived the round trip
    q = query("roundtrip validate exports production")
    assert isinstance(q, str)
    assert "roundtrip" in q.lower() or "validate" in q.lower() or "export" in q.lower()


# ============================================================================
# 11. welcome
# ============================================================================


def test_welcome():
    """Welcome should return a dict without raising."""
    result = welcome()
    assert isinstance(result, dict)


# ============================================================================
# 12. status
# ============================================================================


def test_status():
    """Status should return a dict with expected keys."""
    result = status()
    assert isinstance(result, dict)
    assert "ok" in result
    assert "status" in result
    assert "node_count" in result
    assert "backend" in result
    assert result["backend"] == "sqlite"


# ============================================================================
# 13. diagnostic_report windows
# ============================================================================


def test_diagnostic_windows_count_only_rows_inside_them(monkeypatch):
    """Every diagnostic window compares timestamps in the stored format.

    Rows store isoformat text ('2026-09-22T08:00:00+00:00'); comparing it with
    SQLite datetime() text ('2026-09-22 08:00:01') sorted 'T' after ' ', so rows
    from the cutoff's calendar day landed on the wrong side of every window.
    """
    import sqlite3
    import sys
    import types
    from datetime import datetime, timedelta, timezone

    from omega.bridge import _get_store, diagnostic_report

    now = datetime.now(timezone.utc)

    def just_outside(days):
        return (now - timedelta(days=days, seconds=1)).isoformat()

    def just_inside(days):
        return (now - timedelta(days=days) + timedelta(hours=1)).isoformat()

    db = _get_store()
    node_ids = []
    for content, created_at in [
        ("Postgres vacuum schedule for the analytics replica", just_outside(7)),
        ("Redis eviction policy switched to allkeys-lru", just_inside(7)),
        ("Terraform workspace naming convention for staging", just_outside(14)),
        ("Grafana alert routing to the on-call rotation", just_inside(14)),
    ]:
        node_id = db.store(content=content)
        db._conn.execute("UPDATE memories SET created_at = ? WHERE node_id = ?", (created_at, node_id))
        node_ids.append(node_id)
    db._conn.commit()
    assert len(set(node_ids)) == 4

    # Pro's coordination tables, which Pro writes with isoformat() timestamps.
    coord = sqlite3.connect(":memory:")
    coord.execute("CREATE TABLE coord_audit (tool_name TEXT, latency_ms REAL, created_at TEXT)")
    coord.execute("CREATE TABLE coord_sessions (started_at TEXT)")
    coord.executemany(
        "INSERT INTO coord_audit VALUES ('mcp__omega-memory__omega_query', 5, ?)",
        [(just_outside(30),), (just_inside(30),)],
    )
    coord.executemany(
        "INSERT INTO coord_sessions VALUES (?)",
        [(just_outside(7),), (just_inside(7),), (just_outside(30),), (just_inside(30),)],
    )

    class FakeManager:
        def get_read_connection(self):
            return coord

    coordination = types.ModuleType("omega_platform.orchestrator.coordination")
    coordination.get_manager = lambda: FakeManager()
    monkeypatch.setitem(sys.modules, "omega_platform", types.ModuleType("omega_platform"))
    monkeypatch.setitem(sys.modules, "omega_platform.orchestrator", types.ModuleType("omega_platform.orchestrator"))
    monkeypatch.setitem(sys.modules, "omega_platform.orchestrator.coordination", coordination)

    report = diagnostic_report(days=30)

    assert report["memory_health"]["velocity_total_7d"] == 1
    assert report["memory_health"]["dead_memories"] == 1
    assert report["tool_usage"]["total_calls"] == 1
    assert report["tool_usage"]["omega_calls"] == 1
    assert report["tool_usage"]["top_tools"][0]["calls"] == 1
    assert report["sessions"] == {"total": 4, "week": 1, "month": 3}
