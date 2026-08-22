"""Tests for OMEGA Forgetting Audit Trail (Feature 1)."""

import pytest
from datetime import datetime, timedelta, timezone

from omega.server.handlers import HANDLERS


# ============================================================================
# Fixture: reset bridge singleton between tests
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_bridge(tmp_omega_dir):
    """Reset the bridge singleton so each test gets a fresh store."""
    from omega.bridge import reset_memory

    reset_memory()
    yield
    reset_memory()


def _get_store():
    from omega.bridge import _get_store

    return _get_store()


# ============================================================================
# 1. delete_node logs to forgetting_log
# ============================================================================


def test_delete_node_logs_forgetting():
    """Deleting a memory should create a forgetting_log entry with reason 'user_deleted'."""
    store = _get_store()
    node_id = store.store(content="Test memory to delete", metadata={"event_type": "decision"})
    store.delete_node(node_id)

    entries = store.get_forgetting_log(limit=10)
    assert len(entries) == 1
    assert entries[0]["node_id"] == node_id
    assert entries[0]["reason"] == "user_deleted"
    assert entries[0]["event_type"] == "decision"
    assert "Test memory" in entries[0]["content_preview"]


# ============================================================================
# 2. cleanup_expired (TTL) logs to forgetting_log
# ============================================================================


def test_ttl_expired_logs_forgetting():
    """Expired TTL memories should be logged with reason 'ttl_expired'."""
    store = _get_store()
    node_id = store.store(
        content="Ephemeral memory with TTL",
        metadata={"event_type": "session_summary"},
        ttl_seconds=1,
    )

    # Backdate created_at so it's already expired
    store._conn.execute(
        "UPDATE memories SET created_at = ? WHERE node_id = ?",
        ((datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(), node_id),
    )
    store._conn.commit()

    removed = store.cleanup_expired()
    assert removed == 1

    entries = store.get_forgetting_log(limit=10)
    assert len(entries) == 1
    assert entries[0]["reason"] == "ttl_expired"
    assert entries[0]["node_id"] == node_id


# ============================================================================
# 3. evict_lru logs to forgetting_log
# ============================================================================


def test_lru_eviction_logs_forgetting():
    """LRU eviction should be logged with reason 'lru_evicted'."""
    store = _get_store()
    node_id = store.store(content="Old rarely used memory", metadata={"event_type": "memory"})

    evicted = store.evict_lru(count=1)
    assert evicted == 1

    entries = store.get_forgetting_log(limit=10)
    assert len(entries) == 1
    assert entries[0]["reason"] == "lru_evicted"
    assert entries[0]["node_id"] == node_id


# ============================================================================
# 4. consolidate logs to forgetting_log
# ============================================================================


def test_consolidation_logs_forgetting():
    """Consolidation pruning should be logged with reason 'consolidation_pruned'."""
    store = _get_store()

    # Create stale zero-access memories older than prune threshold
    # Use very different content to avoid embedding dedup
    cutoff = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    diverse_content = [
        "The quantum entanglement experiment produced unexpected results in February",
        "PostgreSQL database migration scripts need careful version management",
        "Kubernetes pod autoscaling thresholds were set to 80 percent CPU usage",
    ]
    ids = []
    for content in diverse_content:
        nid = store.store(
            content=content,
            metadata={"event_type": "memory"},
            skip_inference=True,
        )
        store._conn.execute(
            "UPDATE memories SET access_count = 0, created_at = ? WHERE node_id = ?",
            (cutoff, nid),
        )
        ids.append(nid)
    store._conn.commit()

    stats = store.consolidate(prune_days=30)
    assert stats["pruned_stale"] == 3

    entries = store.get_forgetting_log(limit=10)
    assert len(entries) >= 3
    pruned = [e for e in entries if e["reason"] == "consolidation_pruned"]
    assert len(pruned) == 3


# ============================================================================
# 5. feedback_flagged logs to forgetting_log
# ============================================================================


def test_feedback_flagged_logs_forgetting():
    """Memory crossing feedback threshold should log with reason 'feedback_flagged'."""
    store = _get_store()
    node_id = store.store(content="Potentially wrong advice", metadata={"event_type": "lesson_learned"})

    # Rate it down past the -3 threshold
    store.record_feedback(node_id, "outdated", reason="old info")  # -2
    store.record_feedback(node_id, "unhelpful", reason="not useful")  # -3

    entries = store.get_forgetting_log(limit=10)
    flagged = [e for e in entries if e["reason"] == "feedback_flagged"]
    assert len(flagged) == 1
    assert flagged[0]["node_id"] == node_id
    assert flagged[0]["event_type"] == "lesson_learned"


# ============================================================================
# 6. Self-pruning of forgetting_log entries
# ============================================================================


def test_prune_forgetting_log():
    """Old forgetting log entries should be prunable."""
    store = _get_store()

    # Create and delete a memory to generate a log entry
    node_id = store.store(content="Will be deleted", metadata={"event_type": "memory"})
    store.delete_node(node_id)

    # Backdate the log entry to 100 days ago
    old_date = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    store._conn.execute("UPDATE forgetting_log SET deleted_at = ?", (old_date,))
    store._conn.commit()

    removed = store.prune_forgetting_log(max_age_days=90)
    assert removed == 1

    entries = store.get_forgetting_log(limit=10)
    assert len(entries) == 0


def test_historical_access_cannot_shield_old_memory_from_strength_decay():
    """Decay eligibility depends on age and strength, not access history."""
    store = _get_store()
    node_id = store.store(
        content="Old accessed lifecycle record must remain eligible for decay",
        metadata={"event_type": "memory"},
        skip_inference=True,
    )
    old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    recent = datetime.now(timezone.utc).isoformat()
    store._conn.execute(
        """UPDATE memories SET created_at = ?, last_accessed = ?, access_count = 500
           WHERE node_id = ?""",
        (old, recent, node_id),
    )
    store._conn.commit()

    stats = store.apply_strength_decay(min_strength=2.0, min_age_days=30)

    metadata = store.get_node(node_id, track_access=False).metadata
    assert stats["scanned"] == 1
    assert stats["decayed"] == 1
    assert metadata["superseded_reason"] == "strength_decay"


# ============================================================================
# 7. Reason filter on get_forgetting_log
# ============================================================================


def test_forgetting_log_reason_filter():
    """Filtering by reason should return only matching entries."""
    store = _get_store()

    # Create two deletions with different reasons
    n1 = store.store(content="TTL memory", metadata={"event_type": "session_summary"}, ttl_seconds=1)
    n2 = store.store(content="User deleted memory", metadata={"event_type": "decision"})

    # Expire n1
    store._conn.execute(
        "UPDATE memories SET created_at = ? WHERE node_id = ?",
        ((datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(), n1),
    )
    store._conn.commit()
    store.cleanup_expired()

    # Delete n2
    store.delete_node(n2)

    all_entries = store.get_forgetting_log(limit=10)
    assert len(all_entries) == 2

    ttl_only = store.get_forgetting_log(limit=10, reason="ttl_expired")
    assert len(ttl_only) == 1
    assert ttl_only[0]["reason"] == "ttl_expired"

    deleted_only = store.get_forgetting_log(limit=10, reason="user_deleted")
    assert len(deleted_only) == 1
    assert deleted_only[0]["reason"] == "user_deleted"


# ============================================================================
# 8. Schema migration creates forgetting_log table
# ============================================================================


def test_schema_migration_creates_table():
    """A fresh store should have the forgetting_log table at schema v6."""
    store = _get_store()

    # Verify the table exists by doing an INSERT + SELECT
    store._conn.execute(
        """INSERT INTO forgetting_log (node_id, content_preview, event_type, reason, deleted_at)
           VALUES ('test-node', 'test preview', 'memory', 'user_deleted', '2026-01-01T00:00:00')"""
    )
    store._conn.commit()
    row = store._conn.execute("SELECT * FROM forgetting_log WHERE node_id = 'test-node'").fetchone()
    assert row is not None

    # Verify schema version is 13
    version = store._conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    assert version[0] == 15


# ============================================================================
# 9. Handler integration test
# ============================================================================


@pytest.mark.asyncio
async def test_forgetting_log_handler():
    """The omega_forgetting_log handler should return formatted markdown."""
    store = _get_store()

    # Create and delete a memory
    node_id = store.store(content="Handler test memory", metadata={"event_type": "decision"})
    store.delete_node(node_id)

    result = await HANDLERS["omega_forgetting_log"]({"limit": 10})
    assert not result.get("isError"), result
    text = result["content"][0]["text"]
    assert "Forgetting Log" in text
    assert "user_deleted" in text
    assert node_id[:12] in text


# ============================================================================
# 8. Consolidation protected types and thresholds
# ============================================================================


def test_consolidation_protects_exempt_types():
    """Consolidation must never prune user_preference, error_pattern, behavioral_pattern, constraint, reminder."""
    store = _get_store()

    protected_types = ["user_preference", "error_pattern", "behavioral_pattern", "constraint", "reminder"]
    old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    ids = []

    for i, etype in enumerate(protected_types):
        nid = store.store(
            content=f"Protected content number {i} for {etype} type testing thoroughly",
            metadata={"event_type": etype},
            skip_inference=True,
        )
        store._conn.execute(
            "UPDATE memories SET access_count = 0, created_at = ? WHERE node_id = ?",
            (old_date, nid),
        )
        ids.append(nid)
    store._conn.commit()

    stats = store.consolidate(prune_days=14)
    assert stats["pruned_stale"] == 0

    # All protected memories still exist
    for nid in ids:
        row = store._conn.execute("SELECT 1 FROM memories WHERE node_id = ?", (nid,)).fetchone()
        assert row is not None, f"Protected memory {nid} was incorrectly pruned"


def test_consolidation_prunes_lesson_learned():
    """lesson_learned is no longer protected; stale zero-access lessons get pruned."""
    store = _get_store()

    old_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    nid = store.store(
        content="An old lesson about debugging network timeouts in production environments",
        metadata={"event_type": "lesson_learned"},
        skip_inference=True,
    )
    store._conn.execute(
        "UPDATE memories SET access_count = 0, created_at = ? WHERE node_id = ?",
        (old_date, nid),
    )
    store._conn.commit()

    stats = store.consolidate(prune_days=14)
    assert stats["pruned_stale"] >= 1

    row = store._conn.execute("SELECT 1 FROM memories WHERE node_id = ?", (nid,)).fetchone()
    assert row is None, "Stale lesson_learned should have been pruned"


def test_consolidation_respects_14_day_threshold():
    """Memories younger than 14 days should not be pruned even with 0 access."""
    store = _get_store()

    # 10 days old -- under the 14-day threshold
    recent_date = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    nid = store.store(
        content="Recent memory about configuring the deployment pipeline correctly",
        metadata={"event_type": "memory"},
        skip_inference=True,
    )
    store._conn.execute(
        "UPDATE memories SET access_count = 0, created_at = ? WHERE node_id = ?",
        (recent_date, nid),
    )
    store._conn.commit()

    stats = store.consolidate(prune_days=14)
    assert stats["pruned_stale"] == 0

    row = store._conn.execute("SELECT 1 FROM memories WHERE node_id = ?", (nid,)).fetchone()
    assert row is not None, "Memory under 14-day threshold should survive"


def test_consolidation_preserves_accessed_memories():
    """Memories with access_count > 0 should never be pruned regardless of age."""
    store = _get_store()

    old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    nid = store.store(
        content="A well-used memory about authentication flow design patterns",
        metadata={"event_type": "decision"},
        skip_inference=True,
    )
    store._conn.execute(
        "UPDATE memories SET access_count = 5, created_at = ? WHERE node_id = ?",
        (old_date, nid),
    )
    store._conn.commit()

    stats = store.consolidate(prune_days=14)

    row = store._conn.execute("SELECT 1 FROM memories WHERE node_id = ?", (nid,)).fetchone()
    assert row is not None, "Accessed memory should never be pruned"


def test_consolidation_preserves_priority5_decisions():
    """Priority 5 decisions should survive even with 0 access."""
    store = _get_store()

    old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    nid = store.store(
        content="Critical architecture decision about database schema migration strategy",
        metadata={"event_type": "decision"},
        skip_inference=True,
    )
    store._conn.execute(
        "UPDATE memories SET access_count = 0, priority = 5, created_at = ? WHERE node_id = ?",
        (old_date, nid),
    )
    store._conn.commit()

    stats = store.consolidate(prune_days=14)

    row = store._conn.execute("SELECT 1 FROM memories WHERE node_id = ?", (nid,)).fetchone()
    assert row is not None, "Priority 5 decision should survive consolidation"


def test_phase0_decision_prune_at_14_days():
    """Phase 0 should prune zero-access decisions older than 14 days with priority < 5."""
    store = _get_store()

    old_date = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    nid = store.store(
        content="A routine decision about updating the configuration file format",
        metadata={"event_type": "decision"},
        skip_inference=True,
    )
    store._conn.execute(
        "UPDATE memories SET access_count = 0, priority = 3, created_at = ?, event_type = 'decision' WHERE node_id = ?",
        (old_date, nid),
    )
    store._conn.commit()

    stats = store.consolidate(prune_days=14)
    assert stats["pruned_stale"] >= 1

    row = store._conn.execute("SELECT 1 FROM memories WHERE node_id = ?", (nid,)).fetchone()
    assert row is None, "Stale low-priority decision should be pruned by Phase 0"

    # Check forgetting log has the right reason
    entries = store.get_forgetting_log(limit=5)
    phase0 = [e for e in entries if e["reason"] == "consolidation_phase0_pruned"]
    assert len(phase0) >= 1
