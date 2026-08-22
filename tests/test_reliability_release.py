"""Regression contracts for the Core 1.5.13 reliability release.

These tests intentionally describe the public behaviour before the production
implementation changes land.  They use an isolated SQLite database supplied
by the shared test fixture.
"""

import asyncio
import json
from datetime import datetime, timezone

import pytest


@pytest.fixture
def handler_store(store, monkeypatch):
    """Route bridge-backed handlers to the isolated test store."""
    import omega.bridge as bridge

    monkeypatch.setattr(bridge, "_store_instance", store)
    return store


def _metadata(store, memory_id):
    row = store._conn.execute(
        "SELECT metadata FROM memories WHERE node_id = ?", (memory_id,)
    ).fetchone()
    return json.loads(row[0])


def _edge_count(store, source_id, target_id, edge_type=None):
    sql = "SELECT COUNT(*) FROM edges WHERE source_id = ? AND target_id = ?"
    params = [source_id, target_id]
    if edge_type:
        sql += " AND edge_type = ?"
        params.append(edge_type)
    return store._conn.execute(sql, params).fetchone()[0]


def test_supersede_marks_old_memory_and_links_replacement(handler_store):
    """memory_id is the outdated record; target_id is its replacement."""
    from omega.server.handlers import handle_omega_memory

    old_id = handler_store.store(
        "Legacy certificate rotation runs manually every Tuesday at 03:00 UTC.",
        metadata={"event_type": "decision"},
    )
    new_id = handler_store.store(
        "Approved deployment policy requires automated certificate rotation after every release.",
        metadata={"event_type": "decision"},
    )
    assert old_id != new_id

    response = asyncio.run(
        handle_omega_memory(
            {
                "action": "supersede",
                "memory_id": old_id,
                "target_id": new_id,
                "reason": "policy corrected",
            }
        )
    )

    assert not response.get("isError")
    assert _metadata(handler_store, old_id).get("superseded", False) is True
    assert not _metadata(handler_store, new_id).get("superseded", False)
    assert _edge_count(handler_store, new_id, old_id, "supersedes") == 1


def test_supersede_without_replacement_retires_only_old_memory(handler_store):
    """Omitting target_id remains a supported simple retirement operation."""
    from omega.server.handlers import handle_omega_memory

    old_id = handler_store.store("Retired policy", metadata={"event_type": "decision"})

    response = asyncio.run(
        handle_omega_memory({"action": "supersede", "memory_id": old_id})
    )

    assert not response.get("isError")
    assert _metadata(handler_store, old_id)["superseded"] is True
    assert handler_store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0


def test_cross_entity_supersede_leaves_both_records_unchanged(handler_store):
    """A replacement outside the caller's entity scope must not mutate either record."""
    from omega.server.handlers import handle_omega_memory

    old_id = handler_store.store(
        "Account one retains invoices for seven years under its own retention policy.",
        metadata={"event_type": "decision"},
        entity_id="account-one",
    )
    new_id = handler_store.store(
        "Account two rotates its separate encryption keys every ninety days.",
        metadata={"event_type": "decision"},
        entity_id="account-two",
    )
    assert old_id != new_id

    response = asyncio.run(
        handle_omega_memory(
            {
                "action": "supersede",
                "memory_id": old_id,
                "target_id": new_id,
                "entity_id": "account-one",
            }
        )
    )

    assert response.get("isError")
    assert not _metadata(handler_store, old_id).get("superseded", False)
    assert not _metadata(handler_store, new_id).get("superseded", False)
    assert _edge_count(handler_store, new_id, old_id) == 0


def test_priority_only_edit_preserves_memory_identity_and_history(handler_store):
    """Correcting priority must not rewrite a memory or its relationships."""
    from omega.server.handlers import handle_omega_memory

    memory_id = handler_store.store(
        "Keep this decision and its history", metadata={"event_type": "decision", "priority": 2}
    )
    related_id = handler_store.store("Related decision", metadata={"event_type": "decision"})
    assert handler_store.add_edge(memory_id, related_id, "related")
    accessed_at = "2026-08-22T00:00:00+00:00"
    handler_store._conn.execute(
        "UPDATE memories SET access_count = ?, last_accessed = ? WHERE node_id = ?",
        (7, accessed_at, memory_id),
    )
    handler_store._conn.commit()
    before = handler_store._conn.execute(
        "SELECT node_id, created_at, access_count, last_accessed FROM memories WHERE node_id = ?",
        (memory_id,),
    ).fetchone()

    response = asyncio.run(
        handle_omega_memory({"action": "edit", "memory_id": memory_id, "priority": 5})
    )

    assert not response.get("isError")
    after = handler_store._conn.execute(
        "SELECT node_id, created_at, access_count, last_accessed, priority FROM memories WHERE node_id = ?",
        (memory_id,),
    ).fetchone()
    assert after[:4] == before
    assert after[4] == 5
    assert _metadata(handler_store, memory_id)["priority"] == 5
    assert _edge_count(handler_store, memory_id, related_id, "related") == 1


def test_search_does_not_count_returned_results_as_accesses(store):
    """Discovery must be read-only; explicit retrieval records access separately."""
    memory_id = store.store("The search contract is read only", metadata={"event_type": "memory"})
    before = store._conn.execute(
        "SELECT access_count, last_accessed FROM memories WHERE node_id = ?", (memory_id,)
    ).fetchone()

    results = store.query("search contract read only", limit=5, use_cache=False)

    assert memory_id in [result.id for result in results]
    after = store._conn.execute(
        "SELECT access_count, last_accessed FROM memories WHERE node_id = ?", (memory_id,)
    ).fetchone()
    assert after == before


def test_semantic_rank_beats_high_priority_metadata_outside_a_near_tie(store):
    """Priority is a bounded tie-breaker, never a replacement for relevance."""
    from omega.sqlite_store._types import MemoryResult

    now = datetime.now(timezone.utc)
    semantic_id = "semantic-winner"
    metadata_id = "high-priority-metadata"
    all_results = {
        semantic_id: MemoryResult(
            semantic_id, "the semantically relevant record", {"event_type": "memory", "priority": 1}, now
        ),
        metadata_id: MemoryResult(
            metadata_id, "the metadata-favoured record", {"event_type": "memory", "priority": 5}, now
        ),
    }
    scores = {}

    store._query_phase_fusion(
        "relevant query",
        all_results,
        scores,
        [(semantic_id, 0.90), (metadata_id, 0.10)],
        [(semantic_id, 0.90), (metadata_id, 0.10)],
        [],
        1.0,
        1.0,
        0.0,
        0.0,
        None,
    )

    assert scores[semantic_id] > scores[metadata_id]
