"""Regression contracts for the Core 1.5.13 reliability release.

These tests intentionally describe the public behaviour before the production
implementation changes land.  They use an isolated SQLite database supplied
by the shared test fixture.
"""

import asyncio
import json
import sqlite3

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
    assert "no replacement" in response["content"][0]["text"].lower()
    assert _metadata(handler_store, old_id)["superseded"] is True
    assert handler_store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0


def test_store_supersede_replacement_is_atomic(store):
    """The store changes status and lineage together through one public operation."""
    old_id = store.store(
        "Legacy release approval policy",
        metadata={"event_type": "decision"},
        entity_id="release-owner",
        skip_inference=True,
    )
    new_id = store.store(
        "Current release approval policy",
        metadata={"event_type": "decision"},
        entity_id="release-owner",
        skip_inference=True,
    )

    success, error = store.supersede_with_replacement(
        old_id, new_id, "policy corrected"
    )

    assert success is True
    assert error is None
    assert _metadata(store, old_id)["superseded"] is True
    assert not _metadata(store, new_id).get("superseded", False)
    assert _edge_count(store, new_id, old_id, "supersedes") == 1


def test_store_supersede_rolls_back_status_when_lineage_insert_fails(store, monkeypatch):
    """A failed edge write must leave both memory rows and the graph unchanged."""
    old_id = store.store(
        "Legacy atomic release policy",
        metadata={"event_type": "decision"},
        entity_id="release-owner",
        skip_inference=True,
    )
    new_id = store.store(
        "Current atomic release policy",
        metadata={"event_type": "decision"},
        entity_id="release-owner",
        skip_inference=True,
    )
    original_exec = store._exec

    def fail_supersedes_edge(sql, params=None):
        if "INSERT OR IGNORE INTO edges" in sql and "supersedes" in sql:
            raise sqlite3.IntegrityError("injected lineage failure")
        return original_exec(sql, params)

    monkeypatch.setattr(store, "_exec", fail_supersedes_edge)

    success, error = store.supersede_with_replacement(
        old_id, new_id, "policy corrected"
    )

    assert success is False
    assert "injected lineage failure" in error
    assert not _metadata(store, old_id).get("superseded", False)
    assert not _metadata(store, new_id).get("superseded", False)
    assert _edge_count(store, new_id, old_id, "supersedes") == 0


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


@pytest.mark.parametrize(
    ("scope_argument", "caller_scope"),
    [
        ("entity_id", "account-a"),
        ("project", "project-a"),
        ("caller_session_id", "session-a"),
    ],
)
def test_supersede_rejects_matching_records_outside_caller_scope(
    handler_store, scope_argument, caller_scope
):
    """Two mutually matching records still cannot escape the caller's scope."""
    from omega.server.handlers import handle_omega_memory

    metadata = {"event_type": "decision", "project": "project-b"}
    old_id = handler_store.store(
        "Account B legacy deployment policy",
        metadata=metadata,
        session_id="session-b",
        entity_id="account-b",
        skip_inference=True,
    )
    new_id = handler_store.store(
        "Account B current deployment policy",
        metadata=metadata,
        session_id="session-b",
        entity_id="account-b",
        skip_inference=True,
    )
    arguments = {
        "action": "supersede",
        "memory_id": old_id,
        "target_id": new_id,
        scope_argument: caller_scope,
    }

    response = asyncio.run(handle_omega_memory(arguments))

    assert response.get("isError")
    assert not _metadata(handler_store, old_id).get("superseded", False)
    assert not _metadata(handler_store, new_id).get("superseded", False)
    assert _edge_count(handler_store, new_id, old_id, "supersedes") == 0


@pytest.mark.parametrize("replacement_status", ["superseded", "archived"])
def test_supersede_rejects_inactive_replacement(handler_store, replacement_status):
    """A replacement must still be active when the transaction validates it."""
    from omega.server.handlers import handle_omega_memory

    old_id = handler_store.store(
        "Legacy active policy awaiting correction",
        metadata={"event_type": "decision"},
        entity_id="release-owner",
        skip_inference=True,
    )
    new_id = handler_store.store(
        "Replacement policy that is no longer active",
        metadata={"event_type": "decision"},
        entity_id="release-owner",
        skip_inference=True,
    )
    handler_store._conn.execute(
        "UPDATE memories SET status = ? WHERE node_id = ?",
        (replacement_status, new_id),
    )
    handler_store._conn.commit()

    response = asyncio.run(
        handle_omega_memory(
            {"action": "supersede", "memory_id": old_id, "target_id": new_id}
        )
    )

    assert response.get("isError")
    assert not _metadata(handler_store, old_id).get("superseded", False)
    assert _edge_count(handler_store, new_id, old_id, "supersedes") == 0


def test_memory_schema_describes_supersede_ids_and_editable_priority():
    """The public tool contract must make both IDs and edit fields unambiguous."""
    from omega.server.tool_schemas import TOOL_SCHEMAS

    schema = next(item for item in TOOL_SCHEMAS if item["name"] == "omega_memory")
    properties = schema["inputSchema"]["properties"]

    assert "outdated" in properties["memory_id"]["description"].lower()
    assert "replacement" in properties["target_id"]["description"].lower()
    assert properties["priority"]["minimum"] == 1
    assert properties["priority"]["maximum"] == 5
    assert "caller" in properties["entity_id"]["description"].lower()
    assert "caller" in properties["project"]["description"].lower()
    assert "caller" in properties["caller_session_id"]["description"].lower()


def test_priority_only_edit_preserves_memory_identity_and_history(handler_store):
    """Correcting priority must not rewrite a memory or its relationships."""
    from omega.server.handlers import handle_omega_memory

    memory_id = handler_store.store(
        "Keep this decision and its history",
        metadata={
            "event_type": "decision",
            "priority": 2,
            "revision_history": [{"source": "import"}],
        },
        session_id="release-session",
        entity_id="release-contract",
    )
    related_id = handler_store.store("Related decision", metadata={"event_type": "decision"})
    assert handler_store.add_edge(memory_id, related_id, "related")
    accessed_at = "2026-08-22T00:00:00+00:00"
    handler_store._conn.execute(
        "UPDATE memories SET access_count = ?, last_accessed = ?, updated_at = ? WHERE node_id = ?",
        (7, accessed_at, "2026-08-21T00:00:00+00:00", memory_id),
    )
    handler_store._conn.commit()
    before = handler_store._conn.execute(
        """SELECT node_id, content, entity_id, project, session_id,
                  created_at, updated_at, access_count, last_accessed, metadata
           FROM memories WHERE node_id = ?""",
        (memory_id,),
    ).fetchone()

    response = asyncio.run(
        handle_omega_memory({"action": "edit", "memory_id": memory_id, "priority": 5})
    )

    assert not response.get("isError")
    after = handler_store._conn.execute(
        """SELECT node_id, content, entity_id, project, session_id,
                  created_at, updated_at, access_count, last_accessed, metadata,
                  priority
           FROM memories WHERE node_id = ?""",
        (memory_id,),
    ).fetchone()
    assert after[0] == before[0]
    assert after[1] == before[1]
    assert after[2] == before[2]
    assert after[3] == before[3]
    assert after[4] == before[4]
    assert after[5] == before[5]
    assert after[7] == before[7]
    assert after[8] == before[8]
    assert json.loads(after[9])["revision_history"] == json.loads(before[9])["revision_history"]
    assert after[6] > before[6]
    assert after[10] == 5
    assert _metadata(handler_store, memory_id)["priority"] == 5
    assert _edge_count(handler_store, memory_id, related_id, "related") == 1


def test_priority_edit_preserves_metadata_added_after_bridge_read(
    handler_store, monkeypatch
):
    """A late metadata writer must not be erased by stale bridge state."""
    from omega.server.handlers import handle_omega_memory

    memory_id = handler_store.store(
        "Concurrent metadata preservation",
        metadata={"event_type": "decision", "priority": 2, "source": "seed"},
    )
    original_update_node = handler_store.update_node

    def add_metadata_before_transaction(*args, **kwargs):
        current = _metadata(handler_store, memory_id)
        current["late_audit_marker"] = "must-survive"
        handler_store._conn.execute(
            "UPDATE memories SET metadata = ? WHERE node_id = ?",
            (json.dumps(current), memory_id),
        )
        handler_store._conn.commit()
        return original_update_node(*args, **kwargs)

    monkeypatch.setattr(handler_store, "update_node", add_metadata_before_transaction)

    response = asyncio.run(
        handle_omega_memory(
            {"action": "edit", "memory_id": memory_id, "priority": 5}
        )
    )

    assert not response.get("isError")
    metadata = _metadata(handler_store, memory_id)
    assert metadata["late_audit_marker"] == "must-survive"
    assert metadata["priority"] == 5
    assert metadata["edit_count"] == 1


def test_edit_accepts_content_only(handler_store):
    """Content-only edits retain the existing priority."""
    from omega.server.handlers import handle_omega_memory

    memory_id = handler_store.store(
        "Original content", metadata={"event_type": "decision", "priority": 2}
    )

    response = asyncio.run(
        handle_omega_memory(
            {"action": "edit", "memory_id": memory_id, "new_content": "Corrected content"}
        )
    )

    assert not response.get("isError")
    row = handler_store._conn.execute(
        "SELECT content, priority FROM memories WHERE node_id = ?", (memory_id,)
    ).fetchone()
    assert row == ("Corrected content", 2)


def test_edit_accepts_priority_only(handler_store):
    """Priority-only correction is a first-class edit operation."""
    from omega.server.handlers import handle_omega_memory

    memory_id = handler_store.store(
        "Priority correction", metadata={"event_type": "decision", "priority": 2}
    )

    response = asyncio.run(
        handle_omega_memory({"action": "edit", "memory_id": memory_id, "priority": 4})
    )

    assert not response.get("isError")
    assert handler_store._conn.execute(
        "SELECT priority FROM memories WHERE node_id = ?", (memory_id,)
    ).fetchone()[0] == 4


def test_edit_accepts_content_and_priority(handler_store):
    """A single edit may correct content and priority together."""
    from omega.server.handlers import handle_omega_memory

    memory_id = handler_store.store(
        "Original policy", metadata={"event_type": "decision", "priority": 2}
    )

    response = asyncio.run(
        handle_omega_memory(
            {
                "action": "edit",
                "memory_id": memory_id,
                "new_content": "Corrected policy",
                "priority": 5,
            }
        )
    )

    assert not response.get("isError")
    assert handler_store._conn.execute(
        "SELECT content, priority FROM memories WHERE node_id = ?", (memory_id,)
    ).fetchone() == ("Corrected policy", 5)


def test_edit_rejects_request_with_neither_content_nor_priority(handler_store):
    """An edit must contain at least one mutable field."""
    from omega.server.handlers import handle_omega_memory

    memory_id = handler_store.store("No-op edit", metadata={"event_type": "decision"})

    response = asyncio.run(handle_omega_memory({"action": "edit", "memory_id": memory_id}))

    assert response.get("isError")


@pytest.mark.parametrize("priority", [0, 6, True, 3.5, "4"])
def test_edit_rejects_priority_outside_supported_range(handler_store, priority):
    """Priority is constrained to the documented inclusive range of one through five."""
    from omega.server.handlers import handle_omega_memory

    memory_id = handler_store.store("Range checked priority", metadata={"event_type": "decision"})

    response = asyncio.run(
        handle_omega_memory(
            {
                "action": "edit",
                "memory_id": memory_id,
                "new_content": "Still valid content",
                "priority": priority,
            }
        )
    )

    assert response.get("isError")


def test_priority_edit_history_is_bounded(handler_store):
    """Repeated priority corrections retain only the twenty most recent changes."""
    from omega.server.handlers import handle_omega_memory

    memory_id = handler_store.store(
        "Bounded priority history",
        metadata={"event_type": "decision", "priority": 1},
    )
    priorities = [2, 3, 4, 5, 4] * 5

    for priority in priorities:
        response = asyncio.run(
            handle_omega_memory(
                {"action": "edit", "memory_id": memory_id, "priority": priority}
            )
        )
        assert not response.get("isError")

    history = _metadata(handler_store, memory_id)["priority_edit_history"]
    assert len(history) == 20
    assert history[0]["old_priority"] == 4
    assert history[0]["new_priority"] == 2
    assert history[-1]["old_priority"] == 5
    assert history[-1]["new_priority"] == 4
    assert all(entry["edited_at"].endswith("+00:00") for entry in history)


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


def test_public_query_prefers_semantic_result_over_hot_high_priority_history(store, monkeypatch):
    """The public query path must not let hot/access metadata outrank relevance."""
    query_text = "semantic retrieval contract check"
    semantic_id = store.store(
        "Semantic retrieval contract check canonical answer.",
        metadata={"event_type": "memory", "priority": 1},
    )
    hot_id = store.store(
        "Semantic retrieval contract check legacy stale answer.",
        metadata={"event_type": "memory", "priority": 5},
    )
    store._conn.execute(
        "UPDATE memories SET access_count = ? WHERE node_id = ?", (500, hot_id)
    )
    store._conn.commit()
    store._refresh_hot_cache()
    monkeypatch.setattr(store, "_fast_path_lookup", lambda *_args, **_kwargs: [])

    def fake_vec(
        _query_text,
        _skip_vec,
        _entity_id,
        _limit,
        all_results,
        vec_ranked,
        raw_vec_sims,
        query_embedding=None,
    ):
        del query_embedding
        for memory_id, similarity in ((semantic_id, 0.90), (hot_id, 0.10)):
            all_results[memory_id] = store.get_node(memory_id, track_access=False)
            vec_ranked.append((memory_id, similarity))
            raw_vec_sims[memory_id] = similarity
        return None

    def fake_fts(
        _query_text,
        _temporal_range,
        _entity_id,
        _limit,
        all_results,
        text_ranked,
        _temporal_ranked,
    ):
        for memory_id, relevance in ((semantic_id, 0.80), (hot_id, 0.70)):
            all_results[memory_id] = store.get_node(memory_id, track_access=False)
            text_ranked.append((memory_id, relevance))

    monkeypatch.setattr(store, "_query_phase_vec", fake_vec)
    monkeypatch.setattr(store, "_query_phase_fts", fake_fts)

    results = store.query(query_text, limit=2, use_cache=False, expand_query=False)

    assert results[0].id == semantic_id
