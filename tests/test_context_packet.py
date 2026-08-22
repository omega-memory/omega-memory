"""Tests for task-aware context_packet assembly."""

import ast
import sqlite3

import pytest

from omega.server.context_handlers import (
    build_context_packet,
    handle_context_packet,
    _estimate_tokens,
    _query_by_event_type,
    _query_packet_scope_fallback,
    _render_context_packet,
)


pytestmark = pytest.mark.usefixtures("_reset_bridge")


def _payload(resp: dict) -> dict:
    content = resp.get("content", [])
    assert content, "MCP response missing content"
    return ast.literal_eval(content[0]["text"])


def _is_error(resp: dict) -> bool:
    content = resp.get("content", [])
    return bool(content) and content[0].get("text", "").lower().startswith("error")


def test_context_packet_schema_exposed_by_default():
    from omega.server.tool_schemas import TOOL_SCHEMAS

    names = {schema["name"] for schema in TOOL_SCHEMAS}
    assert "context_packet" in names
    assert "context_assemble" not in names


async def test_context_packet_empty_store_is_well_formed(tmp_omega_dir):
    resp = await handle_context_packet({
        "task": "edit auth handler",
        "files": ["src/auth.py"],
        "budget_tokens": 300,
    })
    assert not _is_error(resp)
    payload = _payload(resp)
    assert payload["packet_markdown"].startswith("[MEMORY_CONTEXT]")
    assert payload["chains"] == []
    assert payload["candidate_ids"] == []
    assert payload["memories_used"] == []
    assert payload["estimated_tokens"] <= payload["budget_tokens"]


async def test_context_packet_rejects_non_list_files(tmp_omega_dir):
    resp = await handle_context_packet({"files": "src/auth.py"})
    assert _is_error(resp)


async def test_context_packet_includes_relevant_seed(tmp_omega_dir):
    from omega.bridge import store

    store(
        content="Decision: auth handlers must validate JWT tokens server-side before any database lookup.",
        event_type="decision",
        entity_id="omega-packet-seed",
    )

    resp = await handle_context_packet({
        "task": "update JWT auth handler",
        "files": ["src/auth.py"],
        "scope": {"entity_id": "omega-packet-seed"},
        "budget_tokens": 500,
    })
    payload = _payload(resp)
    md = payload["packet_markdown"]
    assert "Prior Decisions" in md
    assert "JWT tokens" in md
    assert payload["memories_used"]
    assert payload["memories_used"][0] in payload["candidate_ids"]
    assert payload["candidate_receipts"]
    assert payload["candidate_receipts"][0]["score"] > 0
    assert any(row["rendered"] for row in payload["candidate_receipts"])
    assert payload["metrics"]["memories_used"] == len(payload["memories_used"])
    assert payload["metrics"]["estimated_tokens"] == payload["estimated_tokens"]


def test_top_lessons_and_packet_fallback_ignore_access_popularity(tmp_omega_dir):
    """Context selection must not promote an older record because it was popular."""
    from omega.bridge import _get_store

    db = _get_store()
    old_popular = db.store(
        content="Old popular lesson for context ordering",
        metadata={"event_type": "lesson_learned"},
        entity_id="omega-context-order",
        skip_inference=True,
    )
    recent = db.store(
        content="Recent lesson for context ordering",
        metadata={"event_type": "lesson_learned"},
        entity_id="omega-context-order",
        skip_inference=True,
    )
    db._conn.execute(
        "UPDATE memories SET created_at = ?, access_count = 500 WHERE node_id = ?",
        ("2020-01-01T00:00:00+00:00", old_popular),
    )
    db._conn.execute(
        "UPDATE memories SET created_at = ?, access_count = 0 WHERE node_id = ?",
        ("2026-01-01T00:00:00+00:00", recent),
    )
    db._conn.commit()

    lessons = _query_by_event_type(
        db,
        event_type="lesson_learned",
        entity_id="omega-context-order",
        limit=2,
        since_iso=None,
        order_by="access",
    )
    fallback = _query_packet_scope_fallback(
        db,
        scope={"entity_id": "omega-context-order", "project": None, "session_id": None},
        max_sensitivity="restricted",
        limit=2,
    )

    assert [row["id"] for row in lessons] == [recent, old_popular]
    assert [row["id"] for row in fallback] == [recent, old_popular]


def test_context_packet_records_only_unique_rendered_memory_ids(tmp_omega_dir, monkeypatch):
    """Candidate discovery is read-only; only IDs actually rendered count as access."""
    from omega.bridge import _get_store

    db = _get_store()
    memory_ids = []
    for index in range(8):
        memory_ids.append(db.store(
            content=(
                f"Decision {index}: packet access boundary requires a deliberately long "
                "candidate body so the minimum packet budget cannot render every match. " * 3
            ),
            metadata={"event_type": "decision"},
            entity_id="omega-packet-access",
            skip_inference=True,
        ))
    monkeypatch.setattr(
        db,
        "query",
        lambda *_args, **_kwargs: [
            db.get_node(node_id, track_access=False) for node_id in memory_ids
        ],
    )

    packet = build_context_packet(
        db,
        task="packet access boundary long candidate body",
        scope={"entity_id": "omega-packet-access"},
        budget_tokens=120,
    )

    assert packet["memories_used"]
    assert set(packet["memories_used"]) < set(packet["candidate_ids"])
    rows = db._conn.execute(
        "SELECT node_id, access_count FROM memories WHERE entity_id = ?",
        ("omega-packet-access",),
    ).fetchall()
    access_by_id = dict(rows)
    assert all(access_by_id[node_id] == 1 for node_id in packet["memories_used"])
    assert all(
        access_by_id[node_id] == 0
        for node_id in set(packet["candidate_ids"]) - set(packet["memories_used"])
    )


def test_context_packet_survives_second_connection_access_lock(tmp_omega_dir, monkeypatch):
    """A locked audit write cannot replace a successfully assembled packet with an error."""
    from omega.bridge import _get_store

    db = _get_store()
    memory_id = db.store(
        content="Decision: locked access accounting must preserve context output.",
        metadata={"event_type": "decision"},
        entity_id="omega-packet-lock",
        skip_inference=True,
    )
    monkeypatch.setattr(
        db,
        "query",
        lambda *_args, **_kwargs: [db.get_node(memory_id, track_access=False)],
    )
    db._conn.execute("PRAGMA busy_timeout = 1")
    blocker = sqlite3.connect(str(db.db_path), timeout=0, isolation_level=None)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        blocker.execute(
            "UPDATE memories SET metadata = metadata WHERE node_id = ?", (memory_id,)
        )

        packet = build_context_packet(
            db,
            task="locked access accounting context output",
            scope={"entity_id": "omega-packet-lock"},
            budget_tokens=300,
        )

        assert packet["memories_used"] == [memory_id]
        assert "locked access accounting" in packet["packet_markdown"]
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()


async def test_context_packet_includes_graph_chain(tmp_omega_dir):
    from omega.bridge import _get_store

    db = _get_store()
    parent = db.store(
        content="Decision: use local-first BYOS as the customer acceptance boundary.",
        metadata={"event_type": "decision"},
        entity_id="omega-packet-chain",
    )
    child = db.store(
        content="Lesson: do not cite operator-only infrastructure as customer pilot evidence.",
        metadata={"event_type": "lesson_learned"},
        entity_id="omega-packet-chain",
    )
    db.add_edge(parent, child, edge_type="related", weight=0.95)

    packet = build_context_packet(
        db,
        task="write customer pilot evidence",
        files=["docs/audits/enterprise-readiness-baseline.md"],
        scope={"entity_id": "omega-packet-chain"},
        budget_tokens=700,
    )

    md = packet["packet_markdown"]
    assert "local-first BYOS" in md
    assert "operator-only infrastructure" in md
    assert packet["chains"]
    assert any(
        node["id"] in {parent, child}
        for chain in packet["chains"]
        for node in chain["nodes"]
    )


async def test_context_packet_prefers_validated_typed_edges(tmp_omega_dir):
    from omega.bridge import _get_store

    db = _get_store()
    seed = db.store(
        content="Decision: packet ranking seed for context packet graph scoring.",
        metadata={"event_type": "decision"},
        entity_id="omega-packet-edge-rank",
    )
    generic = db.store(
        content="Lesson: generic related edge should rank below validated packet miss edge.",
        metadata={"event_type": "lesson_learned"},
        entity_id="omega-packet-edge-rank",
    )
    validated = db.store(
        content="Lesson: validated packet miss edge should surface first in graph scoring.",
        metadata={"event_type": "lesson_learned"},
        entity_id="omega-packet-edge-rank",
    )
    db.add_edge(seed, generic, edge_type="related", weight=0.95, metadata={"source": "manual"})
    db.add_edge(
        seed,
        validated,
        edge_type="same_entity",
        weight=0.90,
        metadata={"source": "context_packet_miss_backfill", "auto": True, "typed": "same_entity"},
    )

    packet = build_context_packet(
        db,
        task="packet ranking seed graph scoring",
        files=["src/context.py"],
        scope={"entity_id": "omega-packet-edge-rank"},
        budget_tokens=700,
    )

    first_chain = next(chain for chain in packet["chains"] if chain["seed_id"] == seed)
    assert first_chain["nodes"][0]["id"] == validated
    assert first_chain["nodes"][0]["edge_source"] == "context_packet_miss_backfill"


async def test_context_packet_surfaces_stale_as_warning(tmp_omega_dir):
    from omega.bridge import _get_store

    db = _get_store()
    active = db.store(
        content="Decision: route customer cloud setup through BYOS Supabase.",
        metadata={"event_type": "decision"},
        entity_id="omega-packet-stale",
    )
    stale = db.store(
        content="Old lesson: use the operator Supabase project for customer setup.",
        metadata={"event_type": "lesson_learned"},
        entity_id="omega-packet-stale",
    )
    db.add_edge(active, stale, edge_type="related", weight=0.9)
    db._conn.execute("UPDATE memories SET status = 'stale' WHERE node_id = ?", (stale,))
    db._conn.commit()

    packet = build_context_packet(
        db,
        task="customer Supabase setup",
        files=["docs/setup.md"],
        scope={"entity_id": "omega-packet-stale"},
        budget_tokens=700,
    )

    md = packet["packet_markdown"]
    assert "BYOS Supabase" in md
    assert "Warnings:" in md
    assert "stale" in md
    # The stale content can appear in warning form, but not as a lesson section.
    assert "Lessons:\n- " not in md or "operator Supabase project" not in md.split("Warnings:")[0]


def test_context_packet_compacts_warning_rendering():
    warnings = [
        {
            "reason": "stale",
            "item": {
                "id": f"mem-warning-{i}",
                "content": f"Old lesson {i}: warning compaction stale detail should stay in structured payload.",
            },
        }
        for i in range(4)
    ]

    md = _render_context_packet(
        title="warnings.md",
        admitted=[],
        warnings=warnings,
        budget_tokens=900,
        include_receipt=True,
    )

    assert "4 warning(s) in structured receipt" in md
    assert "3 additional warning(s) omitted from prompt" in md
    assert md.count("top stale:") == 1


def test_context_packet_renders_top_scored_item_before_section_order():
    admitted = [
        {
            "id": "mem-top-relevant",
            "event_type": "memory",
            "content": "Top relevant context should render before lower-scored section items.",
            "score": 9.0,
        },
        *[
            {
                "id": f"mem-decision-{i}",
                "event_type": "decision",
                "content": f"Lower-scored decision {i} should not evict the top relevant item.",
                "score": 1.0,
            }
            for i in range(8)
        ],
    ]

    md = _render_context_packet(
        title="rank.md",
        admitted=admitted,
        warnings=[],
        budget_tokens=1000,
        include_receipt=True,
    )

    assert "`mem-top-rele`" in md
    assert md.find("Relevant Context:") < md.find("Prior Decisions:")
    assert md.count("- `") <= 8


async def test_context_packet_excludes_superseded_memories(tmp_omega_dir):
    from omega.bridge import _get_store

    db = _get_store()
    current = db.store(
        content="Decision: current OAuth callback implementation must validate state nonce.",
        metadata={"event_type": "decision"},
        entity_id="omega-packet-superseded",
        skip_inference=True,
    )
    old = db.store(
        content="Old decision: OAuth callback can skip state nonce validation.",
        metadata={"event_type": "decision"},
        entity_id="omega-packet-superseded",
        skip_inference=True,
    )
    db.add_edge(current, old, edge_type="supersedes", weight=0.98)
    db.mark_superseded(old, current)

    packet = build_context_packet(
        db,
        task="update OAuth callback nonce validation",
        files=["src/auth/oauth.py"],
        scope={"entity_id": "omega-packet-superseded"},
        budget_tokens=700,
    )

    md = packet["packet_markdown"]
    assert "current OAuth callback" in md
    assert "skip state nonce validation" not in md
    assert all(w["reason"] != "superseded" for w in packet["warnings"])


async def test_context_packet_contradiction_edges_are_warnings(tmp_omega_dir):
    from omega.bridge import _get_store

    db = _get_store()
    active = db.store(
        content="Decision: license checks fail closed for remote validation errors.",
        metadata={"event_type": "decision"},
        entity_id="omega-packet-contradiction",
        skip_inference=True,
    )
    contradictory = db.store(
        content="Conflicting note: license checks may fail open during remote validation errors.",
        metadata={"event_type": "lesson_learned"},
        entity_id="omega-packet-contradiction",
        skip_inference=True,
    )
    db.add_edge(active, contradictory, edge_type="contradicts", weight=0.9)

    packet = build_context_packet(
        db,
        task="license remote validation error handling",
        files=["src/license.py"],
        scope={"entity_id": "omega-packet-contradiction"},
        budget_tokens=700,
    )

    md = packet["packet_markdown"]
    assert "fail closed" in md
    assert "Warnings:" in md
    assert "contradicts" in md
    assert "fail open" not in md.split("Warnings:")[0]


async def test_context_packet_respects_sensitivity_limit(tmp_omega_dir):
    from omega.bridge import _get_store

    db = _get_store()
    db.store(
        content="Decision: public launch copy can mention local-first memory packets.",
        metadata={"event_type": "decision"},
        entity_id="omega-packet-sensitivity",
        sensitivity="public",
    )
    db.store(
        content="Decision: confidential enterprise pilot pricing is ACME-only.",
        metadata={"event_type": "decision"},
        entity_id="omega-packet-sensitivity",
        sensitivity="confidential",
    )

    packet = build_context_packet(
        db,
        task="prepare public launch copy for local-first memory packets",
        files=["docs/launch.md"],
        scope={"entity_id": "omega-packet-sensitivity"},
        max_sensitivity="public",
        budget_tokens=700,
    )

    md = packet["packet_markdown"]
    assert "public launch copy" in md
    assert "confidential enterprise pilot pricing" not in md


async def test_context_packet_respects_budget(tmp_omega_dir):
    from omega.bridge import store

    for i in range(8):
        store(
            content=(
                f"Decision {i}: context packet budget test should keep snippets concise "
                "and avoid dumping long raw memory bodies into the prompt. " * 4
            ),
            event_type="decision",
            entity_id="omega-packet-budget",
        )

    resp = await handle_context_packet({
        "task": "context packet budget test",
        "scope": {"entity_id": "omega-packet-budget"},
        "budget_tokens": 180,
    })
    payload = _payload(resp)
    assert _estimate_tokens(payload["packet_markdown"]) <= 180


async def test_context_packet_caps_rendered_memories(tmp_omega_dir):
    from omega.bridge import store

    for i in range(10):
        store(
            content=f"Decision {i}: rendered packet cap test should prefer focused memory packets.",
            event_type="decision",
            entity_id="omega-packet-render-cap",
        )

    resp = await handle_context_packet({
        "task": "rendered packet cap test focused memory packets",
        "scope": {"entity_id": "omega-packet-render-cap"},
        "budget_tokens": 1000,
    })
    payload = _payload(resp)
    assert len(payload["memories_used"]) <= 8
    assert payload["metrics"]["memories_used"] <= 8


async def test_context_packet_handler_tracks_telemetry(tmp_omega_dir, monkeypatch):
    from omega.bridge import store

    calls = []
    monkeypatch.setattr(
        "omega.telemetry.track_context_packet",
        lambda metrics, surface="unknown": calls.append((metrics, surface)),
    )
    store(
        content="Decision: telemetry packet test should count packet usage.",
        event_type="decision",
        entity_id="omega-packet-telemetry",
    )

    resp = await handle_context_packet({
        "task": "telemetry packet test",
        "scope": {"entity_id": "omega-packet-telemetry"},
    })

    assert not _is_error(resp)
    assert calls
    metrics, surface = calls[0]
    assert surface == "mcp"
    assert metrics["memories_used"] >= 1
    assert metrics["estimated_tokens"] > 0


def test_context_packet_rendered_warning_metric_matches_visible_top_lines():
    warnings = [
        {
            "reason": "stale",
            "item": {"id": f"mem-warning-{i}", "content": f"Old stale warning {i}"},
        }
        for i in range(3)
    ]

    md = _render_context_packet(
        title="warnings.md",
        admitted=[],
        warnings=warnings,
        budget_tokens=900,
        include_receipt=True,
    )

    from omega.server.context_handlers import _rendered_packet_warning_count

    assert _rendered_packet_warning_count(md) == 1
