"""Store-time supersession: when may a new memory retire an older one?

Storing used to retire any older memory of the same type at cosine >= 0.75,
with no contradiction signal and no project or entity check (bug-team audit
2026-09-29, finding B1). Related-but-distinct memories were retired, including
another client's, and the store result never said so.

The policy these tests pin:
  * retire only when both memories share the project AND the entity (both
    unset counts as shared) AND the newer text carries an explicit update or
    contradiction signal (``detect_update_signal``);
  * a same-scope match without a signal is recorded as a supersession
    candidate on the new memory and the old one stays active;
  * a memory in another project or entity is never touched;
  * every retirement and candidate is reported to the caller.
"""

import json
import math
import time
from pathlib import Path

import pytest

from omega.contradictions import detect_update_signal
from omega.sqlite_store import EMBEDDING_DIM, SQLiteStore

_PAIRS = json.loads(
    (Path(__file__).parent / "fixtures" / "supersession_pairs.json").read_text()
)


# ---------------------------------------------------------------------------
# detect_update_signal — pure text rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("event_type,older,newer", _PAIRS["distinct"])
def test_distinct_pairs_carry_no_update_signal(event_type, older, newer):
    """Related memories that are both still true must not look like updates."""
    assert detect_update_signal(newer, older) is None


@pytest.mark.parametrize("event_type,older,newer", _PAIRS["updates"])
def test_genuine_updates_carry_an_update_signal(event_type, older, newer):
    assert detect_update_signal(newer, older) is not None


@pytest.mark.parametrize(
    "older,newer,signal",
    [
        ("The project uses Postgres for storage.", "The project now uses SQLite for storage.", "update_marker"),
        ("Rate limit for the public API is 100 requests per minute.",
         "Rate limit for the public API is 300 requests per minute.", "value_change"),
        ("LongMemEval score is 82%", "LongMemEval score is 95.4%", "value_change"),
        ("We will deploy on Fridays.", "We will not deploy on Fridays.", "negation"),
        ("We will not deploy on Fridays.", "We will deploy on Fridays.", "negation"),
        ("Feature flag checkout_v2 is on in production.",
         "Feature flag checkout_v2 is off in production.", "antonym"),
        ("We decided to use PostgreSQL as the primary database.",
         "User prefers SQLite over PostgreSQL as the primary database.", "replacement"),
    ],
)
def test_each_signal_kind(older, newer, signal):
    assert detect_update_signal(newer, older) == signal


def test_update_marker_must_be_anchored_to_the_older_memory():
    """'switched' says something changed, not that *this* memory changed."""
    older = "Use Redis for the session cache in the orders service."
    newer = "Switched the orders service database from PostgreSQL to MySQL."
    assert detect_update_signal(newer, older) is None


def test_marker_already_in_older_memory_is_not_a_signal():
    older = "User now prefers dark mode in the editor."
    newer = "User now prefers a 14px font size in the editor."
    assert detect_update_signal(newer, older) is None


def test_single_word_substitution_is_not_a_signal():
    """'frontend' -> 'backend' is a different subject, not an update."""
    older = "Deploy the frontend on Vercel."
    newer = "Deploy the backend on Vercel."
    assert detect_update_signal(newer, older) is None


def test_identical_text_is_not_a_signal():
    text = "Use Redis for caching."
    assert detect_update_signal(text, text) is None


# ---------------------------------------------------------------------------
# Store-level policy (controlled embeddings: deterministic similarity)
# ---------------------------------------------------------------------------


def _embedding(seed: float) -> list:
    raw = [math.sin(seed * (i + 1)) for i in range(EMBEDDING_DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _near(base: list, amount: float = 0.3) -> list:
    """An embedding close to ``base`` (cosine well above the 0.75 gate)."""
    noise = _embedding(99.0)
    mixed = [b * (1 - amount) + n * amount for b, n in zip(base, noise)]
    norm = math.sqrt(sum(x * x for x in mixed))
    return [x / norm for x in mixed]


@pytest.fixture
def vec_store(tmp_omega_dir):
    s = SQLiteStore(db_path=tmp_omega_dir / "policy.db")
    if not s._vec_available:
        pytest.skip("sqlite-vec not available")
    yield s
    s.close()


def _store_pair(
    store,
    older: str,
    newer: str,
    *,
    event_type: str = "decision",
    newer_type: str | None = None,
    older_scope: dict | None = None,
    newer_scope: dict | None = None,
    seed: float = 1.0,
    **newer_kwargs,
):
    older_scope = older_scope or {"project": "/work/alpha"}
    newer_scope = newer_scope or {"project": "/work/alpha"}
    base = _embedding(seed)
    old_id = store.store(
        content=older,
        metadata={"event_type": event_type, "project": older_scope["project"]},
        entity_id=older_scope.get("entity_id"),
        embedding=base,
    )
    store.get_last_supersession_results()  # drop the first store's (empty) report
    time.sleep(0.01)  # created_at must order the pair
    new_id = store.store(
        content=newer,
        metadata={"event_type": newer_type or event_type, "project": newer_scope["project"]},
        entity_id=newer_scope.get("entity_id"),
        embedding=_near(base),
        **newer_kwargs,
    )
    return old_id, new_id


def _status(store, node_id: str) -> str:
    return store._conn.execute(
        "SELECT status FROM memories WHERE node_id = ?", (node_id,)
    ).fetchone()[0]


def _candidates(store, node_id: str) -> list:
    node = store.get_node(node_id, track_access=False)
    return (node.metadata or {}).get("supersession_candidates", [])


UPDATE = ("The project uses Postgres for storage.", "The project now uses SQLite for storage.")
DISTINCT = (
    "Rate-limit the public API to 100 requests per minute per key.",
    "Rate-limit the admin API to 20 requests per minute per user.",
)
# A genuine update sharing 9 of 11 long words, so word overlap alone calls it a
# duplicate; long enough (>= 80 chars) to pass the noise gate on hook captures.
LONG_UPDATE = (
    "Decision: the project uses Postgres for storage in every environment, including local development.",
    "Decision: the project now uses SQLite for storage in every environment, including local development.",
)
# Long enough (>= 150 chars) for the bridge to add an observation summary.
LONG_DISTINCT = (
    "Rate-limit the public API to 100 requests per minute per key. Bursts above "
    "that return HTTP 429 with a Retry-After header so clients can back off cleanly.",
    "Rate-limit the admin API to 20 requests per minute per user. Admin tools are "
    "interactive, so the limit protects the database from runaway scripted loops.",
)


class TestSameScope:
    def test_genuine_update_retires_the_older_memory(self, vec_store):
        old_id, new_id = _store_pair(vec_store, *UPDATE)

        assert _status(vec_store, old_id) == "superseded"
        old = vec_store.get_node(old_id, track_access=False)
        assert old.metadata["superseded_by"] == new_id
        assert old.metadata["superseded_reason"] == "update_marker"
        edges = vec_store._conn.execute(
            "SELECT source_id, target_id FROM edges WHERE edge_type = 'supersedes'"
        ).fetchall()
        assert (new_id, old_id) in edges

    def test_retirement_is_reported(self, vec_store):
        old_id, _ = _store_pair(vec_store, *UPDATE)

        report = vec_store.get_last_supersession_results()
        assert [(r["node_id"], r["action"], r["signal"]) for r in report] == [
            (old_id, "retired", "update_marker")
        ]
        assert vec_store.get_last_supersession_results() == []  # consume-once

    def test_distinct_memory_is_kept_and_flagged_as_candidate(self, vec_store):
        old_id, new_id = _store_pair(vec_store, *DISTINCT)

        assert _status(vec_store, old_id) == "active"
        old = vec_store.get_node(old_id, track_access=False)
        assert not old.metadata.get("superseded")
        [candidate] = _candidates(vec_store, new_id)
        assert candidate["target_id"] == old_id
        assert candidate["target_event_type"] == "decision"
        assert candidate["similarity"] >= 0.75
        report = vec_store.get_last_supersession_results()
        assert [(r["node_id"], r["action"]) for r in report] == [(old_id, "candidate")]

    def test_same_entity_update_retires(self, vec_store):
        scope = {"project": "/p", "entity_id": "acme"}
        old_id, _ = _store_pair(vec_store, *UPDATE, older_scope=scope, newer_scope=scope)
        assert _status(vec_store, old_id) == "superseded"

    def test_user_fact_is_eligible(self, vec_store):
        old_id, _ = _store_pair(
            vec_store, "User lives in Lisbon.", "User moved from Lisbon to Berlin.",
            event_type="user_fact",
        )
        assert _status(vec_store, old_id) == "superseded"

    def test_preference_may_retire_a_decision(self, vec_store):
        old_id, _ = _store_pair(
            vec_store,
            "We decided to use PostgreSQL as the primary database.",
            "User prefers SQLite over PostgreSQL as the primary database.",
            event_type="decision",
            newer_type="user_preference",
        )
        assert _status(vec_store, old_id) == "superseded"

    def test_decision_never_retires_a_preference(self, vec_store):
        old_id, _ = _store_pair(
            vec_store,
            "User prefers dark mode in the editor.",
            "We now use light mode in the editor.",
            event_type="user_preference",
            newer_type="decision",
        )
        assert _status(vec_store, old_id) == "active"

    def test_below_similarity_gate_is_neither_retired_nor_flagged(self, vec_store):
        old_id = vec_store.store(
            content=UPDATE[0],
            metadata={"event_type": "decision", "project": "/work/alpha"},
            embedding=_embedding(3.0),
        )
        time.sleep(0.01)
        new_id = vec_store.store(
            content=UPDATE[1],
            metadata={"event_type": "decision", "project": "/work/alpha"},
            embedding=_embedding(50.0),
        )
        assert _status(vec_store, old_id) == "active"
        assert _candidates(vec_store, new_id) == []

    def test_ineligible_type_is_untouched(self, vec_store):
        old_id, new_id = _store_pair(vec_store, *UPDATE, event_type="session_summary")
        assert _status(vec_store, old_id) == "active"
        assert _candidates(vec_store, new_id) == []

    def test_already_superseded_memory_is_not_a_candidate(self, vec_store):
        old_id, mid_id = _store_pair(vec_store, *UPDATE)
        assert _status(vec_store, old_id) == "superseded"
        time.sleep(0.01)
        vec_store.store(
            content="The project now uses DuckDB for storage.",
            metadata={"event_type": "decision", "project": "/work/alpha"},
            embedding=_near(_embedding(1.0), 0.25),
        )
        report = vec_store.get_last_supersession_results()
        assert old_id not in {r["node_id"] for r in report}

    def test_hook_capture_never_retires(self, vec_store):
        """allow_supersession=False turns a would-be retirement into a candidate."""
        old_id, new_id = _store_pair(vec_store, *UPDATE, allow_supersession=False)

        assert _status(vec_store, old_id) == "active"
        [candidate] = _candidates(vec_store, new_id)
        assert candidate["target_id"] == old_id
        report = vec_store.get_last_supersession_results()
        assert [(r["node_id"], r["action"], r["signal"]) for r in report] == [
            (old_id, "candidate", "update_marker")
        ]


class TestScopeIsolation:
    """A memory in another project or entity is never retired or flagged."""

    def test_other_project_is_untouched(self, vec_store):
        old_id, new_id = _store_pair(
            vec_store, *UPDATE,
            older_scope={"project": "/work/client-a"},
            newer_scope={"project": "/work/client-b"},
        )
        assert _status(vec_store, old_id) == "active"
        assert _candidates(vec_store, new_id) == []
        assert vec_store.get_last_supersession_results() == []

    def test_other_entity_is_untouched(self, vec_store):
        old_id, new_id = _store_pair(
            vec_store,
            "Invoices are sent on the first business day of each month.",
            "Invoices are now sent on the last business day of each month.",
            older_scope={"project": "/p", "entity_id": "acme"},
            newer_scope={"project": "/p", "entity_id": "globex"},
        )
        assert _status(vec_store, old_id) == "active"
        assert _candidates(vec_store, new_id) == []

    @pytest.mark.parametrize(
        "older_entity,newer_entity", [("acme", None), (None, "acme")]
    )
    def test_entity_set_on_one_side_only_is_not_shared(
        self, vec_store, older_entity, newer_entity
    ):
        old_id, _ = _store_pair(
            vec_store, *UPDATE,
            older_scope={"project": "/p", "entity_id": older_entity},
            newer_scope={"project": "/p", "entity_id": newer_entity},
        )
        assert _status(vec_store, old_id) == "active"


# ---------------------------------------------------------------------------
# Through the bridge: what the caller sees
# ---------------------------------------------------------------------------


@pytest.fixture
def bridge_with_embeddings(_reset_bridge, monkeypatch):
    """Route the bridge through deterministic embeddings keyed by content."""
    import omega.embedding as embedding

    base = _embedding(7.0)
    vectors = {
        UPDATE[0]: base,
        UPDATE[1]: _near(base),
        DISTINCT[0]: _embedding(11.0),
        DISTINCT[1]: _near(_embedding(11.0)),
        LONG_DISTINCT[0]: _embedding(13.0),
        LONG_DISTINCT[1]: _near(_embedding(13.0)),
        LONG_UPDATE[0]: _embedding(17.0),
        LONG_UPDATE[1]: _near(_embedding(17.0)),
    }

    def fake(text, *args, **kwargs):
        return vectors.get(text) or _embedding(float(sum(map(ord, text[:50]))))

    monkeypatch.setattr(embedding, "generate_embedding", fake)
    import omega.bridge as bridge
    return bridge


def _node_id(result: str) -> str:
    return result.split()[1]


def test_bridge_reports_a_retirement(bridge_with_embeddings):
    bridge = bridge_with_embeddings
    first = bridge.store(UPDATE[0], event_type="decision", project="/work/alpha")
    old_id = _node_id(first)

    result = bridge.store(UPDATE[1], event_type="decision", project="/work/alpha")

    assert "[SUPERSEDED]" in result
    assert old_id in result
    store = bridge._get_store()
    assert _status(store, old_id) == "superseded"


def test_bridge_reports_a_candidate_and_keeps_the_old_memory(bridge_with_embeddings):
    bridge = bridge_with_embeddings
    old_id = _node_id(bridge.store(DISTINCT[0], event_type="decision", project="/work/alpha"))

    result = bridge.store(DISTINCT[1], event_type="decision", project="/work/alpha")

    assert "[POSSIBLE UPDATE]" in result
    assert old_id in result
    assert 'omega_memory(action="supersede"' in result
    assert _status(bridge._get_store(), old_id) == "active"


def test_bridge_never_retires_across_projects(bridge_with_embeddings):
    bridge = bridge_with_embeddings
    old_id = _node_id(bridge.store(UPDATE[0], event_type="decision", project="/work/client-a"))

    result = bridge.store(UPDATE[1], event_type="decision", project="/work/client-b")

    assert "[SUPERSEDED]" not in result
    assert _status(bridge._get_store(), old_id) == "active"


def test_candidate_record_survives_the_observation_summary(bridge_with_embeddings):
    """The bridge used to overwrite the new memory's metadata after store()."""
    bridge = bridge_with_embeddings
    old_id = _node_id(bridge.store(LONG_DISTINCT[0], event_type="decision", project="/work/alpha"))

    new_id = _node_id(bridge.store(LONG_DISTINCT[1], event_type="decision", project="/work/alpha"))

    node = bridge._get_store().get_node(new_id, track_access=False)
    assert node.metadata.get("observation")
    assert [c["target_id"] for c in node.metadata.get("supersession_candidates", [])] == [old_id]


def test_update_is_not_swallowed_by_word_overlap_dedup(bridge_with_embeddings):
    """9 of 11 words shared clears the 0.80 dedup bar; the update must still land."""
    bridge = bridge_with_embeddings
    old_id = _node_id(bridge.store(LONG_UPDATE[0], event_type="decision", project="/work/alpha"))

    result = bridge.store(LONG_UPDATE[1], event_type="decision", project="/work/alpha")

    assert result.startswith("Stored"), result
    assert "[SUPERSEDED]" in result and old_id in result


def test_hook_capture_flags_but_never_retires(bridge_with_embeddings):
    """Audit B2: junk captured by a hook could retire a real decision."""
    bridge = bridge_with_embeddings
    assert detect_update_signal(LONG_UPDATE[1], LONG_UPDATE[0]) == "update_marker"
    old_id = _node_id(bridge.store(LONG_UPDATE[0], event_type="decision", project="/work/alpha"))

    result = bridge.auto_capture(
        content=LONG_UPDATE[1],
        event_type="decision",
        metadata={"source": "auto_capture_hook"},
        project="/work/alpha",
    )

    assert "[SUPERSEDED]" not in result
    assert "[POSSIBLE UPDATE]" in result and old_id in result
    assert _status(bridge._get_store(), old_id) == "active"
