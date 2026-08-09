"""Regression tests for the silent-drop store bug.

Embedding-similarity dedup used to collapse textually distinct memories that
scored above a cosine threshold, discard the incoming content, and return the
existing node ID — which bridge.py then reported as "Stored". Records that
differed only in their numbers (successive benchmark runs, version counts,
metrics) were the worst case, because sentence embeddings are near-blind to
digits.

The check was also completely ungated here: it compared against the single
nearest vector neighbour regardless of event_type, agent_type, or project, so
unrelated memories could absorb each other's content.

These tests pin three properties:
  1. Content differing only in numbers is stored as distinct memories.
  2. Memories of different types are never collapsed into one another.
  3. A store that does collapse into an existing memory says so, rather than
     claiming a write happened.
"""

from omega.embedding import generate_embedding

V400 = (
    "Benchmark run v400 complete. Latency p50 120ms, p95 340ms, "
    "throughput 810 rps, error rate 0.4 percent."
)
V401 = (
    "Benchmark run v401 complete. Latency p50 96ms, p95 210ms, "
    "throughput 1150 rps, error rate 0.1 percent."
)


def test_numeric_variants_are_stored_separately(store):
    """Same prose, different measurements — must not collapse.

    These two strings embed at ~0.96 cosine similarity, well above the old
    0.88 dedup threshold, so this is the exact case that silently lost data.
    """
    first = store.store(
        content=V400,
        metadata={"event_type": "memory"},
        embedding=generate_embedding(V400),
    )
    second = store.store(
        content=V401,
        metadata={"event_type": "memory"},
        embedding=generate_embedding(V401),
    )

    assert first != second, "numeric-only differences must not dedup"

    contents = {
        store.get_node(first, track_access=False).content,
        store.get_node(second, track_access=False).content,
    }
    assert contents == {V400, V401}, "both versions must be retrievable verbatim"


def test_similar_content_of_different_types_stays_separate(store):
    """The old check ignored event_type entirely — a decision could absorb a lesson."""
    a = store.store(
        content=V400,
        metadata={"event_type": "decision"},
        embedding=generate_embedding(V400),
    )
    b = store.store(
        content=V401,
        metadata={"event_type": "lesson_learned"},
        embedding=generate_embedding(V401),
    )
    assert a != b


def test_numeric_variant_store_is_not_reported_as_dedup(store):
    """The insert path must not set the dedup flag."""
    store.store(
        content=V400, metadata={"event_type": "memory"}, embedding=generate_embedding(V400)
    )
    store.store(
        content=V401, metadata={"event_type": "memory"}, embedding=generate_embedding(V401)
    )

    assert store.get_last_store_deduped() is False


def test_identical_content_still_dedups_and_reports_it(store):
    """Exact duplicates should still collapse — and admit that they did."""
    first = store.store(
        content=V400, metadata={"event_type": "memory"}, embedding=generate_embedding(V400)
    )
    second = store.store(
        content=V400, metadata={"event_type": "memory"}, embedding=generate_embedding(V400)
    )

    assert first == second, "byte-identical content should dedup via content hash"
    assert store.get_last_store_deduped() is True


def test_dedup_flag_is_consume_once(store):
    """A stale flag would mislabel the next insert as a dedup."""
    store.store(
        content=V400, metadata={"event_type": "memory"}, embedding=generate_embedding(V400)
    )
    store.store(
        content=V400, metadata={"event_type": "memory"}, embedding=generate_embedding(V400)
    )

    assert store.get_last_store_deduped() is True
    assert store.get_last_store_deduped() is False
