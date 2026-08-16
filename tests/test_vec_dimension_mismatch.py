"""Regression tests for silent vector loss on embedding-dimension mismatch.

Reported 2026-08-16 by a customer running a 1024-dim model against vec tables
built at 384. The memories row was inserted and its id returned, then the
``memories_vec`` insert raised, and the handler swallowed it:

    except Exception as e:
        logger.debug(f"Vec insert failed: {e}")

``logger.debug`` is below the default INFO level, so nothing surfaced. Writes
reported success while storing no vector for seven hours. Document ingestion
performs the same insert without a guard, so it failed loudly -- which is the
only reason the problem was noticed at all.

These tests pin two properties:
  1. A dimension mismatch raises instead of reporting a successful store.
  2. The failed store leaves no row behind -- the memory and its vector are
     written atomically, so a rejected write cannot leave an unretrievable
     memory that every later search misses.
"""

import pytest

from omega.exceptions import StorageError
from omega.sqlite_store import EMBEDDING_DIM, SQLiteStore


@pytest.fixture
def store(tmp_path):
    store = SQLiteStore(str(tmp_path / "omega.db"))
    if not store._vec_available:
        pytest.skip("sqlite-vec unavailable; dimension enforcement is vec-specific")
    return store


def _count_memories(store) -> int:
    return store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]


def _count_vectors(store) -> int:
    return store._conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]


def test_mismatched_dimension_raises_instead_of_silently_dropping(store):
    """The seven-hour outage: store() returned an id and stored no vector."""
    wrong = [0.1] * (EMBEDDING_DIM * 2)

    with pytest.raises(StorageError) as excinfo:
        store.store("memory written with a mismatched embedding", embedding=wrong)

    message = str(excinfo.value)
    # The error has to name both numbers, or the operator cannot tell which
    # side is wrong without reading the source.
    assert str(len(wrong)) in message
    assert str(EMBEDDING_DIM) in message


def test_rejected_store_leaves_no_orphaned_memory(store):
    """Atomicity: a rejected write must not leave a vectorless memory row.

    Without an explicit rollback the pending INSERT stays in the writer
    connection's open IMMEDIATE transaction and is committed by the next
    successful write -- trading silent vector loss for silent corruption.
    """
    before = _count_memories(store)

    with pytest.raises(StorageError):
        store.store("rejected write", embedding=[0.1] * (EMBEDDING_DIM * 2))

    assert _count_memories(store) == before

    # A subsequent good write must not resurrect the rejected row.
    store.store("good write", embedding=[0.1] * EMBEDDING_DIM)

    assert _count_memories(store) == before + 1
    assert _count_vectors(store) == 1
    rows = store._conn.execute("SELECT content FROM memories").fetchall()
    assert [r[0] for r in rows] == ["good write"]


def test_correct_dimension_still_stores_vector(store):
    """The happy path stays intact."""
    node_id = store.store("well-formed memory", embedding=[0.1] * EMBEDDING_DIM)

    assert node_id
    assert _count_vectors(store) == 1


def test_store_without_embedding_is_unaffected(store):
    """Callers that pass no embedding must not start failing."""
    node_id = store.store("no embedding supplied")

    assert node_id
    assert _count_memories(store) == 1
