"""A memory keeps its recent feedback signals and a running count, not every signal."""

import json
import sqlite3

from omega.feedback_signals import FEEDBACK_SIGNALS_KEPT
from omega.sqlite_store import SCHEMA_VERSION, SQLiteStore


def _metadata(store, node_id):
    row = store._conn.execute(
        "SELECT metadata FROM memories WHERE node_id = ?", (node_id,)
    ).fetchone()
    return json.loads(row[0])


def test_history_is_capped_and_the_total_keeps_counting(store):
    nid = store.store(content="A memory that surfaces in nearly every session")

    for i in range(FEEDBACK_SIGNALS_KEPT + 15):
        result = store.record_feedback(nid, "helpful", reason=f"signal {i}")

    meta = _metadata(store, nid)
    assert len(meta["feedback_signals"]) == FEEDBACK_SIGNALS_KEPT
    assert meta["feedback_signals"][-1]["reason"] == f"signal {FEEDBACK_SIGNALS_KEPT + 14}"
    assert meta["feedback_signals"][0]["reason"] == "signal 15"
    assert meta["feedback_counts"] == {"helpful": FEEDBACK_SIGNALS_KEPT + 15}
    assert meta["feedback_score"] == FEEDBACK_SIGNALS_KEPT + 15
    assert result["total_signals"] == FEEDBACK_SIGNALS_KEPT + 15


def test_counts_are_kept_per_rating(store):
    nid = store.store(content="A memory with mixed feedback over time")

    store.record_feedback(nid, "helpful")
    store.record_feedback(nid, "helpful")
    result = store.record_feedback(nid, "unhelpful")

    assert _metadata(store, nid)["feedback_counts"] == {"helpful": 2, "unhelpful": 1}
    assert result["total_signals"] == 3
    assert result["new_score"] == 1


def test_batch_feedback_is_capped_too(store):
    nid = store.store(content="A memory that the session-end pass rates in batches")

    store.batch_record_feedback([(nid, "helpful", f"batch {i}") for i in range(FEEDBACK_SIGNALS_KEPT + 5)])

    meta = _metadata(store, nid)
    assert len(meta["feedback_signals"]) == FEEDBACK_SIGNALS_KEPT
    assert meta["feedback_counts"] == {"helpful": FEEDBACK_SIGNALS_KEPT + 5}


def test_a_memory_written_before_counts_existed_is_counted_on_its_next_signal(store):
    nid = store.store(content="A memory with a short history from an older version")
    meta = _metadata(store, nid)
    meta["feedback_signals"] = [
        {"rating": "helpful", "reason": None, "timestamp": "2026-01-01T00:00:00+00:00"},
        {"rating": "unhelpful", "reason": None, "timestamp": "2026-01-02T00:00:00+00:00"},
    ]
    meta["feedback_score"] = 0
    store._conn.execute(
        "UPDATE memories SET metadata = ? WHERE node_id = ?", (json.dumps(meta), nid)
    )
    store._conn.commit()

    result = store.record_feedback(nid, "helpful")

    assert _metadata(store, nid)["feedback_counts"] == {"helpful": 2, "unhelpful": 1}
    assert result["total_signals"] == 3


def test_upgrade_trims_existing_histories_and_keeps_their_totals(tmp_omega_dir):
    db_path = tmp_omega_dir / "upgrade.db"
    old = SQLiteStore(db_path=db_path)
    long_id = old.store(content="A memory with years of feedback behind it")
    short_id = old.store(content="A memory with two signals, left alone by the upgrade")
    untouched_id = old.store(content="A memory nobody has rated")
    old.close()

    signals = [
        {"rating": "unhelpful" if i % 10 == 0 else "helpful", "reason": f"old {i}", "timestamp": f"2026-01-01T00:{i % 60:02d}:00+00:00"}
        for i in range(500)
    ]
    conn = sqlite3.connect(str(db_path))
    for node_id, history in ((long_id, signals), (short_id, signals[:2])):
        meta = json.loads(conn.execute("SELECT metadata FROM memories WHERE node_id = ?", (node_id,)).fetchone()[0])
        meta["feedback_signals"] = history
        meta["feedback_score"] = sum(-1 if s["rating"] == "unhelpful" else 1 for s in history)
        conn.execute("UPDATE memories SET metadata = ? WHERE node_id = ?", (json.dumps(meta), node_id))
    conn.execute("UPDATE schema_version SET version = 15")
    conn.commit()
    conn.close()

    upgraded = SQLiteStore(db_path=db_path)
    try:
        version = upgraded._conn.execute("SELECT version FROM schema_version").fetchone()[0]
        long_meta = _metadata(upgraded, long_id)
        short_meta = _metadata(upgraded, short_id)

        assert version == SCHEMA_VERSION == 16
        assert [s["reason"] for s in long_meta["feedback_signals"]] == [f"old {i}" for i in range(480, 500)]
        assert long_meta["feedback_counts"] == {"helpful": 450, "unhelpful": 50}
        assert long_meta["feedback_score"] == 400
        assert len(short_meta["feedback_signals"]) == 2
        assert "feedback_signals" not in _metadata(upgraded, untouched_id)

        result = upgraded.record_feedback(long_id, "helpful")
        assert result["total_signals"] == 501
        assert result["new_score"] == 401
    finally:
        upgraded.close()
