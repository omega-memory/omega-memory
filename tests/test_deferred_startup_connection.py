"""The deferred-startup thread must never touch the store's primary connection.

Callers such as bridge._check_graduation run statements on ``store._conn``
directly, outside the store lock. While the deferred integrity check shared
that connection, a commit on it could land with the PRAGMA still mid-statement:
"cannot commit transaction - SQL statements in progress". It surfaced as a
flaky failure in unrelated tests on Python 3.13, where it reproduced in about
half of all runs, and the same race is open to every direct caller on any
version during the first seconds after a store opens.
"""
import json
import threading

from omega.sqlite_store import SQLiteStore


class _ForbiddenConnection:
    """Stands in for the primary connection; records every use instead of serving it."""

    def __init__(self) -> None:
        self.uses: list[str] = []

    def __getattr__(self, name: str) -> None:
        self.uses.append(name)
        raise AssertionError(f"deferred startup used the primary connection: {name}")


def _open_settled_store(db_path) -> SQLiteStore:
    store = SQLiteStore(db_path=db_path)
    store._deferred_thread.join(timeout=10)
    assert store._deferred_startup_done
    return store


def test_deferred_startup_reads_through_its_own_connection(tmp_omega_dir):
    store = _open_settled_store(tmp_omega_dir / "omega.db")
    try:
        store.store(content="A memory worth backing up", metadata={"event_type": "decision"})
        primary = store._conn
        forbidden = _ForbiddenConnection()
        store._conn = forbidden
        try:
            store._deferred_startup()
        finally:
            store._conn = primary

        assert forbidden.uses == []
        backups = sorted((tmp_omega_dir / "backups").glob("omega-auto-*.json"))
        assert len(backups) == 1, "the backup path must have run to completion"
        exported = json.loads(backups[0].read_text())
        assert exported["node_count"] == 1
    finally:
        store.close()


def test_deferred_startup_does_not_wait_for_the_store_lock(tmp_omega_dir):
    """Startup work runs on its own connection, so it needs no turn on the lock.

    Holding the lock from another thread would have hung the old
    implementation; the first tool call used to wait behind the integrity
    check the same way.
    """
    store = _open_settled_store(tmp_omega_dir / "omega.db")
    try:
        store._deferred_startup_done = False
        with store._lock:
            worker = threading.Thread(target=store._deferred_startup, daemon=True)
            worker.start()
            worker.join(timeout=10)
            assert not worker.is_alive(), "deferred startup queued behind the store lock"
        assert store._deferred_startup_done
    finally:
        store.close()


def test_export_reads_from_the_connection_it_is_given(tmp_omega_dir):
    store = _open_settled_store(tmp_omega_dir / "omega.db")
    try:
        store.store(content="Exported through a sibling connection", metadata={"event_type": "decision"})
        sibling = store._open_deferred_read_conn()
        try:
            result = store.export_to_file(tmp_omega_dir / "export.json", conn=sibling)
        finally:
            sibling.close()
        assert result["node_count"] == 1
    finally:
        store.close()
