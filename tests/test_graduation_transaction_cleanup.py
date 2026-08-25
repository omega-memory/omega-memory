"""_check_graduation must not strand a write transaction when its commit fails.

The UPDATE in _check_graduation runs directly on the store's long-lived primary
connection, opened isolation_level="IMMEDIATE": the first DML takes a RESERVED
lock held until commit or rollback. The bare `except Exception` around it used
to swallow a failing commit and return None, leaving that transaction open on a
connection that then sits idle. Every other writer to omega.db blocks behind it,
and an idle holder has no SQLite stack frame, so it cannot be identified by
stack sampling. Observed live on 2026-08-25 as a ~30-minute total write stall.

The tests below drive the real failure -- commit() raising -- and assert on the
observable database state, not on log text.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from omega import bridge
from omega.sqlite_store import SQLiteStore


def _assert_isolated(db, isolated_root: Path) -> None:
    """Fail closed unless the store is inside this test's own directory.

    Binding the store explicitly (below) rather than relying on OMEGA_HOME plus
    the process-wide _get_store() singleton: an earlier revision relied on
    import-order and resolved to the owner's real ~/.omega/omega.db.
    """
    real = (Path.home() / ".omega" / "omega.db").resolve()
    actual = Path(db.db_path).resolve()
    assert actual != real, f"REFUSING: store resolved to the live DB {actual}"
    assert str(actual).startswith(str(isolated_root.resolve())), (
        f"REFUSING: store is outside the isolated dir: {actual}"
    )


@pytest.fixture
def node_id(request):
    return f"grad-{request.node.name}"


@pytest.fixture
def store(tmp_path, node_id, monkeypatch):
    """A store bound explicitly to this test's own database file."""
    db = SQLiteStore(db_path=tmp_path / "omega.db")
    _assert_isolated(db, tmp_path)
    # _check_graduation resolves the store through _get_store(); point it here.
    monkeypatch.setattr(bridge, "_get_store", lambda: db)

    signals = (
        '{"feedback_signals": ['
        '{"rating":"helpful","reason":"diff-correlated"},'
        '{"rating":"helpful","reason":"diff-correlated"}]}'
    )
    db._conn.execute(
        "INSERT OR REPLACE INTO memories "
        "(node_id, content, metadata, priority, created_at) "
        "VALUES (?, 'c', ?, 3, datetime('now'))",
        (node_id, signals),
    )
    db._conn.commit()
    assert not db._conn.in_transaction
    yield db
    try:
        if db._conn.in_transaction:
            db._conn.rollback()
    except Exception:
        pass


def _independent_write_lock_free(db_path) -> bool:
    """True if a SEPARATE connection can take the write lock."""
    conn = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=1.0)
        conn.execute("PRAGMA busy_timeout=1000")
        conn.execute("BEGIN IMMEDIATE")
        conn.rollback()
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        if conn is not None:
            conn.close()


class _FailingCommit:
    """Delegates everything, but commit() raises 'database is locked'.

    sqlite3.Connection attributes are read-only, so the commit cannot be
    patched in place. This reproduces the real failure: the UPDATE succeeded
    and the commit did not.
    """

    def __init__(self, conn):
        self._conn = conn

    def commit(self):
        raise sqlite3.OperationalError("database is locked")

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_control_the_harness_actually_creates_the_leak(store, node_id):
    """Red-capability control: with NO cleanup, a failing commit leaks.

    Reproduces the pre-fix behaviour inline -- UPDATE, commit raises, exception
    swallowed -- and proves the harness detects a stranded transaction. Without
    this, the post-fix assertions could pass for the wrong reason.
    """
    real = store._conn
    try:
        real.execute(
            "UPDATE memories SET priority = MIN(COALESCE(priority,3)+1,5) "
            "WHERE node_id = ?", (node_id,))
        try:
            _FailingCommit(real).commit()
        except Exception:
            pass                                   # the old handler: swallow
        assert real.in_transaction, "the leak was not reproduced"
        assert not _independent_write_lock_free(store.db_path), (
            "a stranded transaction must block an independent writer"
        )
    finally:
        real.rollback()


def test_failed_commit_does_not_strand_the_transaction(store, node_id, monkeypatch):
    """The subject: after the fix, a failing commit leaves nothing open."""
    real = store._conn
    monkeypatch.setattr(store, "_conn", _FailingCommit(real), raising=False)

    result = bridge._check_graduation(node_id)

    assert result is None, "a failed graduation must not report success"
    assert not real.in_transaction, "transaction was left open after a failed commit"


def test_independent_writer_succeeds_after_the_failure(store, node_id, monkeypatch):
    """The operational property that actually matters: the DB is usable."""
    real = store._conn
    monkeypatch.setattr(store, "_conn", _FailingCommit(real), raising=False)

    bridge._check_graduation(node_id)

    assert _independent_write_lock_free(store.db_path), (
        "another connection still cannot take the write lock"
    )


def test_failed_update_is_not_durable(store, node_id, monkeypatch):
    """Rolling back must discard the write, not silently keep it."""
    real = store._conn
    before = real.execute(
        "SELECT priority FROM memories WHERE node_id=?", (node_id,)).fetchone()[0]

    monkeypatch.setattr(store, "_conn", _FailingCommit(real), raising=False)
    bridge._check_graduation(node_id)
    monkeypatch.setattr(store, "_conn", real, raising=False)

    after = real.execute(
        "SELECT priority FROM memories WHERE node_id=?", (node_id,)).fetchone()[0]
    assert after == before, "a failed graduation must not be durable"


def test_subsequent_normal_operation_still_succeeds(store, node_id, monkeypatch):
    """Recovery must be complete: the next call works normally."""
    real = store._conn
    monkeypatch.setattr(store, "_conn", _FailingCommit(real), raising=False)
    bridge._check_graduation(node_id)
    monkeypatch.setattr(store, "_conn", real, raising=False)

    before = real.execute(
        "SELECT priority FROM memories WHERE node_id=?", (node_id,)).fetchone()[0]
    result = bridge._check_graduation(node_id)

    assert result == "graduated"
    after = real.execute(
        "SELECT priority FROM memories WHERE node_id=?", (node_id,)).fetchone()[0]
    assert after == before + 1
    assert not real.in_transaction
    assert _independent_write_lock_free(store.db_path)


def test_success_semantics_are_unchanged(store, node_id):
    """The fix must not alter the working path: it still commits, once."""
    before = store._conn.execute(
        "SELECT priority FROM memories WHERE node_id=?", (node_id,)).fetchone()[0]

    assert bridge._check_graduation(node_id) == "graduated"

    after = store._conn.execute(
        "SELECT priority FROM memories WHERE node_id=?", (node_id,)).fetchone()[0]
    assert after == before + 1
    assert not store._conn.in_transaction
    # Durable to an independent reader -> genuinely committed, not just pending.
    other = sqlite3.connect(str(store.db_path))
    try:
        assert other.execute(
            "SELECT priority FROM memories WHERE node_id=?", (node_id,)).fetchone()[0] == after
    finally:
        other.close()
