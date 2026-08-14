"""Public read-connection API required by Pro analyzers."""

import queue
import sqlite3
import threading

import pytest


def test_thread_local_read_connection_is_stable_and_configured(store):
    conn = store.get_thread_local_read_conn()

    assert conn is store.get_thread_local_read_conn()
    assert conn is not store._conn
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] in {5000, 10000}
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_thread_local_read_connection_differs_between_threads(store):
    main_conn = store.get_thread_local_read_conn()
    result = queue.Queue()

    def worker():
        result.put(store.get_thread_local_read_conn())

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert result.get_nowait() is not main_conn


def test_close_closes_current_thread_read_connection(tmp_path):
    from omega.sqlite_store import SQLiteStore

    local_store = SQLiteStore(db_path=tmp_path / "thread-close.db")
    conn = local_store.get_thread_local_read_conn()

    local_store.close()

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
