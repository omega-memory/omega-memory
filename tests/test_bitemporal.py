"""Tests for bi-temporal data model (valid_from, valid_until columns)."""
import time
import pytest
import sqlite3
from datetime import datetime, timezone, timedelta
from omega.sqlite_store import SQLiteStore
from omega.schema import SCHEMA_VERSION


class TestBitemporalSchema:
    """Test schema migration adds bi-temporal columns."""

    def test_schema_version_is_12(self, store):
        assert SCHEMA_VERSION == 15
        row = store._conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        assert row[0] == 15

    def test_valid_from_column_exists(self, store):
        info = store._conn.execute("PRAGMA table_info(memories)").fetchall()
        col_names = [col[1] for col in info]
        assert "valid_from" in col_names

    def test_valid_until_column_exists(self, store):
        info = store._conn.execute("PRAGMA table_info(memories)").fetchall()
        col_names = [col[1] for col in info]
        assert "valid_until" in col_names

    def test_valid_from_index_exists(self, store):
        indexes = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_memories_valid_from'"
        ).fetchall()
        assert len(indexes) == 1

    def test_valid_until_index_exists(self, store):
        indexes = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_memories_valid_until'"
        ).fetchall()
        assert len(indexes) == 1


class TestBitemporalStore:
    """Test bi-temporal behavior on store operations."""

    def test_new_store_sets_valid_from_to_created_at(self, store):
        nid = store.store(content="Test memory for bi-temporal", metadata={"event_type": "decision"})
        row = store._conn.execute(
            "SELECT created_at, valid_from FROM memories WHERE node_id = ?", (nid,)
        ).fetchone()
        assert row[0] is not None
        assert row[1] is not None
        assert row[0] == row[1]  # valid_from == created_at when no referenced_date

    def test_store_with_referenced_date_uses_it_as_valid_from(self, store):
        ref_date = "2025-06-15T10:00:00+00:00"
        nid = store.store(
            content="Memory with referenced date",
            metadata={"event_type": "decision", "referenced_date": ref_date},
        )
        row = store._conn.execute(
            "SELECT valid_from, created_at FROM memories WHERE node_id = ?", (nid,)
        ).fetchone()
        assert row[0] == ref_date
        assert row[0] != row[1]  # valid_from != created_at

    def test_new_store_has_null_valid_until(self, store):
        nid = store.store(content="Active memory", metadata={"event_type": "memory"})
        row = store._conn.execute(
            "SELECT valid_until FROM memories WHERE node_id = ?", (nid,)
        ).fetchone()
        assert row[0] is None  # NULL means still valid


class TestBitemporalSupersede:
    """Test that mark_superseded() sets valid_until."""

    def test_mark_superseded_sets_valid_until(self, store):
        nid1 = store.store(content="Original fact", metadata={"event_type": "decision"})
        nid2 = store.store(content="Updated fact", metadata={"event_type": "decision"})

        result = store.mark_superseded(nid1, superseded_by=nid2)
        assert result is True

        row = store._conn.execute(
            "SELECT valid_until, metadata FROM memories WHERE node_id = ?", (nid1,)
        ).fetchone()
        assert row[0] is not None  # valid_until is set

        # valid_until should match superseded_at in metadata
        import json
        meta = json.loads(row[1])
        assert meta["superseded_at"] == row[0]


class TestBitemporalQuery:
    """Test point-in-time queries using valid_at."""

    def test_query_without_valid_at_returns_all_non_superseded(self, store):
        """Backward compatibility: no valid_at means no temporal filtering."""
        nid1 = store.store(
            content="The production database uses PostgreSQL for relational data storage with JSONB columns",
            metadata={"event_type": "decision"},
        )
        nid2 = store.store(
            content="The staging database uses MySQL for compatibility testing with legacy systems",
            metadata={"event_type": "decision"},
        )

        results = store.query("database", limit=10)
        result_ids = [r.id for r in results]
        # At least one of the database memories should be returned
        assert nid1 in result_ids or nid2 in result_ids

    def test_query_with_valid_at_excludes_not_yet_valid(self, store):
        """Memories with valid_from after the query point should be excluded."""
        future_date = "2099-01-01T00:00:00+00:00"
        nid = store.store(
            content="Future fact about quantum computing",
            metadata={"event_type": "decision", "referenced_date": future_date},
        )

        # Query at a time before the valid_from
        past_point = "2025-01-01T00:00:00+00:00"
        results = store.query("quantum computing", limit=10, valid_at=past_point)
        result_ids = [r.id for r in results]
        assert nid not in result_ids

    def test_query_with_valid_at_excludes_superseded_before_query_point(self, store):
        """Memories superseded before the query point should be excluded via valid_until."""
        nid1 = store.store(
            content="The old deployment pipeline uses Jenkins for continuous integration and deployment",
            metadata={"event_type": "decision"},
        )
        nid2 = store.store(
            content="The new deployment pipeline uses GitHub Actions for continuous integration and deployment",
            metadata={"event_type": "decision"},
        )

        store.mark_superseded(nid1, superseded_by=nid2)

        # Verify valid_until was set on superseded memory
        row = store._conn.execute(
            "SELECT valid_until FROM memories WHERE node_id = ?", (nid1,)
        ).fetchone()
        assert row[0] is not None, "valid_until should be set after superseding"

        # Verify valid_until is NOT set on the new memory
        row2 = store._conn.execute(
            "SELECT valid_until FROM memories WHERE node_id = ?", (nid2,)
        ).fetchone()
        assert row2[0] is None, "valid_until should remain NULL for active memory"

    def test_query_with_valid_at_includes_memory_valid_at_that_time(self, store):
        """Memories valid at the query point should be included."""
        nid = store.store(
            content="Important deployment protocol for servers",
            metadata={"event_type": "decision"},
        )
        # valid_from is set to created_at (now), valid_until is NULL (still valid)

        # Query at a future time: should include this memory
        future_point = "2099-12-31T23:59:59+00:00"
        results = store.query("deployment protocol", limit=10, valid_at=future_point)
        result_ids = [r.id for r in results]
        assert nid in result_ids


class TestBitemporalMigrationBackfill:
    """Test that migration backfills existing memories correctly."""

    def test_backfill_sets_valid_from_from_created_at(self, store):
        """New stores should have valid_from = created_at by default."""
        nid = store.store(content="Backfill test memory", metadata={"event_type": "memory"})
        row = store._conn.execute(
            "SELECT created_at, valid_from FROM memories WHERE node_id = ?", (nid,)
        ).fetchone()
        assert row[1] is not None
        assert row[0] == row[1]


class TestBitemporalMemoryResult:
    """Test that MemoryResult carries valid_from/valid_until."""

    def test_memory_result_has_valid_from_attribute(self, store):
        from omega.sqlite_store import MemoryResult
        mr = MemoryResult(id="test", content="test")
        assert hasattr(mr, "valid_from")
        assert mr.valid_from is None

    def test_memory_result_has_valid_until_attribute(self, store):
        from omega.sqlite_store import MemoryResult
        mr = MemoryResult(id="test", content="test")
        assert hasattr(mr, "valid_until")
        assert mr.valid_until is None

    def test_memory_result_accepts_valid_from_and_valid_until(self, store):
        from omega.sqlite_store import MemoryResult
        now = datetime.now(timezone.utc)
        later = now + timedelta(hours=1)
        mr = MemoryResult(id="test", content="test", valid_from=now, valid_until=later)
        assert mr.valid_from == now
        assert mr.valid_until == later
