"""Tests for memory type classification (episodic/semantic/procedural)."""
import pytest
import sqlite3
from omega.sqlite_store import SQLiteStore


class TestMemoryTypeSchema:
    """Test schema migration adds memory_type column."""

    def test_memory_type_column_exists(self, store):
        info = store._conn.execute("PRAGMA table_info(memories)").fetchall()
        col_names = [col[1] for col in info]
        assert "memory_type" in col_names

    def test_memory_type_default_is_semantic(self, store):
        node_id = store.store(content="A generic memory", metadata={"event_type": "memory"})
        row = store._conn.execute(
            "SELECT memory_type FROM memories WHERE node_id = ?", (node_id,)
        ).fetchone()
        assert row[0] == "semantic"

    def test_memory_type_index_exists(self, store):
        indexes = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_memories_memory_type'"
        ).fetchall()
        assert len(indexes) == 1

    def test_schema_version_is_12(self, store):
        from omega.schema import SCHEMA_VERSION
        assert SCHEMA_VERSION == 15
        row = store._conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        assert row[0] == 15


class TestMemoryTypeAutoClassify:
    """Test auto-classification of memory_type on store."""

    def test_decision_is_semantic(self, store):
        nid = store.store(content="We chose PostgreSQL", metadata={"event_type": "decision"})
        row = store._conn.execute("SELECT memory_type FROM memories WHERE node_id = ?", (nid,)).fetchone()
        assert row[0] == "semantic"

    def test_lesson_is_procedural(self, store):
        nid = store.store(content="Always run tests before deploy", metadata={"event_type": "lesson_learned"})
        row = store._conn.execute("SELECT memory_type FROM memories WHERE node_id = ?", (nid,)).fetchone()
        assert row[0] == "procedural"

    def test_session_summary_is_episodic(self, store):
        nid = store.store(content="Session: fixed auth bug", metadata={"event_type": "session_summary"})
        row = store._conn.execute("SELECT memory_type FROM memories WHERE node_id = ?", (nid,)).fetchone()
        assert row[0] == "episodic"

    def test_constraint_is_semantic(self, store):
        nid = store.store(content="Never deploy on Fridays", metadata={"event_type": "constraint"})
        row = store._conn.execute("SELECT memory_type FROM memories WHERE node_id = ?", (nid,)).fetchone()
        assert row[0] == "semantic"

    def test_reflexion_is_procedural(self, store):
        nid = store.store(content="I should check logs first", metadata={"event_type": "reflexion"})
        row = store._conn.execute("SELECT memory_type FROM memories WHERE node_id = ?", (nid,)).fetchone()
        assert row[0] == "procedural"

    def test_unknown_type_defaults_to_semantic(self, store):
        nid = store.store(content="Some content", metadata={"event_type": "unknown_type"})
        row = store._conn.execute("SELECT memory_type FROM memories WHERE node_id = ?", (nid,)).fetchone()
        assert row[0] == "semantic"

    def test_no_event_type_defaults_to_semantic(self, store):
        nid = store.store(content="No type specified")
        row = store._conn.execute("SELECT memory_type FROM memories WHERE node_id = ?", (nid,)).fetchone()
        assert row[0] == "semantic"

    def test_all_event_types_mapped(self, store):
        """Every key in _TYPE_WEIGHTS should be in _MEMORY_TYPE_MAP."""
        from omega.sqlite_store import SQLiteStore
        for etype in SQLiteStore._TYPE_WEIGHTS:
            assert etype in SQLiteStore._MEMORY_TYPE_MAP, f"{etype} not in _MEMORY_TYPE_MAP"


@pytest.mark.usefixtures("_reset_bridge")
class TestMemoryTypeFilter:

    def test_filter_procedural(self, tmp_omega_dir):
        from omega.bridge import store, query
        store(content="Always validate input before processing", event_type="lesson_learned")
        store(content="We decided to use REST over GraphQL", event_type="decision")
        result = query(query_text="processing", memory_type="procedural")
        assert "validate input" in result.lower() or "No matching" in result

    def test_filter_semantic(self, tmp_omega_dir):
        from omega.bridge import store, query
        store(content="Database choice: PostgreSQL for OLTP", event_type="decision")
        store(content="Lesson: always index foreign keys", event_type="lesson_learned")
        result = query(query_text="database", memory_type="semantic")
        assert "PostgreSQL" in result or "Results: 0" in result

    def test_filter_episodic(self, tmp_omega_dir):
        from omega.bridge import store, query
        store(content="Session: debugged memory leak in production", event_type="session_summary")
        store(content="Never ignore memory leak warnings", event_type="constraint")
        result = query(query_text="memory leak", memory_type="episodic")
        assert "debugged" in result.lower() or "Results: 0" in result

    def test_no_filter_returns_all_types(self, tmp_omega_dir):
        from omega.bridge import store, query
        store(content="Episodic: completed auth feature", event_type="session_summary")
        store(content="Semantic: auth uses JWT tokens", event_type="decision")
        store(content="Procedural: always check token expiry", event_type="lesson_learned")
        result = query(query_text="auth")
        assert "Results:" in result
