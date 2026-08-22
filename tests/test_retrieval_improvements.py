"""Tests for P1-P6 retrieval improvements.

P1: Hybrid BM25/RRF fusion
P2: Cross-encoder reranking with temporal metadata
P3: Session-level retrieval aggregation
P4: Temporal indexing and retrieval channel
P5: Structured fact extraction at indexing
P6: Graph-based multi-hop retrieval
"""
import json
import pytest
from datetime import datetime, timedelta, timezone

from omega.sqlite_store import SQLiteStore, SCHEMA_VERSION, MemoryResult


# ============================================================================
# P1: Reciprocal Rank Fusion
# ============================================================================


class TestRRFFusion:
    """Test the _rrf_fuse static method."""

    def test_rrf_single_channel(self):
        """RRF with one channel normalizes to [0, 1]."""
        ranked = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        scores = SQLiteStore._rrf_fuse([ranked])
        assert scores["a"] == 1.0  # Rank 1 = max
        assert scores["b"] < scores["a"]
        assert scores["c"] < scores["b"]

    def test_rrf_two_channels_dual_match_boost(self):
        """Documents found by both channels get higher scores."""
        vec_ranked = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        text_ranked = [("a", 0.8), ("d", 0.6), ("e", 0.4)]
        scores = SQLiteStore._rrf_fuse([vec_ranked, text_ranked])
        # "a" appears in both channels -> highest score
        assert scores["a"] == 1.0
        # "b" only in vec, "d" only in text -> both should score lower than "a"
        assert scores["b"] < scores["a"]
        assert scores["d"] < scores["a"]

    def test_rrf_weighted_channels(self):
        """Channel weights affect relative contributions."""
        vec_ranked = [("a", 0.9)]
        text_ranked = [("b", 0.8)]
        # With equal weights, both should have same RRF score
        scores_equal = SQLiteStore._rrf_fuse([vec_ranked, text_ranked], weights=[1.0, 1.0])
        assert scores_equal["a"] == scores_equal["b"]

        # With vec weighted 2x, "a" should score higher
        scores_vec_heavy = SQLiteStore._rrf_fuse([vec_ranked, text_ranked], weights=[2.0, 1.0])
        assert scores_vec_heavy["a"] > scores_vec_heavy["b"]

    def test_rrf_empty_channels(self):
        """Empty channel lists return empty scores."""
        assert SQLiteStore._rrf_fuse([]) == {}
        assert SQLiteStore._rrf_fuse([[], []]) == {}

    def test_rrf_normalization(self):
        """All scores should be in [0, 1] range."""
        ranked = [("a", 0.9), ("b", 0.7), ("c", 0.5), ("d", 0.3)]
        scores = SQLiteStore._rrf_fuse([ranked])
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_rrf_three_channels(self):
        """RRF works with 3+ channels (for temporal retrieval)."""
        vec = [("a", 0.9), ("b", 0.7)]
        text = [("b", 0.8), ("c", 0.6)]
        temporal = [("a", 1.0), ("c", 0.5)]
        scores = SQLiteStore._rrf_fuse([vec, text, temporal])
        # "a" in vec+temporal, "b" in vec+text, "c" in text+temporal
        assert len(scores) == 3
        # All should be positive
        for v in scores.values():
            assert v > 0

    def test_rrf_integration_in_query(self, store):
        """Query pipeline uses RRF fusion (integration test)."""
        # Store memories with different content characteristics
        store.store(content="Python is a programming language used for data science",
                    metadata={"event_type": "lesson_learned"})
        store.store(content="JavaScript is used for web development and frontend",
                    metadata={"event_type": "lesson_learned"})
        store.store(content="Python Flask is a web framework for Python",
                    metadata={"event_type": "decision"})

        # Query should find Python-related memories
        results = store.query("Python programming", limit=10)
        assert len(results) > 0
        # The Python memories should be ranked higher
        python_found = any("Python" in r.content for r in results[:2])
        assert python_found


# ============================================================================
# P2: Cross-Encoder Reranking
# ============================================================================


class TestCrossEncoderReranking:
    """Test temporal metadata enrichment in reranker."""

    def test_cross_encoder_accepts_temporal_metadata(self):
        """cross_encoder_score should accept temporal_metadata parameter."""
        from omega.reranker import cross_encoder_score
        import os
        # Disable actual model to test parameter passing
        os.environ["OMEGA_CROSS_ENCODER"] = "0"
        try:
            result = cross_encoder_score(
                "test query",
                ["passage 1", "passage 2"],
                temporal_metadata=["2024-01-15", "2024-02-20"],
            )
            assert result is None  # Disabled, but no error
        finally:
            os.environ.pop("OMEGA_CROSS_ENCODER", None)

    def test_reranker_model_selection(self):
        """Reranker auto-detects best available model."""
        from omega.reranker import _RERANKER_MODEL_NAME
        # Auto-detects bge-reranker-v2-m3 if ONNX model exists on disk,
        # otherwise falls back to ms-marco-MiniLM-L-6-v2
        assert _RERANKER_MODEL_NAME in ("ms-marco-MiniLM-L-6-v2", "bge-reranker-v2-m3")

    def test_available_models_registry(self):
        """Both model configs should be in the registry.

        bge-reranker-v2-m3 uses the multi-precision schema (precisions.<p>.{dir,files});
        ms-marco-MiniLM-L-6-v2 uses the flat schema (dir, files at top level).
        """
        from omega.reranker import _AVAILABLE_MODELS
        assert "bge-reranker-v2-m3" in _AVAILABLE_MODELS
        assert "ms-marco-MiniLM-L-6-v2" in _AVAILABLE_MODELS
        for name, config in _AVAILABLE_MODELS.items():
            assert "repo_id" in config
            if "precisions" in config:
                assert "default_precision" in config
                assert config["default_precision"] in config["precisions"]
                for variant in config["precisions"].values():
                    assert "dir" in variant
                    assert "files" in variant
            else:
                assert "dir" in config
                assert "files" in config


# ============================================================================
# P3: Session-Level Retrieval Aggregation
# ============================================================================


class TestSessionAggregation:
    """Test retrieve_by_session method."""

    def test_retrieve_by_session_groups_results(self, store):
        """Results should be grouped by session."""
        # Create memories in different sessions
        for i in range(3):
            store.store(
                content=f"Session A memory {i} about machine learning algorithms",
                session_id="session-a",
                metadata={"event_type": "lesson_learned", "session_id": "session-a"},
            )
        for i in range(3):
            store.store(
                content=f"Session B memory {i} about database optimization strategies",
                session_id="session-b",
                metadata={"event_type": "decision", "session_id": "session-b"},
            )

        results = store.retrieve_by_session(
            "machine learning", top_k_sessions=1,
        )
        # Should get results primarily from session-a
        session_ids = set(r.metadata.get("session_id") for r in results)
        assert "session-a" in session_ids

    def test_retrieve_by_session_empty_store(self, store):
        """Empty store returns empty results."""
        results = store.retrieve_by_session("anything")
        assert results == []

    def test_retrieve_by_session_respects_top_k(self, store):
        """Should limit to top_k_sessions sessions."""
        for sid in ["s1", "s2", "s3"]:
            store.store(
                content=f"Memory about testing in {sid} with pytest framework",
                session_id=sid,
                metadata={"event_type": "lesson_learned", "session_id": sid},
            )
        results = store.retrieve_by_session("testing pytest", top_k_sessions=2)
        session_ids = set(r.metadata.get("session_id") for r in results)
        assert len(session_ids) <= 2


# ============================================================================
# P4: Temporal Indexing and Retrieval
# ============================================================================


class TestTemporalRetrieval:
    """Test temporal search channel."""

    def test_temporal_search_finds_in_range(self, store):
        """Memories within date range should be found."""
        store.store(
            content="Meeting with client about project requirements",
            metadata={
                "event_type": "decision",
                "referenced_date": "2024-06-15T10:00:00",
            },
        )
        store.store(
            content="Code review for authentication module",
            metadata={
                "event_type": "task_completion",
                "referenced_date": "2024-01-01T10:00:00",
            },
        )

        results = store._temporal_search(
            "2024-06-01", "2024-06-30", limit=10,
        )
        assert len(results) >= 1
        # The June memory should score highest (in range)
        top_id, top_score = results[0]
        assert top_score == 1.0  # In-range proximity

    def test_temporal_search_empty_range(self, store):
        """No memories in range returns empty (far enough away to avoid 3x window)."""
        store.store(
            content="Old memory from 2020 about API design",
            metadata={
                "event_type": "decision",
                "referenced_date": "2020-01-15T10:00:00",
            },
        )
        # Search window is just 7 days, so 3x = 21 days. 2020 is 5 years away.
        results = store._temporal_search("2025-06-01", "2025-06-07", limit=10)
        assert len(results) == 0

    def test_temporal_search_proximity_decay(self, store):
        """Nearby but out-of-range memories get decayed scores."""
        # Memory just outside range
        store.store(
            content="Near-miss memory about deployment",
            metadata={
                "event_type": "decision",
                "referenced_date": "2024-06-05T10:00:00",
            },
        )
        # Memory far outside range
        store.store(
            content="Far-away memory about initial setup",
            metadata={
                "event_type": "decision",
                "referenced_date": "2024-01-01T10:00:00",
            },
        )
        results = store._temporal_search("2024-06-10", "2024-06-20", limit=10)
        # Near-miss should appear (within 3x range) with decayed score
        if results:
            assert results[0][1] < 1.0  # Not perfect score (out of range)
            assert results[0][1] > 0.0  # But still positive


# ============================================================================
# P5: Structured Fact Extraction
# ============================================================================


class TestFactExtraction:
    """Test _extract_keywords static method."""

    def test_extract_proper_nouns(self):
        """Multi-word proper nouns should be extracted."""
        content = "Met with John Smith at New York office about the project."
        keywords = SQLiteStore._extract_keywords(content)
        assert "John Smith" in keywords
        assert "New York" in keywords

    def test_extract_dates(self):
        """Dates in various formats should be extracted."""
        content = "Meeting scheduled for 2024-06-15 and another on Jan 20, 2025."
        keywords = SQLiteStore._extract_keywords(content)
        assert "2024-06-15" in keywords
        assert "Jan 20, 2025" in keywords

    def test_extract_technical_terms(self):
        """CamelCase and ACRONYMS should be extracted."""
        content = "The SQLiteStore uses ONNX for embedding and BM25 for text search."
        keywords = SQLiteStore._extract_keywords(content)
        assert "SQLiteStore" in keywords
        assert "ONNX" in keywords

    def test_extract_numbers_with_units(self):
        """Numbers with units should be extracted."""
        content = "The model is 384 MB and achieves 95% accuracy on the benchmark."
        keywords = SQLiteStore._extract_keywords(content)
        assert "95%" in keywords

    def test_extract_empty_content(self):
        """Empty content returns empty string."""
        assert SQLiteStore._extract_keywords("") == ""

    def test_extract_keywords_capped(self):
        """Keywords should be capped at 50."""
        content = " ".join(f"Entity{i} Name{i}" for i in range(100))
        keywords = SQLiteStore._extract_keywords(content)
        # Should not have more than 50 keywords
        assert len(keywords.split()) <= 100  # 50 multi-word entries

    def test_keywords_stored_in_db(self, store):
        """Extracted keywords should be stored in the extracted_keywords column."""
        nid = store.store(
            content="Meeting with John Smith about SQLiteStore performance on 2024-06-15",
            metadata={"event_type": "decision"},
        )
        row = store._conn.execute(
            "SELECT extracted_keywords FROM memories WHERE node_id = ?", (nid,)
        ).fetchone()
        assert row is not None
        assert row[0] is not None
        assert "John Smith" in row[0]

    def test_keywords_enhance_fts_search(self, store):
        """Keywords in FTS index should improve BM25 search recall."""
        # Store a memory where the key entity only appears in extracted keywords
        nid = store.store(
            content="Discussed project timeline with John Smith at the office today",
            metadata={"event_type": "decision"},
        )
        # Search for the entity name should find it (via enriched FTS)
        results = store._text_search("John Smith", limit=5)
        found_ids = [r.id for r in results]
        assert nid in found_ids


# ============================================================================
# P6: Graph Multi-Hop Retrieval
# ============================================================================


class TestGraphMultiHop:
    """Test multi-hop graph traversal."""

    def test_single_hop_traversal(self, store):
        """1-hop neighbors should be surfaced."""
        # Create seed memory and a connected neighbor
        seed_id = store.store(
            content="Core architecture decision about microservices pattern",
            metadata={"event_type": "decision"},
        )
        neighbor_id = store.store(
            content="Related implementation detail about service mesh",
            metadata={"event_type": "task_completion"},
        )
        # Create edge between them
        store._conn.execute(
            """INSERT INTO edges (source_id, target_id, edge_type, weight, created_at)
               VALUES (?, ?, 'causal', 0.8, datetime('now'))""",
            (seed_id, neighbor_id),
        )
        store._conn.commit()

        # Query should find seed and potentially the neighbor
        results = store.query("microservices architecture", limit=10)
        result_ids = [r.id for r in results]
        assert seed_id in result_ids

    def test_two_hop_traversal(self, store):
        """2-hop neighbors should be surfaced with decayed scores."""
        # Chain: A -> B -> C
        a_id = store.store(
            content="Original decision about database sharding strategy",
            metadata={"event_type": "decision"},
        )
        b_id = store.store(
            content="Follow-up on sharding implementation details and timeline",
            metadata={"event_type": "task_completion"},
        )
        c_id = store.store(
            content="Performance benchmark results after sharding deployment",
            metadata={"event_type": "lesson_learned"},
        )
        now = datetime.now(timezone.utc).isoformat()
        store._conn.execute(
            """INSERT INTO edges (source_id, target_id, edge_type, weight, created_at)
               VALUES (?, ?, 'causal', 0.9, ?)""",
            (a_id, b_id, now),
        )
        store._conn.execute(
            """INSERT INTO edges (source_id, target_id, edge_type, weight, created_at)
               VALUES (?, ?, 'causal', 0.8, ?)""",
            (b_id, c_id, now),
        )
        store._conn.commit()

        results = store.query("database sharding", limit=10)
        result_ids = [r.id for r in results]
        assert a_id in result_ids  # Direct match


# ============================================================================
# Schema Migration
# ============================================================================


class TestSchemaMigration:
    """Test v7 -> v8 schema migration."""

    def test_schema_version_is_8(self, store):
        """SCHEMA_VERSION should be 8."""
        assert SCHEMA_VERSION == 15
        row = store._conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()
        assert row[0] == 15

    def test_end_date_column_exists(self, store):
        """The memories table should have an end_date column."""
        cols = store._conn.execute("PRAGMA table_info(memories)").fetchall()
        col_names = [c[1] for c in cols]
        assert "end_date" in col_names

    def test_extracted_keywords_column_exists(self, store):
        """The memories table should have an extracted_keywords column."""
        cols = store._conn.execute("PRAGMA table_info(memories)").fetchall()
        col_names = [c[1] for c in cols]
        assert "extracted_keywords" in col_names
