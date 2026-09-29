"""Tests for internal bridge.py helper functions.

Covers:
  - _extract_facts: regex-based fact extraction
  - _auto_relate: typed edge creation between similar memories
  - _split_atomic_facts: sentence-level fact splitting
"""
import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from omega.bridge import (
    _auto_relate,
    _extract_facts,
    _split_atomic_facts,
)
from omega.sqlite_store import MemoryResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_store():
    """A MagicMock store — no spec so mark_superseded etc. auto-create."""
    store = MagicMock()
    store.stats = {}
    return store


def _mr(
    node_id=None,
    content="some content",
    metadata=None,
    relevance=0.9,
    created_at=None,
):
    """Build a MemoryResult with sensible defaults."""
    return MemoryResult(
        id=node_id or f"mem-{uuid.uuid4().hex[:12]}",
        content=content,
        metadata=metadata or {},
        relevance=relevance,
        created_at=created_at or datetime.now(timezone.utc),
    )


# ===================================================================
# _extract_facts
# ===================================================================


class TestExtractFacts:
    def test_camelcase_extraction(self):
        facts = _extract_facts("We migrated to SQLiteStore for persistence")
        assert "sqlitestore" in facts

    def test_multiple_camelcase(self):
        facts = _extract_facts("MemoryResult and SQLiteStore are used together")
        assert "memoryresult" in facts
        assert "sqlitestore" in facts

    def test_upper_case_constants(self):
        facts = _extract_facts("Set DEBUG=true and MAX_NODES=5000")
        assert "debug" in facts
        assert "max_nodes" in facts

    def test_upper_case_minimum_length(self):
        facts = _extract_facts("Set AB to 1 and API_KEY to secret")
        assert "ab" not in facts
        assert "api_key" in facts

    def test_backtick_extraction(self):
        facts = _extract_facts("Run `npm install` to set up deps")
        assert "npm install" in facts

    def test_backtick_length_bounds(self):
        """Backtick tokens must be 2-40 chars."""
        facts = _extract_facts("Run `npm install` to set up deps")
        assert "npm install" in facts
        # Single char tokens inside backticks won't match the 2+ char regex
        facts2 = _extract_facts("Use `z` only")
        backtick_facts = [f for f in facts2 if f == "z"]
        assert len(backtick_facts) == 0

    def test_quoted_string_extraction(self):
        facts = _extract_facts('Use "refresh token" for auth')
        assert "refresh token" in facts

    def test_single_quoted_string(self):
        facts = _extract_facts("Prefer 'strict mode' always")
        assert "strict mode" in facts

    def test_decision_verb_chose(self):
        facts = _extract_facts("We chose PostgreSQL for the database.")
        assert any("postgresql" in f for f in facts)

    def test_decision_verb_switched_to(self):
        facts = _extract_facts("We switched to Redis for caching.")
        assert any("redis" in f for f in facts)

    def test_decision_verb_use(self):
        facts = _extract_facts("We use Python 3.11 for all services.")
        assert any("python" in f for f in facts)

    def test_dotted_path_extraction(self):
        facts = _extract_facts("Edit omega.sqlite_store to fix the bug")
        assert "omega.sqlite_store" in facts

    def test_dotted_path_skips_version_numbers(self):
        facts = _extract_facts("Upgrade to version 1.0.0 now")
        assert "1.0.0" not in facts

    def test_hyphenated_compound_terms(self):
        facts = _extract_facts("Enable multi-session support for cross-agent collaboration")
        assert "multi-session" in facts
        assert "cross-agent" in facts

    def test_hyphenated_too_short(self):
        facts = _extract_facts("Use a-b pair")
        assert "a-b" not in facts

    def test_empty_string_returns_empty(self):
        assert _extract_facts("") == []

    def test_deduplication(self):
        facts = _extract_facts("Use SQLiteStore. Then use SQLiteStore again.")
        count = sum(1 for f in facts if f == "sqlitestore")
        assert count <= 1

    def test_stopword_filtering(self):
        _STOP = {"the", "and", "for", "with", "that", "this", "from", "have", "been", "will", "not"}
        facts = _extract_facts("We chose the database for this project with TypeScript")
        for f in facts:
            words = f.split()
            meaningful = [w for w in words if w not in _STOP and len(w) > 1]
            assert len(meaningful) > 0, f"Fact '{f}' should have been filtered"

    def test_cap_at_20(self):
        terms = [f"Module{chr(65+i)}Class{chr(65+i)}" for i in range(25)]
        content = " ".join(terms)
        facts = _extract_facts(content)
        assert len(facts) <= 20

    def test_results_are_sorted(self):
        facts = _extract_facts("Use SQLiteStore and MemoryResult and AutoRelate")
        assert facts == sorted(facts)

    def test_mixed_extraction(self):
        content = "We chose `SQLiteStore` over MAX_NODES. Edit omega.bridge for multi-session support."
        facts = _extract_facts(content)
        assert "sqlitestore" in facts
        assert "max_nodes" in facts
        assert "omega.bridge" in facts
        assert "multi-session" in facts


# ===================================================================
# _auto_relate
# ===================================================================


class TestAutoRelate:
    def test_returns_zero_no_embedding(self, mock_store):
        mock_store.get_embedding.return_value = None
        assert _auto_relate(mock_store, "mem-abc") == 0

    def test_returns_zero_no_similar(self, mock_store):
        mock_store.get_embedding.return_value = [0.1, 0.2, 0.3]
        mock_store.find_similar.return_value = []
        assert _auto_relate(mock_store, "mem-abc") == 0

    def test_returns_zero_only_self(self, mock_store):
        mock_store.get_embedding.return_value = [0.1, 0.2, 0.3]
        self_result = _mr(node_id="mem-abc", relevance=1.0)
        mock_store.find_similar.return_value = [self_result]
        assert _auto_relate(mock_store, "mem-abc") == 0

    def test_returns_zero_below_min_similarity(self, mock_store):
        mock_store.get_embedding.return_value = [0.1, 0.2, 0.3]
        low_sim = _mr(node_id="mem-other", relevance=0.50)
        mock_store.find_similar.return_value = [low_sim]
        assert _auto_relate(mock_store, "mem-abc", min_similarity=0.65) == 0

    def test_same_entity_edge(self, mock_store):
        source_id, target_id = "mem-src", "mem-tgt"
        mock_store.get_embedding.return_value = [0.1, 0.2]
        target = _mr(node_id=target_id, relevance=0.90, metadata={"entity_id": "proj-alpha", "event_type": "decision"})
        mock_store.find_similar.return_value = [target]
        mock_store.get_node.return_value = _mr(node_id=source_id, metadata={"entity_id": "proj-alpha", "event_type": "lesson_learned"})
        mock_store.add_edge.return_value = True

        count = _auto_relate(mock_store, source_id)
        assert count == 1
        mock_store.add_edge.assert_called_once_with(source_id, target_id, "same_entity", 0.90)

    def test_evolution_edge(self, mock_store):
        source_id, target_id = "mem-src", "mem-tgt"
        mock_store.get_embedding.return_value = [0.1, 0.2]
        target = _mr(node_id=target_id, relevance=0.78, metadata={"event_type": "decision"})
        mock_store.find_similar.return_value = [target]
        mock_store.get_node.return_value = _mr(node_id=source_id, metadata={"event_type": "decision"})
        mock_store.add_edge.return_value = True

        count = _auto_relate(mock_store, source_id)
        assert count == 1
        mock_store.add_edge.assert_called_once_with(source_id, target_id, "evolution", 0.78)

    def test_evolution_requires_075(self, mock_store):
        source_id, target_id = "mem-src", "mem-tgt"
        now = datetime.now(timezone.utc)
        mock_store.get_embedding.return_value = [0.1, 0.2]
        target = _mr(node_id=target_id, relevance=0.70, metadata={"event_type": "decision"}, created_at=now - timedelta(hours=10))
        mock_store.find_similar.return_value = [target]
        mock_store.get_node.return_value = _mr(node_id=source_id, metadata={"event_type": "decision"}, created_at=now)
        assert _auto_relate(mock_store, source_id) == 0

    def test_temporal_cluster_edge(self, mock_store):
        source_id, target_id = "mem-src", "mem-tgt"
        now = datetime.now(timezone.utc)
        mock_store.get_embedding.return_value = [0.1, 0.2]
        target = _mr(node_id=target_id, relevance=0.70, metadata={"event_type": "observation"}, created_at=now - timedelta(minutes=30))
        mock_store.find_similar.return_value = [target]
        mock_store.get_node.return_value = _mr(node_id=source_id, metadata={"event_type": "lesson_learned"}, created_at=now)
        mock_store.add_edge.return_value = True

        count = _auto_relate(mock_store, source_id)
        assert count == 1
        mock_store.add_edge.assert_called_once_with(source_id, target_id, "temporal_cluster", 0.70)

    def test_related_edge_high_similarity(self, mock_store):
        source_id, target_id = "mem-src", "mem-tgt"
        now = datetime.now(timezone.utc)
        mock_store.get_embedding.return_value = [0.1, 0.2]
        target = _mr(node_id=target_id, relevance=0.85, metadata={"event_type": "observation"}, created_at=now - timedelta(hours=5))
        mock_store.find_similar.return_value = [target]
        mock_store.get_node.return_value = _mr(node_id=source_id, metadata={"event_type": "lesson_learned"}, created_at=now)
        mock_store.add_edge.return_value = True

        count = _auto_relate(mock_store, source_id)
        assert count == 1
        mock_store.add_edge.assert_called_once_with(source_id, target_id, "related", 0.85)

    def test_skips_below_080_no_signal(self, mock_store):
        source_id, target_id = "mem-src", "mem-tgt"
        now = datetime.now(timezone.utc)
        mock_store.get_embedding.return_value = [0.1, 0.2]
        target = _mr(node_id=target_id, relevance=0.70, metadata={"event_type": "observation"}, created_at=now - timedelta(hours=5))
        mock_store.find_similar.return_value = [target]
        mock_store.get_node.return_value = _mr(node_id=source_id, metadata={"event_type": "lesson_learned"}, created_at=now)
        assert _auto_relate(mock_store, source_id) == 0

    def test_max_related_caps_edges(self, mock_store):
        source_id = "mem-src"
        now = datetime.now(timezone.utc)
        mock_store.get_embedding.return_value = [0.1, 0.2]
        targets = [
            _mr(node_id=f"mem-t{i}", relevance=0.95 - (i * 0.01),
                metadata={"entity_id": "proj-alpha"}, created_at=now - timedelta(minutes=i))
            for i in range(5)
        ]
        mock_store.find_similar.return_value = targets
        mock_store.get_node.return_value = _mr(node_id=source_id, metadata={"entity_id": "proj-alpha"}, created_at=now)
        mock_store.add_edge.return_value = True

        count = _auto_relate(mock_store, source_id, max_related=2)
        assert count == 2
        assert mock_store.add_edge.call_count == 2

    def test_graceful_on_exception(self, mock_store):
        mock_store.get_embedding.side_effect = RuntimeError("DB error")
        assert _auto_relate(mock_store, "mem-abc") == 0

    def test_returns_zero_get_node_none(self, mock_store):
        mock_store.get_embedding.return_value = [0.1, 0.2]
        mock_store.find_similar.return_value = [_mr(node_id="mem-tgt", relevance=0.90)]
        mock_store.get_node.return_value = None
        assert _auto_relate(mock_store, "mem-src") == 0

    def test_same_entity_over_evolution(self, mock_store):
        """same_entity wins over evolution when both conditions match."""
        source_id, target_id = "mem-src", "mem-tgt"
        mock_store.get_embedding.return_value = [0.1, 0.2]
        target = _mr(node_id=target_id, relevance=0.90, metadata={"entity_id": "proj-alpha", "event_type": "decision"})
        mock_store.find_similar.return_value = [target]
        mock_store.get_node.return_value = _mr(node_id=source_id, metadata={"entity_id": "proj-alpha", "event_type": "decision"})
        mock_store.add_edge.return_value = True

        _auto_relate(mock_store, source_id)
        mock_store.add_edge.assert_called_once_with(source_id, target_id, "same_entity", 0.90)

    def test_add_edge_failure_not_counted(self, mock_store):
        source_id, target_id = "mem-src", "mem-tgt"
        mock_store.get_embedding.return_value = [0.1, 0.2]
        target = _mr(node_id=target_id, relevance=0.90, metadata={"entity_id": "proj-alpha"})
        mock_store.find_similar.return_value = [target]
        mock_store.get_node.return_value = _mr(node_id=source_id, metadata={"entity_id": "proj-alpha"})
        mock_store.add_edge.return_value = False

        assert _auto_relate(mock_store, source_id) == 0


# ===================================================================
# _split_atomic_facts
# ===================================================================


class TestSplitAtomicFacts:
    def setup_method(self):
        os.environ["OMEGA_ATOMIC_FACTS"] = "1"

    def teardown_method(self):
        os.environ.pop("OMEGA_ATOMIC_FACTS", None)

    def test_non_applicable_event_type(self):
        assert _split_atomic_facts("We use Python. Our server is fast.", "observation") == []
        assert _split_atomic_facts("We use Python. Our server is fast.", "lesson_learned") == []
        assert _split_atomic_facts("We use Python. Our server is fast.", "session_summary") == []

    def test_short_content_returns_empty(self):
        assert _split_atomic_facts("We use Python.", "decision") == []
        assert _split_atomic_facts("Short.", "user_fact") == []

    def test_splits_multi_sentence_decision(self):
        content = (
            "We use PostgreSQL for the database. "
            "Our API is hosted on AWS. "
            "We adopted Redis for caching."
        )
        facts = _split_atomic_facts(content, "decision")
        assert len(facts) >= 1
        for f in facts:
            assert len(f) >= 15

    def test_user_signal_required(self):
        content = (
            "PostgreSQL is a relational database management system. "
            "Redis stores data in memory for fast access. "
            "SQLite operates without a separate server process."
        )
        assert _split_atomic_facts(content, "decision") == []

    def test_sentence_too_short_filtered(self):
        content = (
            "We use it. "
            "Our PostgreSQL database is running on port 5432 with SSL enabled."
        )
        facts = _split_atomic_facts(content, "user_fact")
        for f in facts:
            assert len(f) >= 15

    def test_sentence_too_long_filtered(self):
        long_sentence = "We use " + "a" * 200 + " for our database."
        short_sentence = "Our API is hosted on AWS."
        content = long_sentence + " " + short_sentence
        facts = _split_atomic_facts(content, "decision")
        for f in facts:
            assert len(f) <= 200

    def test_is_verb_pattern(self):
        content = (
            "Our database is PostgreSQL running on port 5432. "
            "My API keys are stored in the vault securely."
        )
        facts = _split_atomic_facts(content, "user_fact")
        assert len(facts) >= 1

    def test_use_verb_pattern(self):
        content = (
            "We use TypeScript for the frontend application. "
            "We adopted Tailwind CSS for styling components."
        )
        facts = _split_atomic_facts(content, "decision")
        assert len(facts) >= 1

    def test_location_pattern(self):
        content = (
            "I am based in Singapore for work. "
            "Our servers are located in us-east-1 region."
        )
        facts = _split_atomic_facts(content, "user_fact")
        assert len(facts) >= 1

    def test_deduplication(self):
        content = (
            "We use Python for scripting tasks. "
            "We use Python for scripting tasks. "
            "Our database is PostgreSQL on port 5432."
        )
        facts = _split_atomic_facts(content, "decision")
        normalized = [f.strip().lower() for f in facts]
        assert len(normalized) == len(set(normalized))

    def test_max_five_facts(self):
        sentences = [f"We use tool{i} for our project infrastructure setup." for i in range(10)]
        content = " ".join(sentences)
        facts = _split_atomic_facts(content, "decision")
        assert len(facts) <= 5

    def test_decision_type_works(self):
        content = "We use React for our frontend. Our backend is built with FastAPI."
        assert len(_split_atomic_facts(content, "decision")) >= 1

    def test_user_fact_type_works(self):
        content = "My name is Alex and I live in Berlin. Our company was founded in 2020."
        assert len(_split_atomic_facts(content, "user_fact")) >= 1

    def test_mixed_qualifying_and_non(self):
        content = (
            "Python is a programming language. "
            "We use Python for all our backend services. "
            "The weather is nice today."
        )
        facts = _split_atomic_facts(content, "decision")
        assert len(facts) >= 1
        assert any("Python" in f for f in facts)

    def test_db_keyword_triggers_user_signal(self):
        content = (
            "The database is running PostgreSQL version 15. "
            "The server is configured with 16GB RAM total."
        )
        facts = _split_atomic_facts(content, "user_fact")
        assert len(facts) >= 1
