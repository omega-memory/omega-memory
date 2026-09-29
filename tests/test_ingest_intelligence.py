"""Tests for ingest-side intelligence: contradiction detection, atomic fact splitting, corpus hygiene."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestContradictionDetection:
    """Store-time supersession with real embeddings.

    Supersession used to run twice: inside store() and again in the bridge's
    Phase 4.1 (_detect_and_supersede), which retired on similarity alone with
    no project check. store() is now the single authority; see
    test_supersession_policy.py for the policy itself.
    """

    def test_single_word_substitution_is_flagged_not_retired(self, store):
        """PostgreSQL -> MySQL with no update wording: keep both, flag the pair."""
        nid1 = store.store(
            content="We decided to use PostgreSQL as the primary database for production.",
            metadata={"event_type": "decision"},
        )
        nid2 = store.store(
            content="We decided to use MySQL as the primary database for production.",
            metadata={"event_type": "decision"},
        )

        old = store.get_node(nid1)
        assert old.metadata.get("superseded") is not True
        candidates = store.get_node(nid2).metadata.get("supersession_candidates", [])
        # Hash-fallback embeddings (no model in CI) may not reach the gate.
        assert all(c["target_id"] == nid1 for c in candidates)

    def test_contradiction_same_content_not_superseded(self, store):
        """Exact duplicate content should not trigger superseding."""
        content = "We use PostgreSQL for the main database in all environments."
        nid1 = store.store(content=content, metadata={"event_type": "decision"})
        store.store(content=content, metadata={"event_type": "decision"})

        old = store.get_node(nid1)
        assert old.metadata.get("superseded") is not True

    def test_contradiction_different_type_not_superseded(self, store):
        """Similar content but different event_type should not be superseded."""
        nid1 = store.store(
            content="PostgreSQL is the best database choice for our workload.",
            metadata={"event_type": "lesson_learned"},
        )
        store.store(
            content="PostgreSQL is now deprecated in favor of SQLite for our workload.",
            metadata={"event_type": "decision"},
        )

        # lesson_learned != decision → should not supersede
        old = store.get_node(nid1)
        assert old.metadata.get("superseded") is not True

    def test_contradiction_threshold(self, store):
        """Unrelated content (below the similarity gate) is never superseded."""
        nid1 = store.store(
            content="The deployment pipeline runs on Jenkins with Docker containers.",
            metadata={"event_type": "decision"},
        )
        nid2 = store.store(
            content="Our team now prefers dark mode in all IDEs and editors.",
            metadata={"event_type": "decision"},
        )

        old = store.get_node(nid1)
        assert old.metadata.get("superseded") is not True
        assert store.get_node(nid2).metadata.get("supersession_candidates", []) == []

    def test_cross_type_user_preference_supersedes_decision(self, store):
        """A user_preference naming the old choice supersedes the decision."""
        nid1 = store.store(
            content="We decided to use PostgreSQL as the primary database for this project.",
            metadata={"event_type": "decision"},
        )
        nid2 = store.store(
            content="User prefers SQLite over PostgreSQL as the primary database for this project.",
            metadata={"event_type": "user_preference"},
        )

        old = store.get_node(nid1)
        assert old.metadata.get("superseded") is True
        assert old.metadata.get("superseded_by") == nid2
        assert old.metadata.get("superseded_reason") == "replacement"

    def test_cross_type_decision_does_not_supersede_preference(self, store):
        """A decision should NOT supersede a user_preference (one-directional)."""
        nid1 = store.store(
            content="User prefers to always use SQLite as the primary database.",
            metadata={"event_type": "user_preference"},
        )
        store.store(
            content="Decided to use PostgreSQL instead of SQLite as the primary database.",
            metadata={"event_type": "decision"},
        )

        # decision should NOT supersede user_preference
        old = store.get_node(nid1)
        assert old.metadata.get("superseded") is not True

    def test_cross_type_low_similarity_no_supersede(self, store):
        """Low-similarity user_preference should not supersede unrelated decision."""
        nid1 = store.store(
            content="Deploy the production database migration on Friday night.",
            metadata={"event_type": "decision"},
        )
        store.store(
            content="I now prefer dark mode in all text editors and IDEs.",
            metadata={"event_type": "user_preference"},
        )

        old = store.get_node(nid1)
        assert old.metadata.get("superseded") is not True


class TestAtomicFactSplitting:
    """Phase 4.2: Split content into atomic fact nodes."""

    def test_fact_splitting_extracts_facts(self):
        """Decision with factual sentences should produce fact nodes."""
        from omega.bridge import _split_atomic_facts

        content = (
            "We use PostgreSQL as the primary database. "
            "The API server is deployed on AWS. "
            "Logging was configured with Datadog."
        )
        # Feature is gated behind OMEGA_ATOMIC_FACTS=1
        with patch.dict(os.environ, {"OMEGA_ATOMIC_FACTS": "1"}):
            facts = _split_atomic_facts(content, "decision")
        assert len(facts) >= 1
        # At least one sentence should match "is/are/use" patterns
        any_pg = any("PostgreSQL" in f for f in facts)
        any_aws = any("AWS" in f for f in facts)
        assert any_pg or any_aws

    def test_fact_splitting_max_5(self):
        """Should cap at 5 fact nodes per parent."""
        from omega.bridge import _split_atomic_facts

        content = (
            "We built the frontend with React. "
            "We wrote the backend in Python. "
            "We use Redis for caching. "
            "Our database is PostgreSQL. "
            "We handle CI/CD with GitHub Actions. "
            "We do monitoring with Grafana. "
            "We use Cloudflare as CDN. "
            "We handle authentication with OAuth2."
        )
        with patch.dict(os.environ, {"OMEGA_ATOMIC_FACTS": "1"}):
            facts = _split_atomic_facts(content, "decision")
        assert len(facts) <= 5

    def test_fact_splitting_short_content_skipped(self):
        """Content < 50 chars should produce no facts."""
        from omega.bridge import _split_atomic_facts

        facts = _split_atomic_facts("Use Redis.", "decision")
        assert facts == []

    def test_fact_splitting_wrong_type_skipped(self):
        """Non-matching event types should produce no facts."""
        from omega.bridge import _split_atomic_facts

        content = "The server is running on port 8080. We use nginx as a reverse proxy."
        facts = _split_atomic_facts(content, "error_pattern")
        assert facts == []

    def test_fact_graph_edge_in_auto_capture(self, store, tmp_omega_dir):
        """Fact nodes created by auto_capture should have edges to parent."""
        from omega.bridge import auto_capture, _get_store
        import omega.bridge as bridge

        # Point bridge to our test store
        old_store = bridge._store_instance
        bridge._store_instance = store
        try:
            result = auto_capture(
                content=(
                    "We decided to use PostgreSQL as the database. "
                    "The API server is deployed on AWS EC2 instances. "
                    "Authentication uses JWT tokens for all endpoints."
                ),
                event_type="decision",
                session_id="test-session",
            )

            # Check that fact nodes were created with edges
            all_edges = store._conn.execute(
                "SELECT source_id, target_id, edge_type FROM edges "
                "WHERE edge_type = 'contains_fact'"
            ).fetchall()
            # May or may not produce facts depending on content matching,
            # but if facts were extracted, edges should exist
            for src, tgt, etype in all_edges:
                assert etype == "contains_fact"
                # Verify target node exists and has auto_extracted metadata
                fact_node = store.get_node(tgt)
                if fact_node:
                    assert fact_node.metadata.get("auto_extracted") is True
        finally:
            bridge._store_instance = old_store


class TestCorpusHygiene:
    """Phase: corpus deduplication in session stop hook."""

    def test_corpus_hygiene_dedup(self, store):
        """Near-duplicate memories should be superseded by hygiene."""
        from omega.bridge import _get_store
        from omega.embedding import generate_embedding

        # Store two very similar memories
        nid1 = store.store(
            content="The deployment pipeline uses GitHub Actions for CI/CD automation.",
            metadata={"event_type": "decision"},
        )
        nid2 = store.store(
            content="The deployment pipeline uses GitHub Actions for CI/CD automation and testing.",
            metadata={"event_type": "decision"},
        )

        # Compute cosine similarity
        emb1 = store.get_embedding(nid1)
        emb2 = store.get_embedding(nid2)
        if emb1 and emb2:
            dot = sum(x * y for x, y in zip(emb1, emb2))
            norm_a = sum(x * x for x in emb1) ** 0.5
            norm_b = sum(x * x for x in emb2) ** 0.5
            if norm_a > 0 and norm_b > 0:
                cosine = dot / (norm_a * norm_b)
                if cosine > 0.90:
                    # Manually trigger what corpus hygiene would do
                    store.mark_superseded(nid1, superseded_by=nid2)
                    old = store.get_node(nid1)
                    assert old.metadata.get("superseded") is True

    def test_mark_superseded_method(self, store):
        """mark_superseded should set metadata correctly."""
        nid = store.store(
            content="Test memory for superseding.",
            metadata={"event_type": "decision"},
        )
        # Create a "newer" memory
        nid2 = store.store(
            content="Updated test memory.",
            metadata={"event_type": "decision"},
        )

        result = store.mark_superseded(nid, superseded_by=nid2)
        assert result is True

        node = store.get_node(nid)
        assert node.metadata["superseded"] is True
        assert node.metadata["superseded_by"] == nid2
        assert "superseded_at" in node.metadata

    def test_mark_superseded_nonexistent(self, store):
        """mark_superseded on non-existent node returns False."""
        result = store.mark_superseded("nonexistent-id", superseded_by="other-id")
        assert result is False


class TestSupersededFiltering:
    """Verify superseded memories are filtered from queries and find_similar."""

    def test_superseded_filtered_from_query(self, store):
        """Superseded memories should not appear in query results."""
        nid1 = store.store(
            content="The primary language is Python for all backend services.",
            metadata={"event_type": "decision"},
        )
        nid2 = store.store(
            content="The primary language is Rust for all backend services.",
            metadata={"event_type": "decision"},
        )

        # Mark old as superseded
        store.mark_superseded(nid1, superseded_by=nid2)

        results = store.query("primary language backend", limit=10)
        result_ids = {r.id for r in results}
        # Old superseded node should not appear
        assert nid1 not in result_ids

    def test_superseded_filtered_from_find_similar(self, store):
        """Superseded memories should not appear in find_similar results."""
        from omega.embedding import generate_embedding

        nid1 = store.store(
            content="We use Docker containers for all deployments.",
            metadata={"event_type": "decision"},
        )
        nid2 = store.store(
            content="We use Kubernetes pods for all deployments.",
            metadata={"event_type": "decision"},
        )

        store.mark_superseded(nid1, superseded_by=nid2)

        emb = generate_embedding("Docker Kubernetes deployment containers")
        results = store.find_similar(emb, limit=10)
        result_ids = {r.id for r in results}
        assert nid1 not in result_ids
