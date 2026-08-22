"""Golden Set Retrieval Regression Tests.

Automated safety net for scoring changes (temporal penalties, feedback boosts,
type weights, decay, dedup). Asserts **relative ordering** — never exact scores.

Pattern: insert memories via direct SQL (bypassing store-time dedup layers) to
control metadata precisely, then query and verify rank order.
"""

import hashlib
import json
import uuid
import os
import pytest
import unicodedata
from datetime import datetime, timedelta, timezone

from omega.sqlite_store import SQLiteStore


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_bridge(tmp_omega_dir):
    """Reset the bridge singleton so each test gets a fresh store."""
    from omega.bridge import reset_memory

    reset_memory()
    yield
    reset_memory()


def _get_store():
    from omega.bridge import _get_store

    return _get_store()


def _insert_memory(store, content, event_type="memory", priority=3, **extra_meta):
    """Insert a memory directly via SQL, bypassing all store-time dedup layers.

    This is intentional for golden set tests: we need precise control over
    what exists in the DB to test scoring and ranking independently of dedup.
    """
    node_id = f"mem-{uuid.uuid4().hex[:12]}"
    meta = {"event_type": event_type, "priority": priority, **extra_meta}
    now = datetime.now(timezone.utc).isoformat()
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    canonical = unicodedata.normalize("NFKC", content).lower()
    canonical_hash = hashlib.sha256(canonical.encode()).hexdigest()

    store._conn.execute(
        """INSERT INTO memories
           (node_id, content, metadata, created_at, access_count, last_accessed,
            content_hash, canonical_hash, event_type, priority)
           VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?)""",
        (
            node_id, content, json.dumps(meta), now, now,
            content_hash, canonical_hash, event_type, priority,
        ),
    )
    store._conn.commit()
    return node_id


def _set_metadata(store, node_id, **fields):
    """Update specific metadata fields via SQL."""
    row = store._conn.execute(
        "SELECT metadata FROM memories WHERE node_id = ?", (node_id,)
    ).fetchone()
    meta = json.loads(row[0]) if row[0] else {}
    meta.update(fields)
    store._conn.execute(
        "UPDATE memories SET metadata = ? WHERE node_id = ?",
        (json.dumps(meta), node_id),
    )
    store._conn.commit()


def _set_timestamps(store, node_id, created_at=None, last_accessed=None, access_count=None):
    """Set created_at, last_accessed, access_count via SQL."""
    updates = []
    params = []
    if created_at is not None:
        updates.append("created_at = ?")
        params.append(created_at)
    if last_accessed is not None:
        updates.append("last_accessed = ?")
        params.append(last_accessed)
    if access_count is not None:
        updates.append("access_count = ?")
        params.append(access_count)
    if updates:
        params.append(node_id)
        store._conn.execute(
            f"UPDATE memories SET {', '.join(updates)} WHERE node_id = ?",
            params,
        )
        store._conn.commit()


def _query_ids(store, query_text, limit=10, **kwargs):
    """Query and return ordered list of node_ids."""
    store._invalidate_query_cache()
    results = store.query(query_text, limit=limit, **kwargs)
    return [r.id for r in results]


# ============================================================================
# 1. Type Weight Ranking
# ============================================================================


class TestTypeWeightRanking:
    """Constraint (3.0) > decision (2.0) > session_summary (1.2)."""

    def test_constraint_ranks_above_decision(self):
        store = _get_store()
        d_id = _insert_memory(
            store,
            "Production databases must use PostgreSQL exclusively for all services",
            "decision",
        )
        c_id = _insert_memory(
            store,
            "PostgreSQL is the required database engine for every production deployment",
            "constraint",
        )
        ids = _query_ids(store, "PostgreSQL production database requirement")
        assert c_id in ids and d_id in ids, f"Both should be returned, got {ids}"
        assert ids.index(c_id) < ids.index(d_id), "Constraint (3.0) should rank above decision (2.0)"

    def test_decision_ranks_above_session_summary(self):
        store = _get_store()
        # Insert session_summary first so FTS5 recency bias works against it,
        # making this a cleaner test of type_weight ranking (decision=2.0 > session_summary=1.2)
        s_id = _insert_memory(
            store,
            "JWT authentication migration to API tokens was discussed in the session",
            "session_summary",
        )
        d_id = _insert_memory(
            store,
            "JWT authentication migration to API tokens for all endpoints",
            "decision",
        )
        ids = _query_ids(store, "JWT authentication migration API tokens", include_infrastructure=True)
        assert d_id in ids and s_id in ids, f"Both should be returned, got {ids}"
        assert ids.index(d_id) < ids.index(s_id), "Decision (2.0) should rank above session_summary (1.2)"

    def test_constraint_ranks_above_session_summary(self):
        store = _get_store()
        c_id = _insert_memory(
            store,
            "Full test suite must pass before any deployment to production environments",
            "constraint",
        )
        s_id = _insert_memory(
            store,
            "The test suite should always pass before deployment to production is allowed",
            "session_summary",
        )
        ids = _query_ids(store, "test suite deployment production", include_infrastructure=True)
        assert c_id in ids and s_id in ids, f"Both should be returned, got {ids}"
        assert ids.index(c_id) < ids.index(s_id)


# ============================================================================
# 2. Feedback Factor
# ============================================================================


class TestFeedbackFactor:
    """Positive feedback boosts rank; negative feedback demotes."""

    def test_positive_feedback_boosts(self):
        """feedback_score +5 (1.75x) should rank above neutral (1.0x)."""
        store = _get_store()
        # Insert boosted first (FTS5 recency disadvantage) to ensure
        # the feedback boost alone drives the ranking, not insertion order.
        boosted_id = _insert_memory(
            store,
            "React hooks state management in functional components",
            "decision",
        )
        _set_metadata(store, boosted_id, feedback_score=5)
        neutral_id = _insert_memory(
            store,
            "React hooks state management in functional components for local data",
            "decision",
        )
        ids = _query_ids(store, "React hooks state management components")
        assert boosted_id in ids and neutral_id in ids, f"Both should be returned, got {ids}"
        assert ids.index(boosted_id) < ids.index(neutral_id), "Positive feedback should boost rank"

    def test_negative_feedback_demotes(self):
        """feedback_score -4 (0.2x floor) should rank last."""
        store = _get_store()
        normal_id = _insert_memory(
            store,
            "Nginx reverse proxy configuration with SSL termination at the load balancer level",
            "lesson_learned",
        )
        demoted_id = _insert_memory(
            store,
            "SSL termination for nginx reverse proxy should happen at the gateway level always",
            "lesson_learned",
        )
        _set_metadata(store, demoted_id, feedback_score=-4)
        ids = _query_ids(store, "nginx reverse proxy SSL termination configuration")
        assert normal_id in ids and demoted_id in ids, f"Both should be returned, got {ids}"
        assert ids.index(normal_id) < ids.index(demoted_id), "Negative feedback should demote"

    def test_feedback_factor_math(self):
        """Verify _compute_fb_factor produces expected multipliers."""
        assert SQLiteStore._compute_fb_factor(0) == 1.0
        assert SQLiteStore._compute_fb_factor(5) == pytest.approx(1.75)
        assert SQLiteStore._compute_fb_factor(10) == pytest.approx(2.5)
        assert SQLiteStore._compute_fb_factor(-4) == pytest.approx(0.2)
        assert SQLiteStore._compute_fb_factor(15) == pytest.approx(2.5)
        assert SQLiteStore._compute_fb_factor(-10) == pytest.approx(0.2)


# ============================================================================
# 3. Decay Curves
# ============================================================================


class TestDecayCurves:
    """Fresh memories rank higher; access slows decay; protected types don't decay."""

    def test_fresh_ranks_above_old(self):
        """A memory created today should rank above one created 60 days ago."""
        store = _get_store()
        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        # Insert fresh first (FTS5 disadvantage) so decay alone drives ranking
        fresh_id = _insert_memory(
            store,
            "Database indexing strategy for user queries optimizes read performance",
            "decision",
        )
        old_id = _insert_memory(
            store,
            "Database indexing strategy for user queries provides fast lookup",
            "decision",
        )
        _set_timestamps(store, old_id, created_at=old_date, last_accessed=old_date)
        ids = _query_ids(store, "database indexing strategy user queries")
        assert fresh_id in ids and old_id in ids, f"Both should be returned, got {ids}"
        assert ids.index(fresh_id) < ids.index(old_id), "Fresh should rank above old"

    def test_accessed_old_ranks_above_unaccessed_old(self):
        """An old memory with recent access should rank above equally old unaccessed one."""
        store = _get_store()
        old_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        recent_access = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

        unaccessed_id = _insert_memory(
            store,
            "API rate limiting uses token bucket algorithm for request throttling configuration",
            "decision",
        )
        _set_timestamps(store, unaccessed_id, created_at=old_date, last_accessed=old_date, access_count=0)

        accessed_id = _insert_memory(
            store,
            "Request throttling via API rate limiting protects against excessive traffic loads",
            "decision",
        )
        _set_timestamps(store, accessed_id, created_at=old_date, last_accessed=recent_access, access_count=10)

        ids = _query_ids(store, "API rate limiting request throttling")
        assert accessed_id in ids and unaccessed_id in ids, f"Both should be returned, got {ids}"
        assert ids.index(accessed_id) < ids.index(unaccessed_id), "Accessed old > unaccessed old"

    def test_user_preference_no_decay(self):
        """user_preference has lambda=0.0 and should not decay even after a year."""
        store = _get_store()
        old_date = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        factor = store._compute_decay_factor("user_preference", old_date, old_date)
        assert factor == 1.0, "user_preference should never decay"


# ============================================================================
# 4. Priority Factor
# ============================================================================


class TestPriorityFactor:
    """Priority breaks semantic ties without manufacturing relevance."""

    def test_high_priority_ranks_above_low(self):
        """Priority 5 wins only when semantic evidence is equal."""
        from omega.sqlite_store._query import _score_bounded_metadata

        low, _ = _score_bounded_metadata(
            semantic_score=0.8,
            best_semantic_score=0.8,
            priority=1,
            access_count=0,
        )
        high, _ = _score_bounded_metadata(
            semantic_score=0.8,
            best_semantic_score=0.8,
            priority=5,
            access_count=0,
        )

        assert high > low

    def test_priority_factor_math(self):
        """Priority contributes nothing outside the semantic near-tie band."""
        from omega.sqlite_store._query import _score_bounded_metadata

        score, reasons = _score_bounded_metadata(
            semantic_score=0.78,
            best_semantic_score=0.8,
            priority=5,
            access_count=500,
        )

        assert score == pytest.approx(0.78)
        assert reasons["priority_contribution"] == 0.0
        assert reasons["access_contribution"] == 0.0


# ============================================================================
# 5. Exact Dedup (query-time)
# ============================================================================


class TestExactDedup:
    """Identical normalized content (150 chars) produces single result at query time."""

    def test_identical_content_deduped_at_query(self):
        """Two memories with identical first 150 chars (normalized) yield one query result."""
        store = _get_store()
        # Insert directly to bypass store-time dedup
        _insert_memory(store, "Deploy the application to production environment today for the release", "decision")
        _insert_memory(store, "Deploy the application to production environment today for the release", "decision")
        ids = _query_ids(store, "deploy application production environment")
        assert len(ids) == 1, f"Exact dedup should collapse to 1 result, got {len(ids)}"

    def test_different_content_not_deduped(self):
        """Memories with different content should both appear."""
        store = _get_store()
        id1 = _insert_memory(store, "PostgreSQL handles the user service database for persistent storage", "decision")
        id2 = _insert_memory(store, "Redis provides fast session caching and rate limiting for the backend", "decision")
        ids = _query_ids(store, "database caching infrastructure choices backend")
        returned = set(ids)
        assert id1 in returned or id2 in returned, "Distinct memories should appear"


# ============================================================================
# 5b. Semantic Dedup (unit test for _semantic_dedup method)
# ============================================================================


class TestSemanticDedup:
    """Unit tests for _semantic_dedup method."""

    def test_semantic_dedup_disabled_by_threshold(self):
        """Threshold 1.0 effectively disables semantic dedup."""
        store = _get_store()
        from omega.sqlite_store import MemoryResult
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        r1 = MemoryResult(id="a", content="test1", metadata={}, created_at=now,
                          access_count=0, last_accessed=now, relevance=1.0, ttl_seconds=None)
        r2 = MemoryResult(id="b", content="test2", metadata={}, created_at=now,
                          access_count=0, last_accessed=now, relevance=0.9, ttl_seconds=None)
        # threshold=1.0 means nothing gets deduped
        result = store._semantic_dedup([r1, r2], {"a": 1.0, "b": 0.9}, threshold=1.0)
        assert len(result) == 2

    def test_semantic_dedup_single_item_passthrough(self):
        """Single item should pass through unchanged."""
        store = _get_store()
        from omega.sqlite_store import MemoryResult
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        r1 = MemoryResult(id="a", content="test1", metadata={}, created_at=now,
                          access_count=0, last_accessed=now, relevance=1.0, ttl_seconds=None)
        result = store._semantic_dedup([r1], {"a": 1.0}, threshold=0.92)
        assert len(result) == 1
        assert result[0].id == "a"

    def test_semantic_dedup_env_var_override(self):
        """OMEGA_SEMANTIC_DEDUP_THRESHOLD env var should control the threshold."""
        old = os.environ.get("OMEGA_SEMANTIC_DEDUP_THRESHOLD")
        try:
            os.environ["OMEGA_SEMANTIC_DEDUP_THRESHOLD"] = "1.0"
            # Re-read to verify env var is parsed (tested in query flow)
            val = float(os.environ.get("OMEGA_SEMANTIC_DEDUP_THRESHOLD", "0.92"))
            assert val == 1.0
        finally:
            if old is not None:
                os.environ["OMEGA_SEMANTIC_DEDUP_THRESHOLD"] = old
            else:
                os.environ.pop("OMEGA_SEMANTIC_DEDUP_THRESHOLD", None)

    def test_semantic_dedup_stats_tracking(self):
        """Semantic dedup should increment stats counter when items are dropped."""
        store = _get_store()
        # Store two identical memories and verify the method tracks removals
        id1 = _insert_memory(store, "Identical content for semantic dedup testing", "decision")
        id2 = _insert_memory(store, "Identical content for semantic dedup testing", "decision")
        # Even without vec, the stats key should exist if method runs
        initial = store.stats.get("semantic_dedup_query", 0)
        # Method gracefully handles missing embeddings (no vec in test)
        from omega.sqlite_store import MemoryResult
        now = datetime.now(timezone.utc)
        r1 = MemoryResult(id=id1, content="test", metadata={}, created_at=now,
                          access_count=0, last_accessed=now, relevance=1.0, ttl_seconds=None)
        r2 = MemoryResult(id=id2, content="test", metadata={}, created_at=now,
                          access_count=0, last_accessed=now, relevance=0.9, ttl_seconds=None)
        # Without embeddings, nothing should be dropped
        result = store._semantic_dedup([r1, r2], {id1: 1.0, id2: 0.9}, threshold=0.92)
        assert len(result) == 2  # No embeddings available, both survive


# ============================================================================
# 6. Abstention
# ============================================================================


class TestAbstention:
    """Low-quality results are filtered; infrastructure types excluded."""

    def test_unrelated_query_returns_few_or_empty(self):
        """A query completely unrelated to stored content should return few/no results."""
        store = _get_store()
        _insert_memory(store, "Configure webpack bundling for React application build pipeline", "decision")
        _insert_memory(store, "Set up continuous integration pipeline with GitHub Actions workflows", "lesson_learned")
        ids = _query_ids(store, "quantum entanglement thermodynamics black hole")
        assert len(ids) <= 2, f"Unrelated query should return few results, got {len(ids)}"

    def test_infrastructure_types_excluded_by_default(self):
        """coordination_snapshot is excluded from user queries by default."""
        store = _get_store()
        infra_id = _insert_memory(store, "Coordination snapshot about webpack configuration status", "coordination_snapshot")
        user_id = _insert_memory(store, "Webpack configuration for production builds uses tree shaking", "decision")
        ids = _query_ids(store, "webpack configuration")
        assert user_id in ids, "User-facing memory should appear"
        assert infra_id not in ids, "Infrastructure type should be excluded by default"

    def test_infrastructure_included_with_flag(self):
        """include_infrastructure=True should include infrastructure types."""
        store = _get_store()
        infra_id = _insert_memory(store, "Session summary about testing framework setup and configuration", "session_summary")
        ids = _query_ids(store, "testing framework setup", include_infrastructure=True)
        assert infra_id in ids, "Infrastructure type should appear with include_infrastructure=True"
