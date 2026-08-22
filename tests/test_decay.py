"""Tests for OMEGA Automated Decay Curves (Feature 2)."""

import math
import pytest
from datetime import datetime, timedelta, timezone

# ============================================================================
# Fixture: fresh store per test
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


# ============================================================================
# 1. Protected types return 1.0 (no decay)
# ============================================================================


def test_protected_types_no_decay():
    """user_preference, error_pattern, and reminder should never decay."""
    store = _get_store()
    old_date = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()

    for event_type in ("user_preference", "error_pattern", "reminder"):
        factor = store._compute_decay_factor(event_type, None, old_date)
        assert factor == 1.0, f"{event_type} should not decay, got {factor}"


# ============================================================================
# 2. Fresh memory returns ~1.0
# ============================================================================


def test_fresh_memory_near_one():
    """A memory accessed just now should have decay factor close to 1.0."""
    store = _get_store()
    now = datetime.now(timezone.utc).isoformat()

    factor = store._compute_decay_factor("memory", now, now)
    assert factor >= 0.99, f"Fresh memory should be ~1.0, got {factor}"


# ============================================================================
# 3. Old memory approaches floor
# ============================================================================


def test_old_memory_near_floor():
    """A very old, never-accessed memory should approach the decay floor."""
    store = _get_store()
    old_date = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()

    # "memory" type has lambda=0.02, so after 365 days: exp(-0.02*365) ≈ 0.0007
    # Access history no longer changes decay; both use the same floor.
    factor = store._compute_decay_factor("memory", None, old_date, access_count=0)
    assert factor == store._DECAY_FLOOR_NEVER_ACCESSED, f"Expected floor {store._DECAY_FLOOR_NEVER_ACCESSED}, got {factor}"

    factor_accessed = store._compute_decay_factor("memory", None, old_date, access_count=1)
    assert factor_accessed == factor


# ============================================================================
# 4. Never-accessed uses created_at
# ============================================================================


def test_never_accessed_uses_created_at():
    """When last_accessed is None, created_at should be used for decay."""
    store = _get_store()
    # 100 days ago, lambda=0.02 -> exp(-2.0) ~ 0.135
    # Never-accessed (access_count=0) uses lower floor (0.15)
    old_date = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()

    factor = store._compute_decay_factor("memory", None, old_date, access_count=0)
    assert factor == store._DECAY_FLOOR_NEVER_ACCESSED

    factor_accessed = store._compute_decay_factor("memory", None, old_date, access_count=1)
    assert factor_accessed == factor


# ============================================================================
# 5. Floor is enforced
# ============================================================================


def test_floor_enforced():
    """Decay factor should never go below the appropriate floor."""
    store = _get_store()
    ancient = (datetime.now(timezone.utc) - timedelta(days=10000)).isoformat()

    # Never-accessed uses the lower floor
    factor = store._compute_decay_factor("session_summary", None, ancient, access_count=0)
    assert factor == store._DECAY_FLOOR_NEVER_ACCESSED
    assert factor > 0, "Decay factor should never be zero"

    factor_accessed = store._compute_decay_factor("session_summary", None, ancient, access_count=1)
    assert factor_accessed == factor


# ============================================================================
# 6. Ranking is affected by decay
# ============================================================================


def test_ranking_affected_by_decay():
    """Decay should cause old memories to rank lower than recent ones."""
    store = _get_store()
    now_str = datetime.now(timezone.utc).isoformat()
    old_str = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()

    # Store two memories with different ages but identical content relevance
    old_id = store.store(
        content="The API endpoint returns 404 for missing resources",
        metadata={"event_type": "decision"},
        skip_inference=True,
    )
    new_id = store.store(
        content="The API endpoint returns 404 for unknown resource paths",
        metadata={"event_type": "decision"},
        skip_inference=True,
    )

    # Backdate old one
    store._conn.execute(
        "UPDATE memories SET created_at = ?, last_accessed = NULL WHERE node_id = ?",
        (old_str, old_id),
    )
    # Make new one recent
    store._conn.execute(
        "UPDATE memories SET created_at = ?, last_accessed = ? WHERE node_id = ?",
        (now_str, now_str, new_id),
    )
    store._conn.commit()

    # Verify decay factors differ
    old_factor = store._compute_decay_factor("decision", None, old_str)
    new_factor = store._compute_decay_factor("decision", now_str, now_str)
    assert new_factor > old_factor, f"New ({new_factor}) should rank higher than old ({old_factor})"


# ============================================================================
# 7. Permanent types unaffected at any age
# ============================================================================


def test_permanent_types_unaffected_at_any_age():
    """Protected types should return 1.0 regardless of how old they are."""
    store = _get_store()

    ages = [1, 30, 365, 3650]  # days
    for days in ages:
        old_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        for etype in ("user_preference", "error_pattern", "reminder"):
            factor = store._compute_decay_factor(etype, None, old_date)
            assert factor == 1.0, f"{etype} at {days}d should be 1.0, got {factor}"


# ============================================================================
# 8. Medium-age expected range
# ============================================================================


def test_medium_age_expected_range():
    """A medium-age memory (30 days) should have a decay factor between floor and 1.0."""
    store = _get_store()
    # 30 days, "decision" lambda=0.015 → exp(-0.45) ≈ 0.638
    medium_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    factor = store._compute_decay_factor("decision", medium_date, medium_date)

    # Expected: exp(-0.015 * 30) = exp(-0.45) ≈ 0.6376
    expected = math.exp(-0.015 * 30)
    assert abs(factor - expected) < 0.05, f"Expected ~{expected:.3f}, got {factor:.3f}"
    assert store._DECAY_FLOOR < factor < 1.0
