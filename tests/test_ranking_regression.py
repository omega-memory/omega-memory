"""End-to-end ranking regression net.

Every assertion here runs a real ``store.query()`` and checks the order of the
returned records.  That is deliberate: the 1.5.13 recency regression was
invisible to pure-function tests because the scoring helper was correct in
isolation and the ordering only inverted once the full query pipeline —
fusion, word overlap, and the cross-encoder rerank — had run.

Pure-function tests of the scoring helpers still live in
``test_retrieval_golden_set.py``.  They complement these; they do not replace
them.

The paired records below share a base sentence and differ only by a trailing
reference token.  That keeps them semantically equivalent to the reranker
(measured cross-encoder spread ~0.007, versus ~0.71 for the near-paraphrases
that motivated this file) so the signal under test is the only thing separating
them.
"""

import pytest
from datetime import datetime, timedelta, timezone

from test_retrieval_golden_set import (
    _get_store,
    _insert_memory,
    _query_ids,
    _set_timestamps,
)

BASE = "Database indexing strategy for user queries optimizes read performance"
QUERY = "database indexing strategy user queries"
UNRELATED = "The office coffee machine was replaced on Tuesday morning"


@pytest.fixture(autouse=True)
def _reset_bridge(tmp_omega_dir):
    """Fresh store per test, mirroring the golden set fixture."""
    from omega.bridge import reset_memory

    reset_memory()
    yield
    reset_memory()


def _equivalent_pair(store, event_type="decision", **kw):
    """Two records that are semantically equivalent but not byte-identical.

    Byte-identical content is collapsed by canonical-hash dedup, so the
    trailing reference token is required to keep both records queryable.
    """
    first = _insert_memory(store, f"{BASE} ref-1", event_type, **kw)
    second = _insert_memory(store, f"{BASE} ref-2", event_type, **kw)
    return first, second


def _age(store, node_id, days):
    stamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    _set_timestamps(store, node_id, created_at=stamp, last_accessed=stamp)


def _reasons(store, query_text=QUERY, limit=10):
    store._invalidate_query_cache()
    return {
        node.id: (node.metadata or {}).get("_ranking_reasons", {})
        for node in store.query(query_text, limit=limit)
    }


class TestRecencyIsFirstClass:
    """The 1.5.13 regression: a stale record outranked an equivalent fresh one."""

    def test_fresh_outranks_old_when_semantically_equivalent(self):
        store = _get_store()
        fresh_id, old_id = _equivalent_pair(store)
        _age(store, old_id, 60)

        ids = _query_ids(store, QUERY)

        assert fresh_id in ids and old_id in ids, f"both expected, got {ids}"
        assert ids.index(fresh_id) < ids.index(old_id), (
            "a 60-day-old record must not outrank an equivalent fresh one"
        )

    def test_fresh_outranks_old_regardless_of_insertion_order(self):
        """Guards against the ordering accidentally tracking insert order."""
        store = _get_store()
        first_id, second_id = _equivalent_pair(store)
        _age(store, first_id, 60)

        ids = _query_ids(store, QUERY)

        assert ids.index(second_id) < ids.index(first_id)

    def test_recency_applies_outside_the_semantic_near_tie_band(self):
        """Recency is a first-class term, not a near-tie tiebreaker.

        Priority and access are deliberately confined to the near-tie band.
        Recency must not be, or age stops mattering the moment a candidate
        falls outside it.
        """
        store = _get_store()
        fresh_id, old_id = _equivalent_pair(store)
        _age(store, old_id, 60)
        _insert_memory(store, UNRELATED, "decision")

        reasons = _reasons(store)

        for node_id in (fresh_id, old_id):
            assert node_id in reasons, f"{node_id} missing from results"
        fresh_reasons, old_reasons = reasons[fresh_id], reasons[old_id]
        assert fresh_reasons["recency_contribution"] > old_reasons["recency_contribution"], (
            "the fresher record must earn the larger recency contribution"
        )
        outside_band = [r for r in reasons.values() if not r.get("near_tie")]
        assert all(r["recency_contribution"] > 0 for r in outside_band), (
            "recency must still be credited outside the near-tie band"
        )

    def test_decay_is_not_collapsed_into_the_bounded_quality_term(self):
        """The exact shape of the regression: decay folded into a +/-0.0025 term.

        If decay is ever moved back inside the near-tie quality bundle, the
        recency spread between a fresh and an aged record collapses to at most
        that bundle's cap and this fails.
        """
        store = _get_store()
        fresh_id, old_id = _equivalent_pair(store)
        _age(store, old_id, 60)

        reasons = _reasons(store)
        spread = (
            reasons[fresh_id]["recency_contribution"]
            - reasons[old_id]["recency_contribution"]
        )

        assert spread > 0.0025, (
            f"recency spread {spread} collapsed to the bounded-quality scale; "
            "decay must stay a first-class signal"
        )


class TestSemanticRelevanceStillWins:
    """Recency must not become the new dominating signal."""

    def test_relevant_old_beats_irrelevant_fresh(self):
        store = _get_store()
        relevant_id = _insert_memory(store, BASE, "decision")
        _age(store, relevant_id, 60)
        _insert_memory(store, UNRELATED, "decision")

        ids = _query_ids(store, QUERY)

        assert ids[0] == relevant_id, (
            "a clearly more relevant older record must not lose to a fresher "
            f"irrelevant one, got {ids}"
        )

    def test_recency_cannot_promote_an_unrelated_record(self):
        store = _get_store()
        relevant_id = _insert_memory(store, BASE, "decision")
        _age(store, relevant_id, 365)
        unrelated_id = _insert_memory(store, UNRELATED, "decision")

        ids = _query_ids(store, QUERY)

        if unrelated_id in ids:
            assert ids.index(relevant_id) < ids.index(unrelated_id)


class TestPriorityStaysBounded:
    """Restores the end-to-end priority ordering assertion."""

    def test_priority_contribution_is_applied_and_capped(self):
        """Priority earns exactly its specified bounded contribution.

        This asserts the contribution rather than a resulting rank order on
        purpose.  PRIORITY_MAX_ADDITIVE is 0.005, which is smaller than the
        ordinary spread the reranker and word-overlap stages introduce between
        two otherwise equivalent records, so priority is not guaranteed to
        decide an end-to-end tie and a test asserting that it does would be
        flaky rather than protective.
        """
        from omega.sqlite_store._query import PRIORITY_MAX_ADDITIVE

        store = _get_store()
        # priority must be set at insert time so the SQL column and the
        # metadata copy agree; scoring reads the metadata copy.
        low_id = _insert_memory(store, f"{BASE} ref-1", "decision", priority=1)
        high_id = _insert_memory(store, f"{BASE} ref-2", "decision", priority=5)

        reasons = _reasons(store)

        assert reasons[low_id]["priority_contribution"] == 0.0
        assert reasons[high_id]["priority_contribution"] == pytest.approx(
            PRIORITY_MAX_ADDITIVE
        )
        assert all(
            r["priority_contribution"] <= PRIORITY_MAX_ADDITIVE
            for r in reasons.values()
        ), "priority must never exceed its specified bound"

    def test_priority_cannot_outrank_a_clearly_better_match(self):
        store = _get_store()
        relevant_id = _insert_memory(store, BASE, "decision", priority=1)
        _insert_memory(store, UNRELATED, "decision", priority=5)

        ids = _query_ids(store, QUERY)

        assert ids[0] == relevant_id, (
            f"priority 5 must not promote an irrelevant record, got {ids}"
        )


class TestAccessStaysBounded:
    """Restores the end-to-end access ordering assertion."""

    def test_access_cannot_outrank_a_clearly_better_match(self):
        store = _get_store()
        relevant_id = _insert_memory(store, BASE, "decision")
        hot_id = _insert_memory(store, UNRELATED, "decision")
        _set_timestamps(store, hot_id, access_count=500)

        ids = _query_ids(store, QUERY)

        assert ids[0] == relevant_id, (
            f"a heavily accessed irrelevant record must not win, got {ids}"
        )

    def test_access_contribution_is_capped(self):
        store = _get_store()
        capped_id, saturated_id = _equivalent_pair(store)
        _set_timestamps(store, capped_id, access_count=3)
        _set_timestamps(store, saturated_id, access_count=500)

        reasons = _reasons(store)

        assert (
            reasons[capped_id]["access_contribution"]
            == reasons[saturated_id]["access_contribution"]
        ), "access beyond the scoring cap must not buy additional rank"
        assert reasons[saturated_id]["bounded_access_count"] == 3
        assert reasons[saturated_id]["access_count"] == 500, (
            "the audit counter must stay truthful even though scoring caps it"
        )


class TestRerankerCannotManufactureConfidence:
    """The reranker's min-max normalisation is scale-free by construction.

    With a small candidate set it maps the best candidate to 1.0 and the worst
    to 0.0 however close their raw scores are, which is what previously let it
    override recency between two equivalent records.
    """

    def test_negligible_reranker_spread_does_not_override_recency(self):
        store = _get_store()
        fresh_id, old_id = _equivalent_pair(store)
        _age(store, old_id, 60)

        ids = _query_ids(store, QUERY)

        assert ids.index(fresh_id) < ids.index(old_id), (
            "a negligible reranker preference must not outweigh a 60-day age gap"
        )
