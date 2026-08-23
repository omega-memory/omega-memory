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
reference token.  That keeps them semantically equivalent under every supported
reranker, which the near-paraphrases that motivated this file are not:

    model                   ref-1/ref-2   near-paraphrase
    ms-marco-MiniLM-L-6-v2  0.007         0.712
    bge-reranker-v2-m3      0.267         2.487

Both models place the reference-token pair in their tied regime and the
near-paraphrase pair in their separated regime, so the signal under test is the
only thing separating the records here.

Caveat, recorded because it is a known open defect: these are raw-logit spreads
measured on bare passages.  The shipped pipeline prepends "[Date: ...]" to every
passage before scoring, which moves the logits.  See the open-defects banner in
docs/ranking-calibration.md.
"""

import pytest
from datetime import datetime, timedelta, timezone

from omega.sqlite_store._query import _CE_DECISIVE_CONFIDENCE
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

    def test_stronger_near_paraphrase_beats_fresher_weaker_one(self):
        """The genuine converse: a real relevance gap, not an obvious mismatch.

        Which of these two near-paraphrases is "stronger" is a property of the
        resolved reranker, not a constant — ms-marco prefers "provides fast
        lookup" while bge prefers "optimizes read performance". The preference
        is therefore measured at runtime and the preferred record is the one
        aged, so this asserts the invariant rather than one model's opinion.
        """
        from omega.reranker import cross_encoder_score

        alternative = "Database indexing strategy for user queries provides fast lookup"
        scores = cross_encoder_score(QUERY, [BASE, alternative])
        if scores is None:
            pytest.skip("reranker unavailable; this test is about its preference")
        preferred, weaker = (
            (BASE, alternative) if scores[0] >= scores[1] else (alternative, BASE)
        )

        store = _get_store()
        stronger_old = _insert_memory(store, preferred, "decision")
        _age(store, stronger_old, 60)
        weaker_fresh = _insert_memory(store, weaker, "decision")

        ids = _query_ids(store, QUERY)

        assert stronger_old in ids and weaker_fresh in ids, f"both expected, got {ids}"
        assert ids.index(stronger_old) < ids.index(weaker_fresh), (
            "a materially stronger semantic match must beat a fresher weaker one"
        )

    def test_recency_cannot_promote_an_unrelated_record(self):
        """A fresh unrelated record must not be retrieved at all for this query.

        Asserted directly rather than behind an ``if unrelated_id in ids``
        guard.  The guarded form never executed — abstention filters the
        unrelated record out before ranking — so it asserted nothing in any
        configuration.
        """
        store = _get_store()
        relevant_id = _insert_memory(store, BASE, "decision")
        _age(store, relevant_id, 365)
        unrelated_id = _insert_memory(store, UNRELATED, "decision")

        ids = _query_ids(store, QUERY)

        assert relevant_id in ids, f"the relevant record must survive, got {ids}"
        assert unrelated_id not in ids, (
            f"a fresh unrelated record must not be promoted into results, got {ids}"
        )


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


class TestCrossEncoderConfidenceCalibration:
    """The rerank boost must be earned, and comparably so across models.

    ``cross_encoder_score`` returns raw logits, so a spread is only meaningful
    relative to the model that produced it.  The two supported models' regimes
    sit an order of magnitude apart, and because confidence is a ratio rather
    than a threshold, no single global scale satisfies both at once -- see
    docs/ranking-calibration.md.  Hence these tests are keyed to model identity.
    """

    # Measured maxima over five tied pairs and minima over five genuinely
    # separated pairs, per model. See docs/ranking-calibration.md.
    MEASURED_TIED_MAX = {
        "ms-marco-MiniLM-L-6-v2": 0.04495,
        "bge-reranker-v2-m3": 0.38548,
    }
    MEASURED_SEPARATED_MIN = {
        "ms-marco-MiniLM-L-6-v2": 0.71181,
        "bge-reranker-v2-m3": 2.48673,
    }
    # A top-3 boost (ce_w 0.15) overpowers the recency span
    # (RECENCY_MAX_ADDITIVE * (1 - decay floor) = 0.0425) at this confidence,
    # so a tied pair must stay well below it.  Taken from the module so the
    # guard cannot drift away from the value the calibration was derived under.
    DECISIVE_CONFIDENCE = _CE_DECISIVE_CONFIDENCE

    def test_every_supported_model_is_calibrated(self):
        from omega.reranker import _AVAILABLE_MODELS
        from omega.sqlite_store._query import _CE_SPREAD_FULL_SCALE_BY_MODEL

        missing = set(_AVAILABLE_MODELS) - set(_CE_SPREAD_FULL_SCALE_BY_MODEL)
        assert not missing, (
            f"reranker(s) selectable but uncalibrated: {sorted(missing)}. "
            "An uncalibrated scale is the defect this calibration exists to stop."
        )

    @pytest.mark.parametrize("model", sorted(MEASURED_TIED_MAX))
    def test_tied_spread_cannot_outweigh_recency(self, model):
        """Guards against a mis-scaled constant, per model."""
        from omega.sqlite_store._query import _ce_confidence, _ce_full_scale

        confidence = _ce_confidence(self.MEASURED_TIED_MAX[model], _ce_full_scale(model))

        assert confidence < self.DECISIVE_CONFIDENCE, (
            f"{model}: a tied pair earns confidence {confidence}, enough to "
            "overpower recency. The calibrated scale is too small."
        )

    @pytest.mark.parametrize("model", sorted(MEASURED_SEPARATED_MIN))
    def test_genuinely_separated_spread_keeps_reranker_useful(self, model):
        """The other half: scaling must not silence a real preference."""
        from omega.sqlite_store._query import _ce_confidence, _ce_full_scale

        confidence = _ce_confidence(
            self.MEASURED_SEPARATED_MIN[model], _ce_full_scale(model)
        )

        assert confidence > 0.5, (
            f"{model}: a genuinely separated pair earns only {confidence}. "
            "The calibrated scale is too large and suppresses the reranker."
        )

    @pytest.mark.parametrize("precision", ["fp32", "int8"])
    def test_every_selectable_precision_resolves_to_a_calibrated_scale(
        self, precision, monkeypatch
    ):
        """``_resolve_reranker_model`` can select a precision, not just a name.

        The scale is keyed on model identity, and a precision variant keeps its
        model's name, so it inherits that model's scale.  This pins the
        inheritance deliberately rather than leaving it to coincidence: a new
        variant that reported a different name would silently fall through to
        the uncalibrated path and lose the rerank boost entirely.
        """
        from omega.reranker import _AVAILABLE_MODELS, _resolve_reranker_model
        from omega.sqlite_store._query import _ce_full_scale

        for name, config in _AVAILABLE_MODELS.items():
            if precision != "fp32" and precision not in config.get("precisions", {}):
                continue
            monkeypatch.setenv("OMEGA_RERANKER_MODEL", name)
            monkeypatch.setenv("OMEGA_RERANKER_PRECISION", precision)

            resolved_name, _ = _resolve_reranker_model()

            assert _ce_full_scale(resolved_name) is not None, (
                f"{name} at {precision} resolves to '{resolved_name}', which has "
                "no calibrated scale and would run with the rerank boost off."
            )

    def test_uncalibrated_model_disables_the_boost(self):
        """An unknown reranker gets no magnitude boost rather than a guess."""
        from omega.sqlite_store._query import _ce_confidence, _ce_full_scale

        assert _ce_full_scale("some-unmeasured-reranker") is None
        assert _ce_full_scale(None) is None
        assert _ce_confidence(5.0, _ce_full_scale("some-unmeasured-reranker")) == 0.0

    @pytest.mark.parametrize(
        "ce_range",
        [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
    )
    def test_degenerate_spreads_earn_no_boost(self, ce_range):
        """Equal, malformed, and non-finite spreads must not reorder anything."""
        from omega.sqlite_store._query import _ce_confidence

        assert _ce_confidence(ce_range, 1.0) == 0.0

    @pytest.mark.parametrize("full_scale", [0.0, -1.0, float("nan"), float("inf")])
    def test_degenerate_scales_earn_no_boost(self, full_scale):
        from omega.sqlite_store._query import _ce_confidence

        assert _ce_confidence(1.0, full_scale) == 0.0

    def test_confidence_is_bounded_and_monotonic(self):
        from omega.sqlite_store._query import _ce_confidence

        values = [_ce_confidence(r, 1.0) for r in (0.1, 0.5, 1.0, 10.0)]

        assert values == sorted(values)
        assert all(0.0 <= v <= 1.0 for v in values)
        assert values[-1] == 1.0

    def test_single_candidate_is_never_reranked(self):
        """One candidate has no spread, so the reranker cannot act on it."""
        store = _get_store()
        only_id = _insert_memory(store, BASE, "decision")

        ids = _query_ids(store, QUERY)

        assert ids == [only_id]


class TestProtectedTypesAreExemptFromDecay:
    """Types with decay lambda 0 never age, so recency cannot order them.

    This is intended behaviour, not a defect: constraints and preferences are
    meant to persist. It is asserted here because it is a real limit on the
    recency invariant, and because nothing else covers these types end to end.
    """

    PROTECTED = ["constraint", "user_preference", "error_pattern", "reminder"]

    @pytest.mark.parametrize("event_type", PROTECTED)
    def test_protected_type_receives_full_recency_credit_at_any_age(self, event_type):
        from omega.sqlite_store._query import RECENCY_MAX_ADDITIVE

        store = _get_store()
        fresh_id, old_id = _equivalent_pair(store, event_type=event_type)
        _age(store, old_id, 3 * 365)

        reasons = _reasons(store)

        for node_id in (fresh_id, old_id):
            assert reasons[node_id]["decay_factor"] == 1.0
            assert reasons[node_id]["recency_contribution"] == pytest.approx(
                RECENCY_MAX_ADDITIVE
            )

    def test_decaying_type_still_ages(self):
        """Contrast case, so the exemption above cannot silently become global."""
        store = _get_store()
        fresh_id, old_id = _equivalent_pair(store, event_type="decision")
        _age(store, old_id, 3 * 365)

        reasons = _reasons(store)

        assert reasons[old_id]["decay_factor"] < reasons[fresh_id]["decay_factor"]
