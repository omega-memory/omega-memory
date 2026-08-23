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
    ms-marco-MiniLM-L-6-v2  0.036         0.647
    bge-reranker-v2-m3      0.387         1.387

Both models place the reference-token pair in their tied regime and the
near-paraphrase pair in their separated regime, so the signal under test is the
only thing separating the records here.

Those spreads are measured through the production representation -- the query
path prepends "[Date: YYYY-MM-DD] " before scoring, which moves the logits --
at the 60-day gap these tests use.  An earlier version of this file quoted bare
passage spreads, which the pipeline never scores.
"""

import math

import pytest
from datetime import datetime, timedelta, timezone

from omega.sqlite_store._query import (
    _CE_DECISIVE_CONFIDENCE,
    _CE_SEPARATED_MIN_CONFIDENCE,
)
from test_retrieval_golden_set import (
    _get_store,
    _insert_memory,
    _query_ids,
    _set_timestamps,
)

BASE = "Database indexing strategy for user queries optimizes read performance"
QUERY = "database indexing strategy user queries"
UNRELATED = "The office coffee machine was replaced on Tuesday morning"

# The genuinely-separated half of the calibration set, in the same order as
# docs/ranking-calibration.md.  Paired against BASE.
SEPARATED_ALTERNATIVES = [
    "Database indexing strategy for user queries provides fast lookup",
    "Postgres index tuning for slow analytic queries",
    "We migrated the user table to a composite index last quarter",
    "Index maintenance runs nightly on the reporting replica",
    "Query planner statistics are refreshed weekly",
]


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

    # Worst-case maxima over five tied pairs and minima over five genuinely
    # separated pairs, per model, measured through the PRODUCTION scoring path
    # (day-granular date prefix) across three date configurations.  See
    # docs/ranking-calibration.md.  An earlier version of these constants was
    # measured on bare passages, which is not what the pipeline scores.
    MEASURED_TIED_MAX = {
        "ms-marco-MiniLM-L-6-v2": 0.07318,
        "bge-reranker-v2-m3": 0.41323,
    }
    MEASURED_SEPARATED_MIN = {
        "ms-marco-MiniLM-L-6-v2": 0.64687,
        "bge-reranker-v2-m3": 1.35221,
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

        assert confidence > _CE_SEPARATED_MIN_CONFIDENCE, (
            f"{model}: a genuinely separated pair earns only {confidence}. "
            "The calibrated scale is too large and suppresses the reranker."
        )

    @staticmethod
    def _observed_confidence(second_content):
        """Run one real query and return the (spread, confidence) it computed.

        Reads what the pipeline actually produced, including the day-granular
        date prefix ``cross_encoder_score`` applies, rather than feeding a
        stored constant into a helper.
        """
        from omega.sqlite_store import _query as query_module

        observed = []
        real_normalise = query_module._ce_normalise

        def spy(ce_scores, full_scale):
            result = real_normalise(ce_scores, full_scale)
            if ce_scores:
                observed.append((max(ce_scores) - min(ce_scores), result[1]))
            return result

        store = _get_store()
        _insert_memory(store, f"{BASE} ref-1", "decision")
        other = _insert_memory(store, second_content, "decision")
        _age(store, other, 60)

        query_module._ce_normalise = spy
        try:
            _query_ids(store, QUERY)
        finally:
            query_module._ce_normalise = real_normalise

        assert observed, "the reranker did not run, so this guard proved nothing"
        return observed[-1]

    def _skip_if_uncalibrated(self):
        from omega.reranker import _RERANKER_MODEL_NAME
        from omega.sqlite_store._query import _ce_full_scale

        if _ce_full_scale(_RERANKER_MODEL_NAME) is None:
            pytest.skip(f"no calibration for resolved reranker {_RERANKER_MODEL_NAME}")
        return _RERANKER_MODEL_NAME

    def test_tied_pair_stays_indecisive_on_the_real_query_path(self):
        """The guard that would have caught the bare-passage calibration.

        Every other test in this class feeds a stored constant into a pure
        helper, so it proves only that the arithmetic is self-consistent.  This
        one fails if the calibration for the resolved model drifts above its
        production-path window, where a tied pair would start deciding
        rankings.
        """
        model = self._skip_if_uncalibrated()

        spread, confidence = self._observed_confidence(f"{BASE} ref-2")

        assert confidence < _CE_DECISIVE_CONFIDENCE, (
            f"{model}: on the real query path a tied pair (spread "
            f"{spread:.5f}) earns confidence {confidence:.5f}, at or above the "
            f"decisive {_CE_DECISIVE_CONFIDENCE}. The scale is too small for "
            "the representation production actually scores."
        )

    def test_separated_pair_keeps_its_authority_in_the_production_representation(self):
        """The other half, and the one the bare-passage calibration failed.

        This scores through the production *representation* -- the date-prefixed
        passage ``cross_encoder_score`` builds when the query path passes
        temporal metadata -- rather than through a full ``store.query()``.  The
        reason is a retrieval-layer property, recorded here because it is not
        obvious: a genuinely different aged record does not reliably enter the
        fusion candidate set alongside its fresh counterpart unless the two
        share a trailing token, so the pipeline never presents the calibration's
        own separated pairs to the reranker together.  The tied guard above does
        run through the full pipeline; this one pins the half of the window that
        guard cannot reach, against the same representation.
        """
        from omega.reranker import _RERANKER_MODEL_NAME, cross_encoder_score
        from omega.sqlite_store._query import _ce_confidence, _ce_full_scale

        scale = _ce_full_scale(_RERANKER_MODEL_NAME)
        if scale is None:
            pytest.skip(f"no calibration for resolved reranker {_RERANKER_MODEL_NAME}")

        today = datetime.now(timezone.utc).date()
        dates = [today.isoformat(), (today - timedelta(days=60)).isoformat()]
        worst = None
        for alternative in SEPARATED_ALTERNATIVES:
            scores = cross_encoder_score(QUERY, [BASE, alternative], temporal_metadata=dates)
            if not scores:
                pytest.skip(f"reranker {_RERANKER_MODEL_NAME} unavailable")
            spread = max(scores) - min(scores)
            confidence = _ce_confidence(spread, scale)
            if worst is None or confidence < worst[0]:
                worst = (confidence, spread, alternative)

        confidence, spread, alternative = worst
        assert confidence > _CE_SEPARATED_MIN_CONFIDENCE, (
            f"{_RERANKER_MODEL_NAME}: the weakest genuinely separated pair "
            f"({alternative!r}, spread {spread:.5f}) earns only "
            f"{confidence:.5f} at scale {scale}. The scale is too large and "
            "suppresses the reranker where it has a real preference."
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


class TestNonFiniteCrossEncoderScores:
    """A reranker emitting NaN or infinity must not corrupt ranking.

    ``_ce_confidence`` alone was not enough.  ``ce_norm`` used to be computed
    before, and independently of, the confidence, under a bare ``ce_range > 0``
    guard that ``inf`` satisfies.  That produced NaN entries in ``ce_norm``
    which survived a zero confidence, because ``NaN * 0.0`` is ``NaN``, and a
    single NaN collapsed every returned record to zero relevance without
    raising, so no error path caught it.  Normalisation and confidence are now
    decided together, and the fallback is deterministic: a flat 0.5 at zero
    confidence, which makes every multiplier at the call site exactly 1.0.
    """

    NON_FINITE = [float("nan"), float("inf"), float("-inf")]

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_helper_returns_the_neutral_fallback(self, bad):
        from omega.sqlite_store._query import _ce_normalise

        ce_norm, confidence = _ce_normalise([1.0, 2.0, bad], 1.0)

        assert confidence == 0.0
        assert ce_norm == [0.5, 0.5, 0.5], "must be flat, and must not hold NaN"
        assert all(math.isfinite(v) for v in ce_norm)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_mixed_finite_and_non_finite_earns_nothing(self, bad):
        """One bad score makes the whole spread meaningless, not just its own."""
        from omega.sqlite_store._query import _ce_normalise

        ce_norm, confidence = _ce_normalise([0.1, 0.9, 5.0, bad, 2.0], 1.0)

        assert confidence == 0.0
        assert ce_norm == [0.5] * 5

    def test_equal_finite_scores_earn_nothing(self):
        from omega.sqlite_store._query import _ce_normalise

        assert _ce_normalise([2.5, 2.5, 2.5], 1.0) == ([0.5, 0.5, 0.5], 0.0)

    def test_single_score_earns_nothing(self):
        from omega.sqlite_store._query import _ce_normalise

        assert _ce_normalise([7.0], 1.0) == ([0.5], 0.0)

    def test_empty_score_list_is_safe(self):
        from omega.sqlite_store._query import _ce_normalise

        assert _ce_normalise([], 1.0) == ([], 0.0)

    def test_finite_scores_still_normalise(self):
        """The guard must not disable the reranker for well-formed scores."""
        from omega.sqlite_store._query import _ce_normalise

        ce_norm, confidence = _ce_normalise([0.0, 0.5, 1.0], 1.0)

        assert ce_norm == [0.0, 0.5, 1.0]
        assert confidence == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_query_path_survives_a_non_finite_reranker(self, monkeypatch, bad):
        """The property the helper exists to protect, asserted end to end."""
        import omega.reranker as reranker_module

        def scorer(value):
            def score(query, passages, temporal_metadata=None):
                scores = [1.0] * len(passages)
                if scores and value is not None:
                    scores[-1] = value
                return scores

            return score

        # One store for both runs, so record identity is stable and the only
        # thing that differs is what the reranker returns.
        store = _get_store()
        _equivalent_pair(store)
        _insert_memory(store, UNRELATED, "decision")

        def order_under(value):
            # The query path imports cross_encoder_score from omega.reranker
            # inside the function, so that module is the binding that matters.
            monkeypatch.setattr(
                reranker_module, "cross_encoder_score", scorer(value)
            )
            store._invalidate_query_cache()
            return store.query(QUERY, limit=10)

        # Control: all-equal finite scores. The reranker has expressed no
        # preference, confidence is zero, and the ordering is whatever fusion
        # produced. A non-finite set must be indistinguishable from this.
        control = order_under(None)
        results = order_under(bad)

        assert results, "a poisoned reranker must not empty the result set"
        for node in results:
            assert node.relevance is None or math.isfinite(node.relevance), (
                f"non-finite relevance {node.relevance} leaked from the reranker"
            )
        assert [n.id for n in results] == [n.id for n in control], (
            "a non-finite reranker must behave exactly like one that expressed "
            "no preference, not reorder or drop records"
        )
        assert [n.relevance for n in results] == [n.relevance for n in control]
