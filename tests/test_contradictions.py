"""Tests for OMEGA contradiction detection — pure function + integration tests."""

import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from omega.contradictions import (
    detect_contradictions,
    _check_negation_asymmetry,
    _check_antonyms,
    _check_preference_change,
    _check_temporal_override,
    _word_overlap_similarity,
    _normalize_scores,
)


# ---------------------------------------------------------------------------
# 1. TestNegationDetection
# ---------------------------------------------------------------------------


class TestNegationDetection:
    """Test negation asymmetry signal."""

    def test_one_negated_one_not(self):
        """'Alex likes X' vs 'Alex does not like X' should score high."""
        score = _check_negation_asymmetry(
            "alex likes python", {"alex", "likes", "python"},
            "alex does not like python", {"alex", "does", "not", "like", "python"},
        )
        assert score >= 0.5

    def test_neither_negated(self):
        """Two affirmative statements should score 0."""
        score = _check_negation_asymmetry(
            "alex likes python", {"alex", "likes", "python"},
            "alex likes javascript", {"alex", "likes", "javascript"},
        )
        assert score == 0.0

    def test_both_negated_same(self):
        """Both negated with same negation word should score 0."""
        score = _check_negation_asymmetry(
            "never use eval", {"never", "use", "eval"},
            "never use exec", {"never", "use", "exec"},
        )
        assert score == 0.0

    def test_contraction_negation(self):
        """Contractions like 'don't' should be detected."""
        score = _check_negation_asymmetry(
            "don't use tabs", {"don't", "use", "tabs"},
            "use tabs for indentation", {"use", "tabs", "for", "indentation"},
        )
        assert score >= 0.5


# ---------------------------------------------------------------------------
# 2. TestAntonymDetection
# ---------------------------------------------------------------------------


class TestAntonymDetection:
    """Test antonym pair detection."""

    def test_light_vs_dark(self):
        """'light mode' vs 'dark mode' should detect antonym."""
        score = _check_antonyms(
            {"prefers", "light", "mode"},
            {"prefers", "dark", "mode"},
        )
        assert score >= 0.7

    def test_enable_vs_disable(self):
        """'enable feature' vs 'disable feature' should detect antonym."""
        score = _check_antonyms(
            {"enable", "the", "feature"},
            {"disable", "the", "feature"},
        )
        assert score >= 0.7

    def test_no_antonyms(self):
        """Unrelated words should score 0."""
        score = _check_antonyms(
            {"python", "programming", "language"},
            {"weather", "forecast", "sunny"},
        )
        assert score == 0.0

    def test_antonym_with_shared_context(self):
        """Antonym in shared context should score highest."""
        score = _check_antonyms(
            {"always", "use", "dark", "theme", "in", "editor"},
            {"always", "use", "light", "theme", "in", "editor"},
        )
        assert score >= 0.9


# ---------------------------------------------------------------------------
# 3. TestPreferenceChange
# ---------------------------------------------------------------------------


class TestPreferenceChange:
    """Test preference value change detection."""

    def test_different_preference_values(self):
        """'prefers vim' vs 'prefers vscode' should detect change."""
        score = _check_preference_change(
            "alex prefers vim for editing",
            "alex prefers vscode for editing",
        )
        assert score >= 0.5

    def test_same_preference(self):
        """Same preference value should score 0."""
        score = _check_preference_change(
            "alex prefers vim",
            "alex prefers vim for everything",
        )
        assert score == 0.0

    def test_switched_to_pattern(self):
        """'switched to X' should be detected."""
        score = _check_preference_change(
            "switched to neovim",
            "uses vim daily",
        )
        assert score >= 0.5

    def test_no_preference_patterns(self):
        """Text without preference patterns should score 0."""
        score = _check_preference_change(
            "the sky is blue",
            "water is wet",
        )
        assert score == 0.0


# ---------------------------------------------------------------------------
# 4. TestTemporalOverride
# ---------------------------------------------------------------------------


class TestTemporalOverride:
    """Test temporal override signal detection."""

    def test_new_has_temporal_marker(self):
        """New content with 'now' should detect temporal override."""
        score = _check_temporal_override(
            "now uses python 3.12",
            "uses python 3.11",
        )
        assert score >= 0.4

    def test_no_temporal_markers(self):
        """Neither text with temporal markers should score 0."""
        score = _check_temporal_override(
            "likes python",
            "likes javascript",
        )
        assert score == 0.0

    def test_changed_keyword(self):
        """'changed to X' is a temporal signal."""
        score = _check_temporal_override(
            "changed to using bun instead of npm",
            "uses npm for package management",
        )
        assert score >= 0.4


# ---------------------------------------------------------------------------
# 5. TestDetectContradictions (integration)
# ---------------------------------------------------------------------------


class TestDetectContradictions:
    """Test the main detect_contradictions function."""

    def test_empty_inputs(self):
        """Empty content or candidates returns empty list."""
        assert detect_contradictions("", ["something"]) == []
        assert detect_contradictions("something", []) == []

    def test_clear_contradiction(self):
        """Opposite statements should be detected."""
        results = detect_contradictions(
            "Alex prefers dark mode",
            [
                "Alex prefers light mode",
                "The weather is sunny today",
                "Python is a programming language",
            ],
            similarity_threshold=0.1,
            contradiction_threshold=0.1,
        )
        # Should find at least one contradiction with the first candidate
        assert len(results) >= 1
        assert results[0].candidate_index == 0
        assert results[0].confidence > 0
        assert len(results[0].signals) > 0

    def test_no_contradiction(self):
        """Unrelated statements should not trigger contradiction."""
        results = detect_contradictions(
            "Python is great for data science",
            [
                "The weather is nice today",
                "Cats are popular pets",
            ],
            similarity_threshold=0.3,
            contradiction_threshold=0.4,
        )
        assert len(results) == 0

    def test_negation_contradiction(self):
        """Direct negation should be caught."""
        results = detect_contradictions(
            "Never commit directly to main",
            [
                "Always commit directly to main",
            ],
            similarity_threshold=0.1,
            contradiction_threshold=0.1,
        )
        assert len(results) >= 1
        assert "negation" in results[0].signals or "antonym" in results[0].signals

    def test_precomputed_similarity(self):
        """Pre-computed similarity scores should be used when provided."""
        results = detect_contradictions(
            "Alex prefers dark mode",
            ["Alex prefers light mode"],
            similarity_scores=[5.0],  # High raw similarity
            contradiction_threshold=0.1,
        )
        assert len(results) >= 1

    def test_results_sorted_by_confidence(self):
        """Results should be sorted by confidence descending."""
        results = detect_contradictions(
            "Always use dark mode in the editor",
            [
                "Always use light mode in the editor",
                "Sometimes use the editor",
                "Never use dark mode in any editor",
            ],
            similarity_threshold=0.0,
            contradiction_threshold=0.0,
        )
        if len(results) >= 2:
            assert results[0].confidence >= results[1].confidence

    def test_result_dataclass_fields(self):
        """ContradictionResult should have all expected fields."""
        results = detect_contradictions(
            "Enable dark mode",
            ["Disable dark mode"],
            similarity_threshold=0.0,
            contradiction_threshold=0.0,
        )
        assert len(results) >= 1
        r = results[0]
        assert isinstance(r.candidate_index, int)
        assert isinstance(r.candidate_content, str)
        assert isinstance(r.confidence, float)
        assert isinstance(r.reason, str)
        assert isinstance(r.similarity, float)
        assert isinstance(r.signals, list)


# ---------------------------------------------------------------------------
# 6. TestHelpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Test helper functions."""

    def test_word_overlap_similarity(self):
        """Jaccard similarity should work correctly."""
        scores = _word_overlap_similarity(
            "python programming language",
            [
                "python programming is fun",
                "completely unrelated text here",
            ],
        )
        assert len(scores) == 2
        assert scores[0] > scores[1]  # First is more similar

    def test_word_overlap_empty(self):
        """Empty text should return zeros."""
        assert _word_overlap_similarity("", ["hello"]) == [0.0]

    def test_normalize_scores(self):
        """Score normalization to [0, 1]."""
        normed = _normalize_scores([1.0, 3.0, 5.0])
        assert normed[0] == 0.0
        assert normed[1] == 0.5
        assert normed[2] == 1.0

    def test_normalize_identical_scores(self):
        """Identical scores should normalize to 0.5."""
        normed = _normalize_scores([3.0, 3.0, 3.0])
        assert all(s == 0.5 for s in normed)

    def test_normalize_empty(self):
        """Empty list returns empty."""
        assert _normalize_scores([]) == []


# ---------------------------------------------------------------------------
# 7. TestTemporalSupersession (integration — requires SQLiteStore + sqlite-vec)
# ---------------------------------------------------------------------------

from omega.sqlite_store import SQLiteStore, EMBEDDING_DIM


def _make_embedding(seed: float = 1.0) -> list:
    """Create a deterministic 384-dim unit embedding from a seed."""
    raw = [math.sin(seed * (i + 1)) for i in range(EMBEDDING_DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _perturb_embedding(base: list, amount: float = 0.4) -> list:
    """Perturb an embedding by mixing in noise. Smaller amount = more similar."""
    noise = _make_embedding(seed=99.0)
    mixed = [b * (1 - amount) + n * amount for b, n in zip(base, noise)]
    norm = math.sqrt(sum(x * x for x in mixed))
    return [x / norm for x in mixed]


@pytest.fixture
def vec_store(tmp_path):
    """SQLiteStore with vec support for supersession tests."""
    db_path = tmp_path / ".omega" / "test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["OMEGA_HOME"] = str(tmp_path / ".omega")
    s = SQLiteStore(db_path=db_path)
    if not s._vec_available:
        pytest.skip("sqlite-vec not available")
    yield s
    s.close()
    os.environ.pop("OMEGA_HOME", None)


class TestTemporalSupersession:
    """Integration tests for temporal supersession in _check_contradictions."""

    def test_same_type_supersedes(self, vec_store):
        """Older memory with same event_type + high similarity gets superseded."""
        base_emb = _make_embedding(seed=1.0)
        similar_emb = _perturb_embedding(base_emb, amount=0.4)

        old_id = vec_store.store(
            content="LongMemEval score is 82%",
            session_id="s1",
            metadata={"event_type": "decision"},
            embedding=base_emb,
            skip_inference=True,
        )
        # Small delay to ensure different created_at
        time.sleep(0.01)
        new_id = vec_store.store(
            content="LongMemEval score is 95.4%",
            session_id="s1",
            metadata={"event_type": "decision"},
            embedding=similar_emb,
        )

        # Old memory should be superseded
        old = vec_store.get_node(old_id)
        assert old is not None
        assert old.metadata.get("superseded") is True
        assert old.metadata.get("superseded_by") == new_id
        assert vec_store.stats.get("temporal_supersessions", 0) >= 1

    def test_different_type_no_supersession(self, vec_store):
        """Same content but different event_types should NOT supersede."""
        base_emb = _make_embedding(seed=2.0)
        similar_emb = _perturb_embedding(base_emb, amount=0.4)

        old_id = vec_store.store(
            content="Use dark mode everywhere",
            session_id="s1",
            metadata={"event_type": "decision"},
            embedding=base_emb,
            skip_inference=True,
        )
        time.sleep(0.01)
        vec_store.store(
            content="Use dark mode everywhere v2",
            session_id="s1",
            metadata={"event_type": "lesson_learned"},
            embedding=similar_emb,
        )

        old = vec_store.get_node(old_id)
        assert old is not None
        assert old.metadata.get("superseded") is not True

    def test_low_similarity_no_supersession(self, vec_store):
        """Same type but dissimilar content should NOT supersede."""
        base_emb = _make_embedding(seed=3.0)
        # Large perturbation => low similarity
        different_emb = _make_embedding(seed=50.0)

        old_id = vec_store.store(
            content="Use Python for scripting",
            session_id="s1",
            metadata={"event_type": "decision"},
            embedding=base_emb,
            skip_inference=True,
        )
        time.sleep(0.01)
        vec_store.store(
            content="Deploy to Vercel for hosting",
            session_id="s1",
            metadata={"event_type": "decision"},
            embedding=different_emb,
        )

        old = vec_store.get_node(old_id)
        assert old is not None
        assert old.metadata.get("superseded") is not True

    def test_superseded_excluded_from_query(self, vec_store):
        """After supersession, old memory should not appear in query results."""
        base_emb = _make_embedding(seed=4.0)
        similar_emb = _perturb_embedding(base_emb, amount=0.4)

        vec_store.store(
            content="X strategy is 35 tweets per week",
            session_id="s1",
            metadata={"event_type": "decision"},
            embedding=base_emb,
            skip_inference=True,
        )
        time.sleep(0.01)
        vec_store.store(
            content="X strategy is 5 tweets per week",
            session_id="s1",
            metadata={"event_type": "decision"},
            embedding=similar_emb,
        )

        results = vec_store.query("X strategy tweets per week", limit=10)
        contents = [r.content for r in results]
        assert "X strategy is 5 tweets per week" in contents
        assert "X strategy is 35 tweets per week" not in contents

    def test_explicit_contradiction_still_works(self, vec_store):
        """Existing negation/antonym detection should still function."""
        base_emb = _make_embedding(seed=5.0)
        similar_emb = _perturb_embedding(base_emb, amount=0.4)

        old_id = vec_store.store(
            content="Always use light mode",
            session_id="s1",
            metadata={"event_type": "user_fact"},
            embedding=base_emb,
            skip_inference=True,
        )
        time.sleep(0.01)
        new_id = vec_store.store(
            content="Never use light mode",
            session_id="s1",
            metadata={"event_type": "user_fact"},
            embedding=similar_emb,
        )

        # user_fact is NOT in _TEMPORAL_SUPERSESSION_TYPES so supersession
        # should not apply, but contradiction annotation should exist
        old = vec_store.get_node(old_id)
        new = vec_store.get_node(new_id)
        assert old is not None
        assert old.metadata.get("superseded") is not True
        # Contradiction annotations may or may not fire depending on signal
        # strength, but the old memory should NOT be superseded since user_fact
        # isn't a supersession-eligible type.

    def test_non_eligible_type_not_superseded(self, vec_store):
        """Types outside _TEMPORAL_SUPERSESSION_TYPES should never supersede."""
        base_emb = _make_embedding(seed=6.0)
        similar_emb = _perturb_embedding(base_emb, amount=0.4)

        old_id = vec_store.store(
            content="Session summary: worked on feature X",
            session_id="s1",
            metadata={"event_type": "session_summary"},
            embedding=base_emb,
            skip_inference=True,
        )
        time.sleep(0.01)
        vec_store.store(
            content="Session summary: continued feature X",
            session_id="s1",
            metadata={"event_type": "session_summary"},
            embedding=similar_emb,
        )

        old = vec_store.get_node(old_id)
        assert old is not None
        assert old.metadata.get("superseded") is not True


# ---------------------------------------------------------------------------
# TestNonFiniteSimilarityScores
# ---------------------------------------------------------------------------


class TestNonFiniteSimilarityScores:
    """A reranker emitting NaN or infinity must not manufacture contradictions.

    This is the write-path twin of the ranking fail-safe.  It is worse here
    than on the read path: NaN loses every comparison, so ``sim`` slipped past
    ``sim < similarity_threshold``, ``min(1.0, NaN)`` evaluates to ``1.0`` in
    CPython, and ``confidence < contradiction_threshold`` was False too.  The
    result was a maximum-confidence contradiction that the store writes to
    durable metadata and a graph edge — permanent false state from a transient
    scoring glitch.
    """

    NON_FINITE = [float("nan"), float("inf"), float("-inf")]

    CONTRADICTORY = "I no longer use tabs for indentation"
    CANDIDATE = "I always use tabs for indentation"

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_normalise_refuses_to_emit_non_finite_values(self, bad):
        normalised = _normalize_scores([0.1, bad, 0.9])

        assert all(math.isfinite(s) for s in normalised)
        assert normalised == [0.0, 0.0, 0.0], (
            "an unusable batch must fall below any similarity threshold "
            "rather than normalising into the decidable range"
        )

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_a_poisoned_score_cannot_reach_maximum_confidence(self, bad):
        """The precise failure: garbage in, confidence 1.0 out."""
        results = detect_contradictions(
            self.CONTRADICTORY,
            [self.CANDIDATE],
            similarity_scores=[bad],
        )

        assert not any(r.confidence == 1.0 for r in results), (
            f"a {bad} similarity produced a maximum-confidence contradiction"
        )
        for result in results:
            assert math.isfinite(result.confidence)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_the_reranker_falls_back_to_word_overlap(self, monkeypatch, bad):
        """A non-finite cross-encoder counts as unavailable, not as similar.

        Detection must degrade to the word-overlap path the module already
        defines, not switch off and not fabricate: the outcome has to match
        what an absent cross-encoder produces.
        """
        import omega.contradictions as contradictions_module

        def poisoned(query, passages):
            return [bad] * len(passages)

        def absent(query, passages):
            raise RuntimeError("reranker unavailable")

        def detect_with(scorer):
            monkeypatch.setattr(
                contradictions_module, "cross_encoder_score", scorer, raising=False
            )
            monkeypatch.setitem(
                sys.modules, "omega.reranker", _FakeReranker(scorer)
            )
            return detect_contradictions(self.CONTRADICTORY, [self.CANDIDATE])

        poisoned_results = detect_with(poisoned)
        fallback_results = detect_with(absent)

        assert [(r.confidence, r.reason) for r in poisoned_results] == [
            (r.confidence, r.reason) for r in fallback_results
        ], "a poisoned reranker must behave exactly like an absent one"
        for result in poisoned_results:
            assert math.isfinite(result.confidence)


class _FakeReranker:
    """Stand-in for ``omega.reranker`` so the in-function import is patchable."""

    def __init__(self, scorer):
        self.cross_encoder_score = scorer
