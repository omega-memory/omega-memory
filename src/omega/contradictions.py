"""
OMEGA Contradiction Detection — the core heuristic engine (pure functions).

Stateless scoring engine with NO side effects. Detects when content contradicts
candidates using four signals: negation asymmetry, antonym presence, preference
value changes, and temporal override markers. Uses cross-encoder similarity as
a gate, falling back to Jaccard overlap.

Called by:
- sqlite_store._check_contradictions() — inside store.store() (Phase 3)
- reflect.find_contradictions() — query-time pairwise audit

NOT called by conflicts.py, which uses its own lighter-weight implementation
tuned for the pre-storage fast path (Phase 2.5).

Usage:
    from omega.contradictions import detect_contradictions
    results = detect_contradictions("Alex prefers light mode", candidates)

See also:
- conflicts.py — pre-storage conflict gate with auto-resolve side effects (Phase 2.5)
- reflect.py — query-time pairwise audit using this module as its engine
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "detect_contradictions",
    "ContradictionResult",
]

logger = logging.getLogger("omega.contradictions")

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ContradictionResult:
    """A detected contradiction between new content and an existing memory."""

    candidate_index: int
    candidate_content: str
    confidence: float  # 0.0–1.0
    reason: str  # Human-readable explanation
    similarity: float  # Cross-encoder similarity (normalized)
    signals: list[str] = field(default_factory=list)  # Which heuristics fired


# ---------------------------------------------------------------------------
# Negation / antonym patterns
# ---------------------------------------------------------------------------

# Words that flip meaning
_NEGATION_WORDS = frozenset({
    "not", "no", "never", "none", "neither", "nor",
    "don't", "doesn't", "didn't", "won't", "wouldn't",
    "can't", "cannot", "couldn't", "shouldn't", "isn't",
    "aren't", "wasn't", "weren't", "hasn't", "haven't",
    "hadn't",
})

# Common antonym pairs (bidirectional)
_ANTONYM_PAIRS = [
    ("always", "never"),
    ("true", "false"),
    ("enable", "disable"),
    ("enabled", "disabled"),
    ("yes", "no"),
    ("light", "dark"),
    ("on", "off"),
    ("allow", "deny"),
    ("allow", "block"),
    ("accept", "reject"),
    ("include", "exclude"),
    ("prefer", "avoid"),
    ("like", "dislike"),
    ("use", "avoid"),
    ("increase", "decrease"),
    ("add", "remove"),
    ("start", "stop"),
    ("open", "close"),
    ("show", "hide"),
    ("public", "private"),
    ("before", "after"),
]

# Build lookup: word → set of antonyms
_ANTONYM_MAP: dict[str, set[str]] = {}
for _a, _b in _ANTONYM_PAIRS:
    _ANTONYM_MAP.setdefault(_a, set()).add(_b)
    _ANTONYM_MAP.setdefault(_b, set()).add(_a)

# Patterns that extract key-value preferences
# e.g., "prefers dark mode", "uses vim", "default editor is vscode"
_PREFERENCE_PATTERNS = [
    re.compile(r"\b(?:prefer|prefers|preferred)\s+(\w+(?:\s+\w+)?)", re.IGNORECASE),
    re.compile(r"\b(?:use|uses|using)\s+(\w+(?:\s+\w+)?)", re.IGNORECASE),
    re.compile(r"\b(?:default|always)\s+(?:use|is)\s+(\w+(?:\s+\w+)?)", re.IGNORECASE),
    re.compile(r"\b(?:switched?\s+to|moved?\s+to|changed?\s+to)\s+(\w+(?:\s+\w+)?)", re.IGNORECASE),
]

# Temporal override signals — new info supersedes old
_TEMPORAL_OVERRIDE_PATTERNS = [
    re.compile(r"\b(?:now|currently|recently|today)\b", re.IGNORECASE),
    re.compile(r"\b(?:no longer|stopped|quit|switched)\b", re.IGNORECASE),
    re.compile(r"\b(?:used to|previously|formerly|was)\b", re.IGNORECASE),
    re.compile(r"\b(?:changed|updated|revised|corrected)\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------


def detect_contradictions(
    new_content: str,
    candidates: list[str],
    similarity_threshold: float = 0.3,
    contradiction_threshold: float = 0.4,
    similarity_scores: Optional[list[float]] = None,
) -> list[ContradictionResult]:
    """Detect contradictions between new content and existing memory candidates.

    Args:
        new_content: The new memory content about to be stored.
        candidates: List of existing memory content strings to check against.
        similarity_threshold: Minimum cross-encoder similarity to consider
            a candidate as potentially contradictory (0.0–1.0 after normalization).
        contradiction_threshold: Minimum contradiction confidence to include
            in results (0.0–1.0).
        similarity_scores: Pre-computed cross-encoder scores. If None,
            will attempt to compute them via the reranker module.

    Returns:
        List of ContradictionResult for candidates that exceed the
        contradiction threshold, sorted by confidence descending.
    """
    if not new_content or not candidates:
        return []

    # Step 1: Get similarity scores (cross-encoder or fallback)
    if similarity_scores is None:
        similarity_scores = _get_similarity_scores(new_content, candidates)

    if similarity_scores is None:
        # Cross-encoder unavailable — fall back to word-overlap similarity
        similarity_scores = _word_overlap_similarity(new_content, candidates)

    # Normalize similarity scores to [0, 1]
    sim_norm = _normalize_scores(similarity_scores)

    # Step 2: For each sufficiently similar candidate, check for contradiction
    results = []
    new_words = set(new_content.lower().split())
    new_lower = new_content.lower()

    for i, (candidate, sim) in enumerate(zip(candidates, sim_norm)):
        if sim < similarity_threshold:
            continue  # Not similar enough to be a contradiction

        signals = []
        cand_lower = candidate.lower()
        cand_words = set(cand_lower.split())

        # Signal 1: Negation asymmetry
        neg_score = _check_negation_asymmetry(new_lower, new_words, cand_lower, cand_words)
        if neg_score > 0:
            signals.append("negation")

        # Signal 2: Antonym presence
        ant_score = _check_antonyms(new_words, cand_words)
        if ant_score > 0:
            signals.append("antonym")

        # Signal 3: Preference value change
        pref_score = _check_preference_change(new_lower, cand_lower)
        if pref_score > 0:
            signals.append("preference_change")

        # Signal 4: Temporal override
        temp_score = _check_temporal_override(new_lower, cand_lower)
        if temp_score > 0:
            signals.append("temporal_override")

        if not signals:
            continue

        # Compute final contradiction confidence
        # Base: weighted combination of signal scores
        signal_score = (
            neg_score * 0.35
            + ant_score * 0.25
            + pref_score * 0.25
            + temp_score * 0.15
        )

        # Boost by similarity — high similarity + contradiction signals = strong contradiction
        confidence = min(1.0, signal_score * (0.5 + sim * 0.5))

        if confidence < contradiction_threshold:
            continue

        reason = _build_reason(signals, new_content, candidate)

        results.append(ContradictionResult(
            candidate_index=i,
            candidate_content=candidate,
            confidence=round(confidence, 3),
            reason=reason,
            similarity=round(sim, 3),
            signals=signals,
        ))

    # Sort by confidence descending
    results.sort(key=lambda r: r.confidence, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Signal checkers (each returns 0.0–1.0)
# ---------------------------------------------------------------------------


def _check_negation_asymmetry(
    new_lower: str, new_words: set, cand_lower: str, cand_words: set
) -> float:
    """Check if one text negates the other.

    Returns a score 0.0–1.0 based on negation word asymmetry.
    """
    new_negs = new_words & _NEGATION_WORDS
    cand_negs = cand_words & _NEGATION_WORDS

    # Asymmetric negation: one has negation words, the other doesn't
    if bool(new_negs) != bool(cand_negs):
        # Check that the non-negation words overlap (same topic)
        shared_content = (new_words - _NEGATION_WORDS) & (cand_words - _NEGATION_WORDS)
        if len(shared_content) >= 2:
            return 0.8
        elif len(shared_content) >= 1:
            return 0.5
    # Both have negation but different ones
    elif new_negs and cand_negs and new_negs != cand_negs:
        return 0.3

    return 0.0


def _check_antonyms(new_words: set, cand_words: set) -> float:
    """Check if the texts contain antonym pairs.

    Returns a score 0.0–1.0 based on the number and strength of antonym matches.
    """
    score = 0.0
    for word in new_words:
        antonyms = _ANTONYM_MAP.get(word)
        if antonyms and antonyms & cand_words:
            score = max(score, 0.7)
            # Check if the antonym pair is the main differentiator
            non_antonym_overlap = (new_words - {word}) & (cand_words - antonyms)
            if len(non_antonym_overlap) >= 2:
                score = 0.9  # Same context, opposite value
                break
    return score


def _check_preference_change(new_lower: str, cand_lower: str) -> float:
    """Check if both texts express preferences for different values.

    Returns 0.0–1.0 based on whether a preference value changed.
    """
    new_prefs = set()
    cand_prefs = set()

    for pattern in _PREFERENCE_PATTERNS:
        for m in pattern.finditer(new_lower):
            new_prefs.add(m.group(1).strip().lower())
        for m in pattern.finditer(cand_lower):
            cand_prefs.add(m.group(1).strip().lower())

    if not new_prefs or not cand_prefs:
        return 0.0

    # Check if any preference value from one is a substring/prefix of the other
    # (handles "vim" matching "vim for" as the same preference)
    def _prefs_overlap(set_a: set, set_b: set) -> bool:
        for a in set_a:
            for b in set_b:
                if a == b or a.startswith(b) or b.startswith(a):
                    return True
        return False

    # Same preference verb but different values
    if new_prefs and cand_prefs and not _prefs_overlap(new_prefs, cand_prefs):
        return 0.8

    return 0.0


def _check_temporal_override(new_lower: str, cand_lower: str) -> float:
    """Check for temporal override signals.

    Returns 0.0–1.0 based on presence of temporal markers.
    """
    new_temporal = sum(1 for p in _TEMPORAL_OVERRIDE_PATTERNS if p.search(new_lower))
    cand_temporal = sum(1 for p in _TEMPORAL_OVERRIDE_PATTERNS if p.search(cand_lower))

    if new_temporal > 0 and cand_temporal == 0:
        return 0.6  # New memory has temporal markers, old doesn't
    elif new_temporal > 0 and cand_temporal > 0:
        return 0.4  # Both have temporal markers
    return 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_similarity_scores(query: str, passages: list[str]) -> Optional[list[float]]:
    """Get cross-encoder similarity scores, or None if unavailable.

    A non-finite score counts as unavailable rather than as a similarity.
    ``cross_encoder_score`` returns raw model logits with no finiteness
    sanitation, and a single NaN reaching the caller is worse here than on the
    read path: NaN loses every comparison, so it passes the similarity filter,
    ``min(1.0, NaN)`` evaluates to ``1.0``, and the result is a
    maximum-confidence contradiction written durably to the store.  Returning
    None instead routes the whole batch through the word-overlap fallback the
    module already defines, so detection degrades rather than fabricating.
    """
    try:
        from omega.reranker import cross_encoder_score
        scores = cross_encoder_score(query, passages)
    except ImportError:
        return None
    except Exception as e:
        logger.debug("Cross-encoder scoring failed: %s", e)
        return None
    if scores is not None and not all(math.isfinite(s) for s in scores):
        logger.debug("Cross-encoder returned a non-finite score; using word overlap")
        return None
    return scores


def _word_overlap_similarity(text_a: str, candidates: list[str]) -> list[float]:
    """Fallback similarity using Jaccard word overlap."""
    words_a = set(text_a.lower().split())
    if not words_a:
        return [0.0] * len(candidates)

    scores = []
    for cand in candidates:
        words_b = set(cand.lower().split())
        if not words_b:
            scores.append(0.0)
            continue
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        scores.append(intersection / union if union > 0 else 0.0)
    return scores


def _normalize_scores(scores: list[float]) -> list[float]:
    """Normalize a list of scores to [0, 1] range.

    Scores arrive either from the cross-encoder or from a caller, so the
    non-finite guard is repeated here for the caller-supplied case.  ``inf``
    satisfies a bare ``rng <= 0`` test and NaN fails it, so both used to reach
    the division and emit NaN.  An unusable batch normalises to 0.0, below any
    similarity threshold, which suppresses detection rather than inventing it.
    """
    if not scores:
        return []
    if not all(math.isfinite(s) for s in scores):
        return [0.0] * len(scores)
    min_s = min(scores)
    max_s = max(scores)
    rng = max_s - min_s
    if rng <= 0:
        return [0.5] * len(scores)
    return [(s - min_s) / rng for s in scores]


def _build_reason(signals: list[str], new_content: str, candidate: str) -> str:
    """Build a human-readable reason string from fired signals."""
    parts = []
    if "negation" in signals:
        parts.append("negation detected (one affirms, the other denies)")
    if "antonym" in signals:
        parts.append("opposing terms found")
    if "preference_change" in signals:
        parts.append("different preference values")
    if "temporal_override" in signals:
        parts.append("temporal update detected")
    return "; ".join(parts)
