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
    "detect_update_signal",
    "distinguishing_tokens",
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
# Update signals — may a newer memory retire an older one?
# ---------------------------------------------------------------------------
#
# Embedding similarity cannot tell an update from a related-but-distinct
# memory. On the bug team's synthetic pairs (tests/fixtures/
# supersession_pairs.json) distinct pairs reached cosine 0.935 while genuine
# updates went as low as 0.753, so a similarity threshold alone retired 11 of
# 30 distinct pairs. Retirement therefore needs explicit evidence in the text.
# Each rule below names one kind of evidence; the rules are deliberately
# narrower than detect_contradictions(), whose preference-change signal fires
# on pairs such as "Use JWT access tokens ..." / "Use refresh tokens ...".

# Words that announce a change. Only counted when the older memory lacks the
# same marker and the newer one still talks about the older one's subject.
_UPDATE_MARKERS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bnow\b",
        r"\bno longer\b",
        r"\banymore\b",
        r"\bgoing forward\b",
        r"\bswitch(?:ed|ing)?\b",
        r"\bmov(?:ed|ing)\b",
        r"\bmigrat(?:ed|ing)\b",
        r"\bchang(?:ed|ing)\b",
        r"\breplac(?:ed|ing)\b",
        r"\bstop(?:ped|ping)?\b",
        r"\b(?:raised|lowered|increased|decreased|reduced|bumped)\b",
    )
)

# "X instead of Y", "prefers X over Y": Y must be something the older memory said.
_REPLACEMENT = re.compile(
    r"\b(?:instead of|rather than|in favou?r of|in place of"
    r"|prefer(?:s|red)?\b[^.;]*?\bover)\s+(?:the |a |an )?([a-z0-9][\w.+/-]*)"
)

_UPDATE_NEGATIONS = frozenset({
    "not", "no", "never", "don't", "doesn't", "didn't", "won't", "can't",
    "cannot", "shouldn't", "isn't", "aren't", "wasn't", "weren't", "mustn't",
})

_UPDATE_ANTONYM_PAIRS = (
    ("on", "off"), ("enable", "disable"), ("enabled", "disabled"),
    ("true", "false"), ("light", "dark"), ("allow", "deny"),
    ("always", "never"), ("yes", "no"), ("first", "last"),
    ("before", "after"), ("include", "exclude"), ("accept", "reject"),
    ("public", "private"), ("show", "hide"), ("open", "closed"),
)
_UPDATE_ANTONYMS: dict[str, set[str]] = {}
for _a, _b in _UPDATE_ANTONYM_PAIRS:
    _UPDATE_ANTONYMS.setdefault(_a, set()).add(_b)
    _UPDATE_ANTONYMS.setdefault(_b, set()).add(_a)

_POLARITY_WORDS = _UPDATE_NEGATIONS | set(_UPDATE_ANTONYMS)

_DIGIT_RUN = re.compile(r"\d+(?:[.,:]\d+)*")
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
    "has", "have", "had", "will", "into", "onto", "its", "our", "your", "their",
    "but", "not", "all", "any", "can", "too", "via", "per",
})

# Share of the older memory's content words the newer one must repeat before
# a bare change marker counts: "switched" alone says something changed, not
# that *this* memory changed.
_MARKER_ANCHOR_OVERLAP = 0.5
# Share of words (after removing the swapped pair) two texts must share for an
# antonym swap to count as the same statement with the opposite value.
_ANTONYM_CONTEXT_OVERLAP = 0.6


def detect_update_signal(new_content: str, old_content: str) -> Optional[str]:
    """Name the explicit evidence that ``new_content`` updates ``old_content``.

    Returns one of ``"value_change"``, ``"update_marker"``, ``"negation"``,
    ``"antonym"`` or ``"replacement"``, or None when the text carries no such
    evidence. None does not mean the memories are unrelated, only that nothing
    in the text justifies retiring the older one automatically.
    """
    new_words = _words(new_content)
    old_words = _words(old_content)
    if not new_words or not old_words or new_words == old_words:
        return None

    if _masked_digits(new_words) == _masked_digits(old_words):
        return "value_change"

    new_lower = new_content.lower()
    old_lower = old_content.lower()
    if any(m.search(new_lower) and not m.search(old_lower) for m in _UPDATE_MARKERS):
        if _content_overlap(old_words, new_words) >= _MARKER_ANCHOR_OVERLAP:
            return "update_marker"

    if _negates(new_words, old_words) or _negates(old_words, new_words):
        return "negation"

    if _antonym_swap(set(new_words), set(old_words)):
        return "antonym"

    replaced = _REPLACEMENT.search(new_lower)
    if replaced and replaced.group(1).strip(".,;:") in set(old_words):
        return "replacement"

    return None


def distinguishing_tokens(text: str, include_numbers: bool = True) -> frozenset[str]:
    """Numbers and polarity words in ``text``.

    Word-overlap similarity treats these as noise because they are short and
    change little of the wording, yet each one flips what a statement says:
    "100" vs "300", "deploy" vs "not deploy", "on" vs "off". Two texts whose
    distinguishing tokens differ are not duplicates of each other.
    """
    tokens = {w for w in _words(text) if w in _POLARITY_WORDS}
    if include_numbers:
        tokens.update(_DIGIT_RUN.findall(text))
    return frozenset(tokens)


def _words(text: str) -> list[str]:
    """Lowercase words with surrounding punctuation stripped."""
    stripped = (w.strip(".,;:!?()[]{}\"'`") for w in text.lower().split())
    return [w for w in stripped if w]


def _masked_digits(words: list[str]) -> list[str]:
    return [_DIGIT_RUN.sub("#", w) for w in words]


def _content_overlap(old_words: list[str], new_words: list[str]) -> float:
    """Share of the older text's content words that the newer text repeats."""
    old_content = {w for w in old_words if len(w) >= 2 and w not in _STOPWORDS}
    if not old_content:
        return 0.0
    return len(old_content & set(new_words)) / len(old_content)


def _negates(negated: list[str], plain: list[str]) -> bool:
    """True when ``negated`` denies a phrase that ``plain`` states outright."""
    if not set(negated) & _UPDATE_NEGATIONS or set(plain) & _UPDATE_NEGATIONS:
        return False
    for i, word in enumerate(negated):
        if word in _UPDATE_NEGATIONS:
            phrase = negated[i + 1:i + 3]
            if phrase and _contains_sequence(plain, phrase):
                return True
    return False


def _contains_sequence(words: list[str], phrase: list[str]) -> bool:
    n = len(phrase)
    return any(words[i:i + n] == phrase for i in range(len(words) - n + 1))


def _antonym_swap(new_set: set[str], old_set: set[str]) -> bool:
    """True when the texts match except for one word flipped to its opposite."""
    for word in new_set - old_set:
        for opposite in _UPDATE_ANTONYMS.get(word, ()):
            if opposite in old_set and opposite not in new_set:
                rest_new = new_set - {word}
                rest_old = old_set - {opposite}
                union = rest_new | rest_old
                if union and len(rest_new & rest_old) / len(union) >= _ANTONYM_CONTEXT_OVERLAP:
                    return True
    return False


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
