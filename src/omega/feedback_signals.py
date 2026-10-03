"""A bounded feedback history in a memory's metadata.

Every feedback event used to be appended to ``metadata["feedback_signals"]``
and never removed, and each append rewrote the whole list. A memory that
surfaced often carried thousands of entries, so the list came to dominate the
memories table and recording feedback slowed down with every signal.

The metadata now keeps the most recent signals plus a running count per
rating in ``metadata["feedback_counts"]``. ``feedback_score`` is unchanged:
it was always a running total.
"""

from typing import Any, Dict, List

__all__ = [
    "FEEDBACK_SIGNALS_KEPT",
    "add_feedback_signal",
    "cap_feedback_signals",
    "feedback_signal_total",
]

FEEDBACK_SIGNALS_KEPT = 20


def _count_ratings(signals: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for signal in signals:
        rating = signal.get("rating") or "unknown"
        counts[rating] = counts.get(rating, 0) + 1
    return counts


def add_feedback_signal(meta: Dict[str, Any], signal: Dict[str, Any]) -> None:
    """Count ``signal`` in ``meta`` and keep it among the most recent signals."""
    signals = meta.setdefault("feedback_signals", [])
    if "feedback_counts" not in meta:
        # Written before counts existed: the list is still the full history.
        meta["feedback_counts"] = _count_ratings(signals)
    counts = meta["feedback_counts"]
    rating = signal.get("rating") or "unknown"
    counts[rating] = counts.get(rating, 0) + 1
    signals.append(signal)
    del signals[:-FEEDBACK_SIGNALS_KEPT]


def cap_feedback_signals(meta: Dict[str, Any]) -> bool:
    """Trim an over-long feedback history in place, counting it first.

    Returns True if ``meta`` changed. Used by the schema migration.
    """
    signals = meta.get("feedback_signals")
    if not isinstance(signals, list) or len(signals) <= FEEDBACK_SIGNALS_KEPT:
        return False
    if "feedback_counts" not in meta:
        meta["feedback_counts"] = _count_ratings(signals)
    del signals[:-FEEDBACK_SIGNALS_KEPT]
    return True


def feedback_signal_total(meta: Dict[str, Any]) -> int:
    """How many feedback signals a memory has received in all."""
    counts = meta.get("feedback_counts")
    if counts is not None:
        return sum(counts.values())
    return len(meta.get("feedback_signals", []))
