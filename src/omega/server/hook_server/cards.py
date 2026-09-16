"""Intelligence Card formatters -- structured output for OMEGA hook system.

Replaces ad-hoc [MEMORY], [LEARNED], [RECALL] tags with structured [OMEGA ...]
cards that adapt their verbosity to session complexity.

Card types:
    MEMORY     -- Surfaced memories relevant to current file
    DECISIONS  -- Prior decision trail for context
    LEARNED    -- Auto-captured insight from assistant response
    WARNING    -- Known error patterns or advisory warnings
    SESSION    -- Session summary with outcome stats
"""

from enum import Enum
from typing import List, Optional


class TransparencyLevel(Enum):
    """Adaptive transparency level driven by session complexity."""

    MINIMAL = "minimal"
    NORMAL = "normal"
    VERBOSE = "verbose"


# -- Complexity -> transparency mapping ------------------------------------

_INTENT_WEIGHTS = {
    "coding": 2.0,
    "logic": 2.0,
    "exploration": 1.0,
    "creative": 0.5,
}


def compute_transparency(
    files_edited: int = 0,
    error_count: int = 0,
    intent: str = "",
    has_prior_decisions: bool = False,
) -> TransparencyLevel:
    """Compute transparency level from session complexity score.

    Formula: files_edited * 2 + error_count * 1.5 + intent_weight + has_prior_decisions
    """
    score = (
        files_edited * 2.0
        + error_count * 1.5
        + _INTENT_WEIGHTS.get(intent, 0.0)
        + (1.0 if has_prior_decisions else 0.0)
    )
    if score < 3.0:
        return TransparencyLevel.MINIMAL
    if score < 8.0:
        return TransparencyLevel.NORMAL
    return TransparencyLevel.VERBOSE


# -- Card formatters -------------------------------------------------------


def format_memory_card(
    memories: list[dict],
    filename: str,
    level: TransparencyLevel = TransparencyLevel.NORMAL,
    linked: Optional[list[dict]] = None,
) -> list[str]:
    """Format surfaced memories as an [OMEGA MEMORY] card.

    MINIMAL: top 1 memory (>= 0.85 relevance only).
    NORMAL:  top 2-3 memories.
    VERBOSE: all memories + linked.
    """
    if not memories:
        return []

    lines: list[str] = []

    if level == TransparencyLevel.MINIMAL:
        top = [m for m in memories if m.get("relevance", 0) >= 0.85][:1]
        if not top:
            return []
        lines.append(f"\n[OMEGA MEMORY] Context for {filename}:")
        for m in top:
            lines.append(_format_memory_line(m))
    elif level == TransparencyLevel.NORMAL:
        shown = memories[:3]
        lines.append(f"\n[OMEGA MEMORY] Relevant context for {filename}:")
        for m in shown:
            lines.append(_format_memory_line(m))
    else:  # VERBOSE
        lines.append(f"\n[OMEGA MEMORY] Full context for {filename}:")
        for m in memories:
            lines.append(_format_memory_line(m))
        if linked:
            for ln in linked:
                etype = (ln.get("metadata") or {}).get("event_type", "memory")
                preview = ln.get("content", "")[:100].replace("\n", " ")
                lines.append(f"  [linked] {etype}: {preview}")

    return lines


def format_decision_card(
    decisions: list[dict],
    level: TransparencyLevel = TransparencyLevel.NORMAL,
) -> list[str]:
    """Format prior decisions as an [OMEGA DECISIONS] card.

    MINIMAL: suppressed entirely.
    NORMAL:  latest decision only.
    VERBOSE: full trail (up to 5).
    """
    if not decisions or level == TransparencyLevel.MINIMAL:
        return []

    lines: list[str] = []

    if level == TransparencyLevel.NORMAL:
        d = decisions[0]
        preview = d.get("content", "")[:120].replace("\n", " ")
        lines.append(f"[OMEGA DECISIONS] Prior: {preview}")
    else:  # VERBOSE
        lines.append("[OMEGA DECISIONS] Decision trail:")
        for d in decisions[:5]:
            preview = d.get("content", "")[:120].replace("\n", " ")
            age = d.get("age", "")
            age_part = f" ({age})" if age else ""
            lines.append(f"  - {preview}{age_part}")

    return lines


def format_learning_card(
    matched_type: str,
    content: str,
    confidence: float = 0.0,
    level: TransparencyLevel = TransparencyLevel.NORMAL,
) -> list[str]:
    """Format an auto-capture event as an [OMEGA LEARNED] card.

    MINIMAL: suppressed (capture still happens silently).
    NORMAL+: one-liner with confidence.
    """
    if level == TransparencyLevel.MINIMAL:
        return []

    preview = content[:80].replace("\n", " ").strip()
    conf_str = f" ({confidence:.0%} confidence)" if confidence > 0 else ""
    return [f"[OMEGA LEARNED] {matched_type}: {preview}{conf_str}"]


def format_warning_card(
    warnings: list[dict],
    level: TransparencyLevel = TransparencyLevel.NORMAL,
) -> list[str]:
    """Format advisory warnings as [OMEGA WARNING] cards.

    MINIMAL: only warnings with >= 3 occurrences.
    NORMAL:  any known warning.
    VERBOSE: full detail including memory IDs.
    """
    if not warnings:
        return []

    lines: list[str] = []

    for w in warnings:
        count = w.get("count", 1)
        message = w.get("message", "")
        tag = w.get("tag", "WARNING")

        if level == TransparencyLevel.MINIMAL and count < 3:
            continue

        if level == TransparencyLevel.VERBOSE:
            mem_ids = w.get("memory_ids", [])
            id_str = f" (ids: {', '.join(mid[:8] for mid in mem_ids)})" if mem_ids else ""
            lines.append(f"[OMEGA WARNING] [{tag}] {message}{id_str}")
        else:
            lines.append(f"[OMEGA WARNING] {message}")

    return lines


def format_session_card(
    captured: int = 0,
    surfaced_count: int = 0,
    surfaced_unique_ids: int = 0,
    surfaced_unique_files: int = 0,
    diff_correlated: int = 0,
    diff_total: int = 0,
    type_breakdown: Optional[dict] = None,
    top_decisions: Optional[list[str]] = None,
) -> list[str]:
    """Format session summary as an [OMEGA SESSION] card.

    Always verbose. Includes diff-correlation stats when available.
    """
    lines: list[str] = []

    # Header
    if captured > 0:
        lines.append(f"[OMEGA SESSION] Complete: {captured} captured, {surfaced_count} surfaced")
    else:
        lines.append(f"[OMEGA SESSION] Complete: {surfaced_count} memories surfaced")

    # Type breakdown
    if type_breakdown:
        _LABELS = {
            "decision": ("decision", "decisions"),
            "lesson_learned": ("lesson", "lessons"),
            "error_pattern": ("error", "errors"),
        }
        parts = []
        other = 0
        for key, (singular, plural) in _LABELS.items():
            n = type_breakdown.get(key, 0)
            if n:
                parts.append(f"{n} {plural if n > 1 else singular}")
        for key, n in type_breakdown.items():
            if key not in _LABELS and n > 0:
                other += n
        if other:
            parts.append(f"{other} other")
        if parts:
            lines.append(f"  Stored: {', '.join(parts)}")

    # Key decisions
    if top_decisions:
        lines.append(f"  Key: {'; '.join(top_decisions)}")

    # Diff correlation stats
    if diff_total > 0:
        pct = (diff_correlated / diff_total * 100) if diff_total else 0
        lines.append(f"  Outcome: {diff_correlated}/{diff_total} surfaced memories correlated with commits ({pct:.0f}%)")

    # Recall stats
    if surfaced_unique_ids > 0:
        file_s = "s" if surfaced_unique_files != 1 else ""
        lines.append(
            f"  Recalled: {surfaced_unique_ids} unique memories across {surfaced_unique_files} file{file_s}"
        )

    return lines


# -- Internal helpers ------------------------------------------------------


def _format_memory_line(m: dict) -> str:
    """Format a single memory result into a card line."""
    score = m.get("relevance", 0.0)
    etype = m.get("event_type", "memory")
    preview = m.get("content", "")[:120].replace("\n", " ")
    nid = m.get("id", "")[:8]
    age = m.get("age", "")
    age_part = f" ({age})" if age else ""

    # High-relevance cross-session memories
    is_remembered = m.get("is_remembered", False)
    prefix = "[REMEMBERED]" if is_remembered else f"[{score:.0%}]"
    return f"  {prefix} {etype}{age_part}: {preview} (id:{nid})"


# -- Compact card formatters (user-facing [OMEGA] blocks) ------------------
# These produce single-string cards that Claude surfaces directly to the user.


def _truncate(text: str, max_len: int = 120) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _days_label(days: int) -> str:
    if days == 0:
        return "today"
    if days == 1:
        return "1d ago"
    return f"{days}d ago"


def format_compact_memory_card(
    content: str,
    verified_count: int = 0,
    last_accessed_days: int = 0,
    project: Optional[str] = None,
) -> str:
    """Format a compact Memory Card -- surfaces when Claude uses a retrieved memory."""
    meta_parts = []
    if verified_count > 0:
        meta_parts.append(f"verified {verified_count}x")
    meta_parts.append(f"last used {_days_label(last_accessed_days)}")
    if project:
        meta_parts.append(f"project: {project}")
    meta = " | ".join(meta_parts)
    return f'[OMEGA] Used: "{_truncate(content)}"\n  {meta}'


def format_decision_trail_card(
    topic: str,
    decisions: List[dict],
) -> str:
    """Format a Decision Trail Card -- surfaces before decisions on topics with history."""
    if not decisions:
        return ""
    lines = [f'[OMEGA] Prior decisions on "{topic}":']
    for d in decisions[:5]:  # Cap at 5 most recent
        status = d.get("status", "active")
        lines.append(f'  -> {d["date"]}: {_truncate(d["content"], 80)} ({status})')
    lines.append("  ! New decision should be consistent with these.")
    return "\n".join(lines)


def format_compact_learning_card(
    content: str,
    event_type: str = "lesson_learned",
) -> str:
    """Format a Learning Card -- surfaces when OMEGA auto-captures from Claude's response."""
    if event_type == "decision":
        return f'[OMEGA] Captured decision: "{_truncate(content)}"\n  auto-captured | tracking for consistency'
    if event_type == "advisor_insight":
        return f'[OMEGA] Captured insight: "{_truncate(content)}"\n  auto-captured | will surface in related contexts'
    return f'[OMEGA] Learned: "{_truncate(content)}"\n  auto-captured | will verify in future sessions'


def format_compact_warning_card(
    filename: str,
    error_count: int,
    pattern: str,
    last_fix_session: Optional[str] = None,
    last_fix_date: Optional[str] = None,
) -> str:
    """Format a Warning Card -- surfaces proactively for files with known issues."""
    error_word = "error" if error_count == 1 else "errors"
    lines = [f"[OMEGA] Warning: Known issues in {filename}:"]
    lines.append(f'  {error_count} prior {error_word} related to "{pattern}"')
    if last_fix_session and last_fix_date:
        lines.append(f"  Last fix: session {last_fix_session[:7]}, {last_fix_date}")
    elif last_fix_date:
        lines.append(f"  Last fix: {last_fix_date}")
    return "\n".join(lines)


def format_session_summary_card(
    memories_surfaced: int,
    memories_used: int,
    lessons_captured: int,
    contradictions: int,
    repeated_mistakes: int,
    verified_lessons_this_week: int,
    dedup_count: int = 0,
    evolution_count: int = 0,
) -> str:
    """Format a Session Summary Card -- surfaces at session end."""
    lines = ["[OMEGA] Session intelligence:"]
    lines.append(
        f"  {memories_surfaced} memories surfaced | {memories_used} used"
        f" | {lessons_captured} new lessons captured"
    )
    lines.append(
        f"  {contradictions} contradictions detected"
        f" | {repeated_mistakes} repeated mistakes"
    )
    if dedup_count > 0 or evolution_count > 0:
        pipeline_parts = []
        if dedup_count > 0:
            pipeline_parts.append(f"{dedup_count} deduped (saved)")
        if evolution_count > 0:
            pipeline_parts.append(f"{evolution_count} evolved")
        lines.append(f"  Pipeline: {' | '.join(pipeline_parts)}")
    if verified_lessons_this_week > 0:
        lines.append(f"  Learning rate: +{verified_lessons_this_week} verified lessons this week")
    return "\n".join(lines)
