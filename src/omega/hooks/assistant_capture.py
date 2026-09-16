#!/usr/bin/env python3
"""OMEGA Stop hook — capture high-value assistant responses.

Fires on every Stop event. Detects fix, decision, lesson, and recommendation
patterns in ``last_assistant_message`` and stores them via bridge.auto_capture.

``run(payload)`` is the single implementation. The hook daemon calls it
in-process; ``main()`` wraps it for the standalone fallback path.
"""
import json
import logging
import re
import sys

from omega.hooks._output import emit

logger = logging.getLogger("omega.hooks.assistant_capture")

FIX_PATTERNS = [
    r"the (?:fix|issue|problem|bug) was\b",
    r"root cause (?:was|is)\b",
    r"the error (?:occurred|happens|was caused) because\b",
    r"fixed (?:by|this by)\b",
]

DECISION_PATTERNS = [
    r"(?:decided|choosing) to\b",
    r"going with\b",
    r"switched to\b",
    r"using \S+ instead of\b",
    r"chose \S+ because\b",
]

LESSON_PATTERNS = [
    r"(?:note|notice) that\b",
    r"important:\s",
    r"be careful\b",
    r"gotcha:\s",
    r"caveat:\s",
    r"key takeaway\b",
]

MIN_MESSAGE_LENGTH = 200
MIN_CONTENT_CHARS = 40
MIN_CONTENT_WORDS = 8

_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```", re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_INSIGHT_OPEN_RE = re.compile(r"[★✦]\s*Insight\s*─+", re.IGNORECASE)
_INSIGHT_CLOSE_RE = re.compile(r"─{10,}")

# Per-session capture cap. Keyed by session because the daemon serves many
# sessions from one process.
MAX_CAPTURES = 10
_captures_by_session: dict[str, int] = {}


def _clean(text):
    return _FENCED_CODE_RE.sub("", text).strip()


def _find_match(text, patterns):
    for pat in patterns:
        compiled = re.compile(pat, re.IGNORECASE)
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            sentence = sentence.strip()
            if compiled.search(sentence):
                if len(sentence) >= MIN_CONTENT_CHARS and len(sentence.split()) >= MIN_CONTENT_WORDS:
                    return sentence
    return None


def _extract_insight_blocks(text):
    """Extract ★ Insight delimited blocks from assistant text."""
    blocks = []
    search_start = 0
    while True:
        open_match = _INSIGHT_OPEN_RE.search(text, search_start)
        if not open_match:
            break
        body_start = open_match.end()
        close_match = _INSIGHT_CLOSE_RE.search(text, body_start)
        if not close_match:
            body = text[body_start:body_start + 2000].strip()
        else:
            body = text[body_start:close_match.start()].strip()
        if body and len(body) >= MIN_CONTENT_CHARS:
            blocks.append(body[:2000])
        search_start = close_match.end() if close_match else len(text)
    return blocks


def _store(content: str, event_type: str, metadata: dict, session_id: str, cwd: str) -> bool:
    """Store one capture; return True when it counted against the session cap."""
    try:
        from omega.bridge import auto_capture
    except ImportError:
        return False
    try:
        auto_capture(
            content=content,
            event_type=event_type,
            metadata=metadata,
            session_id=session_id,
            project=cwd,
        )
    except Exception:
        logger.warning("assistant_capture hook failed to store a %s", event_type, exc_info=True)
        return False
    _captures_by_session[session_id] = _captures_by_session.get(session_id, 0) + 1
    return True


def run(payload: dict) -> None:
    """Capture insight blocks or fix/decision/lesson sentences from one Stop payload."""
    message = payload.get("last_assistant_message", "")
    if not message or len(message) < MIN_MESSAGE_LENGTH:
        return

    session_id = payload.get("session_id", "")
    cwd = payload.get("cwd") or payload.get("project") or ""

    if _captures_by_session.get(session_id, 0) >= MAX_CAPTURES:
        return

    # Pre-pass: detect ★ Insight delimited blocks
    insight_blocks = _extract_insight_blocks(message)
    if insight_blocks:
        for block in insight_blocks:
            if _captures_by_session.get(session_id, 0) >= MAX_CAPTURES:
                break
            stored = _store(
                f"Insight: {block}",
                "advisor_insight",
                {"source": "assistant_capture_hook", "project": cwd, "capture_confidence": "high"},
                session_id,
                cwd,
            )
            if stored:
                preview = block[:80].replace("\n", " ").strip()
                emit(f"[LEARNED] insight: {preview}")
        return

    cleaned = _clean(message)
    if not cleaned:
        return

    # Try pattern groups in priority order
    for label, event_type, patterns in [
        ("fix", "lesson_learned", FIX_PATTERNS),
        ("decision", "decision", DECISION_PATTERNS),
        ("lesson", "lesson_learned", LESSON_PATTERNS),
    ]:
        content = _find_match(cleaned, patterns)
        if content:
            stored = _store(
                f"Assistant {label}: {content[:500]}",
                event_type,
                {"source": "assistant_capture_hook", "project": cwd},
                session_id,
                cwd,
            )
            if stored:
                preview = content[:80].replace("\n", " ").strip()
                emit(f"[LEARNED] {label}: {preview}")
            return


def main(data: dict | None = None) -> None:
    """Standalone entry point: read the hook payload from stdin when not given."""
    if data is None:
        try:
            raw = sys.stdin.read()
            if not raw.strip():
                return
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError, ValueError):
            return
    run(data)


if __name__ == "__main__":
    main()
