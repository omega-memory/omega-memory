#!/usr/bin/env python3
"""OMEGA UserPromptSubmit hook — Auto-capture decisions and lessons from user prompts.

Fires on every user prompt. Detects decision and lesson patterns and stores them
as 'decision' or 'lesson_learned' event type in OMEGA memory. Uses conservative
matching to avoid noise.

``run(payload)`` is the single implementation. The hook daemon calls it
in-process; ``main()`` wraps it for the standalone fallback path.
"""
import json
import logging
import re
import sys

from omega.hooks._output import emit

logger = logging.getLogger("omega.hooks.auto_capture")


# Decision indicators (case-insensitive patterns)
DECISION_PATTERNS = [
    r"\blet'?s?\s+(?:go\s+with|use|switch\s+to|stick\s+with|move\s+to)\b",
    r"\bi\s+(?:decided?|chose|picked|went\s+with|prefer)\b",
    r"\bwe\s+(?:should|will|are\s+going\s+to)\s+(?:use|go\s+with|switch|adopt|implement)\b",
    r"\b(?:decision|approach|strategy):\s*\S",
    r"\binstead\s+of\s+\S+[,\s]+(?:use|let'?s|we'?ll)\b",
    r"\bfrom\s+now\s+on\b",
    r"\bremember\s+(?:that|this)\b",
]

# Lesson indicators (case-insensitive patterns)
LESSON_PATTERNS = [
    r"\bi\s+learned\s+that\b",
    r"\bturns?\s+out\b",
    r"\bthe\s+trick\s+is\b",
    r"\bnote\s+to\s+self\b",
    r"\btil\b|\btoday\s+i\s+learned\b",
    r"\bthe\s+fix\s+was\b",
    r"\bthe\s+problem\s+was\b",
    r"\bdon'?t\s+forget\b",
    r"\bimportant:\s*\S",
    r"\bkey\s+(?:insight|takeaway|learning)\b",
    r"\bnever\s+(?:again|do|use)\b",
    r"\balways\s+(?:make\s+sure|remember|check)\b",
]

# Minimum prompt length to avoid matching on short commands
MIN_PROMPT_LENGTH = 20

# Maximum prompts to store per session (avoid runaway storage). Keyed by
# session because the daemon serves many sessions from one process.
MAX_CAPTURES_PER_SESSION = 20
_captures_by_session: dict[str, int] = {}


def _summarize_content(prompt: str, max_len: int = 60) -> str:
    """Extract a concise summary from the prompt for the echo line."""
    # Strip common prefixes like "Decision: " or "Lesson: "
    text = re.sub(r"^(Decision|Lesson):\s*", "", prompt, flags=re.IGNORECASE).strip()
    # Take first sentence or first max_len chars
    first_sentence = re.split(r"[.!?\n]", text)[0].strip()
    if len(first_sentence) <= max_len:
        return first_sentence
    return first_sentence[:max_len].rsplit(" ", 1)[0] + "..."


def _echo_capture(result: str, event_type: str, prompt: str) -> None:
    """Emit a 1-line capture confirmation visible to the user.

    bridge.auto_capture() reports what it did as a short string:
    - "Stored <id> ..."        → [OMEGA] Captured: decision — X
    - "Evolved <id> (#N)"      → [OMEGA] Memory evolved: decision updated (evolution #N) — X
    - Deduped / Reconfirmed / Blocked → silent
    """
    if not result:
        return

    summary = _summarize_content(prompt)

    if result.startswith("Evolved"):
        evo_match = re.search(r"\(#(\d+)\)", result)
        evo_num = evo_match.group(1) if evo_match else "?"
        emit(f"[OMEGA] Memory evolved: {event_type} updated (evolution #{evo_num}) — {summary}")
    elif result.startswith("Stored"):
        emit(f"[OMEGA] Captured: {event_type} — {summary}")


def _detect_decision(prompt: str) -> bool:
    """Check if prompt contains a decision pattern."""
    if len(prompt) < MIN_PROMPT_LENGTH:
        return False
    prompt_lower = prompt.lower()
    return any(re.search(pat, prompt_lower) for pat in DECISION_PATTERNS)


def _detect_lesson(prompt: str) -> bool:
    """Check if prompt contains a lesson/insight pattern."""
    if len(prompt) < MIN_PROMPT_LENGTH:
        return False
    prompt_lower = prompt.lower()
    return any(re.search(pat, prompt_lower) for pat in LESSON_PATTERNS)


def _capture(content: str, event_type: str, label: str, prompt: str, session_id: str, cwd: str) -> None:
    try:
        from omega.bridge import auto_capture
    except ImportError:
        return
    try:
        result = auto_capture(
            content=content,
            event_type=event_type,
            metadata={"source": "auto_capture_hook", "project": cwd},
            session_id=session_id,
            project=cwd,
        )
    except Exception:
        logger.warning("auto_capture hook failed to store a %s", label, exc_info=True)
        return
    _captures_by_session[session_id] = _captures_by_session.get(session_id, 0) + 1
    _echo_capture(result, label, prompt)


def run(payload: dict) -> None:
    """Capture a decision or lesson from one UserPromptSubmit payload."""
    prompt = payload.get("prompt", "")
    session_id = payload.get("session_id", "")
    cwd = payload.get("cwd") or payload.get("project") or ""

    if not prompt:
        return
    if _captures_by_session.get(session_id, 0) >= MAX_CAPTURES_PER_SESSION:
        return

    # Decision takes priority if both match
    if _detect_decision(prompt):
        _capture(f"Decision: {prompt[:500]}", "decision", "decision", prompt, session_id, cwd)
        return

    if _detect_lesson(prompt):
        # Lesson quality gate: min 60 chars, >= 8 words, substance validation
        if len(prompt) < 60 or len(prompt.split()) < 8:
            return
        _tech_signals = ["/", "`", "Error", "error", ".py", ".js", ".ts", "import ", "def ", "class "]
        if len(prompt) < 100 and not any(s in prompt for s in _tech_signals):
            return
        _capture(f"Lesson: {prompt[:500]}", "lesson_learned", "lesson", prompt, session_id, cwd)


def main(payload: dict | None = None) -> None:
    """Standalone entry point: read the hook payload from stdin when not given."""
    if payload is None:
        try:
            raw = sys.stdin.read()
            if not raw.strip():
                return
            payload = json.loads(raw)
        except (json.JSONDecodeError, OSError, ValueError):
            return
    run(payload)


if __name__ == "__main__":
    main()
